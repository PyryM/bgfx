// INPUTS/SETUP:
//   buffers:
//     0: lightBuffer [vec4f color, vec4f pos, vec4f rad]*nlights
//     1: outLightIndices [float]*(MAX_LIGHTS_PER_TILE*NUM_TILES)
//     2: s_depthMap [r32f "depth" buffer]
//
//  uniforms:
//     u_screenSize: vec2f (in pixels)
//     u_lightCount: vec4 (x: number of lights)
//     u_projectionInvMat: inverse camera projection matrix
//     u_viewMat: camera view matrix
//     u_dispatchParams: vec4f (x: n_tiles_x, y: n_tiles_y)
//
//  Dispatch:
//  	workGroupsX = (SCREEN_SIZE.x + (SCREEN_SIZE.x % 16)) / 16;
//    workGroupsY = (SCREEN_SIZE.y + (SCREEN_SIZE.y % 16)) / 16;
//    (workGroupsX, workGroupsY, 1)

// Largely adapted from https://github.com/GPUOpen-LibrariesAndSDKs/ForwardPlus11
// MIT License,
// Copyright (c) 2016 Advanced Micro Devices, Inc. All rights reserved.
// adapted to bgfx 2017 by Pyry Matikainen

// TILE_SIZE, MAX_LIGHTS_PER_TILE, {LIGHT_}
#include "tile_params.sh"
#include "bgfx_compute.sh"
#define NUM_THREADS_PER_TILE (TILE_SIZE*TILE_SIZE)

// sampler/buffer type uniforms
BUFFER_RO(lightBuffer, vec4, 0);
BUFFER_WR(outLightIndices, float, 1);
IMAGE2D_RO(s_depthMap, rgba32f, 2);

//SAMPLER2D(s_depthMap, 2);

uniform vec4 u_screenSize;
uniform vec4 u_lightCount;
uniform vec4 u_dispatchParams; // hlsl doesn't seem to make this available
uniform mat4 u_projectionInvMat;
uniform mat4 u_viewMat;

//------------------------------------------------------------------------------
// Helper Functions
//------------------------------------------------------------------------------
vec3 convertProjToView( vec4 p )
{
    p = mul( p, u_projectionInvMat );
    p /= p.w;
    return p.xyz;
}

// this creates the standard Hessian-normal-form plane equation from three points,
// except it is simplified for the case where the first point is the origin
vec4 createPlaneEquation( vec3 b, vec3 c )
{
    // normalize(cross( b-a, c-a )), except we know "a" is the origin
    // also, typically there would be a fourth term of the plane equation,
    // -(n dot a), except we know "a" is the origin
    // (so we set it to zero)
    return vec4(normalize(cross(b,c)), 0.0);
}



//------------------------------------------------------------------------------
// Group Shared Memory (aka local data share, or LDS)
//------------------------------------------------------------------------------
SHARED uint depthMin;
SHARED uint depthMax;

SHARED uint lightIdxCounter;
SHARED uint lightIdx[MAX_LIGHTS_PER_TILE];

SHARED vec4 planes[6];

// assume non-MSAA (see the full amd example for MSAA)
void calculateMinMaxDepth(ivec2 pixelpos)
{
    float depth = imageLoad(s_depthMap, pixelpos).r;
    uint z = floatBitsToUint(depth); // glsl builtin

    // TODO: get the bgfx macros for these to actually work
  	//atomicMin(depthMin, z);
  	//atomicMax(depthMax, z);
    InterlockedMin(depthMin, z);
  	InterlockedMax(depthMax, z);
}

//-----------------------------------------------------------------------------------------
// Light culling shader
//-----------------------------------------------------------------------------------------
// WEIRD: this macro specifically needs there to be no ; at the end
NUM_THREADS(TILE_SIZE, TILE_SIZE, 1)

void main()
{
    ivec2 tileIdx = ivec2(gl_WorkGroupID.xy);
    ivec2 tileCounts = ivec2(u_dispatchParams.xy);
    ivec2 pixelPos = ivec2(gl_GlobalInvocationID.xy);

    uint localIdxFlattened = gl_LocalInvocationIndex;

    if(localIdxFlattened == 0)
    {
    		viewProjection = mul(u_projectionMat, u_viewMat);
        minDepth = 0x7f7fffff;  // FLT_MAX as a uint
        maxDepth = 0;
        lightIdxCounter = 0;
    }
    barrier();

    // calculate the min and max depth for this tile,
    // to form the front and back of the frustum
    calculateMinMaxDepth(pixelPos);
    barrier();

    if(localIdxFlattened == 0)
    {
        float minZ = uintBitsToFloat( depthMin );
        float maxZ = uintBitsToFloat( depthMax );

        // construct frustum for this tile
        // four corners of the tile, clockwise from top-left
        // ASSUMING the screen is a whole number of tiles,
        vec2 bottomLeft = (gl_WorkGroupID.xy) / (u_dispatchParams.xy);
        vec2 topRight   = (gl_WorkGroupID.xy + vec2(1.0, 1.0)) / (u_dispatchParams.xy);
        // convert from [0,1] => [-1,1]
        bottomLeft = (bottomLeft * 2.0) - 1.0;
        topRight   = (topRight * 2.0) - 1.0;

        vec3 frustum0 = convertProjToView( vec4( bottomLeft.x, bottomLeft.y, 1.f, 1.f) );
        vec3 frustum1 = convertProjToView( vec4( topRight.x, bottomLeft.y, 1.f, 1.f) );
        vec3 frustum2 = convertProjToView( vec4( topRight.x, topRight.y, 1.f, 1.f) );
        vec3 frustum3 = convertProjToView( vec4( bottomLeft.x, topRight.y, 1.f, 1.f) );

        // create plane equations for the four sides of the frustum,
        // with the positive half-space outside the frustum (and remember,
        // view space is left handed, so use the left-hand rule to determine
        // cross product direction)
        planes[0] = createPlaneEquation( frustum0, frustum1 );
        planes[1] = createPlaneEquation( frustum1, frustum2 );
        planes[2] = createPlaneEquation( frustum2, frustum3 );
        planes[3] = createPlaneEquation( frustum3, frustum0 );

        // depth planes:
        // -z > minDepth  :=  -z - minDepth > 0.0
        // -z < maxDepth  :=   z + maxDepth > 0.0
        planes[4] = vec4(0.0, 0.0, -1.0, -minZ);
        planes[5] = vec4(0.0, 0.0,  1.0,  maxZ);

        // here's a trick:
        // we want to compute
        // plane dot (view * light_pos)
        // =
        // plane^T * view * light_pos
        // =
        // (plane^T * view) dot light_pos
        // this means we can avoid multiplying each
        // light position by the view matrix by
        // instead multiplying the planes ahead of time
        // by the view matrix transpose

        for(uint i = 0; i < 6; ++i)
        {
          planes[i] = mul(planes[i], u_viewMat);
        }
    }

    barrier();

    // loop over the lights and do a sphere vs. frustum intersection test
    uint numLights = u_lightCount.x;
    for(uint i = localIdxFlattened; i < numLights; i += NUM_THREADS_PER_TILE)
    {
        if(i >= numLights) {
          break;
        }

        // we aren't using fancy structured buffers, so we need to stride by how
    		// many vec4's in each light
    		uint rawIndex = i * LIGHT_STRIDE;
        vec4 position = vec4(lightBuffer[rawIndex + LIGHT_POS_OFFSET].xyz, 1.0);
    		float negativeRadius =  -1.0 * lightBuffer[rawIndex + LIGHT_RAD_OFFSET].x;
        float distance = 0.0;

    		for (uint j = 0; j < 6; ++j) {
    			distance = dot(position, planes[j]);
    			if (distance < negativeRadius) {
    				break;
    			}
    		}

    		if (distance >= negativeRadius) {
    			uint offset;
    			InterlockedAdd(lightIdxCounter, 1, offset);
    			// TODO: fix bgfx macros atomicAdd(lightIdxCounter, 1);
    			lightIdx[offset] = int(i);
    		}
    }

    barrier();

    {   // write back
        uint tileIdxFlattened = tileIdx.x + tileIdx.y * tileCount.x;
        uint startOffset = MAX_LIGHTS_PER_TILE * tileIdxFlattened;

        for(uint i = localIdxFlattened; i < lightIdxCounter && i < MAX_LIGHTS_PER_TILE; i += NUM_THREADS_PER_TILE)
        {
            // per-tile list of light indices
            outLightIndices[startOffset + i] = lightIdx[i];
        }

        // add a sentinel if necessary (fewer than max lights)
        if( localIdxFlattened == 0 && lightIdxCounter < MAX_LIGHTS_PER_TILE)
        {
            // mark the end of each per-tile list with a sentinel
            outLightIndices[startOffset + lightIdxCounter] = -1.0;
        }
    }
}
