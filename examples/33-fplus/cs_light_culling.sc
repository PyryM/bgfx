// INPUTS/SETUP:
//   buffers:
//     0: lightBuffer [vec4f color, vec4f pos, vec4f rad]*nlights
//     1: visibleLightIndicesBuffer [float]*(MAX_LIGHTS_PER_TILE*NUM_TILES)
//     2: s_depthMap [r32f "depth" buffer]
//
//  uniforms:
//     u_screenSize: vec2f (in pixels)
//     u_lightCount: vec4 (x: number of lights)
//     u_projectionMat: camera projection matrix
//     u_viewMat: camera view matrix
//     u_dispatchParams: vec4f (x: n_tiles_x, y: n_tiles_y)
//
//  Dispatch:
//  	workGroupsX = (SCREEN_SIZE.x + (SCREEN_SIZE.x % 16)) / 16;
//    workGroupsY = (SCREEN_SIZE.y + (SCREEN_SIZE.y % 16)) / 16;
//    (workGroupsX, workGroupsY, 1)

// this is largely adapted from
//   https://github.com/bcrusco/Forward-Plus-Renderer
// and
//   https://github.com/GPUOpen-LibrariesAndSDKs/ForwardPlus11

// TILE_SIZE, MAX_LIGHTS_PER_TILE, {LIGHT_}
#include "tile_params.sh"
#include "bgfx_compute.sh"

// sampler/buffer type uniforms
BUFFER_RO(lightBuffer, vec4, 0);
BUFFER_WR(visibleLightIndicesBuffer, float, 1);
IMAGE2D_RO(s_depthMap, rgba32f, 2);

//SAMPLER2D(s_depthMap, 2);

uniform vec4 u_screenSize;
uniform vec4 u_lightCount;
uniform vec4 u_dispatchParams; // hlsl doesn't seem to make this available
uniform mat4 u_projectionMat;
uniform mat4 u_viewMat;

// Shared values between all the threads in the group
SHARED uint minDepthInt;
SHARED uint maxDepthInt;
SHARED uint visibleLightCount;
SHARED vec4 frustumPlanes[6];
// Shared local storage for visible indices, will be written out to the global buffer at the end
SHARED int visibleLightIndices[MAX_LIGHTS_PER_TILE];
SHARED mat4 viewProjection;

// Took some light culling guidance from Dice's deferred renderer
// http://www.dice.se/news/directx-11-rendering-battlefield-3/

// WEIRD: this macro specifically needs there to be no ; at the end
NUM_THREADS(TILE_SIZE, TILE_SIZE, 1)
void main() {
	ivec2 location = ivec2(gl_GlobalInvocationID.xy);
	ivec2 itemID = ivec2(gl_LocalInvocationID.xy);
	ivec2 tileID = ivec2(gl_WorkGroupID.xy);
	ivec2 tileNumber = ivec2(u_dispatchParams.xy);
	uint index = tileID.y * tileNumber.x + tileID.x;
	uint lightCount = uint(u_lightCount.x); // .x?

	// Initialize shared global values for depth and light count
	if (gl_LocalInvocationIndex == 0) {
		minDepthInt = 0xFFFFFFFF;
		maxDepthInt = 0;
		visibleLightCount = 0;
		viewProjection = mul(u_projectionMat, u_viewMat);
	}

	barrier();

	// Step 1: Calculate the minimum and maximum depth values (from the depth buffer) for this group's tile
	float maxDepth, minDepth;
	vec2 text = vec2(location) / u_screenSize.xy;
	//float depth = texture2DLod(s_depthMap, text, 0).r; // hmm not sure about this
	//float depth = imageLoad(s_depthMap, location).r;     // do this instead?
	//float depth = s_depthMap.SampleLevel(text, 0, 0 ).x;   // maybe this?
	float depth = s_depthMap.Load(ivec3(location, 0)).x;

	// NOTE: just store a linear depth in the first place
	// depth = (0.5 * u_projectionMat[3][2]) / (depth + 0.5 * u_projectionMat[2][2] - 0.5);

	// Convert depth to uint so we can do atomic min and max comparisons between the threads
	uint depthInt = floatBitsToUint(depth); // floatBitsToUint is a builtin

	// TODO: get the bgfx macros for these to actually work
	//atomicMin(minDepthInt, depthInt);
	//atomicMax(maxDepthInt, depthInt);
	InterlockedMin(minDepthInt, depthInt);
	InterlockedMax(maxDepthInt, depthInt);

	barrier();

	// Step 2: One thread should calculate the frustum planes to be used for this tile
	if (gl_LocalInvocationIndex == 0) {
		// Convert the min and max across the entire tile back to float
		minDepth = uintBitsToFloat(minDepthInt);
		maxDepth = uintBitsToFloat(maxDepthInt);

		// Steps based on tile sale
		vec2 negativeStep = (2.0 * vec2(tileID)) / vec2(tileNumber);
		vec2 positiveStep = (2.0 * vec2(tileID + ivec2(1, 1))) / vec2(tileNumber);

		// Set up starting values for planes using steps and min and max z values
		frustumPlanes[0] = vec4(1.0, 0.0, 0.0, 1.0 - negativeStep.x); // Left
		frustumPlanes[1] = vec4(-1.0, 0.0, 0.0, -1.0 + positiveStep.x); // Right

		// WEIRD: had to sign flip y   |
		//                             v
		frustumPlanes[2] = vec4(0.0, -1.0, 0.0,  1.0 - negativeStep.y); // Bottom
		frustumPlanes[3] = vec4(0.0,  1.0, 0.0, -1.0 + positiveStep.y); // Top
		frustumPlanes[4] = vec4(0.0, 0.0,  1.0,  minDepth); // Near
		frustumPlanes[5] = vec4(0.0, 0.0, -1.0, -maxDepth); // Far

		// WEIRD: left multiplying the planes (e.g., multiplying by transpose)
		//        this is not the correct inverse!
		// TODO: just actually pass in inverse matrices?
		// Transform the first four planes

		for (uint i = 0; i < 4; i++) {
			frustumPlanes[i] = mul(frustumPlanes[i], viewProjection);
			frustumPlanes[i] /= length(frustumPlanes[i].xyz);
		}

		// Transform the depth planes
		frustumPlanes[4] = mul(frustumPlanes[4], u_viewMat);
		frustumPlanes[4] /= length(frustumPlanes[4].xyz);
		frustumPlanes[5] = mul(frustumPlanes[5], u_viewMat);
		frustumPlanes[5] /= length(frustumPlanes[5].xyz);
	}

	barrier();

	// Step 3: Cull lights.
	// Parallelize the threads against the lights now.
	// If more than TILE_SIZE^2 lights, additional passes are performed
	uint threadCount = TILE_SIZE * TILE_SIZE;
	uint passCount = (u_lightCount + threadCount - 1) / threadCount;
	for (uint i = 0; i < passCount; i++) {
		// Get the lightIndex to test for this thread / pass.
		// If the index is >= light count, then this thread can stop testing lights
		uint lightIndex = i * threadCount + gl_LocalInvocationIndex;
		if (lightIndex >= lightCount) {
			break;
		}
		// we aren't using fancy structured buffers, so we need to stride by how
		// many vec4's in each light
		uint rawIndex = lightIndex * LIGHT_STRIDE;

		vec4 position = vec4(lightBuffer[rawIndex + LIGHT_POS_OFFSET].xyz, 1.0);
		//position = mul(u_viewMat, position);
		float radius =  lightBuffer[rawIndex + LIGHT_RAD_OFFSET].x;

		// We check if the light exists in our frustum
		float distance = 0.0;
		for (uint j = 0; j < 6; ++j) { // DEBUG: j < 6
			distance = dot(position, frustumPlanes[j]) + radius;

			// If one of the tests fails, then there is no intersection
			if (distance < 0.0) {
				break;
			}
		}

		// If greater than zero, then it is a visible light
		if (distance >= 0.0) {
			// Add index to the shared array of visible indices
			uint offset;
			InterlockedAdd(visibleLightCount, 1, offset);
			// TODO: fix bgfx macros atomicAdd(visibleLightCount, 1);
			visibleLightIndices[offset] = int(lightIndex);
		}
	}

	barrier();

	// One thread should fill the global light buffer
	if (gl_LocalInvocationIndex == 0) {
		uint offset = index * MAX_LIGHTS_PER_TILE; // Index in global buffer
		if(visibleLightCount > MAX_LIGHTS_PER_TILE) {
		  visibleLightCount =	MAX_LIGHTS_PER_TILE;
		}
		for (uint i = 0; i < visibleLightCount; ++i) {
			visibleLightIndicesBuffer[offset + i] = float(visibleLightIndices[i]);
		}

		if (visibleLightCount != MAX_LIGHTS_PER_TILE) {
			// Unless we have totally filled the entire array, mark its end with -1
			// Final shader step will use this to determine where to stop
			visibleLightIndicesBuffer[offset + visibleLightCount] = -1.0;
		}
	}
}
