//-----------------------------------------------------------------------------------------
// Helper functions
//-----------------------------------------------------------------------------------------
// adapted from: https://github.com/GPUOpen-LibrariesAndSDKs/ForwardPlus11/blob/master/forwardplus11/src/Shaders/ForwardPlus11Tiling.hlsl

// this creates the standard Hessian-normal-form plane equation from three points,
// except it is simplified for the case where the first point is the origin
vec3 createPlaneEquation( vec3 b, vec3 c )
{
  // normalize(cross( b-a, c-a )), except we know "a" is the origin
  // also, typically there would be a fourth term of the plane equation,
  // -(n dot a), except we know "a" is the origin
  return normalize(cross(b,c));
}

// point-plane distance, simplified for the case where
// the plane passes through the origin
float getSignedDistanceFromPlane( vec3 p, vec3 eqn )
{
    // dot(eqn.xyz,p) + eqn.w, , except we know eqn.w is zero
    // (see CreatePlaneEquation above)
    return dot(eqn,p);
}

bool TestFrustumSides( vec3 c, float r, vec3 plane0, vec3 plane1, vec3 plane2, vec3 plane3 )
{
    bool intersectingOrInside0 = getSignedDistanceFromPlane( c, plane0 ) < r;
    bool intersectingOrInside1 = getSignedDistanceFromPlane( c, plane1 ) < r;
    bool intersectingOrInside2 = getSignedDistanceFromPlane( c, plane2 ) < r;
    bool intersectingOrInside3 = getSignedDistanceFromPlane( c, plane3 ) < r;

    return (intersectingOrInside0 && intersectingOrInside1 &&
            intersectingOrInside2 && intersectingOrInside3);
}

// calculate the number of tiles in the horizontal direction
uint GetNumTilesX()
{
    return (uint)( ( g_uWindowWidth + TILE_RES - 1 ) / (float)TILE_RES );
}

// calculate the number of tiles in the vertical direction
uint GetNumTilesY()
{
    return (uint)( ( g_uWindowHeight + TILE_RES - 1 ) / (float)TILE_RES );
}

// convert a point from post-projection space into view space
float4 ConvertProjToView( float4 p )
{
    p = mul( p, g_mProjectionInv );
    p /= p.w;
    return p;
}

// convert a depth value from post-projection space into view space
float ConvertProjDepthToView( float z )
{
    z = 1.f / (z*g_mProjectionInv._34 + g_mProjectionInv._44);
    return z;
}
