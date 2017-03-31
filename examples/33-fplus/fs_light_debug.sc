$input v_wpos, v_wnormal, v_viewdir

#include "common.sh"
#include "tile_params.sh"
#include "bgfx_compute.sh"

BUFFER_RO(lightBuffer, vec4, 0);
BUFFER_RO(visibleLightIndicesBuffer, float, 1);

uniform vec4 u_dispatchParams; // hlsl doesn't seem to make this available
uniform vec4  u_lightCount;

void main() {
	// Determine which tile this pixel belongs to
	ivec2 location = ivec2(gl_FragCoord.xy);
	ivec2 tileID = location / ivec2(16, 16);
	uint index = tileID.y * u_dispatchParams.x + tileID.x;

	uint offset = index * MAX_LIGHTS_PER_TILE;
	uint i;
	for (i = 0; i < MAX_LIGHTS_PER_TILE && visibleLightIndicesBuffer[offset + i] != -1; ++i);

	float ratio = float(i) / u_lightCount.x;
	gl_FragColor = vec4(ratio, ratio, ratio, 1.0);
}
