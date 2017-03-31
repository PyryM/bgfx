$input v_wpos, v_wnormal, v_viewdir

#include "common.sh"
#include "tile_params.sh"
#include "bgfx_compute.sh"

BUFFER_RO(lightBuffer, vec4, 0);
BUFFER_RO(visibleLightIndicesBuffer, float, 1);

uniform vec4 u_dispatchParams; // hlsl doesn't seem to make this available
uniform vec4 u_diffuseColor;
uniform vec4 u_ambientColor;

// Attenuate the point light intensity
float attenuate(vec3 lightDirection, float radius) {
	float cutoff = 0.5;
	float attenuation = dot(lightDirection, lightDirection) / (100.0 * radius);
	attenuation = 1.0 / (attenuation * 15.0 + 1.0);
	attenuation = (attenuation - cutoff) / (1.0 - cutoff);

	return clamp(attenuation, 0.0, 1.0);
}

float attenuate2(vec3 lightDirection, float radius) {
	float d2 = dot(lightDirection, lightDirection);
	float d = sqrt(d2);
	float rawatten = 1.0 / d2;
	float lin = clamp(2.0 - (2.0*d / radius), 0.0, 1.0);
	return rawatten * lin;
}

void main() {
	// Determine which tile this pixel belongs to
	ivec2 location = ivec2(gl_FragCoord.xy);
	ivec2 tileID = location / ivec2(TILE_SIZE, TILE_SIZE);
	uint index = tileID.y * u_dispatchParams.x + tileID.x;

	vec3 color = vec3(0.0, 0.0, 0.0);
	// normalize normal
	//vec3 worldNormal = normalize(v_wnormal);
	vec3 worldNormal = -normalize(cross(dFdx(v_wpos.xyz), dFdy(v_wpos.xyz)));

	// The offset is this tile's position in the global array of valid light indices.
	// Loop through all these indices until we hit max number of lights or the end (indicated by an index of -1)
	// Calculate the lighting contribution from each visible point light
	uint offset = index * MAX_LIGHTS_PER_TILE;
	for (uint i = 0; i < MAX_LIGHTS_PER_TILE && visibleLightIndicesBuffer[offset + i] >= 0.0; ++i) {
		uint lightIndex = uint(visibleLightIndicesBuffer[offset + i]) * LIGHT_STRIDE;

		vec4 lightPos = lightBuffer[lightIndex + LIGHT_POS_OFFSET];
		vec4 lightColor = lightBuffer[lightIndex + LIGHT_COL_OFFSET];
		float lightRadius = lightBuffer[lightIndex + LIGHT_RAD_OFFSET].x / 2.0;

		// Calculate the light attenuation on the pre-normalized lightDirection
		vec3 lightDirection = lightPos.xyz - v_wpos.xyz;
		float attenuation = attenuate2(lightDirection, lightRadius);

		// Basic lambertian contribution
		lightDirection = normalize(lightDirection);
		float dp = max(dot(lightDirection, worldNormal), 0.0);
		color += lightColor.xyz * dp * attenuation;
	}

	gl_FragColor = (vec4(color, 1.0) * u_diffuseColor) + u_ambientColor;
}
