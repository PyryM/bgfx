$input a_position, a_normal
$output v_wpos, v_wnormal, v_viewdir

#include "common.sh"

void main() {
    vec3 wpos = mul(u_model[0], vec4(a_position, 1.0) ).xyz;
    gl_Position = mul(u_viewProj, vec4(wpos, 1.0) );

    // bgfx example specific: the normal comes in as a uint8,
    // which means we have to transform [0,1] => [-1,1]
    vec3 trueNormal = (a_normal.xyz * 2.0) - 1.0;

    vec3 wnormal = mul(u_model[0], vec4(trueNormal, 0.0) ).xyz;
    vec3 campos = mul(u_invView, vec4(0.0, 0.0, 0.0, 1.0)).xyz;

    v_wpos = wpos;
    v_wnormal = wnormal;
    v_viewdir = normalize(campos - wpos);
}
