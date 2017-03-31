$input a_position
$output v_viewpos

#include "common.sh"

void main() {
	v_viewpos = mul(u_modelView, vec4(a_position, 1.0));
	gl_Position = mul(u_modelViewProj, vec4(a_position, 1.0));
}
