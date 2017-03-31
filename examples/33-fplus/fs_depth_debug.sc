// bgfx doesn't like it if you declare outputs from vs that aren't consumed
$input v_viewpos

#include "common.sh"

uniform vec4 u_debugParams;

void main() {
	// actually, here we're going to MRT:
	//  rt0: color (linear color, float32 * 4)
	//  rt1: depth (linear depth, float32 * 1?)
	gl_FragData[0] = vec4_splat(0.0);
	gl_FragData[1] = vec4_splat(v_viewpos.z);
}
