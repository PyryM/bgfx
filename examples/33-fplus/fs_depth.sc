$input v_viewpos

#include "common.sh"

void main() {
	// write float depth
	float z = -v_viewpos.z; // / 10.0;
	gl_FragColor = vec4(z, z, z, 1.0); // vec4_splat(v_viewpos.z);
}
