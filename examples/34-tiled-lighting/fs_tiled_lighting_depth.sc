$input v_viewpos

#include "common.sh"

void main() {
    // write float depth
    float z = -v_viewpos.z; // want positive depths
    gl_FragColor = vec4(z, z, z, 1.0);
}
