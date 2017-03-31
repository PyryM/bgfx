// shared tile parameters

#define TILE_SIZE 16
#define MAX_LIGHTS_PER_TILE 1024

// lightBuffer stride 3:
//   color
//   position
//   paddingAndRadius
#define LIGHT_STRIDE 3
#define LIGHT_COL_OFFSET 0
#define LIGHT_POS_OFFSET 1
#define LIGHT_RAD_OFFSET 2
