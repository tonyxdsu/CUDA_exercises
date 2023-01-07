#ifndef CONVOLUTION_CUH
#define CONVOLUTION_CUH

#include "tensor.h"

// TODO calculate total registers available or kernel will not launch
#define TILE_DIM 4

// Can only accomodate MASK_DIM^3 mask 
// TODO how to declare shared memory dynamically with dynamic tile size and mask width?
#define MASK_DIM 3
__constant__ float MASK_CONSTANT[MASK_DIM][MASK_DIM][MASK_DIM];

#define MASK_RADIUS MASK_DIM / 2

#define TILE_WITH_HALO_DIM TILE_DIM + 2 * MASK_RADIUS

/**
 * Performs convolution on input with mask. Input must be allocated using unified memory.
 * Allocates calculated output tensor using unified memory.
 * @param input Assumed to be allocated on unified memory.
 * @param mask Assumed to be allocated on unified memory. Dimensions must be odd. Mask must be a cube.
 * @return Result of convolution allocated on unified memory.
*/
Tensor3D* convolution(Tensor3D* input, Tensor3D* mask);

#endif