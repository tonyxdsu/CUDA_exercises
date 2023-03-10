#ifndef CONVOLUTION_CUH
#define CONVOLUTION_CUH

#include "tensor.h"

// TODO calculate total registers available or kernel will not launch
#define BLOCK_DIM 4

/**
 * Performs convolution on input with mask. Input must be allocated using unified memory.
 * Allocates calculated output tensor using unified memory.
 * @param input Assumed to be allocated on unified memory.
 * @param mask Assumed to be allocated on unified memory. Dimensions must be odd. Mask must be a cube.
 * @return Result of convolution allocated on unified memory.
*/
Tensor3D* convolution(Tensor3D* input, Tensor3D* mask);

#endif