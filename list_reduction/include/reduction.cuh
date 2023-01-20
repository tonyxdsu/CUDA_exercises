#ifndef CONVOLUTION_CUH
#define CONVOLUTION_CUH

#include "tensor1D.h"

#define TILE_DIM 4

/**
 * @brief Performs a reduction (sum) on a 1D tensor (vector).
 * @param input Assumed to be allocated on unified memory.
 * @return Result of reduction allocated on unified memory.
*/
Tensor1D* reduction(Tensor1D* input);

#endif