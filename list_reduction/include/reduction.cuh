#ifndef CONVOLUTION_CUH
#define CONVOLUTION_CUH

#include "tensor1D.h"

#define BLOCK_DIM 32

/**
 * @brief Performs a reduction sum on a 1D tensor (vector).
 * @param input Assumed to be allocated on unified memory.
 * @return 1D tensor with TODO
*/
Tensor1D* reductionSum(Tensor1D* input);

#endif