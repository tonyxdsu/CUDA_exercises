#ifndef CONVOLUTION_CUH
#define CONVOLUTION_CUH

#include "tensor1D.h"

#define BLOCK_DIM 32

/**
 * @brief Performs a prefix sum on the input vector using the Brent-Kung algorithm.
 * @param input Assumed to be allocated on unified memory.
 * @return Vector with index i being the sum of the first i elements of input. Allocated on unified memory.
 */
Tensor1D* prefixSumBrentKung(Tensor1D* input);

#endif