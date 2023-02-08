#ifndef PREFIX_SUM_CUH
#define PREFIX_SUM_CUH

#include "tensor1D.h"

#define BLOCK_DIM 512


/**
 * @brief Performs a prefix sum on the input vector.
 * @param input Assumed to be allocated on unified memory.
 * @return Vector with index i being the sum of the first i elements of input. Allocated on unified memory and must be freed by caller.
 */
Tensor1D<unsigned int>* prefixSum(Tensor1D<unsigned int>* input);

#endif