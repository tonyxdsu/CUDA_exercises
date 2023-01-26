#ifndef BRENT_KUNG_CUH
#define BRENT_KUNG_CUH

#include "tensor1D.h"

/**
 * @brief Performs a 
 * @param input Assumed to be allocated on unified memory.
 * @return Vector with index i being the sum of the first i elements of input. Allocated on unified memory.
 */
Tensor1D* blockPrefixSumsBrentKung(Tensor1D* input);

#endif