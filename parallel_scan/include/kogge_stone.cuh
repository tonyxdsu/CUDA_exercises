#ifndef KOGGE_STONE_CUH
#define KOGGE_STONE_CUH

#include "tensor1D.h"

/**
 * @brief Performs a prefix sum on each block of the input vector using the Kogge-Stone algorithm.
 * @param input Assumed to be allocated on unified memory.
 * @return Vector of prefix sums in each block. Allocated on unified memory.
 */
Tensor1D<float>* blockPrefixSumsKoggeStone(Tensor1D<float>* input);

#endif