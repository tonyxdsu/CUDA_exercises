#ifndef HISTOGRAM_CUH
#define HISTOGRAM_CUH

#include "ppm_image.h"
#include "tensor1D.h"

#define BLOCK_SIZE_HISTOGRAM 256


/**
 * @brief Creates a histogram of the frequency of value occurances in input.
 * @param input Assumed to be allocated on unified memory.
 * @param size Number of elements in input.
 * @param maxInputVal Maximum value in input. Used to determine number of bins in histogram.
 * @return returns a pointer to the histogram. Allocated on unified memory and must be freed by caller.
 */
Tensor1D<unsigned int>* createHistogram(Tensor1D<unsigned char>* input, unsigned int maxInputVal);

#endif