#ifndef HISTOGRAM_CUH
#define HISTOGRAM_CUH

#include "ppm_image.h"

#define BLOCK_SIZE_HISTOGRAM 256

/**
 * @brief Creates a histogram of the frequency of value occurances in input
 * @param input Assumed to be allocated on unified memory.
 * @return 
 */
unsigned int* createHistogram(unsigned char* input, unsigned int size, unsigned int maxInputVal);

#endif