#ifndef GREYSCALE_CUH
#define GREYSCALE_CUH

#include "ppm_image.h"
#include "tensor1D.h"

#define BLOCK_DIM_GREYSCALE 32

/**
 * @brief Converts a PPMImage to a Tensor1D of greyscale values.
 * @param input The PPMImage to convert.
 * @return A Tensor1D of greyscale values. Allocated on unified memory and must be freed by the caller.
*/
Tensor1D<unsigned char>* toGreyscaleValues(PPMImage* input);

#endif