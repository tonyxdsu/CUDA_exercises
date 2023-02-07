#ifndef GREYSCALE_CUH
#define GREYSCALE_CUH

#include "ppm_image.h"
#include "tensor1D.h"

#define BLOCK_DIM 32

Tensor1D<unsigned char>* toGreyscaleValues(PPMImage* input);

#endif