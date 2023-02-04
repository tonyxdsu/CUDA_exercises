#ifndef GREYSCALE_CUH
#define GREYSCALE_CUH

#include "ppm_image.h"
#define BLOCK_DIM 32

unsigned char* toGreyscaleValues(PPMImage* input);

#endif