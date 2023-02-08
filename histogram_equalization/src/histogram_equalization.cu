#include <cuda_runtime.h>

#include "../include/histogram_equalization.cuh"
#include "../include/histogram.cuh"
#include "../include/greyscale.cuh"
#include "../include/prefix_sum.cuh"
#include "../include/tensor1d.h"

float probabilityInHistogram(unsigned int occurance, unsigned int totalImageSize) {
    return (float)occurance / (float)totalImageSize;
}

/**
 * @param cdfMin the minimum value in the cdf will be placed here
*/
void calculateCDF(Tensor1D<float>* cdf, float* cdfMin, Tensor1D<unsigned int>* histogram, unsigned int totalImageSize) {
    // TODO is sequential CPU implementation probably better for this small 256 element histogram?
    // okay we're just being lazy now...might write a proper kernel later
    // Tensor1D<unsigned int>* cdf = prefixSum(histogram);

    cdf->elements[0] = probabilityInHistogram(histogram->elements[0], histogram->totalSize);
    float currentMin = cdf->elements[0];

    for (int i = 1; i < histogram->totalSize; i++) {
        cdf->elements[i] = cdf->elements[i - 1] + probabilityInHistogram(histogram->elements[i], totalImageSize);
        if (cdf->elements[i] < currentMin) {
            currentMin = cdf->elements[i];
        }
    }

    *cdfMin = currentMin;
}

float clamp(float value, float min, float max) {
    return value < min ? min : (value > max ? max : value);
}

unsigned char correctColor(unsigned char value, Tensor1D<float>* cdf, float cdfMin) {
    return clamp(255.0f * (cdf->elements[value] - cdfMin) / (1.0f - cdfMin), 0.0f, 255.0f);
}

void histogramEqualization(PPMImage* image) {
    Tensor1D<unsigned char>* greyscaleValues = toGreyscaleValues(image);
    Tensor1D<unsigned int>* histogram = createHistogram(greyscaleValues, 256);

    Tensor1D<float>* cdf = new Tensor1D<float>(histogram->totalSize);
    float cdfMin;
    calculateCDF(cdf, &cdfMin, histogram, image->width * image->height);

    for (int i = 0; i < image->height * image->width * 3; i++) {
        auto corrected = correctColor(image->data[i], cdf, cdfMin);
        image->data[i] = corrected;
    }

    delete greyscaleValues;
    delete histogram;
    delete cdf;
}

