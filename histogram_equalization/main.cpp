#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"

#include "include/PPM_image.h"
#include "include/histogram.cuh"
#include "include/histogram_equalization.cuh"
#include "include/greyscale.cuh"
#include "include/tensor1D.h"

void generateTestPPMImage() {
    PPMImage* testImage = new PPMImage(3, 3);
    for (int i = 0; i < 3 * 3 * 3; i++) {
        testImage->data[i] = i;
    }
    testImage->write("test.ppm");
    delete testImage;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        printf("Usage: ./executableName [inputFilePath] [outputFilePath]\n");
        return 0;
    }   

    char* inputFileName = argv[1];
    char* testFileName = argv[2];
    char* outputFileName = argv[3];

    PPMImage* input = new PPMImage(inputFileName);
    PPMImage* test = new PPMImage(testFileName);

    histogramEqualization(input);

    input->write(outputFileName);

    if (*input == *test) {
        printf("Matches test output!\n");
    }
    else {
        printf("Does not match test output!\n");
    }

    delete input;
    delete test;

    return 0;
}