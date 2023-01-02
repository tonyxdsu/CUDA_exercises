#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"

#include "include/convolution.cuh"
#include "include/tensor.h"

int main(int argc, char** argv) {

    if (argc != 4) {
        printf("Usage: ./executableName [inputFilePath1] [inputFilePath2] [outputFilePath]\n");
        return 0;
    }  

    char* inputFileName = argv[1];
    char* maskFileName = argv[2];
    char* outputFileName = argv[3];

    // pointer itself not on unified memory?????
    // TODO can prefetch to GPU after creation of each Tensor to move data while reading next file?
    Tensor3D* input = new Tensor3D(inputFileName);
    Tensor3D* mask = new Tensor3D(maskFileName);
    Tensor3D* expectedOutput = new Tensor3D(outputFileName);

    if (input->xDim != expectedOutput->xDim || input->yDim != expectedOutput->yDim || input->zDim != expectedOutput->zDim) {
        fprintf(stderr, "input tensor and output tensor dimensions mismatch\n");
        return -1;
    }

    Tensor3D* calculatedOutput = convolution(input, mask);
    
    if (*calculatedOutput == *expectedOutput) {
        printf("Correct\n");
    }
    else {
        printf("Incorrect\n");
    }

    delete input;
    delete mask;
    delete expectedOutput;
    delete calculatedOutput;

    return 0;
}