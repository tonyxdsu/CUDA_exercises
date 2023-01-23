#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"

#include "include/reduction.cuh"
#include "include/tensor1D.h"

int main(int argc, char** argv) {

    if (argc != 3) {
        printf("Usage: ./executableName [inputFilePath1] [outputFilePath]\n");
        return 0;
    }   

    char* inputFileName = argv[1];
    char* outputFileName = argv[2];

    // TODO can prefetch to GPU after creation of each Tensor to move data while reading next file?
    Tensor1D* input = new Tensor1D(inputFileName);
    Tensor1D* expectedOutput = new Tensor1D(outputFileName);
    Tensor1D* calculatedOutput = reductionSum(input);

    // sum all elements of output partial sums
    float calculatedOutputTotalSum = 0;
    for (int i = 0; i < calculatedOutput->totalSize; i++) {
        calculatedOutputTotalSum += calculatedOutput->elements[i];
    }
    
    if (calculatedOutputTotalSum == (*expectedOutput).elements[0]) {
        printf("Correct\n");
    }
    else {
        printf("Incorrect\n");
        printf("Expected: %f\n", (*expectedOutput).elements[0]);
        printf("Calculated: %f\n", calculatedOutputTotalSum);
    }

    delete input;
    delete expectedOutput;
    delete calculatedOutput;

    return 0;
}