#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"

#include "include/prefix_sum.cuh"
#include "include/kogge_stone.cuh"
#include "include/tensor1D.h"

int main(int argc, char** argv) {

    if (argc != 3) {
        printf("Usage: ./executableName [inputFilePath] [outputFilePath]\n");
        return 0;
    }   

    char* inputFileName = argv[1];
    char* outputFileName = argv[2];

    // TODO can prefetch to GPU after creation of each Tensor to move data while reading next file?
    Tensor1D<float>* input = new Tensor1D<float>(inputFileName);
    Tensor1D<float>* expectedOutput = new Tensor1D<float>(outputFileName);
    Tensor1D<float>* calculatedOutput = prefixSum(input);
    

    if (*calculatedOutput == *expectedOutput) {
        printf("Correct\n");
    }
    else {
        printf("Incorrect\n");
    }

    delete input;
    delete expectedOutput;
    delete calculatedOutput;

    return 0;
}