#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"

#include "include/tensor1D.h"

int main(int argc, char** argv) {

    if (argc != 3) {
        printf("Usage: ./executableName [inputFilePath1] [inputFilePath2] [outputFilePath]\n");
        return 0;
    }   

    char* inputFileName = argv[1];
    char* outputFileName = argv[2];

    // TODO can prefetch to GPU after creation of each Tensor to move data while reading next file?
    Tensor1D* input = new Tensor1D(inputFileName);
    Tensor1D* expectedOutput = new Tensor1D(outputFileName);

    input->print();
    expectedOutput->print();

    
    // if (*calculatedOutput == *expectedOutput) {
    //     printf("Correct\n");
    // }
    // else {
    //     printf("Incorrect\n");
    // }

    delete input;
    delete expectedOutput;
    // delete calculatedOutput;

    return 0;
}