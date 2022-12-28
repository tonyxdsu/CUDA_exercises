#include <stdio.h>
#include <stdlib.h>

#include "include/convolution.cuh"
#include "include/parser.h"

int main(int argc, char** argv) {

    if (argc != 4) {
        printf("Usage: ./executableName [inputFilePath1] [inputFilePath2] [outputFilePath]\n");
        return 0;
    }  

    char* inputFileName = argv[1];
    char* kernelFileName = argv[2];
    char* outputFileName = argv[3];

    Tensor3D* tensorInput = parseFileToTensor3D(inputFileName);
    Tensor3D* tensorKernel = parseFileToTensor3DCube(kernelFileName);
    Tensor3D* tensorOutput = parseFileToTensor3D(outputFileName);

    // printTensor(tensorInput);
    printTensor(tensorKernel);
    // printTensor(tensorOutput);

    freeTensor(tensorInput);
    freeTensor(tensorKernel);
    freeTensor(tensorOutput);

    return 0;
}