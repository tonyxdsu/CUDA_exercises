#include <stdio.h>
#include <stdlib.h>

#include "include/matrix_multiply.cuh"
#include "include/parser.h"

int main(int argc, char** argv) {

    if (argc != 4) {
        printf("Usage: ./executableName [inputFilePath1] [inputFilePath2] [outputFilePath]\n");
        return 0;
    }  

    char* inputFileName1 = argv[1];
    char* inputFileName2 = argv[2];
    char* outputFileName = argv[3];

    int height1, height2, heightRes;
    int width1, width2, widthRes;
    float* matrix1_h = parseFileToMatrix(inputFileName1, &height1, &width1);
    float* matrix2_h = parseFileToMatrix(inputFileName2, &height2, &width2);
    float* matrixExpectedRes_h = parseFileToMatrix(outputFileName, &heightRes, &widthRes);

    if (width1 != height2) {
        printf("Cannot multiply matrices with dimensions %d by %d and %d by %d\n", height1, width1, height2, width2);
        return 0;
    }

    if (widthRes != width2 || heightRes != height1) {
        printf("Expected output incorrect size.\n");
        return 0;
    }

    float* matrixCalculatedRes_h = (float*) malloc(heightRes * widthRes * sizeof(float));
    matrixMultiply(matrix1_h, matrix2_h, matrixCalculatedRes_h, height1, width1, height2, width2);

    if (isEqual(matrixExpectedRes_h, matrixCalculatedRes_h, heightRes, widthRes)) {
        printf("Correct\n");
    }
    else {
        printf("Incorrect\n");
    }

    free(matrix1_h);
    free(matrix2_h);
    free(matrixExpectedRes_h);
    free(matrixCalculatedRes_h);

    return 0;
}