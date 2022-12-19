#include <stdio.h>
#include <stdlib.h>

#include "cuda.h"
#include "cuda_runtime.h"

#include "include/vector_addition.cuh"
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
    float* vector1_h = parseFileToMatrix(inputFileName1, &height1, &width1);
    float* vector2_h = parseFileToMatrix(inputFileName2, &height2, &width2);
    float* vectorExpectedRes_h = parseFileToMatrix(outputFileName, &heightRes, &widthRes);

    if (height1 != height2 || height1 != heightRes || 
        width1  != width2  || width1  != widthRes) {
        printf("Unequal input vector sizes\n");
        return 0;
    }

    float* vectorCalculatedRes_h = (float*) malloc(heightRes * widthRes * sizeof(float));
    vectorAddition(vector1_h, vector2_h, vectorCalculatedRes_h, heightRes, widthRes);

    if (isEqual(vectorExpectedRes_h, vectorCalculatedRes_h, heightRes, widthRes)) {
        printf("Correct\n");
    }
    else {
        printf("Incorrect\n");
    }

    free(vector1_h);
    free(vector2_h);
    free(vectorExpectedRes_h);
    free(vectorCalculatedRes_h);

    return 0;
}