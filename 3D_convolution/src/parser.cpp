#include <stdio.h>     
#include <stdlib.h> 
#include <math.h>
#include <cmath>
#include "../include/parser.h"

/**
Parse contents of a file of floats representing a 3D tensor into a linear array of floats.
The first three lines will be the dimensions: z y x.
This function allocates memory and the struct returned may be deleted with the function freeTensor3D.
@param inputFileName string representing the file path
@return a pointer to allocated struct Tensor3D
*/
Tensor3D* parseFileToTensor3D(char* inputFileName) {
    FILE* inputFile = fopen(inputFileName, "r");

    if (inputFile == NULL) {
        printf("%s cannot be opened\n", inputFileName);
        return NULL;
    }

    Tensor3D* tensor = (Tensor3D*) malloc(sizeof(Tensor3D));
    fscanf(inputFile, "%d", &tensor->zDim);
    fscanf(inputFile, "%d", &tensor->yDim);
    fscanf(inputFile, "%d", &tensor->xDim);

    unsigned int totalLength = tensor->xDim * tensor->yDim * tensor->zDim;
    tensor->elements = (float*) malloc(totalLength * sizeof(float));

    for (int i = 0; i < totalLength; i++) {
        fscanf(inputFile, "%f", &tensor->elements[i]);
    }

    return tensor;
}

/**
Parse contents of a file of floats representing a cube 3D tensor into a linear array of floats.
The first line is the total size (must have integer cube root).
This function allocates memory and the struct returned may be deleted with the function freeTensor3D.
@param inputFileName string representing the file path
@return a pointer to allocated struct Tensor3D
*/
Tensor3D* parseFileToTensor3DCube(char* inputFileName) {
    FILE* inputFile = fopen(inputFileName, "r");

    if (inputFile == NULL) {
        printf("%s cannot be opened\n", inputFileName);
        return NULL;
    }

    Tensor3D* tensor = (Tensor3D*) malloc(sizeof(Tensor3D));

    unsigned int totalLength = 1;
    fscanf(inputFile, "%d", &totalLength);
    unsigned int dim = std::cbrt(totalLength);

    tensor->zDim = dim;
    tensor->yDim = dim;
    tensor->xDim = dim;

    tensor->elements = (float*) malloc(totalLength * sizeof(float));

    for (int i = 0; i < totalLength; i++) {
        fscanf(inputFile, "%f", &tensor->elements[i]);
    }

    return tensor;
}

void freeTensor(Tensor3D* tensor) {
    free(tensor->elements);
    free(tensor);
}

void printTensor(Tensor3D* tensor) {
    for (int z = 0; z < tensor->zDim; z++) {
        printf("z = %d\n", z);
        printMatrix(&tensor->elements[z * tensor->yDim * tensor->xDim], tensor->yDim, tensor->xDim);
        printf("\n");
    }
}

void printMatrix(float* matrix, int height, int width) {
    if (matrix == NULL) {
        printf("input to printMatrix is NULL\n");
        return;
    }

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            printf("%.2f ", matrix[y * width + x]);
        }
        printf("\n");
    }
}

int isEqual(float* matrix1, float* matrix2, int height, int width) {
    if (matrix1 == NULL || matrix2 == NULL) return -1;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (fabs(matrix1[y * width + x] - matrix2[y * width + x]) >= MATRIX_EPSILON) {
                printf("Result at row %d col %d is matrix1: %f\n and matrix2: %f\n", y, x, 
                    matrix1[y * width + x], matrix2[y * width + x]);
                return 0;
            }
        }
    }

    return 1;
}
