#include <stdio.h>     
#include <stdlib.h> 
#include <math.h>
#include "../include/parser.h"

/**
Parse contents of a file of floats (max 2 decimal places) representing a matrix into a linear array of floats.
The first line of the file will specify: height width.
Subsequent rows in the file will be rows of the matrix and each entry must be
seperated by a space.
@param inputFileName string representing the file path
@param height a pointer to read the height from the file into
@param width a pointer to read the width from the file into
*/
float* parseFileToMatrix(char* inputFileName, int* height, int* width) {
    FILE* inputFile = fopen(inputFileName, "r");

    if (inputFile == NULL) {
        printf("%s cannot be opened\n", inputFileName);
        return NULL;
    }

    char firstLine[MAX_FIRST_LINE_LENGTH];
    fgets(firstLine, MAX_FIRST_LINE_LENGTH, inputFile);
    char* pointsToWidthDimension;
    *height = strtol(firstLine, &pointsToWidthDimension, 10);
    if (*pointsToWidthDimension != '\n') {
        *width = strtol(pointsToWidthDimension, NULL, 10);
    }
    else {
        *width = 1;
    }

    float* matrix = (float*) malloc(*height * *width * sizeof(float));

    for (int y = 0; y < *height; y++) {
        for (int x = 0; x < *width; x++) {
            fscanf(inputFile, "%f", &matrix[y * *width + x]);
        }
    }

    return matrix;
}

void printMatrix(float* matrix, int height, int width) {
    if (matrix == NULL) return;

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
