#include <stdio.h>     
#include <stdlib.h> 

#define MAX_FIRST_LINE_LENGTH 50
#define NUM_DECIMAL_PLACES_IN_FILE 2



/**
Parse contents of a file of floats (max 2 decimal places) representing a matrix into a linear array of floats
The first line of the file will specify: height width
Subsequent rows in the file will be rows of the matrix and each entry will be
seperated by a space
*/
float* parseFileToMatrix(char* inputFileName, int* height, int* width) {
    FILE* inputFile = fopen(inputFileName, "r");

    if (inputFile == NULL) {
        printf("%s cannot be opened\n", inputFileName);
    }

    char firstLine[MAX_FIRST_LINE_LENGTH];
    fgets(firstLine, MAX_FIRST_LINE_LENGTH, inputFile);
    char* pointsToWidthDimension;
    *height = strtol(firstLine, &pointsToWidthDimension, 10);
    if (*pointsToWidthDimension != 0) {
        *width = strtol(pointsToWidthDimension, NULL, 10);
    }
    else {
        *width = 1;
    }

    float* matrix = malloc(*height * *width * sizeof(float));

    for (int y = 0; y < *height; y++) {
        for (int x = 0; x < *width; x++) {
            fscanf(inputFile, "%f", &matrix[y * *width + x]);
        }
    }

    return matrix;
}

void printMatrix(float* matrix, int height, int width) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            printf("%.2f ", matrix[y * width + x]);
        }
        printf("\n");
    }
}

int main() {
    char inputFileName[] = "input0.raw";
    int height;
    int width;
    float* matrix = parseFileToMatrix(inputFileName, &height, &width);

    printMatrix(matrix, height, width);

    return 0;
}