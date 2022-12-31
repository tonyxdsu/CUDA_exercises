#include "cuda_runtime.h"

#include <stdio.h>     

#include "../include/tensor.h"

Tensor3D::Tensor3D(unsigned int xDim, unsigned int yDim, unsigned int zDim) {
    this->xDim = xDim;
    this->yDim = yDim;
    this->zDim = zDim;
    totalSize = xDim * yDim * zDim;

    // TODO error checking?
    cudaMallocManaged(&elements, totalSize);
    cudaDeviceSynchronize();
}

Tensor3D::Tensor3D(char* fileName) {
    FILE* inputFile = fopen(fileName, "r");

    if (inputFile == NULL) {
        fprintf(stderr, "%s cannot be opened\n", fileName);
        return;
    }

    fscanf(inputFile, "%d", &zDim);
    fscanf(inputFile, "%d", &yDim);
    fscanf(inputFile, "%d", &xDim);
    totalSize = xDim * yDim * zDim;

    // TODO error checking?
    cudaMallocManaged(&elements, totalSize);
    cudaDeviceSynchronize();

    for (int i = 0; i < totalSize; i++) {
        fscanf(inputFile, "%f", &elements[i]);
    }
}

void Tensor3D::print() {
    for (int z = 0; z < zDim; z++) {
        printf("z = %d\n", z);
        for (int y = 0; y < yDim; y++) {
            for (int x = 0; x < xDim; x++) {
                printf("%.2f ", elements[z * (yDim * xDim) + y * xDim + x]);
            }
            printf("\n");
        }
        printf("\n");
    }
}