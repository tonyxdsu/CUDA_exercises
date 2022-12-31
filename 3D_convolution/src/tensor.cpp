#include "cuda_runtime.h"

#include <stdio.h>
#include <math.h>

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

Tensor3D::~Tensor3D() {
    // TODO error checking?
    cudaDeviceSynchronize();
    cudaFree(elements);
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

bool Tensor3D::operator==(const Tensor3D& rhs) {
    if (xDim != rhs.xDim || yDim != rhs.yDim || zDim != rhs.zDim) {
        return false;
    }

    for (int z = 0; z < zDim; z++) {
        for (int y = 0; y < yDim; y++) {
            for (int x = 0; x < xDim; x++) {
                float lhsElement = elements[z * (yDim * xDim) + y * xDim + x];
                float rhsElement = rhs.elements[z * (yDim * xDim) + y * xDim + x];
                if (fabs(lhsElement - rhsElement) >= TENSOR_ACCURACY_EPSILON) {
                    printf("Result at lhs z = %d y = %d x = %d is: %f\n and rhs is: %f\n", z, y, x, 
                        lhsElement, rhsElement);
                    return false;
                }
            }
        }
    }

    return true;
}