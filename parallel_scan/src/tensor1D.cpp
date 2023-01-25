#include "cuda_runtime.h"

#include <stdio.h>
#include <math.h>

#include "../include/tensor1D.h"

Tensor1D::Tensor1D(unsigned int totalSize) {
    this->totalSize = totalSize;

    // TODO error checking?
    cudaMallocManaged(&elements, totalSize * sizeof(float));
    cudaDeviceSynchronize();
}

Tensor1D::Tensor1D(char* fileName) {
    FILE* inputFile = fopen(fileName, "r");

    if (inputFile == NULL) {
        fprintf(stderr, "%s cannot be opened\n", fileName);
        return;
    }

    fscanf(inputFile, "%d", &totalSize);

    // TODO error checking?
    cudaMallocManaged(&elements, totalSize * sizeof(float));
    cudaDeviceSynchronize();

    for (int i = 0; i < totalSize; i++) {
        fscanf(inputFile, "%f", &elements[i]);
    }
}

Tensor1D::~Tensor1D() {
    // TODO error checking?
    cudaDeviceSynchronize();
    cudaFree(elements);
}

void Tensor1D::print() {
    for (int i = 0; i < totalSize; i++) {
        printf("%f ", elements[i]);
    }
    printf("\n");
}

bool Tensor1D::operator==(const Tensor1D& rhs) {
    if (totalSize != rhs.totalSize) {
        return false;
    }

    for (int i = 0; i < totalSize; i++) {
        if (fabs(elements[i] - rhs.elements[i]) > TENSOR_ACCURACY_EPSILON) {
            printf("Wrong element at index %d: %f vs %f\n", i, elements[i], rhs.elements[i]);
            return false;
        }
    }

    return true;
}