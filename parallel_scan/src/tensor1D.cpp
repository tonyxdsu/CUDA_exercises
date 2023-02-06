#include "cuda_runtime.h"

#include <stdio.h>
#include <math.h>
#include <iostream>

#include "../include/tensor1D.h"

template <typename T>
Tensor1D<T>::Tensor1D(unsigned int totalSize) {
    this->totalSize = totalSize;

    // TODO error checking?
    cudaMallocManaged(&elements, totalSize * sizeof(T));
    cudaDeviceSynchronize();
}

template <typename T>
Tensor1D<T>::Tensor1D(char* fileName) {
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
        fscanf(inputFile, fmt, &elements[i]);
    }
}

template <typename T>
Tensor1D<T>::~Tensor1D() {
    // TODO error checking?
    cudaDeviceSynchronize();
    cudaFree(elements);
}

template <typename T>
void Tensor1D<T>::print() {
    for (int i = 0; i < totalSize; i++) {
        std::cout << elements[i] << " ";
    }
    printf("\n");
}

template <typename T>
bool Tensor1D<T>::operator==(const Tensor1D<T>& rhs) {
    if (totalSize != rhs.totalSize) {
        printf("Wrong size: %d vs %d\n", totalSize, rhs.totalSize);
        return false;
    }

    for (int i = 0; i < totalSize; i++) {
        if (fabs(elements[i] - rhs.elements[i]) > TENSOR_ACCURACY_EPSILON) {
            printf("Wrong value at index %d: %f vs %f\n", i, elements[i], rhs.elements[i]);
            return false;
        }
    }

    return true;
}

template class Tensor1D<float>;
template class Tensor1D<unsigned int>;

template<> const char* Tensor1D<unsigned int>::fmt = "%u";
template<> const char* Tensor1D<float>::fmt = "%f";
