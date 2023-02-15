#include "cuda_runtime.h"

#include <stdio.h>
#include <math.h>
#include <iostream>

#include "../include/tensor1D.h"

template <typename T>
Tensor1D<T>::Tensor1D(unsigned int length) {
    this->length = length;

    // TODO error checking?
    cudaMallocManaged(&data, length * sizeof(T));
    cudaDeviceSynchronize();
}

template <typename T>
Tensor1D<T>::Tensor1D(char* fileName) {
    FILE* inputFile = fopen(fileName, "r");

    if (inputFile == NULL) {
        fprintf(stderr, "%s cannot be opened\n", fileName);
        return;
    }

    fscanf(inputFile, "%d", &length);

    // TODO error checking?
    cudaMallocManaged(&data, length * sizeof(float));
    cudaDeviceSynchronize();

    for (int i = 0; i < length; i++) {
        fscanf(inputFile, fmt, &data[i]);
    }
}

template <typename T>
Tensor1D<T>::~Tensor1D() {
    // TODO error checking?
    cudaDeviceSynchronize();
    cudaFree(data);
}

template <typename T>
void Tensor1D<T>::print() {
    for (int i = 0; i < length; i++) {
        if (fmt == "%hhu") {
            std::cout << (int)data[i] << " ";
        }
        else {
            std::cout << data[i] << " ";
        }
    }
    printf("\n");
}

template <typename T>
bool Tensor1D<T>::operator==(const Tensor1D<T>& rhs) {
    if (length != rhs.length) {
        printf("Wrong size: %d vs %d\n", length, rhs.length);
        return false;
    }

    bool isEqual = true;

    for (int i = 0; i < length; i++) {
        if (fabs(data[i] - rhs.data[i]) > TENSOR_ACCURACY_EPSILON) {
            std::cout << "Wrong value at " << i << ": " << data[i] << " vs " << rhs.data[i] << std::endl;
            isEqual = false;
        }
    }

    return isEqual;
}

template class Tensor1D<float>;
template class Tensor1D<unsigned int>;
template class Tensor1D<unsigned char>;

template<> const char* Tensor1D<unsigned int>::fmt = "%u";
template<> const char* Tensor1D<float>::fmt = "%f";
template<> const char* Tensor1D<unsigned char>::fmt = "%hhu";
