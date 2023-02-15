#include "cuda_runtime.h"

#include <stdio.h>
#include <math.h>
#include <iostream>

#include "../include/csr_matrix.h"

template <typename T>
CSRMatrix<T>::CSRMatrix(char* dataFileName, char* rowPtrFileName, char* colIndFileName) {
    FILE* dataFile = fopen(dataFileName, "r");
    FILE* rowPtrFile = fopen(rowPtrFileName, "r");
    FILE* colIndFile = fopen(colIndFileName, "r");


    if (dataFile == NULL) {
        fprintf(stderr, "%s cannot be opened\n", dataFileName);
        return;
    }

    if (rowPtrFile == NULL) {
        fprintf(stderr, "%s cannot be opened\n", rowPtrFileName);
        return;
    }

    if (colIndFile == NULL) {
        fprintf(stderr, "%s cannot be opened\n", colIndFileName);
        return;
    }

    fscanf(rowPtrFile, "%d", &numRows);
    numRows -= 1; // last element of the input rowPtr array points to one past the end of the data array

    unsigned int numDataElements;
    fscanf(dataFile, "%d", &numDataElements);

    unsigned int numColIndElements;
    fscanf(colIndFile, "%d", &numColIndElements);

    if (numDataElements != numColIndElements) {
        fprintf(stderr, "data and colInd arrays are not the same size\n");
        return;
    }

    // TODO error checking?
    cudaMallocManaged(&data, numDataElements * sizeof(T));
    cudaMallocManaged(&rowPtr, numRows * sizeof(unsigned int));
    cudaMallocManaged(&colInd, numDataElements * sizeof(unsigned int));
    cudaDeviceSynchronize();

    for (int i = 0; i < numDataElements; i++) {
        fscanf(dataFile, fmt, &data[i]);
    }

    for (int i = 0; i < numRows + 1; i++) {
        fscanf(rowPtrFile, "%d", &rowPtr[i]);
    }

    for (int i = 0; i < numDataElements; i++) {
        fscanf(colIndFile, "%d", &colInd[i]);
    }
}

template <typename T>
CSRMatrix<T>::~CSRMatrix() {
    cudaDeviceSynchronize();
    cudaFree(data);
    cudaFree(rowPtr);
    cudaFree(colInd);
}

template <typename T>
void CSRMatrix<T>::print() {
    printf("data: ");
    for (int i = 0; i < rowPtr[numRows]; i++) {
        printf(fmt, data[i]);
        printf("\n");
    }
}

template class CSRMatrix<float>;
template class CSRMatrix<unsigned int>;
template class CSRMatrix<unsigned char>;

template<> const char* CSRMatrix<unsigned int>::fmt = "%u";
template<> const char* CSRMatrix<float>::fmt = "%f";
template<> const char* CSRMatrix<unsigned char>::fmt = "%hhu";
