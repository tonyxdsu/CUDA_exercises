#include "cuda_runtime.h"

#include <stdio.h>
#include <math.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <numeric>

#include "../include/jds_matrix.h"
#include "../include/csr_matrix.h"

// template <typename T>
// JDSMatrix<T>::JDSMatrix(char* dataFileName, char* rowPtrFileName, char* colIndFileName) {
//     FILE* dataFile = fopen(dataFileName, "r");
//     FILE* rowPtrFile = fopen(rowPtrFileName, "r");
//     FILE* colIndFile = fopen(colIndFileName, "r");


//     if (dataFile == NULL) {
//         fprintf(stderr, "%s cannot be opened\n", dataFileName);
//         return;
//     }

//     if (rowPtrFile == NULL) {
//         fprintf(stderr, "%s cannot be opened\n", rowPtrFileName);
//         return;
//     }

//     if (colIndFile == NULL) {
//         fprintf(stderr, "%s cannot be opened\n", colIndFileName);
//         return;
//     }

//     fscanf(rowPtrFile, "%d", &numRows);
//     numRows -= 1; // last element of the input rowPtr array points to one past the end of the data array

//     unsigned int numDataElements;
//     fscanf(dataFile, "%d", &numDataElements);

//     unsigned int numColIndElements;
//     fscanf(colIndFile, "%d", &numColIndElements);

//     if (numDataElements != numColIndElements) {
//         fprintf(stderr, "data and colInd arrays are not the same size\n");
//         return;
//     }

//     // TODO error checking?
//     cudaMallocManaged(&data, numDataElements * sizeof(T));
//     cudaMallocManaged(&rowPtr, (numRows + 1) * sizeof(unsigned int));
//     cudaMallocManaged(&colInd, numDataElements * sizeof(unsigned int));
//     cudaDeviceSynchronize();

//     for (int i = 0; i < numDataElements; i++) {
//         fscanf(dataFile, fmt, &data[i]);
//     }

//     for (int i = 0; i < numRows + 1; i++) {
//         fscanf(rowPtrFile, "%d", &rowPtr[i]);
//     }

//     for (int i = 0; i < numDataElements; i++) {
//         fscanf(colIndFile, "%d", &colInd[i]);
//     }
// }

template <typename T>
JDSMatrix<T>::JDSMatrix(const CSRMatrix<T>& csrMatrix) {
    numRows = csrMatrix.numRows;
    numDataElements = csrMatrix.numDataElements;

    cudaMallocManaged(&data, numDataElements * sizeof(T));
    cudaMallocManaged(&rowPtr, (numRows + 1) * sizeof(unsigned int)); // TODO 1 extra
    cudaMallocManaged(&colInd, numDataElements * sizeof(unsigned int));
    cudaMallocManaged(&rowPerm, numRows * sizeof(unsigned int)); // TODO last elem not allocated
    cudaDeviceSynchronize();

    convertCSRToJDS(csrMatrix.data, csrMatrix.rowPtr, csrMatrix.colInd, numRows, numDataElements);
}


template <typename T>
JDSMatrix<T>::~JDSMatrix() {
    cudaDeviceSynchronize();
    cudaFree(data);
    cudaFree(rowPtr);
    cudaFree(colInd);
}

/**
 * Converts the input CSR matrix data into JDS format.
 * O(nlogn) time complexity.
*/
template <typename T>
void JDSMatrix<T>::convertCSRToJDS(T* dataCSR, const unsigned int* rowPtrCSR, const unsigned int* colIndCSR, const unsigned int numRowsCSR, const unsigned int numDataElementsCSR) {
    int lengthOfRowsForEachElementInData[numDataElementsCSR];
    int rowIndexforEachElementInData[numDataElementsCSR];

    for (int r = 0; r < numRowsCSR; r++) {
        for (int i = rowPtrCSR[r]; i < rowPtrCSR[r + 1]; i++) {
            lengthOfRowsForEachElementInData[i] = rowPtrCSR[r + 1] - rowPtrCSR[r];
            rowIndexforEachElementInData[i] = r;
        }
    }

    // ie) 
    // dataCSR = {3, 1, 2, 4, 1, 1, 1}
    // rowPtrCSR = {0, 2, 2, 5, 7}
    // lengthOfRowsForEachElementInData = {2, 2, 3, 3, 3, 1, 1}
    // rowIndexforEachElementInData = {0, 0, 2, 2, 2, 3, 3}

    // sort indices by length of rows
    std::vector<int> indices(numDataElementsCSR);
    std::iota(indices.begin(), indices.end(), 0);

    std::stable_sort(indices.begin(), indices.end(), [&](int a, int b) {
        return lengthOfRowsForEachElementInData[a] > lengthOfRowsForEachElementInData[b];
    });

    // indices before sorting = {0, 1, 2, 3, 4, 5, 6}
    // indices after sorting =  {2, 3, 4, 0, 1, 5, 6}

    // apply indices to this.data, this.colInd and permutedRowIndexforEachElementInData
    int permutedRowIndexforEachElementInData[numDataElementsCSR];
    for (int i = 0; i < numDataElementsCSR; i++) {
        data[i] = dataCSR[indices[i]];
        colInd[i] = colIndCSR[indices[i]];
        permutedRowIndexforEachElementInData[i] = rowIndexforEachElementInData[indices[i]];
    }

    // colIndCSR = {0, 2, 1, 2, 3, 0 3}
    // data = {2, 4, 1, 3, 1, 1, 1}
    // colInd = {1, 2, 3, 0, 2, 0, 3}
    // permutedRowIndexforEachElementInData = {2, 2, 2, 0, 0, 3, 3}

    // update this.rowPtr and this.rowPerm
    int curElemIdx = 0;
    int rowIdxPermutedRowIndexforEachElementInData = permutedRowIndexforEachElementInData[curElemIdx];
    int idxRowPtr = 0;
    int idxRowPerm = 0;

    do {
        rowIdxPermutedRowIndexforEachElementInData = permutedRowIndexforEachElementInData[curElemIdx];

        rowPtr[idxRowPtr] = curElemIdx;
        rowPerm[idxRowPerm] = rowIdxPermutedRowIndexforEachElementInData;
        idxRowPtr++;
        idxRowPerm++;

        while (permutedRowIndexforEachElementInData[curElemIdx] == rowIdxPermutedRowIndexforEachElementInData && curElemIdx < numDataElementsCSR) {
            curElemIdx++;
        }
    } while (curElemIdx < numDataElementsCSR);

    // rowPtr = {0, 3, 5}
    
    // ptr to one beyond last element of last row (used to find length of last row); then zero element rows
    while (idxRowPtr < numRowsCSR + 1) {
        rowPtr[idxRowPtr] = curElemIdx;
        idxRowPtr++;
    }

    // rowPtr = {0, 3, 5, 7, 7} 
    // rowPerm = {2, 0, 3}

    // find indices of rows with zero elements, and set end elements of rowPerm to those indices
    for (int idxRowPtrCSR = 1; idxRowPtrCSR < numRowsCSR; idxRowPtrCSR++) {
        if (rowPtrCSR[idxRowPtrCSR] == rowPtrCSR[idxRowPtrCSR - 1]) {
            rowPerm[idxRowPerm] = idxRowPtrCSR - 1;
            idxRowPerm++;
        }
    }

    /**
     * TODO
     * Why can I NOT do this?
     * - Lambda capture by reference, variables &a and &b are not in address [data, data + numDataElements] from gdb output
     * 
     * std::stable_sort(csrMat->data, csrMat->data + csrMat->numDataElements, [&](float a, float b) {
     *     return lengthOfRowsForEachElementInData[&a - csrMat->data] < lengthOfRowsForEachElementInData[&b - csrMat->data];
     * });
     * 
     * Have to sort an array of indices instead, then apply the indices to the data array.
     * Seems clunky. But still O(n log n) time, but O(n) additional space and O(n) time to apply the indices to the data array.
    */

}

template <typename T>
void JDSMatrix<T>::print() {
    printf("data: ");
    for (int i = 0; i < numDataElements; i++) {
        printf(fmt, data[i]);
        printf(" ");
    }
    printf("\n");
    // print rowPtr, colInd, rowPerm
    printf("rowPtr: ");
    for (int i = 0; i < numRows + 1; i++) {
        printf("%d ", rowPtr[i]);
    }
    printf("\n");
    printf("colInd: ");
    for (int i = 0; i < numDataElements; i++) {
        printf("%d ", colInd[i]);
    }
    printf("\n");
    printf("rowPerm: ");
    for (int i = 0; i < numRows; i++) {
        printf("%d ", rowPerm[i]);
    }
    printf("\n");
    printf("numRows: %d", numRows);
    printf("\n");
}

template class JDSMatrix<float>;
template class JDSMatrix<unsigned int>;
template class JDSMatrix<unsigned char>;

template<> const char* JDSMatrix<unsigned int>::fmt = "%u";
template<> const char* JDSMatrix<float>::fmt = "%f";
template<> const char* JDSMatrix<unsigned char>::fmt = "%hhu";
