#include "cuda_runtime.h"
#include "../include/spmv_csr.cuh"

#include <stdio.h>

template <typename T>
__global__ void spmvCSRKernel(const CSRMatrix<T>* mat, const Tensor1D<T>* vec, Tensor1D<T>* res) {
    // TODO vec small enough to be cached? No need for shared memory? Test performance for large vec.
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < mat->numRows) {
        T dot = 0;
        unsigned int rowStart = mat->rowPtr[row];
        unsigned int rowEnd = mat->rowPtr[row + 1];

        for (unsigned int i = rowStart; i < rowEnd; i++) {
            unsigned int colMat = mat->colInd[i];
            dot += mat->data[i] * vec->data[colMat];
        }

        res->data[row] = dot;
    }
}

template <typename T>
void spmvCSR(const CSRMatrix<T>* mat, const Tensor1D<T>* vec, Tensor1D<T>* res) {
    unsigned int gridX = mat->numRows / BLOCK_DIM_CSR;
    if (mat->numRows % BLOCK_DIM_CSR != 0) {
        gridX++;
    }

    dim3 gridDim(gridX, 1, 1);
    dim3 blockDim(BLOCK_DIM_CSR, 1, 1);

    spmvCSRKernel<<<gridDim, blockDim>>>(mat, vec, res);

    cudaDeviceSynchronize();
}

template void spmvCSR<float>(const CSRMatrix<float>* mat, const Tensor1D<float>* vec, Tensor1D<float>* res);

