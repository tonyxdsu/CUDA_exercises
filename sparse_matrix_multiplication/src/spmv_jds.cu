#include "cuda_runtime.h"
#include "../include/spmv_jds.cuh"

#include <stdio.h>

template <typename T>
__global__ void spmvJDSKernel(const JDSMatrix<T>* mat, const Tensor1D<T>* vec, Tensor1D<T>* res) {
    // TODO vec small enough to be cached? No need for shared memory? Test performance for large vec.
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < mat->numRows) {
        unsigned int rowStart = mat->rowPtr[row];
        unsigned int rowEnd = mat->rowPtr[row + 1];
        T dot = 0; // How to initialize as 0.0f? Oh well minor performance hit from template or is compiler smart enough to not cast int to float?

        for (unsigned int i = rowStart; i < rowEnd; i++) {
            unsigned int colMat = mat->colInd[i];
            dot += mat->data[i] * vec->data[colMat];
        }

        unsigned int origRow = mat->rowPerm[row];
        res->data[origRow] = dot;
    }
}

template <typename T>
void spmvJDS(const JDSMatrix<T>* mat, const Tensor1D<T>* vec, Tensor1D<T>* res) {
    unsigned int gridX = mat->numRows / BLOCK_DIM_JDS;
    if (mat->numRows % BLOCK_DIM_JDS != 0) {
        gridX++;
    }

    dim3 gridDim(gridX, 1, 1);
    dim3 blockDim(BLOCK_DIM_JDS, 1, 1);

    spmvJDSKernel<float><<<gridDim, blockDim>>>(mat, vec, res);

    cudaDeviceSynchronize();
}

template void spmvJDS<float>(const JDSMatrix<float>* mat, const Tensor1D<float>* vec, Tensor1D<float>* res);

