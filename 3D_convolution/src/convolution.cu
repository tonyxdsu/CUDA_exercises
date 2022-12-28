#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "../include/convolution.cuh"

__global__ void convolutionKernel(float* matrix1, float* matrix2, float* res, int heightRes, int widthRes, int width1height2) {

}

void convolution(float* matrix1_h, float* matrix2_h, float* matrixCalculatedRes_h, int height1, int width1, int height2, int width2) {
    cudaError_t cudaStatus;

    float* matrix1_d = 0;
    float* matrix2_d = 0;
    float* matrixCalculatedRes_d = 0;

    cudaStatus = cudaMalloc(&matrix1_d, height1 * width1 * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc matrix1_d error");
    }

    cudaStatus = cudaMalloc(&matrix2_d, height2 * width2 * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc matrix2_d error");
    }

    cudaStatus = cudaMalloc(&matrixCalculatedRes_d, height1 * width2 * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc matrixCalculatedRes_d error");
    }

    cudaStatus = cudaMemcpy(matrix1_d, matrix1_h, height1 * width1 * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy matrix1_d error");
    }

    cudaStatus = cudaMemcpy(matrix2_d, matrix2_h, height2 * width2 * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy matrix2_h error");
    }

    int gridX = width2 / BLOCK_DIM;
    if (width2 % BLOCK_DIM != 0) {
        gridX += 1;
    }

    int gridY = height1 / BLOCK_DIM;
    if (height1 % BLOCK_DIM != 0) {
        gridY +=1;
    }

    dim3 dimGrid(gridX, gridY, 1);
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM, 1);

    // TODO occupancy calculator
    convolutionKernel<<<dimGrid, dimBlock>>>(matrix1_d, matrix2_d, matrixCalculatedRes_d, height1, width2, width1);
    
    cudaStatus = cudaMemcpy(matrixCalculatedRes_h, matrixCalculatedRes_d, height1 * width2 * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy matrixCalculatedRes_h error");
    }

    cudaStatus = cudaFree(matrix1_d);
    if (cudaStatus != cudaSuccess) {
        printf("cudaFree matrix1_d error");
    }

    cudaStatus = cudaFree(matrix2_d);
    if (cudaStatus != cudaSuccess) {
        printf("cudaFree matrix2_d error");
    }

    cudaStatus = cudaFree(matrixCalculatedRes_d);
    if (cudaStatus != cudaSuccess) {
        printf("cudaFree matrixCalculatedRes_d error");
    }
}

