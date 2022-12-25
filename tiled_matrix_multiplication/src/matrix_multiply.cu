#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "../include/matrix_multiply.cuh"

__global__ void multiplyKernel(float* matrix1, float* matrix2, float* res, int heightRes, int widthRes, int width1height2) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float tile1[TILE_HEIGHT][TILE_WIDTH];
    __shared__ float tile2[TILE_WIDTH][TILE_HEIGHT];

    float sum = 0;

    int numTilesInMatrix1Width = width1height2 / TILE_WIDTH;
    if (width1height2 % TILE_WIDTH != 0) {
        numTilesInMatrix1Width++;
    }

    for (int k = 0; k < numTilesInMatrix1Width; k++) {
        if (row < heightRes && k * blockDim.x + threadIdx.x < width1height2) {
            tile1[threadIdx.y][threadIdx.x] = matrix1[row * width1height2 + k * blockDim.x + threadIdx.x];
        }
        else {
            tile1[threadIdx.y][threadIdx.x] = 0;
        }

        if (k * blockDim.y + threadIdx.y < width1height2 && col < widthRes) {
            tile2[threadIdx.y][threadIdx.x] = matrix2[(k * blockDim.y + threadIdx.y) * widthRes + col]; 
        }
        else {
            tile2[threadIdx.y][threadIdx.x] = 0; 
        }

        __syncthreads();
        
        for (int xTile1 = 0; xTile1 < TILE_WIDTH; xTile1++) {
            sum += tile1[threadIdx.y][xTile1] * tile2[xTile1][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < heightRes && col < widthRes) {
        res[row * widthRes + col] = sum;
    }
}

void matrixMultiply(float* matrix1_h, float* matrix2_h, float* matrixCalculatedRes_h, int height1, int width1, int height2, int width2) {
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

    int gridX = width2 / TILE_WIDTH;
    if (width2 % TILE_WIDTH != 0) {
        gridX += 1;
    }

    int gridY = height1 / TILE_HEIGHT;
    if (height1 % TILE_HEIGHT != 0) {
        gridY +=1;
    }

    dim3 dimGrid(gridX, gridY, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_HEIGHT, 1);

    // TODO occupancy calculator
    multiplyKernel<<<dimGrid, dimBlock>>>(matrix1_d, matrix2_d, matrixCalculatedRes_d, height1, width2, width1);
    
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

