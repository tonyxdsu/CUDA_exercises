#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "../include/vector_addition.cuh"

__global__ void additionKernel(float* vector1, float* vector2, float* res) {
    int i = threadIdx.x;
    res[i] = vector1[i] + vector2[i];
}

void vectorAddition(float* vector1_h, float* vector2_h, float* vectorCalculatedRes_h, int height, int width) {
    cudaError_t cudaStatus;

    float* vector1_d = 0;
    float* vector2_d = 0;
    float* vectorCalculatedRes_d = 0;

    cudaStatus = cudaMalloc(&vector1_d, height * width * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc vector1_d error");
    }

    cudaStatus = cudaMalloc(&vector2_d, height * width * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc vector2_d error");
    }

    cudaStatus = cudaMalloc(&vectorCalculatedRes_d, height * width * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc vectorCalculatedRes_d error");
    }

    cudaStatus = cudaMemcpy(vector1_d, vector1_h, height * width * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy vector1_d error");
    }

    cudaStatus = cudaMemcpy(vector2_d, vector2_h, height * width * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy vector2_h error");
    }

    // TODO occupancy calculator
    additionKernel<<<1, height * width>>>(vector1_d, vector2_d, vectorCalculatedRes_d);
    
    cudaStatus = cudaMemcpy(vectorCalculatedRes_h, vectorCalculatedRes_d, height * width * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        printf("cudaMemcpy vectorCalculatedRes_h error");
    }

    cudaStatus = cudaFree(vector1_d);
    if (cudaStatus != cudaSuccess) {
        printf("cudaFree vector1_d error");
    }

    cudaStatus = cudaFree(vector2_d);
    if (cudaStatus != cudaSuccess) {
        printf("cudaFree vector2_d error");
    }

    cudaStatus = cudaFree(vectorCalculatedRes_d);
    if (cudaStatus != cudaSuccess) {
        printf("cudaFree vectorCalculatedRes_d error");
    }
}

