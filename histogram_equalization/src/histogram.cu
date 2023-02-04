#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../include/histogram.cuh"

/**
 * blockDim.x >= numBins since first numBins threads in each block will write to global memory
 * 
 * TODO try textbook's algorithm instead where they use total number of threads in kernel << length
 * This version has too many accesses to shared memory; textbook has each thread calculate multiple elements and update an accumulator in register memory, then write once to shared memory
*/
__global__ void createHistogramKernelSimple(unsigned char* input, unsigned int length, unsigned int numBins, unsigned int* histogramOutput) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= length) { 
        return;
    }

    // TODO compared vs statically allocated size since we're just dealing with images maxVal == 255
    extern __shared__ unsigned int histogramShared[];

    atomicAdd(&(histogramShared[input[index]]), 1);

    __syncthreads();

    // first numBins threads of each block will write to global memory
    if (threadIdx.x < numBins) {
        atomicAdd(&(histogramOutput[threadIdx.x]), histogramShared[threadIdx.x]);
    }
}

unsigned int* createHistogram(unsigned char* input, unsigned int size, unsigned int maxInputVal) {
    unsigned int* histogramOutput = 0;
    cudaMallocManaged(&histogramOutput, maxInputVal * sizeof(unsigned int));

    unsigned int gridXDim = size / BLOCK_SIZE_HISTOGRAM;
    if (size % BLOCK_SIZE_HISTOGRAM != 0) {
        gridXDim++;
    }

    createHistogramKernelSimple<<<gridXDim, BLOCK_SIZE_HISTOGRAM, maxInputVal * sizeof(unsigned int)>>>(input, size, maxInputVal, histogramOutput);

    cudaDeviceSynchronize();

    return histogramOutput;
}

