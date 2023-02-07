#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../include/histogram.cuh"
#include "../include/tensor1D.h"

/**
 * blockDim.x >= numBins since first numBins threads in each block will write to global memory
 * 
 * TODO try textbook's algorithm instead where they use total number of threads in kernel << length
 * This version has too many accesses to shared memory; textbook has each thread calculate multiple elements and update an accumulator in register memory, then write once to shared memory
*/
__global__ void createHistogramKernelSimple(Tensor1D<unsigned char>* input, unsigned int numBins, Tensor1D<unsigned int>* histogramOutput) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    


    // TODO compared vs statically allocated size since we're just dealing with images maxVal == 255
    extern __shared__ unsigned int histogramShared[];

    if (index < input->totalSize) { 
        atomicAdd(&(histogramShared[input->elements[index]]), 1);
    }

    __syncthreads();

    // first numBins threads of each block will write to global memory
    if (threadIdx.x < numBins) {
        atomicAdd(&(histogramOutput->elements[threadIdx.x]), histogramShared[threadIdx.x]);
    }
}

Tensor1D<unsigned int>* createHistogram(Tensor1D<unsigned char>* input, unsigned int maxInputVal) {
    Tensor1D<unsigned int>* histogramOutput = new Tensor1D<unsigned int>(maxInputVal);

    unsigned int gridXDim = input->totalSize / BLOCK_SIZE_HISTOGRAM;
    if (input->totalSize % BLOCK_SIZE_HISTOGRAM != 0) {
        gridXDim++;
    }

    createHistogramKernelSimple<<<gridXDim, BLOCK_SIZE_HISTOGRAM, maxInputVal * sizeof(unsigned int)>>>(input, maxInputVal, histogramOutput);

    cudaDeviceSynchronize();

    return histogramOutput;
}




