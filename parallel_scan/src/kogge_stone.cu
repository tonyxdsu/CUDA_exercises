#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../include/kogge_stone.cuh"
#include "../include/prefix_sum.cuh"

#include <stdio.h>

/**
 * TODO can I pass in a function pointer to a kernel to perform different operations?
*/
__global__ void blockPrefixSumsKoggeStone(Tensor1D* input, Tensor1D* output) {
    int start = blockIdx.x * blockDim.x;
    int tid   = threadIdx.x; // otherwise need to typecast before comparison with 0

    if (start + tid >= input->totalSize) {
        return;
    }

    __shared__ float tempSumsBufferA[BLOCK_DIM];
    __shared__ float tempSumsBufferB[BLOCK_DIM];

    tempSumsBufferA[tid] = input->elements[start + tid];
    tempSumsBufferB[tid] = input->elements[start + tid];

    float* tempSumsBufferPtr1 = &tempSumsBufferA[0];
    float* tempSumsBufferPtr2 = &tempSumsBufferB[0];

    for (int stride = 1; stride < BLOCK_DIM; stride <<= 1) {
        __syncthreads();
        if (tid - stride >= 0) {
            // double buffering removes a need for a syncthreads here to wait for all threads to finish loading into a temp float
            tempSumsBufferPtr1[tid] = tempSumsBufferPtr2[tid] + tempSumsBufferPtr2[tid - stride];

            float* tmp = tempSumsBufferPtr1;
            tempSumsBufferPtr1 = tempSumsBufferPtr2;
            tempSumsBufferPtr2 = tmp;
        }
        else {
            // maintain the same value for non-participating threads to keep consistency in other buffer for other threads' use
            tempSumsBufferPtr1[tid] = tempSumsBufferPtr2[tid];
            // TODO else break? I guess it doesn't matter to speed since other threads still need to complete.
            // if I break here, can GPU schedule other warps (in another process?) while, for example, half the block of this kernel is turned off?
        }
    }

    output->elements[start + tid] = tempSumsBufferPtr2[tid];
}

Tensor1D* blockPrefixSumsKoggeStone(Tensor1D* input) {
    Tensor1D* output = new Tensor1D(input->totalSize);
    
    dim3 dimBlock(BLOCK_DIM, 1, 1);
    
    unsigned int gridX = input->totalSize / BLOCK_DIM;
    if (input->totalSize % BLOCK_DIM != 0) {
        gridX++;
    }
    dim3 dimGrid(gridX, 1, 1);

    blockPrefixSumsKoggeStone<<<dimGrid, dimBlock>>>(input, output);

    cudaDeviceSynchronize();

    return output;
}

