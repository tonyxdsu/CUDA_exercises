#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "../include/reduction.cuh"

/**
 * TODO can I pass in a function pointer to a kernel to perform different operations?
*/
__global__ void reductionSumKernel(Tensor1D* input, Tensor1D* output) {
   // TODO try not loading first iteration into shared memory but directly storing its sum and compare with loading first iteration into shared memory like textbook implementation
   // nvm can't easily do that because I need to load zeroes into shared memory for indicies that are out of bounds

   // for each block, load 2 elements BLOCK_DIM apart into shared memory
   __shared__ float partialSums[BLOCK_DIM * 2];

   unsigned int inputOffset = blockIdx.x * blockDim.x * 2;
   if (inputOffset + threadIdx.x < input->totalSize) {
      partialSums[threadIdx.x] = input->elements[inputOffset + threadIdx.x];
   }
   else {
      partialSums[threadIdx.x] = 0;
   }

   if (inputOffset + blockDim.x + threadIdx.x < input->totalSize) {
      partialSums[blockDim.x + threadIdx.x] = input->elements[inputOffset + blockDim.x + threadIdx.x];
   }
   else {
      partialSums[blockDim.x + threadIdx.x] = 0;
   }

   // calculate partial sums
   for (unsigned int stride = blockDim.x; stride > 0; stride >>= 1) {
      __syncthreads();

      if (threadIdx.x < stride) {
         partialSums[threadIdx.x] += partialSums[threadIdx.x + stride];
      }
   }

   // store partial sum into output
   if (threadIdx.x == 0) {
      output->elements[blockIdx.x] = partialSums[0];
   }
}

Tensor1D* reductionSum(Tensor1D* input) {
   // TODO try accumulating into an array of floats of partial sums then summing with CPU
   unsigned int outputSize = input->totalSize / (BLOCK_DIM * 2);
   if (input->totalSize % (BLOCK_DIM * 2) != 0) {
      outputSize++;
   }

   Tensor1D* output = new Tensor1D(outputSize);

   unsigned int gridX = input->totalSize / BLOCK_DIM;
   if (gridX % BLOCK_DIM != 0) {
      gridX++;
   }

   dim3 dimGrid(gridX, 1, 1);
   dim3 dimBlock(BLOCK_DIM, 1, 1);

   reductionSumKernel<<<dimGrid, dimBlock>>>(input, output);
   // TODO call the kernel again if outputSize > BLOCK_DIM * N? N being some number to justify whether it's worth launching the kernel again; otherwise let CPU calculate the partial sums.

   
   // TODO why is it a segfault at for (int i = 0; i < calculatedOutput->totalSize; i++) if I don't have this?
   // surely it should just be incorrect result? Isn't the object already created at this point? There's cudaDeviceSynchronize() in the Tensor1D constructor.
   cudaDeviceSynchronize(); 
   return output;
}

