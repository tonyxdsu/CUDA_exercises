#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../include/prefix_sum.cuh"
#include "../include/kogge_stone.cuh"

#include <stdio.h>

/**
 * TODO can I pass in a function pointer to a kernel to perform different operations?
*/
__global__ void addBlockSumsToOriginal(Tensor1D<float>* blockPrefixSums, Tensor1D<float>* blockTotalSums,Tensor1D<float>* prefixSums) {
   unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

   // TODO every block uses same element from blockTotalSums, it will be cached automatically right? No need to use shared memory?

   if (index < blockPrefixSums->totalSize) {
      if (blockIdx.x == 0) {
         prefixSums->elements[index] = blockPrefixSums->elements[index];
      }
      else {
         prefixSums->elements[index] = blockPrefixSums->elements[index] + blockTotalSums->elements[blockIdx.x - 1];
      }
   }
}

/**
 * @param blockPrefixSums element[k * BLOCK_DIM - 1] will have the block sum for each block k, k in [0, blockIdx.x)
 * @param blockTotalSums output element [k] will have the block sums up from block [0, k] of blockPrefixSums
*/
__global__ void getBlockTotals(Tensor1D<float>* blockPrefixSums, Tensor1D<float>* blockTotalSums) {
   unsigned int indexBlockTotals  = blockIdx.x * blockDim.x + threadIdx.x;
   unsigned int indexBlockPrefixSums = (indexBlockTotals + 1) * BLOCK_DIM - 1; 

   if (indexBlockTotals < blockTotalSums->totalSize) {
      if (indexBlockPrefixSums < blockPrefixSums->totalSize) {
         blockTotalSums->elements[indexBlockTotals] = blockPrefixSums->elements[indexBlockPrefixSums];
      }
      else {
         blockTotalSums->elements[indexBlockTotals] = blockPrefixSums->elements[blockPrefixSums->totalSize - 1];
      }
   }
}

Tensor1D<float>* prefixSum(Tensor1D<float>* input) {
   // kernel 1: produce prefix sums for each block
   // =================================================================================================
   Tensor1D<float>* blockPrefixSums = blockPrefixSumsKoggeStone(input);
   
   unsigned int gridXBlockTotal= input->totalSize / BLOCK_DIM;
   if (input->totalSize % BLOCK_DIM != 0) {
      gridXBlockTotal++;
   }

   // kernel 2: extract the totals of each block such that element[k] will have the sum of all elements in each kth block in input
   // =================================================================================================
   Tensor1D<float>* blockTotals = new Tensor1D<float>(gridXBlockTotal);

   dim3 dimGridBlockTotal(gridXBlockTotal, 1, 1);
   dim3 dimBlockBlockTotal(BLOCK_DIM, 1, 1);
   getBlockTotals<<<dimGridBlockTotal, dimBlockBlockTotal>>>(blockPrefixSums, blockTotals);

   cudaDeviceSynchronize(); // TODO not needed if using same default streams for both kernel calls?

   // kernel 3: sum the block totals such that element[k] will have the sum of all block totals in input[0, k]
   // =================================================================================================

   // TODO handle case where this does not fit inside a single block
   Tensor1D<float>* blockTotalSums = blockPrefixSumsKoggeStone(blockTotals);

   // kernel 4: add the summed block totals to the original input to produce the final prefix sums vector
   // =================================================================================================
   Tensor1D<float>* prefixSums = new Tensor1D<float>(input->totalSize);

   unsigned int gridXSumToOriginal = input->totalSize / BLOCK_DIM;
   if (input->totalSize % BLOCK_DIM != 0) {
      gridXSumToOriginal++;
   }

   dim3 dimGridSumToOriginal(gridXSumToOriginal, 1, 1);
   dim3 dimBlockSumToOriginal(BLOCK_DIM, 1, 1);
   addBlockSumsToOriginal<<<dimGridSumToOriginal, dimBlockSumToOriginal>>>(blockPrefixSums, blockTotalSums, prefixSums);

   cudaDeviceSynchronize();

   delete blockTotals;
   delete blockTotalSums;

   return prefixSums;
}

