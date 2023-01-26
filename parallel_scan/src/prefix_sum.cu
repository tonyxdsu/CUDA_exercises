#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../include/prefix_sum.cuh"

/**
 * TODO can I pass in a function pointer to a kernel to perform different operations?
*/
__global__ void addBlockSumsToOriginal(Tensor1D* input, Tensor1D* output) {


}

Tensor1D* prefixSum(Tensor1D* input) {
   Tensor1D* output = new Tensor1D(input->totalSize);
   return output;
}

