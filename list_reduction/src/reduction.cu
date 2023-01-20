#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "../include/reduction.cuh"

/**
 * TODO can I pass in a function pointer to a kernel to perform different operations?
*/
__global__ void reductionSumKernel(Tensor1D* input, Tensor1D* output) {
   // TODO
}

Tensor1D* reductionSum(Tensor1D* input) {
   return nullptr;
}

