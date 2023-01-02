#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "../include/convolution.cuh"

__global__ void convolutionKernel(Tensor3D* input, Tensor3D* mask, Tensor3D* output) {
    int column = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int aisle = blockDim.z * blockIdx.z + threadIdx.z;

    int maskDimDivide2 = mask->xDim / 2;

    // TODO no way explicitly storing input->xyzDim in variable is faster right?
    // will be kept in register for every iteration right?

    if (column < input->xDim && row < input->yDim && aisle < input->zDim) {

        float sum = 0;

        for (int z = -maskDimDivide2; z <= maskDimDivide2; z++) {
            for (int y = -maskDimDivide2; y <= maskDimDivide2; y++) {
                for (int x = -maskDimDivide2; x <= maskDimDivide2; x++) {
                    if (x + column >= 0 && x + column < input->xDim &&
                        y + row >= 0 && y + row < input->yDim &&
                        z + aisle >= 0 && z + aisle < input->zDim) {
                        
                        sum += input->elements[(z + aisle) * input->xDim * input->yDim + (y + row) * input->xDim + (x + column)] *
                            mask->elements[(z + maskDimDivide2) * mask->xDim * mask->yDim + (y + maskDimDivide2) * mask->xDim + (x + maskDimDivide2)];
                    }
                }
            }
        }

        output->elements[aisle * input->xDim * input->yDim + row * input->xDim + column] = sum;
    }
}

Tensor3D* convolution(Tensor3D* input, Tensor3D* mask) {
    Tensor3D* output = new Tensor3D(input->xDim, input->yDim, input->zDim);

    unsigned int gridX = input->xDim / BLOCK_DIM;
    if (input->xDim % BLOCK_DIM != 0) {
        gridX++;
    }

    unsigned int gridY = input->yDim / BLOCK_DIM;
    if (input->yDim % BLOCK_DIM != 0) {
        gridY++;
    }

    unsigned int gridZ = input->zDim / BLOCK_DIM;
    if (input->zDim % BLOCK_DIM != 0) {
        gridZ++;
    }

    // TODO occupancy calculation
    dim3 dimGrid(gridX, gridY, gridZ);
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM, BLOCK_DIM);

    // TODO calculate available resources (ie registers per block) or kernel does not run with no error messages
    // TODO how to get error messages for kernel not running? nvprof?
    convolutionKernel<<<dimGrid, dimBlock>>>(input, mask, output);

    // TODO kernel calls are async so does this function return early without synchronize?
    cudaDeviceSynchronize();

    return output;
}

