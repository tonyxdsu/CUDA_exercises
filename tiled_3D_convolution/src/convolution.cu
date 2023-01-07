#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include "../include/convolution.cuh"

/**
 * "Strategy 2", using smaller subset of threads to calculate but all threads will load into tile.
 * TODO Textbook version might have less redundant if statements, not sure.
*/
__global__ void convolutionKernel(Tensor3D* input, Tensor3D* output) {
    int column = TILE_DIM * blockIdx.x + threadIdx.x - MASK_RADIUS;
    int row = TILE_DIM * blockIdx.y + threadIdx.y - MASK_RADIUS;
    int aisle = TILE_DIM * blockIdx.z + threadIdx.z - MASK_RADIUS;

    __shared__ float tile[TILE_WITH_HALO_DIM][TILE_WITH_HALO_DIM][TILE_WITH_HALO_DIM];

    bool isWithinTensor = column >= 0 && column < input->xDim && row >= 0 && row < input->yDim && aisle >= 0 && aisle < input->zDim;

    if (isWithinTensor) {
        tile[threadIdx.z][threadIdx.y][threadIdx.x] = input->elements[aisle * (input->yDim * input->xDim) + row * input->xDim + column];
    }
    else {
        tile[threadIdx.z][threadIdx.y][threadIdx.x] = 0;
    }

    __syncthreads();

    bool isValidTileToCalculate = threadIdx.x >= MASK_RADIUS && threadIdx.x <= TILE_DIM && 
                                  threadIdx.y >= MASK_RADIUS && threadIdx.y <= TILE_DIM && 
                                  threadIdx.z >= MASK_RADIUS && threadIdx.z <= TILE_DIM &&
                                  isWithinTensor;

    if (isValidTileToCalculate) {
        float sum = 0;
        for (int z = -MASK_RADIUS; z <= MASK_RADIUS; z++) {
            for (int y = -MASK_RADIUS; y <= MASK_RADIUS; y++) {
                for (int x = -MASK_RADIUS; x <= MASK_RADIUS; x++) {
                    sum += tile[threadIdx.z + z][threadIdx.y + y][threadIdx.x + x] * MASK_CONSTANT[z + MASK_RADIUS][y + MASK_RADIUS][x + MASK_RADIUS];
                }
            }
        }

        output->elements[aisle * (input->yDim * input->xDim) + row * input->xDim + column] = sum;
    }
}

Tensor3D* convolution(Tensor3D* input, Tensor3D* mask) {
    // TODO async then sync faster?
    cudaMemcpyToSymbol(MASK_CONSTANT, mask->elements, mask->totalSize * sizeof(float), 0, cudaMemcpyHostToDevice);

    Tensor3D* output = new Tensor3D(input->xDim, input->yDim, input->zDim);

    unsigned int gridX = input->xDim / TILE_DIM;
    if (input->xDim % TILE_DIM != 0) {
        gridX++;
    }

    unsigned int gridY = input->yDim / TILE_DIM;
    if (input->yDim % TILE_DIM != 0) {
        gridY++;
    }

    unsigned int gridZ = input->zDim / TILE_DIM;
    if (input->zDim % TILE_DIM != 0) {
        gridZ++;
    }

    // TODO occupancy calculation
    dim3 dimGrid(gridX, gridY, gridZ);
    dim3 dimBlock(TILE_WITH_HALO_DIM, TILE_WITH_HALO_DIM, TILE_WITH_HALO_DIM);

    // TODO calculate available resources (ie registers per block) or kernel does not run with no error messages
    // TODO how to get error messages for kernel not running? nvprof?
    convolutionKernel<<<dimGrid, dimBlock>>>(input, output);

    // TODO kernel calls are async so does this function return early without synchronize?
    cudaDeviceSynchronize();

    return output;
}

