#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../include/greyscale.cuh"
#include "../include/ppm_image.h"

__global__ void greyscaleKernel(PPMImage* input, unsigned char* output) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int index = row * input->width + col;

    if (row < input->height && col < input->width) {
        unsigned char r = input->data[3 * index];
        unsigned char g = input->data[3 * index + 1];
        unsigned char b = input->data[3 * index + 2];
        output[index] = (unsigned char) (0.21 * r + 0.71 * g + 0.07 * b);
    }
}

unsigned char* toGreyscaleValues(PPMImage* input) {
    // TODO greyscale image does not need 3 channels
    unsigned char* output = 0;
    cudaMallocManaged(&output, input->width * input->height * sizeof(unsigned char));
    cudaDeviceSynchronize();

    unsigned int gridX = input->width / BLOCK_DIM;
    if (input->width % BLOCK_DIM != 0) {
        gridX++;
    }

    unsigned int gridY = input->height / BLOCK_DIM;
    if (input->height % BLOCK_DIM != 0) {
        gridY++;
    }

    dim3 dimGrid(gridX, gridY, 1);
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM, 1);

    greyscaleKernel<<<dimGrid, dimBlock>>>(input, output);

    cudaDeviceSynchronize();
    return output;
}

