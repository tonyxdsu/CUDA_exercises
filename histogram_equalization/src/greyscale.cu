#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../include/greyscale.cuh"
#include "../include/ppm_image.h"
#include "../include/tensor1D.h"

__global__ void greyscaleKernel(PPMImage* input, Tensor1D<unsigned char>* output) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int index = row * input->width + col;

    if (row < input->height && col < input->width) {
        unsigned char r = input->data[3 * index];
        unsigned char g = input->data[3 * index + 1];
        unsigned char b = input->data[3 * index + 2];
        output->elements[index] = (unsigned char) (0.21 * r + 0.71 * g + 0.07 * b);
    }
}

Tensor1D<unsigned char>* toGreyscaleValues(PPMImage* input) {
    Tensor1D<unsigned char>* output = new Tensor1D<unsigned char>(input->width * input->height);

    unsigned int gridX = input->width / BLOCK_DIM_GREYSCALE;
    if (input->width % BLOCK_DIM_GREYSCALE != 0) {
        gridX++;
    }

    unsigned int gridY = input->height / BLOCK_DIM_GREYSCALE;
    if (input->height % BLOCK_DIM_GREYSCALE != 0) {
        gridY++;
    }

    dim3 dimGrid(gridX, gridY, 1);
    dim3 dimBlock(BLOCK_DIM_GREYSCALE, BLOCK_DIM_GREYSCALE, 1);

    greyscaleKernel<<<dimGrid, dimBlock>>>(input, output);

    cudaDeviceSynchronize();
    return output;
}

