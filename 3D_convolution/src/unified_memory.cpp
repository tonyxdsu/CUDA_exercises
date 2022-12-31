// based on https://developer.nvidia.com/blog/unified-memory-in-cuda-6/

#include "cuda_runtime.h"
#include "../include/unified_memory.h"

void* UnifiedMemory::operator new(size_t len) {
    void* ptr;
    cudaMallocManaged(&ptr, len);
    cudaDeviceSynchronize();
    return ptr;
}

void UnifiedMemory::operator delete(void* ptr) {
    cudaDeviceSynchronize();
    cudaFree(ptr);
}