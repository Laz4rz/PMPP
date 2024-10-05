#include "../include/cuda_helper.h"

void cudaMallocGuard( void **devPtr, size_t size ) {
    cudaError_t err = cudaMalloc(devPtr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
