#ifndef CUDA_HELPER_H
#define CUDA_HELPER_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

void cudaMallocGuard(void **devPtr, size_t size);

#endif // CUDA_HELPER_H