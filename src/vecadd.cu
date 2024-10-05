#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include "cuda_helper.h"

// could be defined as a macro to reduce function call overhead
// # define cudaMallocGuard(devPtr, size) { \...
// void cudaMallocGuard( void **devPtr, size_t size ) {
//     cudaError_t err = cudaMalloc(devPtr, size);
//     if (err != cudaSuccess) {
//         fprintf(stderr, "Error: %s\n", cudaGetErrorString(err));
//         exit(EXIT_FAILURE);
//     }
// }

__global__
void vecAddKernel(float *A, float *B, float *C, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        printf("Thread %d: %f + %f\n", i, A[i], B[i]);
        C[i] = A[i] + B[i];
    }
}

__global__
void vecSubKernel(float *A, float *B, float *C, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        printf("Thread %d: %f + %f\n", i, A[i], B[i]);
        C[i] = A[i] - B[i];
    }
}

__global__
void vecMulKernel(float *A, float *B, float *C, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        printf("Thread %d: %f + %f\n", i, A[i], B[i]);
        C[i] = A[i] * B[i];
    }
}

// GPU -- device
// Cuda kernels have access to: threadIdx, blockIdx, blockDim, gridDim
// _h, and _d are used to denote host and device memory (host -- CPU, device -- GPU)
// __global__ is a CUDA keyword that indicates a function that runs on the GPU and is called from the CPU
// __device__ is a CUDA keyword that indicates a function that runs on the GPU and is called from the GPU
// __host__ is a CUDA keyword that indicates a function that runs on the CPU and is called from the CPU (default)
// above keywords can be combined, e.g. __host__ __device__ to produce two versions of the object code
// <<<...>>> is a CUDA syntax to specify the number of blocks and threads per block
// current max threads per block is 1024 (4th edition of PMPP)
void devvecadd(float *A_h, float *B_h, float *C_h, int n) {
    int size = n * sizeof(float);
    float * A_d, *B_d, *C_d;

    cudaMallocGuard((void **)&A_d, size);
    cudaMallocGuard((void **)&B_d, size);
    cudaMallocGuard((void **)&C_d, size);

    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    vecAddKernel<<<ceil(n / 256.0), 256>>>(A_d, B_d, C_d, n);

    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

void vec_vec_elementwise_op(void (*kernel)(float*, float*, float*, int), float *A_h, float *B_h, float *C_h, int n) {
    int size = n * sizeof(float);
    float *A_d, *B_d, *C_d;

    cudaMallocGuard((void **)&A_d, size);
    cudaMallocGuard((void **)&B_d, size);
    cudaMallocGuard((void **)&C_d, size);

    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    dim3 dimGrid(ceil(n / 256.0), 1, 1);
    dim3 dimBlock(256, 1, 1);
    printf("dimGrid: %d, %d, %d\ndimBlock: %d, %d, %d\n", dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);
    kernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, n);

    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

// CPU -- host
void vecadd(int n, float *a, float *b, float *c) {
    for (int i=0; i<n; i++) {
        c[i] = a[i] + b[i];
    }
}


void vecprint(int n, float *a) {
    for (int i=0; i<n; i++) {
        printf("%f ", a[i]);
    }
    printf("\n");
}

int main() {
    srand(0);

    int n = 4;
    float *a = (float *)malloc(n * sizeof(float));
    float *b = (float *)malloc(n * sizeof(float));
    float *c = (float *)malloc(n * sizeof(float));

    if (a == NULL || b == NULL || c == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    for (int i=0; i<n; i++) {
        a[i] = rand() % 5;
        b[i] = rand() % 5;
    } 

    vecprint(n, a);
    vecprint(n, b);

    // vecadd(n, a, b, c);
    // devvecadd(a, b, c, n);
    vec_vec_elementwise_op(vecAddKernel, a, b, c, n);

    vecprint(n, c);

    free(a);
    free(b);
    free(c);

    return 0;
}