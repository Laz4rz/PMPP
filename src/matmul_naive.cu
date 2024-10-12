#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include "cuda_helper.h"

// Assuming row-major matrices 

__global__
void matmulKernel(float *A, float *B, float *C, int n) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x < n && y < n) {
        int acc = 0;
        for(int i=0; i<n; i++) {
            acc += A[y * n + i] * B[i * n + x];
        }
        printf("acc: %d\n", acc);
        C[y * n + x] = acc;
    }
}

void vecprint(int n, float *a) {
    for (int i=0; i<n; i++) {
        printf("%f ", a[i]);
    }
    printf("\n");
}

void matmul(float *A_h, float *B_h, float *C_h, int n) {
    int size = n * n * sizeof(float);
    float *A_d, *B_d, *C_d;

    cudaMallocGuard((void **)&A_d, size);
    cudaMallocGuard((void **)&B_d, size);
    cudaMallocGuard((void **)&C_d, size);

    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);

    float maxThreadsPerDim = sqrt(prop.maxThreadsPerBlock);
    printf("maxThreadsPerDim: %f\n", maxThreadsPerDim);
    dim3 dimGrid(ceil(n / maxThreadsPerDim), ceil(n / maxThreadsPerDim), 1);
    dim3 dimBlock(maxThreadsPerDim, maxThreadsPerDim, 1);
    printf("dimGrid: %d, %d, %d\ndimBlock: %d, %d, %d\n", dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);

    matmulKernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Kernel Launch Error: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main() {
    srand(0);

    int n = 2;
    int nn = n * n;
    float *a = (float *)malloc(nn * sizeof(float));
    float *b = (float *)malloc(nn * sizeof(float));
    float *c = (float *)malloc(nn * sizeof(float));

    for (int i=0; i<nn; i++) {
        a[i] = i;
        b[i] = i;
    }

    vecprint(nn, a);

    matmul(a, b, c, n);

    vecprint(nn, c);

    free(a);
    free(b);
    free(c);

    return 0;
}

