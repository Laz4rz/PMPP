#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include "cuda_helper.h"

#define TILE_WIDTH 16

// Assuming row-major matrices 

// that is quite convoluted in PMPP, but the code looks as it does
// cause the TILE_WIDTH is equal to blockDim.x and blockDim.y
// it doesnt have to be (and usually isnt?), but... simplicity 
// we only care about 
__global__
void matmulKernel(float *A, float *B, float *C, int n) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x; 
    int ty = threadIdx.y;
    int bx = blockIdx.x; 
    int by = blockIdx.y;

    int row = by * TILE_WIDTH + ty;
    int column = bx * TILE_WIDTH + tx;

        float acc = 0.0;

        // this for-loop is called strip-mining
        // takes a long running loop and breaks it into parts
        for (int tile=0; tile<(n + TILE_WIDTH - 1)/TILE_WIDTH; ++tile) {
            // moving horizontally
            // [row][column]
            if (row < n && tile * TILE_WIDTH + tx < n)
                Mds[ty][tx] = A[row * n + tile * TILE_WIDTH + tx];
            // moving vertically
            if (column < n && (ty + tile * TILE_WIDTH ) < n) 
                Nds[ty][tx] =  B[tile * TILE_WIDTH * n + ty * n + column];
            __syncthreads(); // read-after-write (true dependence)
            
            for (int i=0; i<TILE_WIDTH; ++i) {
                acc += Mds[ty][i] * Nds[i][tx];
            }
            __syncthreads();
            // write-after-read (false dependence)
        // decreases the number of global memory accesses by factor of TILE_WIDTH
        }
    if (row < n && column < n) {
        C[row * n + column] = acc;
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
    float threads = TILE_WIDTH;
    dim3 dimGrid((n + TILE_WIDTH - 1) / TILE_WIDTH, (n + TILE_WIDTH - 1) / TILE_WIDTH, 1);
    dim3 dimBlock(threads, threads, 1);
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

