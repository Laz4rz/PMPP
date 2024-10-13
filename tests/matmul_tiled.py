import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np

# Define the CUDA kernel as a string
cuda_code = """
#define TILE_WIDTH 16

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
"""

# Compile the CUDA kernel
module = SourceModule(cuda_code)
matmul_kernel = module.get_function("matmulKernel")

# Allocate and initialize arrays on the host
N = 6
M = 6
a = np.arange(N*M).astype(np.float32)
b = np.arange(N*M).astype(np.float32)
c = np.zeros_like(a)

# Allocate device memory and transfer data
a_gpu = cuda.mem_alloc(a.nbytes)
b_gpu = cuda.mem_alloc(b.nbytes)
c_gpu = cuda.mem_alloc(c.nbytes)

cuda.memcpy_htod(a_gpu, a)
cuda.memcpy_htod(b_gpu, b)

# Launch the kernel
TILE_WIDTH = 16
block_size = (TILE_WIDTH, TILE_WIDTH, 1)
grid_size = ((N + TILE_WIDTH - 1) // TILE_WIDTH, (N + TILE_WIDTH - 1) // TILE_WIDTH, 1)

matmul_kernel(a_gpu, b_gpu, c_gpu, np.int32(N), block=block_size, grid=grid_size)

# Catch CUDA errors
try:
    cuda.Context.synchronize()
except cuda.Error as e:
    print(f"CUDA Error: {e}")

# Retrieve the result
cuda.memcpy_dtoh(c, c_gpu)

gt = np.dot(a.reshape(N, M), b.reshape(M, N)).reshape(-1)

assert np.allclose(c, gt), f"Error: {c} != {gt}"
print("Result:", c)
