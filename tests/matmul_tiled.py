import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np

def print_row_major(matrix, m, n):
    for i in range(m):
        for j in range(n):
            print(matrix[i * n + j], end=" ")
        print()

# Define the CUDA kernel as a string
cuda_code = """
#define TILE_WIDTH 2

__global__
void matmulKernel(float *A, float *B, float *C, int m, int n, int o) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x; 
    int ty = threadIdx.y;
    int bx = blockIdx.x; 
    int by = blockIdx.y;

    int row = by * TILE_WIDTH + ty;
    int column = bx * TILE_WIDTH + tx;

    float acc = 0.0;
    int globMemAcc = 0;
    float temp;

    for (int tile=0; tile<(n + TILE_WIDTH - 1)/TILE_WIDTH; ++tile) {

        if (row < m && tile * TILE_WIDTH + tx < n) {
            Mds[ty][tx] = A[row * n + tx + tile * TILE_WIDTH];
            globMemAcc += 1;
        }
        else {
            Mds[ty][tx] = 0.0;
        }

        if (column < o && (ty + tile * TILE_WIDTH ) < n) {
            Nds[ty][tx] =  B[column + ty * o + tile * TILE_WIDTH * o];
            globMemAcc += 1;
        }
        else {
            Nds[ty][tx] = 0.0;
        }
        __syncthreads();                           
  
        for (int i=0; i<TILE_WIDTH; ++i) {
            temp = Mds[ty][i] * Nds[i][tx];
            // if (column == 0 && row == 0) 
                // printf("Mds[%d][%d] * Nds[%d][%d] = %f * %f = %d\\n", ty, i, i, tx, Mds[ty][i], Nds[i][tx], temp);
            acc += temp;
        }
        __syncthreads();
    }
    // printf("Thread (%d, %d) accessed %d global memory locations\\n", row, column, globMemAcc);
    if (row < m && column < o) {
        C[row * o + column] = acc;
    }
}
"""

# Compile the CUDA kernel
module = SourceModule(cuda_code)
matmul_kernel = module.get_function("matmulKernel")

# Allocate and initialize arrays on the host
N = 5
M = 7
O = 9
a = np.arange(N*M).astype(np.float32)
b = np.arange(M*O).astype(np.float32)
c = np.zeros(N*O).astype(np.float32)

print("A:")
print_row_major(a, N, M)
print("B:")
print_row_major(b, M, O)
print("C:")
print_row_major(c, N, O)

# Allocate device memory and transfer data
a_gpu = cuda.mem_alloc(a.nbytes)
b_gpu = cuda.mem_alloc(b.nbytes)
c_gpu = cuda.mem_alloc(c.nbytes)

cuda.memcpy_htod(a_gpu, a)
cuda.memcpy_htod(b_gpu, b)

# Launch the kernel
TILE_WIDTH = 2
block_size = (TILE_WIDTH, TILE_WIDTH, 1)
grid_size = ((O + TILE_WIDTH - 1) // TILE_WIDTH, (N + TILE_WIDTH - 1) // TILE_WIDTH, 1)

print("Block size:", block_size)
print("Grid size:", grid_size)

matmul_kernel(a_gpu, b_gpu, c_gpu, np.int32(N), np.int32(M), np.int32(O), block=block_size, grid=grid_size)

# Catch CUDA errors
try:
    cuda.Context.synchronize()
except cuda.Error as e:
    print(f"CUDA Error: {e}")

# Retrieve the result
cuda.memcpy_dtoh(c, c_gpu)

gt = np.dot(a.reshape(N, M), b.reshape(M, O)).reshape(-1)

passed = np.allclose(c, gt)
assert passed, f"Error: {c} != {gt}"
print(f"Result ({passed}):")
print_row_major(c, N, O)

# tests
if True:
    for i in range(50):
        M = np.random.randint(1, 100)
        N = np.random.randint(1, 100)
        O = np.random.randint(1, 100)
        a = np.random.rand(N * M).astype(np.float32)
        b = np.random.rand(M * O).astype(np.float32)
        c = np.zeros(N*O).astype(np.float32)

        a_gpu = cuda.mem_alloc(a.nbytes)
        b_gpu = cuda.mem_alloc(b.nbytes)
        c_gpu = cuda.mem_alloc(c.nbytes)

        cuda.memcpy_htod(a_gpu, a)
        cuda.memcpy_htod(b_gpu, b)

        block_size = (TILE_WIDTH, TILE_WIDTH, 1)
        grid_size = ((O + TILE_WIDTH - 1) // TILE_WIDTH, (N + TILE_WIDTH - 1) // TILE_WIDTH, 1)

        matmul_kernel(a_gpu, b_gpu, c_gpu, np.int32(N), np.int32(M), np.int32(O), block=block_size, grid=grid_size)
        cuda.Context.synchronize()

        cuda.memcpy_dtoh(c, c_gpu)

        gt = np.dot(a.reshape(N, M), b.reshape(M, O)).reshape(-1)
        print(f"({M}, {N}, {O}) Passed: {np.allclose(c, gt)}")
