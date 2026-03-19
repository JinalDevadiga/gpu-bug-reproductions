__global__ __launch_bounds__(1024) void main_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C) {
  extern __shared__ float buf_dyn_shmem[];

  // Pipeline prologue - load stages 0 and 1
  for (int i = 0; i < 4; ++i)
    buf_dyn_shmem[i * blockDim.x + threadIdx.x] = A[i * blockDim.x + threadIdx.x];

  for (int i = 0; i < 4; ++i)
    buf_dyn_shmem[4096 + i * blockDim.x + threadIdx.x] = B[i * blockDim.x + threadIdx.x];

  // Pipelined loop - WAR hazard here
  for (int ko = 0; ko < 254; ++ko) {
    __syncthreads();
    // Write to future stage (ko+2)%3
    // BUG: this slot may still be read by mma from previous iteration
    for (int i = 0; i < 4; ++i)
      buf_dyn_shmem[((ko + 2) % 3) * 4096 + i * blockDim.x + threadIdx.x] =
        A[ko * blockDim.x * 4 + i * blockDim.x + threadIdx.x];

    __syncthreads();
    // Read from current stage (ko%3) and compute
    float sum = 0.0f;
    for (int i = 0; i < 4; ++i)
      sum += buf_dyn_shmem[(ko % 3) * 4096 + i * blockDim.x + threadIdx.x] *
             buf_dyn_shmem[(ko % 3) * 4096 + i * blockDim.x + threadIdx.x + 12288];

    C[blockIdx.x * blockDim.x + threadIdx.x] += sum;
  }
}
