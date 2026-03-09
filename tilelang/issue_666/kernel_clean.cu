__global__ void main_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C) {
  extern __shared__ float buf_dyn_shmem[];

  // Step 1: Clear shared memory (T.clear)
  for (int i = 0; i < 4; ++i) {
    buf_dyn_shmem[threadIdx.x + i * blockDim.x] = 0.0f;
  }
  __syncthreads();  // Only ONE syncthreads — insufficient for async pipeline

  // Step 2: Async loads into same shared memory (pipeline stage 0)
  for (int i = 0; i < 4; ++i) {
    buf_dyn_shmem[threadIdx.x + i * blockDim.x] = A[threadIdx.x + i * blockDim.x];
  }

  // Step 3: Pipeline loop — overlaps with clearing on H100
  for (int ko = 0; ko < 30; ++ko) {
    __syncthreads();
    for (int i = 0; i < 4; ++i) {
      buf_dyn_shmem[threadIdx.x + i * blockDim.x] = B[threadIdx.x + i * blockDim.x + ko * 128];
    }
    __syncthreads();
    C[threadIdx.x] += buf_dyn_shmem[threadIdx.x];
  }
}
