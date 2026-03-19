__global__ __launch_bounds__(64) void test_kernel_kernel(int* __restrict__ a) {
  extern __shared__ int shared[];
  shared[(((int)threadIdx.x) ^ 1)] = 0;
  __syncthreads();
  atomicAdd(&(shared[((int)threadIdx.x)]), 1);
  a[((int)threadIdx.x)] = shared[(((int)threadIdx.x) ^ 32)];
}
