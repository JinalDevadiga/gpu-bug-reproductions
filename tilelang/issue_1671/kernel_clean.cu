__global__ void buggy_kernel_kernel(const float* __restrict__ A, float* __restrict__ B) {
  for (int j = 0; j < 64; ++j) {
    B[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 64)) + j)] =
    (A[(((((int)blockIdx.x) * 4096) + (((int)threadIdx.x) * 64)) + j)] * 2.0f);
  }
}
