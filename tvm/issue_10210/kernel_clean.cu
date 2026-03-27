/*
 * TVM Issue #10210 — Parallel Reduction Axis WAW Race
 *
 * Manually written to model the buggy TIR pattern.
 *
 * Bug: TVM's parallel() on reduction axis k launches 64 CPU threads
 * all writing to the same accumulator C[i*64+j] without synchronisation.
 *
 * Modelled here as a CUDA kernel to enable GPUVerify analysis.
 * In reality this runs on CPU OpenMP threads — no CUDA kernel is generated.
 *
 * Block: 64 threads (one per k iteration)
 * Grid:  1 block
 */

#include <cuda.h>

__global__ void parallel_reduction_buggy(
    float* C,
    const float* A,
    const float* B)
{
    int k = threadIdx.x;  // 64 threads, one per k
    int i = 0, j = 0;

    /* BUG: all 64 threads read and write C[i*64+j] = C[0]
     * simultaneously with no atomic or __syncthreads()
     * Thread 0:  C[0] = C[0] + A[0]*B[0]
     * Thread 1:  C[0] = C[0] + A[1]*B[64]   <- reads same stale C[0]
     * ...
     * Last writer wins -> 63 updates lost -> wrong result */
    C[i*64+j] = C[i*64+j] + A[i*64+k] * B[k*64+j];
}