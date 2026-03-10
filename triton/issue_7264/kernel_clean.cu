/*
 * Triton Issue #7264 — WAW race in tl.sum butterfly shuffle reduction
 *
 * Manually translated from Triton IR (reduction_kernel.ttir).
 * tt.reduce with addf over 1024 elements — butterfly shuffle sum.
 *
 * Bug: Missing __syncthreads() between butterfly stages causes threads
 * to read shared memory being written by other threads in the same stage.
 * All threads eventually write their accumulated sum to s_data[0] — WAW.
 *
 * Detected by compute-sanitizer: 2 hazards.
 *
 * Grid:  1 block
 * Block: 1024 threads
 */

#include <cuda.h>

#define N 1024

__global__ void reduction_buggy(
    const float* __restrict__ input,
    float*       __restrict__ output)
{
    __shared__ float s_data[N];

    int tid = threadIdx.x;

    s_data[tid] = input[tid];
    __syncthreads();

    /*
     * BUG: Missing __syncthreads() between butterfly stages.
     * Thread tid reads s_data[tid + stride] while thread tid+stride
     * may still be writing s_data[tid+stride] from the previous stage.
     * In the final stage all threads write to s_data[0] — WAW hazard.
     */
    for (int stride = N / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_data[tid] += s_data[tid + stride];
            /* MISSING __syncthreads() here */
        }
    }

    if (tid == 0) {
        *output = s_data[0];
    }
}