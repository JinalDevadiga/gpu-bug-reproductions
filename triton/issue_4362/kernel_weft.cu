/*
 * Triton Issue #4362 — tl.associative_scan Wrong Results with reverse=True
 *
 * Manually translated from Triton's lowering of associative_scan.
 *
 * Bug: When reverse=True, Triton scans elements in the wrong order —
 * effectively broadcasting the final accumulated value to all positions
 * instead of computing a proper reverse prefix scan.
 *
 * Grid:  1 block
 * Block: 4 threads (one per element)
 */

#include <cuda.h>

#define N 4

__global__ __launch_bounds__(4) void associative_scan_buggy(
    const float* __restrict__ exp_in,
    const float* __restrict__ vals_in,
    float* __restrict__ exp_out,
    float* __restrict__ vals_out)
{
    __shared__ float s_exp[N];
    __shared__ float s_vals[N];

    int tid = threadIdx.x;

    s_exp[tid]  = exp_in[tid];
    s_vals[tid] = vals_in[tid];
    __syncthreads();

    float acc_exp  = s_exp[N - 1 - tid];
    float acc_vals = s_vals[N - 1 - tid];

    for (int i = N - 1; i > (N - 1 - tid); i--) {
        float fr = s_exp[i];
        float xr = s_vals[i];
        acc_exp  = fr * acc_exp;
        acc_vals = fr * acc_vals + xr;
        /* MISSING __syncthreads() between iterations */
    }
    /* No final sync before writeback */
    exp_out[tid]  = acc_exp;
    vals_out[tid] = acc_vals;
}