/*
 * Triton Issue #4233 — scatter_add WAW race (non-atomic read-modify-write)
 *
 * Manually translated from Triton IR (scatter_add_racy.ttir).
 *
 * Bug: non-atomic load → add → store on duplicate indices causes WAW race.
 * When multiple threads map to the same output index, each reads the same
 * stale value, computes its update, and last writer wins — updates lost.
 *
 * index = [0, 1, 2, 2, 1, 0]
 * src   = [1, 1, 1, 1, 1, 1]
 * expected out = [2, 2, 2]   (each index appears twice)
 * buggy    out = [1, 1, 1]   (last writer wins — one update lost per index)
 *
 * Grid:  1 block
 * Block: 6 threads (one per element, BLOCK_SIZE=8 with mask for 6 valid)
 */

#include <cuda.h>

#define BLOCK_SIZE 8
#define N 6

__global__ __launch_bounds__(8) void scatter_add_racy(
    const int*   __restrict__ index,   // [6] scatter indices
    const float* __restrict__ src,     // [6] values to add
    float*       __restrict__ out)     // [3] output accumulator
{
    int tid = threadIdx.x;
    if (tid >= N) return;

    int   idx = index[tid];
    float val = src[tid];

    /* BUG: non-atomic read-modify-write
     * Thread 0 and Thread 5 both read out[0] = 0.0
     * Both compute 0.0 + 1.0 = 1.0
     * Both write 1.0 to out[0] — one update lost
     * Expected: out[0] = 2.0 */
    float cur = out[idx];        /* <- read: stale value for duplicate indices */
    out[idx]  = cur + val;       /* <- write: WAW race on duplicate indices */
}