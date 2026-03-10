/*
 * Triton Issue #4736 — WAW race in tl.min butterfly shuffle reduction
 *
 * Manually translated from Triton IR (compute_min_distance_coord.ttir).
 *
 * The kernel finds the nearest coordinate to each input point:
 *   1. tl.sum: reduce squared distances over 8 dims -> [512x32] dist_sq
 *   2. tl.min: find minimum distance coord -> [512] min_idx
 *
 * Bug: tl.min lowering generates a butterfly shuffle where all threads
 * write to the same shared memory address simultaneously — WAW hazard.
 * Detected by compute-sanitizer: 2 hazards.
 *
 * Grid:  1 block
 * Block: 32 threads (one per coordinate)
 */

#include <cuda.h>
#include <float.h>

#define N_COORDS 32

__global__ void min_reduction_buggy(
    const float* __restrict__ dist_sq,  // [32] distances
    const int*   __restrict__ indices,  // [32] coordinate indices
    float*       __restrict__ out_val,  // scalar min distance
    int*         __restrict__ out_idx)  // scalar min index
{
    __shared__ float s_val[N_COORDS];
    __shared__ int   s_idx[N_COORDS];

    int tid = threadIdx.x;

    s_val[tid] = dist_sq[tid];
    s_idx[tid] = indices[tid];
    __syncthreads();

    /*
     * BUG: Triton's butterfly shuffle reduction for tl.min writes
     * to shared memory without proper synchronisation between stages.
     * All threads write to s_val[0] / s_idx[0] simultaneously in the
     * final stage — WAW hazard detected by compute-sanitizer.
     *
     * Missing __syncthreads() between butterfly stages causes threads
     * to read stale values and write to the same location concurrently.
     */
    for (int stride = N_COORDS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other_val = s_val[tid + stride];
            int   other_idx = s_idx[tid + stride];

            /* tie-breaking: prefer lower index on equal values */
            bool tie = (s_val[tid] == other_val) && (s_idx[tid] < other_idx);
            bool lt  = s_val[tid] < other_val;
            bool keep_left = lt || tie;

            /* BUG: no __syncthreads() before writing back —
             * thread tid writes s_val[tid] while thread tid+stride
             * may still be reading s_val[tid+stride] which aliases
             * into the same cache line -> WAW race */
            s_val[tid] = keep_left ? s_val[tid] : other_val;
            s_idx[tid] = keep_left ? s_idx[tid] : other_idx;
            /* MISSING __syncthreads() here */
        }
    }

    if (tid == 0) {
        *out_val = s_val[0];
        *out_idx = s_idx[0];
    }
}