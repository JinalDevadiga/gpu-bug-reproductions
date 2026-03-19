/*
 * Triton Issue #8311 — Missing sync barrier in warp_specialize producer/consumer
 *
 * Manually translated from TTGIR (bad_kernel.ttgir).
 *
 * Bug: warp_specialize=True splits warps into producer (TMA load) and
 * consumer (tt.dot) teams. No synchronisation barrier exists between
 * the async load completing and the consumer warps reading shared memory.
 * Consumer warps proceed to tt.dot before the TMA load has finished —
 * reading stale or partially-written shared memory.
 *
 * Modelled here as a standard shared memory producer-consumer pattern
 * with missing __syncthreads() between store and load phases.
 *
 * Grid:  1 block
 * Block: 32 threads (16 producers + 16 consumers, modelling warp split)
 */

#include <cuda.h>

#define TILE 16

__global__ __launch_bounds__(32) void warp_specialize_buggy(
    const float* __restrict__ x_tile,  // [16] input tile
    const float* __restrict__ y_tile,  // [16] input tile
    float*       __restrict__ out)     // [16] output
{
    __shared__ float s_x[TILE];
    __shared__ float s_y[TILE];

    int tid = threadIdx.x;

    /* Producer warps (threads 0-15): load tiles into shared memory */
    if (tid < TILE) {
        s_x[tid] = x_tile[tid];
        s_y[tid] = y_tile[tid];
        /* BUG: no __syncthreads() after store —
         * consumer warps proceed before load completes */
    }

    /* Consumer warps (threads 16-31): compute dot product */
    if (tid >= TILE) {
        int idx = tid - TILE;
        /* MISSING __syncthreads() here — reads stale s_x / s_y */
        out[idx] = s_x[idx] * s_y[idx];
    }
}