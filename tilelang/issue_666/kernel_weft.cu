/*
 * TileLang Issue #666 — Incorrect Results When Clearing Shared Memory
 * Before Pipelined Loops
 *
 * Simplified version for GPUVerify analysis.
 * Preserves the key buggy pattern from the real kernel:
 * - uint4 shared memory loads
 * - 3-stage pipeline (ko % 3) modulo indexing
 * - T.clear zeroing BEFORE pipeline loads
 * - Only ONE __syncthreads() between clear and first pipeline load
 *
 * Simplifications for GPUVerify tractability:
 * - Block size reduced: 128 -> 4 threads
 * - Pipeline iterations reduced: 30 -> 3
 * - Stage size reduced accordingly
 *
 * Bug: On H100, pipeline uses TMA async transfers. __syncthreads()
 * does NOT wait for TMA to complete — need mbarrier instead.
 * Clear and async loads overlap -> stale shared memory -> wrong results.
 *
 * Block: 4 threads
 * Grid:  1 block
 */

#include <cuda.h>

#define BLOCK_SIZE 4
#define NUM_STAGES 3
#define STAGE_SIZE 4
#define NUM_ITER   3

__global__ __launch_bounds__(4) void main_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C)
{
    extern __shared__ float buf_dyn_shmem[];

    int tid = threadIdx.x;

    /* Step 1: T.clear(B_shared) — zero shared memory
     * This is the BUGGY operation from the real kernel:
     * *(uint4*)(buf_dyn_shmem + ...) = make_uint4(0,0,0,0) */
    for (int i = 0; i < NUM_STAGES; ++i) {
        buf_dyn_shmem[i * STAGE_SIZE + tid] = 0.0f;
    }

    /* Only ONE __syncthreads() — insufficient on H100!
     * Real kernel: __syncthreads() after T.clear, before pipeline loads
     * On H100: TMA async transfers are NOT covered by __syncthreads() */
    __syncthreads();

    /* Step 2: Prologue — load first two stages
     * Real kernel: *(uint4*)(buf_dyn_shmem + ...) = *(uint4*)(A + ...) */
    buf_dyn_shmem[0 * STAGE_SIZE + tid] = A[tid];
    buf_dyn_shmem[1 * STAGE_SIZE + tid] = B[tid];

    /* Step 3: Main pipeline loop — 3-stage rotating buffer
     * Real kernel: for (int ko = 0; ko < 30; ++ko)
     *   prefetch stage (ko+2)%3, compute stage ko%3 */
    float acc = 0.0f;

    for (int ko = 0; ko < NUM_ITER; ++ko) {
        __syncthreads();

        /* Prefetch into (ko+2)%3 — matches real kernel:
         * *(uint4*)(buf_dyn_shmem + ((ko+2)%3)*4096 + ...) = *(uint4*)(A+...) */
        int prefetch_stage = (ko + 2) % NUM_STAGES;
        buf_dyn_shmem[prefetch_stage * STAGE_SIZE + tid] = A[ko * BLOCK_SIZE + tid];

        /* BUG: On H100, the write above via TMA async can overlap with
         * the T.clear zeroing — __syncthreads() does NOT prevent this.
         * Missing mbarrier.arrive/wait here. */
        __syncthreads();

        /* Compute using stage ko%3 */
        int compute_stage = ko % NUM_STAGES;
        acc += buf_dyn_shmem[compute_stage * STAGE_SIZE + tid];
    }

    C[tid] = acc;
}