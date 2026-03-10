/*
 * TVM Issue #17439 — ThreadStorageSync Must Run After MergeSharedMemoryAllocations
 *
 * Manually translated from TVM TIR (After MergeSharedMemory stage).
 *
 * Bug: ThreadSync inserted barriers for separate A.shared / B.shared / C.shared.
 * After MergeSharedMemoryAllocations, C.shared reuses the same address space
 * as A.shared. The barrier placed by ThreadSync no longer protects the aliased
 * region — Store C.shared can overwrite A.shared before other threads finish
 * reading it.
 *
 * Grid:  blockIdx.x  in [0,4),  blockIdx.y in [0,4)
 * Block: threadIdx.x in [0,16), threadIdx.y in [0,16)
 */

#include <cuda.h>

__global__ void tvm_matmul_buggy(
    const float* __restrict__ A,   // [64][64]
    const float* __restrict__ B,   // [64][64]
    float*       __restrict__ C)   // [64][64]
{
    /* After MergeSharedMemoryAllocations:
     * C.shared occupies [0..255], A.shared and B.shared are allocated
     * separately but their base address aliases into C.shared's region.
     * We model this aliasing explicitly with a single shared buffer. */
    __shared__ float C_shared[256];   // C.shared — 16x16 accumulator tile
    __shared__ float A_shared[1];     // A.shared — aliases C_shared base
    __shared__ float B_shared[1];     // B.shared — aliases C_shared base

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    /* Initialise accumulator tile */
    for (int ic = 0; ic < 16; ic++) {
        for (int jc = 0; jc < 16; jc++) {
            __syncthreads();   /* tvm_storage_sync — placed pre-merge, now misplaced */
            C_shared[ic * 16 + jc] = 0.0f;
        }
    }

    /* Reduction loop over k */
    for (int ic = 0; ic < 16; ic++) {
        for (int jc = 0; jc < 16; jc++) {
            int cse_var_1 = ic * 16 + jc;
            for (int k = 0; k < 64; k++) {
                /* Load A tile into A.shared */
                __syncthreads();   /* tvm_storage_sync — pre-merge position */
                if (ty < 1 && tx < 1) {
                    A_shared[ty + tx] = A[
                        blockIdx.y * 1024 + ty * 64 + ic * 64 + tx + k];
                }

                /* Load B tile into B.shared */
                if (ty < 1 && tx < 1) {
                    B_shared[ty + tx] = B[
                        ty * 64 + k * 64 + blockIdx.x * 16 + tx + jc];
                }

                /* BUG: After merge, Store C_shared[cse_var_1] writes to the
                 * same memory region as A_shared / B_shared.
                 * No __syncthreads() between Load A_shared and Store C_shared
                 * -> threads reading A_shared race with threads writing C_shared
                 * -> stale values -> wrong result. */
                __syncthreads();   /* tvm_storage_sync — pre-merge position */
                C_shared[cse_var_1] = C_shared[cse_var_1]
                                      + A_shared[0] * B_shared[0];
                /* MISSING __syncthreads() here after merge invalidates layout */
            }
        }
    }

    /* Write back */
    __syncthreads();   /* tvm_storage_sync */
    C[blockIdx.y * 1024 + ty * 64 + blockIdx.x * 16 + tx] =
        C_shared[ty * 16 + tx];
}
