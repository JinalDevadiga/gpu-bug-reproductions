/*
 * Triton Issue #7402 — tl.atomic_add return value wrong across threads
 *
 * Manually translated from Triton TTGIR (my_kernel).
 *
 * Bug: tt.atomic_rmw produces result in #blocked layout with
 * sizePerThread=[1] across 128 threads (4 warps x 32). Only thread 0
 * holds the real atomic return value; threads 1-127 hold 0.
 * convert_layout spreads this wrong state instead of broadcasting
 * thread 0's value -> threads 1-127 write to out[0,:] instead of out[1,:]
 *
 * Grid:  1 block
 * Block: 128 threads (4 warps x 32)
 */

#include <cuda.h>

#define BLOCK_SIZE 128
#define ROW_WIDTH  8

__global__ __launch_bounds__(128) void atomic_add_buggy(
    int*          __restrict__ index_ptr,   // atomic counter, init=1
    long long int* __restrict__ out)        // output [2][8]
{
    int tid = threadIdx.x;

    /* Only thread 0 performs the atomic — others get 0 (buggy layout) */
    int write_index = 0;
    if (tid == 0) {
        write_index = atomicAdd(index_ptr, 1);  // returns old value = 1
    }
    /* BUG: write_index is NOT broadcast to all threads.
     * Threads 1-127 still have write_index=0.
     * All threads should write to out[1,:] but threads 1-127
     * write to out[0,:] instead — corrupting that row. */

    /* All threads write their (possibly wrong) row */
    for (int i = 0; i < ROW_WIDTH; i++) {
        out[write_index * ROW_WIDTH + i] = 2;
        /* WAW race: threads 1-127 all write to out[0,:] simultaneously */
    }
}