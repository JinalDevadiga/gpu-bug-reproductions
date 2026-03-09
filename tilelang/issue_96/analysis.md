# Analysis: Issue #96 — Race Condition in Pipelined Matmul

## Bug Summary
TileLang's `T.Pipelined` with `num_stages=3` generates a CUDA kernel
with insufficient synchronization between pipeline stages, causing a
Write-After-Read (WAR) hazard in shared memory.

## Affected Version
- **Buggy version:** TileLang (early versions)
- **Hardware:** RTX 4070 Ti, A100 (SM80+)
- **Note:** MX450 cannot execute the kernel (lacks SM80 MMA support)
  but the problematic code pattern is reproducible via compilation

## Root Cause
The pipelined loop reuses shared memory slots using modulo indexing
`(ko+2)%3`. A future stage writes into a buffer slot that may still
be in use by MMA computation from a previous iteration. The two
`__syncthreads()` barriers surrounding the write are insufficient
to prevent overlap between:
- Shared memory writes (pipeline prefetch into stage (ko+2)%3)
- Shared memory reads (MMA computation from stage (ko%3))

## Problematic Pattern:
```cuda
for (int ko = 0; ko < 254; ++ko) {
    __syncthreads();
    // Write to future stage (ko+2)%3
    // BUG: this slot may still be in use from previous iteration
    buf_dyn_shmem[((ko+2)%3) * 4096 + ...] = A[...];

    __syncthreads();
    // Read from current stage (ko%3)
    tl::ptx_ldmatrix_x4(...(ko%3)*4096...);
    tl::mma_sync(...);  // WAR hazard here
}
```

## GPUVerify Result
**RACE DETECTED — Partial True Positive**

GPUVerify detected a write-write race on `buf_dyn_shmem[8258]`:
- Thread 66 writes to `buf_dyn_shmem[((ko+2)%3)*4096 + ...]`
- Thread 64 writes to the same location simultaneously
- This indicates incorrect modulo indexing causing two threads
  to compute the same target address

## Classification
| Property | Value |
|----------|-------|
| Result | Race Detected |
| Classification | Partial True Positive |
| Race Type | Write-Write Race (in simplified kernel) |
| Original Race Type | Write-After-Read (WAR) Hazard |
| Memory | Shared Memory |
| Hardware | SM80+ (A100/H100) required for runtime manifestation |

## Important Notes
1. GPUVerify detected a race but it is a write-write race in our
   simplified kernel, not exactly the WAR hazard in the original
2. The original kernel uses `tl::ptx_ldmatrix_x4` and `tl::mma_sync`
   which GPUVerify cannot reason about
3. The simplified kernel captures the essence of the bug — incorrect
   modulo indexing leading to shared memory conflicts
4. The actual WAR hazard in the original kernel requires SM80+ hardware
   to observe incorrect numerical output

## Conclusion
GPUVerify successfully detected a race condition in the simplified
version of this kernel, confirming that the shared memory access
pattern is indeed unsafe. The detected write-write race is related
to but not identical to the original WAR hazard, which requires
SM80+ hardware and async pipeline primitives to fully manifest.
