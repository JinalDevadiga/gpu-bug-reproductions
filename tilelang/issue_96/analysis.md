# Analysis: Issue #96 — Race Condition in Pipelined Matmul

## Bug Summary
TileLang's `T.Pipelined` with `num_stages=3` generates a CUDA kernel
with insufficient synchronization between pipeline stages, causing a
Write-After-Read (WAR) hazard in shared memory on SM80+ GPUs.

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

## Two Levels of Race Conditions

### Level 1 — Original Kernel (WAR Hazard)
The original generated kernel has a Write-After-Read hazard:
- A buffer slot is overwritten (future pipeline stage)
- While it is still being read (current MMA computation)
- This requires SM80+ hardware and async pipeline primitives to manifest
- GPUVerify CANNOT detect this — uses tl::ptx_ldmatrix_x4 and
  tl::mma_sync which are beyond GPUVerify's understanding

### Level 2 — Simplified Kernel (Write-Write Race)
Our simplified kernel exposed a different but related race:
- The modulo indexing `(ko+2)%3` causes two threads to compute
  the same target shared memory address
- Thread 66 and Thread 64 both write to buf_dyn_shmem[8258]
  simultaneously
- This is a write-write race caused by incorrect address computation
- GPUVerify successfully detected this

## GPUVerify Result
**RACE DETECTED — True Positive (Write-Write Race in simplified kernel)**
```
write-write race on buf_dyn_shmem[8258]:
- Thread 66 writes to buf_dyn_shmem[((ko+2)%3)*4096 + ...]
- Thread 64 writes to the same location simultaneously
```

## Classification
| Property | Value |
|----------|-------|
| Result | Race Detected |
| Classification | True Positive |
| Race Type (simplified kernel) | Write-Write Race |
| Race Type (original kernel) | Write-After-Read (WAR) Hazard |
| Memory | Shared Memory |
| Hardware | SM80+ required for original runtime manifestation |

## Important Notes
1. The original bug is a WAR hazard requiring SM80+ hardware —
   GPUVerify cannot detect this due to async pipeline primitives
2. The simplified kernel revealed a write-write race from incorrect
   modulo address computation — GPUVerify correctly detected this
3. Both races stem from the same root cause: unsafe shared memory
   reuse across pipeline stages
4. The write-write race in the simplified kernel is a valid finding
   and confirms the shared memory access pattern is unsafe

## Conclusion
GPUVerify correctly detected a write-write race in the simplified
kernel, confirming unsafe shared memory access patterns. The original
kernel's WAR hazard is beyond GPUVerify's scope due to its reliance
on modern async pipeline primitives. Both races share the same root
cause — insufficient synchronization in TileLang's pipelined loop
code generation.
