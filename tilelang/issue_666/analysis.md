# Analysis: Issue #666 — Incorrect Results When Clearing Shared Memory Before Pipelined Loops

## Bug Summary
TileLang 0.1.8 generates CUDA code where a shared memory clear operation
(`T.clear`) is not properly synchronized with asynchronous pipeline stages,
causing incorrect results on NVIDIA H100 GPUs.

## Affected Version
- **Buggy version:** TileLang 0.1.8
- **Hardware:** Only manifests on NVIDIA H100 (not reproducible on MX450)

## Root Cause
The generated kernel clears shared memory and then immediately starts
asynchronous pipeline loads into the same memory region with only one
`__syncthreads()` between them. On H100, the async pipeline can overlap
with the clearing operation, causing threads to read a mix of:
- Newly loaded data
- Partially cleared (zeroed) values
- Stale data

## Problematic Pattern in Generated Kernel:
```cuda
// Step 1: Clear shared memory
for (...) { buf_dyn_shmem[...] = 0; }
__syncthreads();  // Only ONE barrier — insufficient for async pipeline

// Step 2: Async pipeline loads into SAME memory (no additional barrier)
for (...) { buf_dyn_shmem[...] = A[...]; }

// Step 3: Pipeline loop begins — overlaps with clearing on H100
for (int ko = 0; ko < 30; ++ko) { ... }
```

## GPUVerify Result
**VERIFIED — False Negative**

GPUVerify reported no races, but the bug is real and confirmed by the
TileLang developers (issue closed with fix).

## Why GPUVerify Cannot Detect This Bug
1. **Hardware-specific behavior** — the race only manifests on H100 due to
   its async pipeline execution model
2. **Modern async primitives** — uses `tl::ptx_ldmatrix_x4`, `tl::mma_sync`
   which are post-2018 CUDA features GPUVerify doesn't understand
3. **Pipeline-level race** — the race occurs at the hardware pipeline level,
   not at the thread synchronization level that GPUVerify reasons about
4. **Simplified kernel limitation** — we had to simplify the kernel to even
   run GPUVerify, losing the async pipeline semantics in the process

## Classification
| Property | Value |
|----------|-------|
| Result | Verified (no races found) |
| Classification | False Negative |
| Race Type | Async Pipeline Race |
| Memory | Shared Memory |
| Hardware | H100 specific |
| GPUVerify Limitation | Cannot reason about async hardware pipelines |

## Conclusion
This bug represents a class of races that GPUVerify fundamentally cannot
detect — hardware-specific async pipeline races introduced by modern GPU
architectures. This is a known limitation of static analysis tools built
before async pipeline primitives became common in GPU programming.
