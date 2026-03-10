# GPUVerify Analysis — Triton Issue #7264

## Bug Summary

Triton's `tl.sum` reduction followed by a layout conversion generates a
butterfly shuffle where threads write their accumulated result to shared
memory without `__syncthreads()` between stages. Threads in one stage
read locations being written by threads in the previous stage — WAW/RAW
hazard. Results are non-deterministic across runs due to FP non-associativity.

## Triton IR Evidence

`reduction_kernel.ttir` shows a single `tt.reduce` with `addf` over 1024
elements — a pure butterfly shuffle sum reduction.

## Reproduction Results
```
Plain run:         Numerically CORRECT (race is silent)
compute-sanitizer: RACECHECK SUMMARY: 2 hazards displayed (2 errors, 0 warnings)
```

## GPUVerify Result

**RACE DETECTED — 2 errors**
```
write-read race on s_data[513]:
  thread 1 reads s_data[tid+stride] while thread 513 writes s_data[tid]
read-write race on s_data[1]:
  thread 1 writes s_data[tid] while thread 129 reads s_data[tid+stride]
GPUVerify kernel analyser finished with 0 verified, 2 errors
```

GPUVerify result matches compute-sanitizer exactly — 2 errors each.

## Classification

| Field              | Value |
|--------------------|-------|
| GPUVerify Result   | 0 verified, 2 errors |
| Classification     | ✅ True Positive |
| Bug Type           | WAW/RAW race in butterfly shuffle sum reduction (missing __syncthreads) |
| Race Location      | Shared memory s_data between butterfly stages |
| Buggy Version      | Triton 3.0.0 |
| Fixed Version      | Unknown |
| Bug Reproduced?    | ✅ Yes — compute-sanitizer 2 hazards + GPUVerify 2 errors |

## Comparison with Issue #4736

Both issues share the same root cause — missing `__syncthreads()` in
butterfly shuffle reduction. #4736 uses `tl.min` + `tl.sum` (4 GPUVerify
errors), while #7264 uses plain `tl.sum` (2 GPUVerify errors).

## Root Cause

Triton's butterfly shuffle reduction lowering omits `__syncthreads()`
between stages. Threads read shared memory locations that other threads
in the same warp are still writing, causing non-deterministic results.