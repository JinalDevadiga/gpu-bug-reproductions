# GPUVerify Analysis — Triton Issue #4736

## Bug Summary

`tl.min` combined with `tl.sum` in a Triton kernel generates a butterfly
shuffle reduction in `triton/language/standard.py:237` where threads write
to shared memory without proper `__syncthreads()` between butterfly stages.
Threads in one stage read shared memory locations that threads in the
previous stage are still writing — a WAW/RAW hazard confirmed by both
`compute-sanitizer` (2 hazards) and GPUVerify (4 errors).

## Triton IR Evidence

The `.ttir` shows two `tt.reduce` operations:
1. `addf` reduction over dim 2 — `tl.sum` for squared distances (clean)
2. `cmpf/select` reduction over dim 1 — `tl.min` for nearest coordinate
   (this is where the butterfly shuffle WAW race occurs)

## Reproduction Results
```
Plain run:     Numerically correct: True  (silent race)
compute-sanitizer: RACECHECK SUMMARY: 2 hazards displayed (2 errors, 0 warnings)
```

## GPUVerify Result

**RACE DETECTED — 4 errors**
```
write-read race on s_val[8]: thread 0 reads while thread 8 writes
write-read race on s_idx[8]: thread 0 reads while thread 8 writes
read-write race on s_val[0]: thread 0 writes while thread 3 reads
read-write race on s_idx[0]: thread 0 writes while thread 3 reads
GPUVerify kernel analyser finished with 0 verified, 4 errors
```

GPUVerify detected more races (4) than compute-sanitizer (2), catching
both the write-read hazards between butterfly stages and the read-write
hazards on the final accumulation slot.

## Classification

| Field              | Value |
|--------------------|-------|
| GPUVerify Result   | 0 verified, 4 errors |
| Classification     | ✅ True Positive |
| Bug Type           | WAW/RAW race in butterfly shuffle reduction (missing __syncthreads) |
| Race Location      | Shared memory s_val / s_idx between butterfly stages |
| Buggy Version      | Triton 3.6.0 (issue still open) |
| Fixed Version      | Not fixed |
| Bug Reproduced?    | ✅ Yes — compute-sanitizer 2 hazards + GPUVerify 4 errors |

## Root Cause

`triton/language/standard.py:237` — `tl.min` butterfly shuffle reduction
missing `__syncthreads()` between stages. Each stage reads locations being
written by the previous stage, causing WAW and RAW races on shared memory.