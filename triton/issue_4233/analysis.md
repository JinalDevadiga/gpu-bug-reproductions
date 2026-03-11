# GPUVerify Analysis — Triton Issue #4233

## Bug Summary

Triton has no built-in `scatter_add`. The naive implementation uses a
non-atomic read-modify-write (`tt.load → arith.addf → tt.store`). When
multiple threads map to the same output index, each reads the same stale
value, computes its update independently, and the last writer wins — all
other updates are silently lost. This is a WAW race on global memory.

## Triton IR Evidence
```
%cur_9 = tt.load %cur_8, %xmask_3    <- read current value (stale for duplicates)
%0     = arith.addf %cur_9, %val_7   <- add source value
tt.store %cur_8, %0, %xmask_3        <- write back — non-atomic, WAW race
```

## Reproduction Results (Triton 3.6.0)
```
scatter_add_racy  : result=[1.0, 1.0, 1.0], expected=[2.0, 2.0, 2.0]
-> BUG CONFIRMED: duplicate indices cause lost updates due to WAW race

scatter_add_atomic: result=[2.0, 2.0, 2.0], expected=[2.0, 2.0, 2.0]
-> PASSED
```

## GPUVerify Result

**RACE DETECTED — 2 errors**
```
read-write race on out[0]:
  Write by thread 0: out[idx] = cur + val;
  Read  by thread 1: float cur = out[idx];

write-write race on out[0]:
  Write by thread 0: out[idx] = cur + val;
  Write by thread 1: out[idx] = cur + val;

GPUVerify kernel analyser finished with 0 verified, 2 errors
```

GPUVerify correctly identifies both the read-write and write-write races
on `out[0]` — thread 0 writes while thread 1 is still reading, and both
threads write to the same location with no synchronisation.

## Classification

| Field              | Value |
|--------------------|-------|
| GPUVerify Result   | 0 verified, 2 errors |
| Classification     | ✅ True Positive |
| Bug Type           | WAW/RAW race on global memory (non-atomic scatter_add) |
| Race Location      | Global memory out[idx] — duplicate indices, last writer wins |
| Buggy Version      | Triton 3.6.0 (issue still open) |
| Fixed Version      | Not fixed — use tl.atomic_add as workaround |
| Bug Reproduced?    | ✅ Yes — wrong output confirmed + GPUVerify 2 errors |

## Root Cause

No built-in atomic scatter_add in Triton. Naive implementation uses
non-atomic `tt.load → arith.addf → tt.store` — when duplicate indices
exist, multiple threads read the same stale value and the last writer wins.
Fix: use `tl.atomic_add` instead of `tl.load + tl.store`.