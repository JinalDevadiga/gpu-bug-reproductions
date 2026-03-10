# GPUVerify Analysis — Triton Issue #7402

## Bug Summary

`tl.atomic_add` atomically increments a counter and returns the old value,
intended as a unique ticket dispenser so each thread writes to a distinct
slot. Due to a layout mismatch in Triton 3.0.0, only thread 0 receives the
correct atomic return value; all other 127 threads receive 0. All non-zero
threads then write to `out[0, :]`, corrupting that row — a WAW race.

## TTGIR Evidence (Triton 3.0.0 — buggy)
```
%2 = tt.atomic_rmw add, relaxed, gpu, %1, %cst, %cst_0 :
     (...) -> tensor<1xi32, #blocked>
     // #blocked: sizePerThread=[1], 128 threads total
     // Only thread 0 holds the real atomic return value (1)
     // Threads 1-127 hold 0

%3 = triton_gpu.convert_layout %2 :
     tensor<1xi32, #blocked> ->
     tensor<1xi32, #triton_gpu.slice<...>>
     // Spreads wrong state instead of broadcasting thread 0's value
```

## Reproduction Results (Triton 3.0.0)
```
Result  : [[0, 2, 2, 2, 2, 2, 2, 2], [2, 0, 0, 0, 0, 0, 0, 0]]
Expected: [[0, 0, 0, 0, 0, 0, 0, 0], [2, 2, 2, 2, 2, 2, 2, 2]]
BUG CONFIRMED: only thread 0 gets correct atomic_add return value
```

## GPUVerify Result

**RACE DETECTED — 1 error**
```
write-write race on out[0]:
  Write by thread 0: out[write_index * ROW_WIDTH + i] = 2;
  Write by thread 1: out[write_index * ROW_WIDTH + i] = 2;
GPUVerify kernel analyser finished with 0 verified, 1 error
```

GPUVerify correctly identifies the WAW race — threads 0 and 1 both
compute `write_index = 0` due to the layout mismatch and write to
the same output location simultaneously.

## Classification

| Field              | Value |
|--------------------|-------|
| GPUVerify Result   | 0 verified, 1 error |
| Classification     | ✅ True Positive |
| Bug Type           | WAW race from atomic_add layout mismatch (wrong broadcast) |
| Race Location      | Global memory out[0] — multiple threads write same slot |
| Buggy Version      | Triton 3.0.0 |
| Fixed Version      | Triton 3.1.0+ (PR #7460) |
| Bug Reproduced?    | ✅ Yes — wrong output confirmed + GPUVerify 1 error |

## Root Cause

`tt.atomic_rmw` produces result in `#blocked` layout with 128 threads,
but the tensor has only 1 element — only thread 0 holds the real return
value. `triton_gpu.convert_layout` spreads this incorrect state instead
of broadcasting thread 0's value to all threads. Fixed in PR #7460.