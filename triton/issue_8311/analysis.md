# GPUVerify Analysis — Triton Issue #8311

## Bug Summary

`warp_specialize=True` splits warps into producer and consumer teams.
Producer warps perform async TMA loads into shared memory; consumer warps
compute `tt.dot`. No synchronisation barrier exists between the async load
completing and the consumer warps reading shared memory — consumer warps
proceed to `tt.dot` before the TMA load has finished, reading stale or
partially-written shared memory. On sm90+ (H100/RTX 5090) this causes
99.3% of output values to be wrong. On sm75 (MX450) TMA hardware is
unavailable so the race cannot fire at runtime.

## TTGIR Evidence
```
%y_tile_51 = tt.load %y_tile_45, ...         <- async load (no completion guarantee)
...
%acc_58 = tt.dot %acc_54, %acc_57, %acc_30   <- proceeds without waiting
scf.yield %acc_58
} {tt.warp_specialize}                        <- no sync barrier before dot
```

## Reproduction Results
```
On sm75 (MX450): PASSED — TMA hardware unavailable, race cannot fire
On sm90+ (H100): BUG CONFIRMED — 99.3% wrong values (from issue report)
```

## GPUVerify Result

**RACE DETECTED — 2 errors**
```
write-read race on s_x[1]:
  Write by thread 1  (producer): s_x[tid] = x_tile[tid];
  Read  by thread 17 (consumer): out[idx] = s_x[idx] * s_y[idx];

write-read race on s_y[1]:
  Write by thread 1  (producer): s_y[tid] = y_tile[tid];
  Read  by thread 17 (consumer): out[idx] = s_x[idx] * s_y[idx];

GPUVerify kernel analyser finished with 0 verified, 2 errors
```

GPUVerify correctly identifies the producer-consumer race — consumer
threads read shared memory while producer threads are still writing it,
exactly modelling the missing barrier between TMA load and tt.dot.

## Classification

| Field              | Value |
|--------------------|-------|
| GPUVerify Result   | 0 verified, 2 errors |
| Classification     | ✅ True Positive |
| Bug Type           | Producer-consumer race (missing barrier between async load and dot) |
| Race Location      | Shared memory s_x / s_y between producer and consumer warps |
| Buggy Version      | Triton ~3.2-3.3 (commit 8ee5840) |
| Fixed Version      | Closed |
| CUDA Kernel?       | Yes — manually translated from TTGIR |
| Bug Reproduced?    | ✅ TTGIR confirmed + GPUVerify 2 errors (runtime needs sm90+) |

## Root Cause

`{tt.warp_specialize}` on the inner `scf.for` loop splits warps into
producer/consumer teams but omits the synchronisation barrier between
the async TMA load completing and the consumer warps beginning `tt.dot`.
Fix: insert a barrier after the TMA load to ensure all producer writes
are visible to consumer warps before computation begins.