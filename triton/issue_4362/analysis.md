# GPUVerify Analysis — Triton Issue #4362

## Bug Summary

`tl.associative_scan` with `reverse=True` produces wrong results in
Triton 3.0.0. The reverse scan scans elements in the wrong order,
effectively broadcasting the final accumulated value to all positions
instead of computing a proper reverse prefix scan.

## Reproduction Results (Triton 3.0.0)
```
exp  result  : [2.4, 2.4, 2.4, 2.4]       <- final value broadcast, wrong
exp  expected: [2.4, 2.4, 1.6, 2.0]
vals result  : [3.15, 4.55, 2.1, 3.5]     <- wrong
vals expected: [3.15, 2.15, 2.1, 2.0]

BUG CONFIRMED on Triton 3.0.0, FIXED on Triton 3.6.0
```

## GPUVerify Result

**FALSE NEGATIVE — 1 verified, 0 errors**

GPUVerify found no races because this is an algorithmic ordering bug,
not a synchronisation bug. Each thread reads/writes a distinct index
with proper barriers around shared memory — no race is present.
The wrong results come from incorrect accumulation logic which is
outside GPUVerify's detection scope.

## Classification

| Field            | Value |
|------------------|-------|
| GPUVerify Result | 1 verified, 0 errors |
| Classification   | ❌ False Negative |
| Bug Type         | Algorithmic ordering error (wrong scan direction) |
| Race Location    | N/A — no data race |
| Buggy Version    | Triton 3.0.0 |
| Fixed Version    | Triton 3.6.0 |
| Bug Reproduced?  | ✅ Yes — wrong output confirmed on Triton 3.0.0 |

## Root Cause

Triton 3.0.0 lowered `reverse=True` by scanning left-to-right on reversed
indices instead of right-to-left, causing the final accumulated value to
be broadcast to all positions rather than computing the correct reverse
prefix scan.