# Issue #4362 — `tl.associative_scan` Wrong Results with `reverse=True`

## Source
- **GitHub Issue:** https://github.com/triton-lang/triton/issues/4362
- **Repo:** triton-lang/triton
- **Reproduced on:** Triton 3.0.0 + PyTorch 2.4.0 + NVIDIA GeForce MX450 (sm75)

## What is the Bug?

`tl.associative_scan` produces incorrect results when `reverse=True`. The
forward direction (`reverse=False`) works correctly. The bug is in how
Triton lowers the reverse scan — elements are not scanned in the right
order, causing wrong intermediate and final values.

Using a first-order linear recurrence combine function:
```python
def op(fl, xl, fr, xr):
    return fr * fl, fr * xl + xr
```

With inputs:
```
exp  = [1.0, 1.5, 0.8, 2.0]
vals = [1.0, -1.0, 0.5, 2.0]
```

## Results

### reverse=False (correct):
```
exp  : [1.0, 1.5, 1.2, 2.4]
vals : [1.0, 0.5, 0.9, 3.8]
```

### reverse=True (buggy):
```
exp  result  : [2.4, 2.4, 2.4, 2.4]      ← all identical, wrong
exp  expected: [2.4, 2.4, 1.6, 2.0]
vals result  : [3.15, 4.55, 2.1, 3.5]    ← wrong
vals expected: [3.15, 2.15, 2.1, 2.0]
```

The `exp` output is all 2.4 — the final accumulated value broadcast to
every position — indicating the scan is not respecting element ordering
when reversing.

## Requirements

- Python 3.10+
- CUDA 12.x
- Any NVIDIA GPU
- Triton 3.0.0

## How to Run

```bash
python reproduce.py
```

---

## Static Analysis Results

### GPUVerify

| Property | Value |
|----------|-------|
| Result | **Verified (no races found)** |
| Classification | ❌ False Negative |
| Reason | Algorithmic ordering bug — not a data race |

GPUVerify output:
```
GPUVerify kernel analyser finished with 1 verified, 0 errors
- no data races within thread blocks
- no data races between thread blocks
```

#### Why GPUVerify Missed This Bug

Each thread operates on a distinct shared memory index with proper
`__syncthreads()` around shared memory loads. The wrong results stem from
incorrect scan direction logic in Triton's lowering, not from missing
synchronization. GPUVerify only detects sync-based races.

### Faial

| Property | Value |
|----------|-------|
| Result | **DRF (Data-Race Free)** |
| Classification | ❌ False Negative |
| Reason | Algorithmic ordering bug — not a data race |
| Command | `faial-drf --cu-to-json=/usr/local/bin/cu-to-json kernel_clean.cu` |

Faial output:
```
Kernel 'associative_scan_buggy' is DRF!
```

Same reason as GPUVerify — the kernel is genuinely race-free. Every thread
writes to and reads from distinct memory locations with correct barriers.
The bug is semantic, not synchronization-based.

### Weft

| Property | Value |
|----------|-------|
| Result | **No races detected** |
| Classification | ❌ False Negative |
| Reason | Algorithmic ordering bug — not a data race |
| Weft Version | Built from source (github.com/lightsighter/Weft) |
| Command | `weft -f kernel_clean.ptx -t 4 -i -d` |
| PTX Compiled With | `nvcc -ptx kernel_weft.cu` (added `__launch_bounds__(4)`) |

Weft output:
```
WEFT INFO: No deadlocks detected in kernel associative_scan_buggy!
WEFT INFO: No races detected in kernel associative_scan_buggy!
WEFT STATISTICS for Kernel associative_scan_buggy
  CTA Thread Count:                        4
  Shared Memory Locations:                 5
  Physical Named Barriers;                 1
  Dynamic Barrier Instances:               1
  Weft Statements:                        50
  Total Race Tests:                      195
```

#### Why Weft Missed This Bug

Unlike issue #4233, this kernel does use shared memory (5 locations,
195 race tests performed). Weft fully analyzed the shared memory accesses
and correctly found no races — because there genuinely are none. The bug
is an **algorithmic ordering error** in Triton's reverse scan lowering:
the scan traverses elements in the wrong direction, broadcasting the final
accumulated value to all positions instead of computing a proper reverse
prefix scan. No two threads access the same shared memory address
simultaneously without a barrier between them, so no race exists at the
synchronization level.

This is a **false negative shared by all three tools** — not because any
tool is deficient, but because the bug is simply not a data race. It is a
semantic correctness bug that requires understanding the intended algorithm,
which is outside the scope of any synchronization-based race detector.

---

## Tool Comparison Summary

| Tool | Result | Classification | Notes |
|------|--------|----------------|-------|
| GPUVerify | Verified (DRF) | ❌ False Negative | Algorithmic bug — not a sync race |
| Faial | DRF | ❌ False Negative | Algorithmic bug — not a sync race |
| Weft | No races detected | ❌ False Negative | Algorithmic bug — not a sync race; 195 race tests performed, all clean |