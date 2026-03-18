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
exp  result  : [2.4, 2.4, 2.4, 2.4]      <- all identical, wrong
exp  expected: [2.4, 2.4, 1.6, 2.0]
vals result  : [3.15, 4.55, 2.1, 3.5]    <- wrong
vals expected: [3.15, 2.15, 2.1, 2.0]
```

The `exp` output is all `2.4` — the final accumulated value broadcast to
every position — indicating the scan is not respecting element ordering when
reversing, effectively reducing rather than scanning.

## Requirements

- Python 3.10+
- CUDA 12.x
- Any NVIDIA GPU
- Triton 3.0.0

## Setup
```bash
conda create -n triton-7402 python=3.12 -y
conda activate triton-7402
pip install torch==2.4.0
pip install triton==3.0.0
```

## How to Run
```bash
python reproduce.py
```

---

## Static Analysis Results

### GPUVerify

| Property | Value |
|----------|-------|
| Result | **VERIFIED (no races found)** |
| Classification | ❌ False Negative |
| Race Type Missed | N/A — this is not a data race |

GPUVerify output:
```
GPUVerify kernel analyser finished with 1 verified, 0 errors
- no data races within thread blocks
- no data races between thread blocks
- no barrier divergence
- no assertion failures
```

### Faial

| Property | Value |
|----------|-------|
| Result | **DRF (Data-Race Free)** |
| Classification | ❌ False Negative |
| Race Type Missed | N/A — this is not a data race |
| Faial Version | faial-drf (built from source, gitlab.com/umb-svl/faial) |
| Command | `faial-drf --cu-to-json=/usr/local/bin/cu-to-json kernel_clean.cu` |

Faial output:
```
Kernel 'associative_scan_buggy' is DRF!
```

#### Why Both Tools Missed This Bug

This bug is fundamentally different from all other issues in this repository
— it is **not a data race at all**. Both Faial and GPUVerify are data race
detectors, so neither can detect it. The bug is an **algorithmic ordering
error** in Triton's compiler lowering of `reverse=True`:

- Each thread reads and writes a **distinct index** — no two threads touch
  the same memory location simultaneously
- Barriers (`__syncthreads()`) are present around shared memory accesses
- The kernel is genuinely race-free from a synchronisation perspective

The wrong results come from Triton 3.0.0 scanning elements left-to-right
on reversed indices instead of right-to-left, causing the final accumulated
value to be broadcast to all positions. This is a **semantic/logical bug**
in the compiler's code generation — the generated kernel is correctly
synchronized but computes the wrong thing.

No static race detector can catch this class of bug because the kernel
contains no race. Detecting it would require reasoning about the
**semantic correctness** of the algorithm itself — a fundamentally different
and much harder problem than race detection.

---

## Tool Comparison Summary

| Tool | Result | Classification | Notes |
|------|--------|----------------|-------|
| GPUVerify | Verified (DRF) | ❌ False Negative | Not a race — algorithmic bug outside tool scope |
| Faial | DRF | ❌ False Negative | Same — not a race, both tools correctly report DRF |