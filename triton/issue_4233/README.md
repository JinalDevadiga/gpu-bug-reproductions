# Issue #4233 — `scatter_add` Data Race Without `tl.atomic_add`

## Source
- **GitHub Issue:** https://github.com/triton-lang/triton/issues/4233
- **Repo:** triton-lang/triton
- **Reproduced on:** Triton 3.6.0 + PyTorch 2.4.0 + NVIDIA GeForce MX450 (sm75)
- **Status:** Open

## What is the Bug?

Triton has no built-in `scatter_add` operation. The naive implementation uses
a non-atomic read-modify-write (`tl.load` → add → `tl.store`). When multiple
threads map to the same output index, each thread reads the same stale value,
computes its update independently, and the last writer wins — all other
updates are silently lost. This is a write-write (WAW) data race.

Example:
```
index_tensor  = [0, 1, 2, 2, 1, 0]
to_add_tensor = [1, 1, 1, 1, 1, 1]
expected      = [2, 2, 2]   # index 0 appears twice → src[0] = 1+1 = 2
```

The correct implementation uses `tl.atomic_add` so every update is applied
atomically regardless of duplicate indices.

## Results

### Racy version (non-atomic):
```
result  : [1.0, 1.0, 1.0]
expected: [2.0, 2.0, 2.0]
BUG CONFIRMED: duplicate indices cause lost updates due to WAW race
```

### Atomic version (`tl.atomic_add`):
```
result  : [2.0, 2.0, 2.0]
expected: [2.0, 2.0, 2.0]
PASSED
```

## Requirements

- Python 3.10+
- CUDA 12.x
- Any NVIDIA GPU
- Triton 3.6.0

## Setup
```bash
conda activate triton-bugs
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
| Result | **RACE DETECTED — 2 errors** |
| Classification | ✅ True Positive |
| Race Type | WAW race + RAW race on global memory `out[0]` |

GPUVerify output:
```
error: possible read-write race on out[0]:
  Write by thread 0: out[idx] = cur + val;
  Read  by thread 1: float cur = out[idx];

error: possible write-write race on out[0]:
  Write by thread 0: out[idx] = cur + val;
  Write by thread 1: out[idx] = cur + val;

GPUVerify kernel analyser finished with 0 verified, 2 errors
```

### Faial

| Property | Value |
|----------|-------|
| Result | **RACE DETECTED — 1 data-race** |
| Classification | ✅ True Positive |
| Race Type | WAW race on global memory `out[idx]` |
| Faial Version | faial-drf (built from source, gitlab.com/umb-svl/faial) |
| Command | `faial-drf --cu-to-json=/usr/local/bin/cu-to-json kernel_clean.cu` |

Faial output:
```
Kernel 'scatter_add_racy' has 1 data-race.
~~~~ Data-race 1 (CIDD) ~~~~
41 |     out[idx]  = cur + val;
Globals
  out[] = 0  |  blockIdx: x=0, y=0, z=0
Locals
  idx (D) = 0  |  idx (D) = 0
  threadIdx: x=1, y=0, z=0  |  threadIdx: x=0, y=0, z=0
WARNING: potential alarm, index depends on input, see variables with (D).
```

#### How Faial Found This Bug
Faial detected that Thread 0 and Thread 1 both write to `out[idx]` on line 41,
with `idx = 0` for both threads — a write-write race.

The race is classified as **CIDD (Conditionally Independent, Data Dependent)**
because the array index `idx` is loaded from the `index[]` input array at
runtime, meaning its value depends on input data rather than being a fixed
expression of `threadIdx`. Faial correctly flags this as a **potential alarm**
with a warning, rather than a guaranteed race for all inputs. However, since
the input `index = [0, 1, 2, 2, 1, 0]` does contain duplicates, this is a
**true positive** — the race is real and confirmed at runtime.

This is an important distinction: Faial is being conservative and honest —
the race exists when duplicate indices are present, which is exactly the
bug scenario described in the issue.

---

## Tool Comparison Summary

| Tool | Result | Classification | Notes |
|------|--------|----------------|-------|
| GPUVerify | RACE DETECTED (2 errors) | ✅ True Positive | Found both RAW and WAW races |
| Faial | RACE DETECTED (1 race, CIDD) | ✅ True Positive | Found WAW race; flagged as data-dependent index |