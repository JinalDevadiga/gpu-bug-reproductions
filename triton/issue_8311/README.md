# Issue #8311 — Incorrect Results from `warp_specialize=True`

## Source
- **GitHub Issue:** https://github.com/triton-lang/triton/issues/8311
- **Repo:** triton-lang/triton
- **Reported on:** Triton commit 8ee5840 (between v3.2 and v3.3, Sep 2025)
- **GPU required:** RTX 5090 or H100 (sm90+) to observe wrong numerical output
- **Status:** Closed

## What is the Bug?

`warp_specialize=True` splits warps into producer/consumer teams:
- **Producer warps** load data via TMA (Tensor Memory Accelerator), an async
  bulk-transfer engine available only on sm90+ GPUs (H100, RTX 5090).
- **Consumer warps** compute (`tl.dot`).

The bug: the synchronization between producer and consumer warps is broken.
Consumer warps proceed to `tl.dot` before the TMA load for the current tile
has completed, reading stale or partially-written shared memory — a classic
**producer-consumer race condition**.

Result on RTX 5090 (as reported):
```
warp_specialize=True  -->  99.3% of output values are WRONG
```

## Key Evidence in the TTGIR

The `{tt.warp_specialize}` attribute appears on the inner `scf.for` loop
but there is no synchronization barrier before `tt.dot`:

```
%acc = scf.for %ko = ... iter_args(...) -> (...) {
  ...
  %y_tile_51 = tt.load %y_tile_45, ...   <- async TMA load
  ...
  %acc_58 = tt.dot %acc_54, %acc_57, ... <- proceeds WITHOUT waiting for load
  scf.yield %acc_58 ...
} {tt.warp_specialize}                    <- warp split enabled, no sync barrier
```

On sm75 (this machine) TMA hardware does not exist, so the warp split never
happens and the race cannot fire. The TTGIR dump confirms the
`{tt.warp_specialize}` attribute is present, proving the buggy code path
was compiled.

## Requirements

- Python 3.10+
- CUDA 12.x
- Any NVIDIA GPU (to compile and inspect IR)
- **NVIDIA H100 or RTX 5090 (sm90+)** to observe wrong output at runtime

## Setup

```bash
conda create -n triton-bugs python=3.12 -y
conda activate triton-bugs
pip install torch==2.4.0
pip install triton
```

## How to Run

```bash
python reproduce.py
```

## Expected Output

### On sm75 (MX450) — this machine:
```
PASSED (sm90+ required to trigger bug — see README)
```

### On sm90+ (H100 / RTX 5090) — required to trigger bug:
```
BUG CONFIRMED
  Mismatched elements: 1041126 / 1048576 (99.3%)
  Greatest absolute difference: 139.0 at index (673, 708)
```

---

## Static Analysis Results

### GPUVerify

| Property | Value |
|----------|-------|
| Result | **RACE DETECTED — 2 errors** |
| Classification | ✅ True Positive |
| Race Type | Write-Read races on shared memory `s_x` and `s_y` between producer and consumer warps |

GPUVerify output:
```
error: possible write-read race on s_x[1]:
  Write by thread 1  (producer): s_x[tid] = x_tile[tid];
  Read  by thread 17 (consumer): out[idx] = s_x[idx] * s_y[idx];

error: possible write-read race on s_y[1]:
  Write by thread 1  (producer): s_y[tid] = y_tile[tid];
  Read  by thread 17 (consumer): out[idx] = s_x[idx] * s_y[idx];

GPUVerify kernel analyser finished with 0 verified, 2 errors
```

### Faial

| Property | Value |
|----------|-------|
| Result | **RACE DETECTED — 2 data-races** |
| Classification | ✅ True Positive |
| Race Type | Write-Read races on shared memory `s_x` and `s_y` between producer and consumer warps |
| Faial Version | faial-drf (built from source, gitlab.com/umb-svl/faial) |
| Command | `faial-drf --cu-to-json=/usr/local/bin/cu-to-json kernel_clean.cu` |

Faial output:
```
Kernel 'warp_specialize_buggy' has 2 data-races.
~~~~ Data-race 1 (CIDI) ~~~~
35 |         s_x[tid] = x_tile[tid];
45 |         out[idx] = s_x[idx] * s_y[idx];
Locals: threadIdx x=16 (consumer) vs threadIdx x=0 (producer)
True alarm detected!

~~~~ Data-race 2 (CIDI) ~~~~
36 |         s_y[tid] = y_tile[tid];
45 |         out[idx] = s_x[idx] * s_y[idx];
Locals: threadIdx x=16 (consumer) vs threadIdx x=0 (producer)
True alarm detected!
```

#### How Faial Found These Races
Faial correctly identified the classic producer-consumer pattern with a
missing barrier. Thread 0 (producer) writes to `s_x[0]` on line 35 while
Thread 16 (consumer) reads `s_x[0]` on line 45 — with no `__syncthreads()`
between them. The same conflict occurs on `s_y`. Both are true alarms.

Faial's result matches GPUVerify's exactly — **2 races**, both pinpointing
the same missing barrier between the producer store and consumer load phases.

---

## Tool Comparison Summary

| Tool | Result | Classification | Races Found | Notes |
|------|--------|----------------|-------------|-------|
| GPUVerify | RACE DETECTED | ✅ True Positive | 2 errors | Producer-consumer race on s_x and s_y |
| Faial | RACE DETECTED | ✅ True Positive | 2 races | Exact match with GPUVerify |