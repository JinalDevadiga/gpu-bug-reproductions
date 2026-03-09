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
  %y_tile_51 = tt.load %y_tile_45, ...   ← async TMA load
  ...
  %acc_58 = tt.dot %acc_54, %acc_57, ... ← proceeds WITHOUT waiting for load
  scf.yield %acc_58 ...
} {tt.warp_specialize}                    ← warp split enabled, no sync barrier
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
The TTGIR dump shows `{tt.warp_specialize}` on the inner loop confirming
the buggy code path was compiled, but TMA hardware is unavailable on sm75
so the race never fires.

### On sm90+ (H100 / RTX 5090) — required to trigger bug:
```
BUG CONFIRMED
  Mismatched elements: 1041126 / 1048576 (99.3%)
  Greatest absolute difference: 139.0 at index (673, 708)
```