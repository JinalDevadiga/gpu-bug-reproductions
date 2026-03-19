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

### Atomic version (tl.atomic_add):
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

## How to Run

```bash
python reproduce.py
```

---

## Static Analysis Results

### GPUVerify

| Property | Value |
|----------|-------|
| Result | **RACE DETECTED** |
| Classification | ✅ True Positive |
| Race Type | WAW/RAW race on global memory (`out[idx]`) |
| Errors | 2 (write-write + read-write on `out[0]`) |

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
| Result | **RACE DETECTED** |
| Classification | ✅ True Positive (with caveat) |
| Race Type | WAW race on global memory (data-dependent index) |
| Command | `faial-drf --cu-to-json=/usr/local/bin/cu-to-json kernel_clean.cu` |

Faial output:
```
Kernel 'scatter_add_racy' has 1 data-race.
~~~~ Data-race 1 (CIDD) ~~~~
41 |     out[idx]  = cur + val;
  idx (D) = 0 for both threads
WARNING: potential alarm, index depends on input, see variables with (D).
True alarm detected!
```

Faial found the race but issued a `(CIDD)` classification and a "potential
alarm" warning, noting that the index `idx` is data-dependent. This means
the race only occurs for specific input values (duplicate indices) rather
than for all possible inputs.

### Weft

| Property | Value |
|----------|-------|
| Result | **No races detected** |
| Classification | ❌ False Negative |
| Race Type Missed | WAW/RAW race on global memory |
| Weft Version | Built from source (github.com/lightsighter/Weft) |
| Command | `weft -f kernel_clean.ptx -t 4 -i -d` |
| PTX Compiled With | `nvcc -ptx kernel_weft.cu` (added `__launch_bounds__(8)`) |

Weft output:
```
WEFT INFO: No deadlocks detected in kernel scatter_add_racy!
WEFT INFO: No races detected in kernel scatter_add_racy!
WEFT STATISTICS for Kernel scatter_add_racy
  CTA Thread Count:                        8
  Shared Memory Locations:                 0
  Physical Named Barriers;                 1
  Dynamic Barrier Instances:               0
  Weft Statements:                         0
  Total Race Tests:                        0
```

#### Why Weft Missed This Bug

The race in issue #4233 is a **global memory race** — threads racing on
`out[idx]` which is a global memory array passed as a kernel parameter.
Weft is designed exclusively to verify **shared memory** accesses
(`ld.shared` / `st.shared` in PTX). It has no visibility into global
memory accesses whatsoever.

The Weft output confirms this: `Shared Memory Locations: 0` and
`Total Race Tests: 0` — there is literally nothing for Weft to check
in this kernel because it contains no shared memory accesses at all.

This is a **fundamental scope limitation** of Weft, not a reasoning
failure. Weft was built specifically for warp-specialized kernels that
use named barriers and shared memory for producer-consumer communication.
Global memory races are outside its design scope entirely.

---

## Tool Comparison Summary

| Tool | Result | Classification | Notes |
|------|--------|----------------|-------|
| GPUVerify | RACE DETECTED | ✅ True Positive | 2 errors (RAW + WAW on global memory) |
| Faial | RACE DETECTED | ✅ True Positive | 1 race (CIDD — data-dependent index warning) |
| Weft | No races detected | ❌ False Negative | Global memory race — outside Weft's shared-memory-only scope |