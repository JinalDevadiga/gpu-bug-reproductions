# Issue #7402 — `tl.atomic_add` Return Value Wrong Across Threads

## Source
- **GitHub Issue:** https://github.com/triton-lang/triton/issues/7402
- **Repo:** triton-lang/triton
- **Fixed in:** PR #7460 (Atomic RMW Broadcasting)
- **Reproduced on:** Triton 3.0.0 + PyTorch 2.4.0 + NVIDIA GeForce MX450 (sm75)

## What is the Bug?

`tl.atomic_add` atomically increments a counter and returns the old value,
intended as a ticket dispenser where each thread gets a unique write slot.
Due to a layout mismatch in Triton 3.0.0, only thread 0 receives the correct
atomic return value; all other 127 threads receive 0. All non-zero threads
then write to `out[0, :]`, corrupting that row — a WAW race.

## Results

### Expected output:
```
Result  : [[0, 0, 0, 0, 0, 0, 0, 0], [2, 2, 2, 2, 2, 2, 2, 2]]
```

### Buggy output (Triton 3.0.0):
```
Result  : [[0, 2, 2, 2, 2, 2, 2, 2], [2, 0, 0, 0, 0, 0, 0, 0]]
BUG CONFIRMED: only thread 0 gets correct atomic_add return value
```

## Requirements

- Python 3.10+
- CUDA 12.x
- Any NVIDIA GPU
- Triton 3.0.0 (bug is fixed in 3.1.0+)

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
| Race Type | WAW race on global memory (`out[0]`) |
| Errors | 1 |

GPUVerify output:
```
error: possible write-write race on out[0]:
  Write by thread 1: out[write_index * ROW_WIDTH + i] = 2;
  Write by thread 0: out[write_index * ROW_WIDTH + i] = 2;
GPUVerify kernel analyser finished with 0 verified, 1 error
```

GPUVerify detected the race because it can reason about global memory
accesses and determined that both threads compute `write_index = 0`.

### Faial

| Property | Value |
|----------|-------|
| Result | **DRF (Data-Race Free)** |
| Classification | ❌ False Negative |
| Race Type Missed | WAW race on global memory (data-dependent write index) |
| Command | `faial-drf --cu-to-json=/usr/local/bin/cu-to-json kernel_clean.cu` |

Faial output:
```
Kernel 'atomic_add_buggy' is DRF!
```

Faial missed the race because `write_index` is treated as an unconstrained
symbolic variable in its memory access protocol. Faial cannot derive the
runtime constraint that threads 1-127 all receive `write_index=0` due to
Triton's broken layout — that information is only known at runtime.

### Weft

| Property | Value |
|----------|-------|
| Result | **No races detected** |
| Classification | ❌ False Negative |
| Race Type Missed | WAW race on global memory |
| Weft Version | Built from source (github.com/lightsighter/Weft) |
| Command | `weft -f kernel_clean.ptx -t 4 -i -d` |
| PTX Compiled With | `nvcc -ptx kernel_weft.cu` (added `__launch_bounds__(128)`) |

Weft output:
```
WEFT INFO: No deadlocks detected in kernel atomic_add_buggy!
WEFT INFO: No races detected in kernel atomic_add_buggy!
WEFT STATISTICS for Kernel atomic_add_buggy
  CTA Thread Count:                      128
  Shared Memory Locations:                 0
  Physical Named Barriers;                 1
  Dynamic Barrier Instances:               0
  Weft Statements:                         0
  Total Race Tests:                        0
```

#### Why Weft Missed This Bug

The race in issue #7402 is a **global memory race** — all 128 threads write
to `out[write_index * ROW_WIDTH + i]` which is a global memory array.
Weft only analyzes shared memory (`ld.shared`/`st.shared` in PTX) and has
no visibility into global memory accesses whatsoever.

`Shared Memory Locations: 0` and `Total Race Tests: 0` confirm that Weft
found nothing to analyze in this kernel — the same fundamental scope
limitation seen in issue #4233.

#### Comparing Weft vs Faial on This Issue

Both Weft and Faial missed this race, but for completely different reasons:
- **Weft** missed it because the race is on global memory — entirely outside
  Weft's shared-memory-only scope.
- **Faial** missed it because `write_index` becomes an unconstrained symbolic
  variable in its analysis — it cannot derive the runtime constraint that
  all non-zero threads receive index 0 due to Triton's layout bug.

Only GPUVerify caught this race, by reasoning about global memory accesses
and concretely evaluating the thread-dependent write index.

---

## Tool Comparison Summary

| Tool | Result | Classification | Notes |
|------|--------|----------------|-------|
| GPUVerify | RACE DETECTED | ✅ True Positive | Global memory WAW race detected |
| Faial | DRF | ❌ False Negative | `write_index` treated as unconstrained symbolic variable |
| Weft | No races detected | ❌ False Negative | Global memory race — outside Weft's shared-memory-only scope |