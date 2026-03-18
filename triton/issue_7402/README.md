# Issue #7402 — `tl.atomic_add` Return Value Wrong Across Threads

## Source
- **GitHub Issue:** https://github.com/triton-lang/triton/issues/7402
- **Repo:** triton-lang/triton
- **Fixed in:** PR #7460 (Atomic RMW Broadcasting)
- **Reproduced on:** Triton 3.0.0 + PyTorch 2.4.0 + NVIDIA GeForce MX450 (sm75)

## What is the Bug?

`tl.atomic_add` atomically increments a counter and returns the old value,
intended as a ticket dispenser where each thread gets a unique write slot:
```python
write_index = tl.atomic_add(index_ptr + tl.arange(0, 1), val=1, sem="relaxed")
tl.store(out_ptr + write_index[:, None] * 8 + tl.arange(0, 8)[None, :], ...)
```

With `index[0] = 1`, the expected return from `atomic_add` is `1` for all
threads, so all threads should write to `out[1, :]`. Instead:

- Thread 0 gets the correct return value (`1`) and writes to `out[1, :]`
- All other threads get `0` and write to `out[0, :]`, corrupting that row

The root cause is a layout mismatch: `tt.atomic_rmw` produces its result in
`#blocked` layout with `sizePerThread = [1]` across 128 threads (4 warps x
32), but the tensor only has 1 element. Only thread 0 holds the real atomic
return value; the remaining 127 threads hold 0. The subsequent
`triton_gpu.convert_layout` spreads this incorrect state rather than
broadcasting thread 0's value to all threads. PR #7460 fixed the atomic RMW
lowering to properly broadcast the return value to all participating threads.

## Requirements

- Python 3.10+
- CUDA 12.x
- Any NVIDIA GPU
- **Triton 3.0.0** (bug is fixed in 3.1.0+)

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

## Expected Output (bug firing on Triton 3.0.0)
```
Result  : [[0, 2, 2, 2, 2, 2, 2, 2], [2, 0, 0, 0, 0, 0, 0, 0]]
Expected: [[0, 0, 0, 0, 0, 0, 0, 0], [2, 2, 2, 2, 2, 2, 2, 2]]
BUG CONFIRMED: tl.atomic_add return value is wrong across threads
```

---

## Static Analysis Results

### GPUVerify

| Property | Value |
|----------|-------|
| Result | **RACE DETECTED — 1 error** |
| Classification | ✅ True Positive |
| Race Type | WAW race on global memory `out[0]` |

GPUVerify output:
```
error: possible write-write race on out[0]:
  Write by thread 1: out[write_index * ROW_WIDTH + i] = 2;
  Write by thread 0: out[write_index * ROW_WIDTH + i] = 2;
GPUVerify kernel analyser finished with 0 verified, 1 error
```

### Faial

| Property | Value |
|----------|-------|
| Result | **DRF (Data-Race Free)** |
| Classification | ❌ False Negative |
| Race Type Missed | WAW race on global memory `out[write_index * 8 + i]` |
| Faial Version | faial-drf (built from source, gitlab.com/umb-svl/faial) |
| Command | `faial-drf --cu-to-json=/usr/local/bin/cu-to-json kernel_clean.cu` |

Faial output:
```
Kernel 'atomic_add_buggy' is DRF!
```

#### Why Faial Missed This Bug

Inspection of Faial's memory access protocol (`--show-map`) reveals the
root cause:

```
int write_index1;   <- treated as unconstrained symbolic variable
foreach (i in 0..7) {
    rw(2) out[(write_index1 * 8) + i]
}
```

Faial abstracted `write_index` as a **free symbolic variable** with no
constraints on its value. Since two threads can have different values of
`write_index1`, Faial concludes they may write to **different** memory
locations — and therefore sees no race.

The actual bug is that at runtime, threads 1-127 all receive `write_index=0`
due to the layout mismatch in Triton's atomic return value broadcasting.
This is a **runtime value constraint** — the race only exists because the
layout bug forces all non-zero threads to have the same index value.

Faial cannot reason about this because:
1. `write_index` is initialized from `atomicAdd` which Faial models as a
   non-deterministic value (correctly — atomics return different values in
   general)
2. The constraint that "threads 1-127 all get 0 due to the layout bug"
   is a property of Triton's specific broken lowering, not something
   derivable from the CUDA kernel structure alone
3. This is similar to the `(CIDD)` data-dependent index case in issue
   #4233, except here Faial doesn't even flag a potential alarm because
   there's no structural evidence that the indices must collide

This is a **fundamental limitation** of symbolic analysis with
unconstrained input variables — the race is only detectable if the
analyzer knows the specific runtime constraint imposed by the Triton bug.

---

## Tool Comparison Summary

| Tool | Result | Classification | Notes |
|------|--------|----------------|-------|
| GPUVerify | RACE DETECTED (1 error) | ✅ True Positive | Detected WAW on out[0] |
| Faial | DRF | ❌ False Negative | `write_index` treated as unconstrained symbolic variable |