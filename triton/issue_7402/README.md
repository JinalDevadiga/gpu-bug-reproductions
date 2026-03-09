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
`#blocked` layout with `sizePerThread = [1]` across 128 threads (4 warps ×
32), but the tensor only has 1 element. Only thread 0 holds the real atomic
return value; the remaining 127 threads hold 0. The subsequent
`triton_gpu.convert_layout` spreads this incorrect state rather than
broadcasting thread 0's value to all threads. PR #7460 fixed the atomic RMW
lowering to properly broadcast the return value to all participating threads.

## Key Evidence in the TTGIR (Triton 3.0.0 — buggy)
```
%2 = tt.atomic_rmw add, relaxed, gpu, %1, %cst, %cst_0 :
     (...) -> tensor<1xi32, #blocked>          ← only thread 0 has real value
%3 = triton_gpu.convert_layout %2 :
     tensor<1xi32, #blocked> ->
     tensor<1xi32, #triton_gpu.slice<...>>     ← spreads wrong state, not a broadcast
%4 = tt.expand_dims %3 ...                     ← other threads still have 0
```

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
  Only thread 0 gets the correct atomic_add return value;
  all other threads in the block receive 0 instead.
```
