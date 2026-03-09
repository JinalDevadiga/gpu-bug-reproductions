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
