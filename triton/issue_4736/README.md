# Issue #4736 — Racecheck Bug when `tl.min` Used with `tl.sum`

## Source
- **GitHub Issue:** https://github.com/triton-lang/triton/issues/4736
- **Repo:** triton-lang/triton
- **Reproduced on:** Triton 3.6.0 + PyTorch 2.4.0 + NVIDIA GeForce MX450 (sm75)
- **Status:** Open

## What is the Bug?

When `tl.min` is combined with `tl.sum` in a kernel, the `tl.min` reduction
lowering in `triton/language/standard.py:237` generates a butterfly shuffle
where all threads write to the same shared memory address simultaneously —
a write-write (WAW) hazard.

The kernel finds the nearest coordinate to each input point by computing
squared distances, summing across dimensions with `tl.sum`, then finding
the minimum across coordinates with `tl.min`. The race occurs inside the
`tl.min` reduction lowering.

On Triton 3.6.0 the race causes numerically wrong output in addition to
the WAW hazard detected by `compute-sanitizer`.

## Results

### Plain run (Triton 3.6.0):
```
Numerically correct: False
```

### Under compute-sanitizer:
```
========= RACECHECK SUMMARY: 2 hazards displayed (2 errors, 0 warnings)
```

Note: "Device not supported" and "WDDM debugger interface" errors appear
because this machine runs WSL2 — the full debugger cannot attach, but the
racecheck tool still detects the WAW hazards.

## Requirements

- Python 3.10+
- CUDA 12.x
- Any NVIDIA GPU
- Triton 3.6.0
- `compute-sanitizer` (included with CUDA toolkit)

## Setup
```bash
conda activate triton-bugs
```

## How to Run
```bash
# Plain run — observe wrong output
python reproduce.py

# Racecheck — observe WAW hazards
compute-sanitizer --tool=racecheck python reproduce.py
```
