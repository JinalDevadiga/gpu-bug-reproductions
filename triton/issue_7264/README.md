# Issue #7264 — Write-Write Data Race in Reduction (Butterfly Shuffle)

## Source
- **GitHub Issue:** https://github.com/triton-lang/triton/issues/7264
- **Repo:** triton-lang/triton
- **Reproduced on:** Triton 3.0.0 + PyTorch 2.4.0 + NVIDIA GeForce MX450 (sm75)

## What is the Bug?

When lowering a `tl.sum` reduction followed by a layout conversion, Triton
generates a butterfly shuffle so every thread accumulates the full result.
All threads then write their result to the **same shared memory address**.

The race: because threads accumulate in different orders during the butterfly
shuffle, they may compute slightly different floating-point values due to
FP non-associativity. Multiple threads then write **different values to the
same address** simultaneously — a write-write (WAW) hazard.

The output is often numerically close to correct, but:
- The result is **non-deterministic** across runs
- `compute-sanitizer --tool racecheck` reports the hazards explicitly

## Reproduction

### Plain run (non-deterministic result):
```
Kernel result : 6.850529
Reference     : 6.850526
Numerically CORRECT (race is silent)
```
```
Kernel result : 41.672100     ← different value on second run!
Reference     : 41.672104
Numerically CORRECT (race is silent)
```

### Under compute-sanitizer (race detected):
```
========= RACECHECK SUMMARY: 2 hazards displayed (2 errors, 0 warnings)
```

Note: "Device not supported" and "WDDM debugger interface" errors appear
because this machine runs WSL2 — the full debugger interface cannot attach,
but the racecheck tool still detects the WAW hazards.

## Requirements

- Python 3.10+
- CUDA 12.x
- Any NVIDIA GPU
- Triton 3.0.0
- `compute-sanitizer` (included with CUDA toolkit)

## Setup
```bash
conda create -n triton-7402 python=3.10 -y
conda activate triton-7402
pip install torch==2.4.0
pip install triton==3.0.0
```

## How to Run
```bash
# Plain run — observe non-deterministic output
python reproduce.py

# Racecheck — observe WAW hazards
compute-sanitizer --tool racecheck python reproduce.py
```
