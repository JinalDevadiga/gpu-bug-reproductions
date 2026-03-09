# Issue #4362 — `tl.associative_scan` Wrong Results with `reverse=True`

## Source
- **GitHub Issue:** https://github.com/triton-lang/triton/issues/4362
- **Repo:** triton-lang/triton
- **Reproduced on:** Triton 3.0.0 + PyTorch 2.4.0 + NVIDIA GeForce MX450 (sm75)

## What is the Bug?

`tl.associative_scan` produces incorrect results when `reverse=True`. The
forward direction (`reverse=False`) works correctly. The bug is in how
Triton lowers the reverse scan — elements are not scanned in the right
order, causing wrong intermediate and final values.

Using a first-order linear recurrence combine function:
```python
def op(fl, xl, fr, xr):
    return fr * fl, fr * xl + xr
```

With inputs:
```
exp  = [1.0, 1.5, 0.8, 2.0]
vals = [1.0, -1.0, 0.5, 2.0]
```

## Results

### reverse=False (correct):
```
exp  : [1.0, 1.5, 1.2, 2.4]
vals : [1.0, 0.5, 0.9, 3.8]
```

### reverse=True (buggy):
```
exp  result  : [2.4, 2.4, 2.4, 2.4]      ← all identical, wrong
exp  expected: [2.4, 2.4, 1.6, 2.0]
vals result  : [3.15, 4.55, 2.1, 3.5]    ← wrong
vals expected: [3.15, 2.15, 2.1, 2.0]
```

The `exp` output is all `2.4` — the final accumulated value broadcast to
every position — indicating the scan is not respecting element ordering when
reversing, effectively reducing rather than scanning.

## Requirements

- Python 3.10+
- CUDA 12.x
- Any NVIDIA GPU
- Triton 3.0.0

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
=== reverse=False (should be correct) ===
  exp  : [1.0, 1.5, 1.2000000476837158, 2.4000000953674316]
  vals : [1.0, 0.5, 0.9000000357627869, 3.799999952316284]

=== reverse=True (buggy) ===
  exp  result  : [2.4000000953674316, 2.4000000953674316, 2.4000000953674316, 2.4000000953674316]
  exp  expected: [2.4000000953674316, 2.4000000953674316, 1.600000023841858, 2.0]
  vals result  : [3.1499998569488525, 4.550000190734863, 2.0999999046325684, 3.5]
  vals expected: [3.1499900817871094, 2.1499900817871094, 2.0999999046325684, 2.0]

BUG CONFIRMED: tl.associative_scan reverse=True produces wrong results
  exp  diff: [0.0, 0.0, 0.8000000715255737, 0.40000009536743164]
  vals diff: [9.775161743164062e-06, 2.400010108947754, 0.0, 1.5]
```
