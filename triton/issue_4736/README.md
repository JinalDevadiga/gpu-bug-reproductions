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

---

## Static Analysis Results

### GPUVerify

| Property | Value |
|----------|-------|
| Result | **RACE DETECTED — 4 errors** |
| Classification | ✅ True Positive |
| Race Type | WAW/RAW races on shared memory `s_val` and `s_idx` between butterfly stages |

GPUVerify output:
```
error: possible write-read race on s_val[8]: thread 0 reads while thread 8 writes
error: possible write-read race on s_idx[8]: thread 0 reads while thread 8 writes
error: possible read-write race on s_val[0]: thread 0 writes while thread 3 reads
error: possible read-write race on s_idx[0]: thread 0 writes while thread 3 reads
GPUVerify kernel analyser finished with 0 verified, 4 errors
```

### Faial

| Property | Value |
|----------|-------|
| Result | **RACE DETECTED — 2 data-races** |
| Classification | ✅ True Positive |
| Race Type | RAW/WAW races on shared memory `s_val` and `s_idx` between butterfly stages |
| Faial Version | faial-drf (built from source, gitlab.com/umb-svl/faial) |
| Command | `faial-drf --cu-to-json=/usr/local/bin/cu-to-json kernel_clean.cu` |

Faial output:
```
Kernel 'min_reduction_buggy' has 2 data-races.
~~~~ Data-race 1 (CIDI) ~~~~
50 |             int   other_idx = s_idx[tid + stride];
62 |             s_idx[tid] = keep_left ? s_idx[tid] : other_idx;
Locals: stride=16 | threadIdx x=1  vs  stride=1 | threadIdx x=0
True alarm detected!

~~~~ Data-race 2 (CIDI) ~~~~
49 |             float other_val = s_val[tid + stride];
61 |             s_val[tid] = keep_left ? s_val[tid] : other_val;
Locals: stride=16 | threadIdx x=7  vs  stride=4 | threadIdx x=3
True alarm detected!
```

#### How Faial Found These Races
Faial detected both races in the butterfly shuffle loop — one on `s_idx`
and one on `s_val`. In both cases, one thread reads `s_val[tid + stride]`
or `s_idx[tid + stride]` while another thread from a different stride
iteration writes to the overlapping location, with no `__syncthreads()`
between butterfly stages.

Notably, Faial found exactly **2 races** — matching `compute-sanitizer`'s
count of 2 hazards — while GPUVerify found 4 errors. Both counts are
correct: GPUVerify reports each direction of the conflict separately
(read-write AND write-read), while Faial and compute-sanitizer report
each conflicting pair once.

---

## Tool Comparison Summary

| Tool | Result | Classification | Races Found | Notes |
|------|--------|----------------|-------------|-------|
| compute-sanitizer | 2 hazards | ✅ True Positive | 2 | Runtime detection |
| GPUVerify | RACE DETECTED | ✅ True Positive | 4 errors | Counts each direction separately |
| Faial | RACE DETECTED | ✅ True Positive | 2 races | Matches compute-sanitizer count |