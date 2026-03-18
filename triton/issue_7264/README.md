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

---

## Static Analysis Results

### GPUVerify

| Property | Value |
|----------|-------|
| Result | **RACE DETECTED — 2 errors** |
| Classification | ✅ True Positive |
| Race Type | WAW/RAW races on shared memory `s_data` between butterfly stages |

GPUVerify output:
```
error: possible write-read race on s_data[513]:
  Read  by thread 1,   Write by thread 513: s_data[tid] += s_data[tid + stride]
error: possible read-write race on s_data[1]:
  Write by thread 1,   Read  by thread 129: s_data[tid] += s_data[tid + stride]
GPUVerify kernel analyser finished with 0 verified, 2 errors
```

### Faial

| Property | Value |
|----------|-------|
| Result | **RACE DETECTED — 1 data-race** |
| Classification | ✅ True Positive |
| Race Type | RAW race on shared memory `s_data` between butterfly stages |
| Faial Version | faial-drf (built from source, gitlab.com/umb-svl/faial) |
| Command | `faial-drf --cu-to-json=/usr/local/bin/cu-to-json kernel_clean.cu` |

Faial output:
```
Kernel 'reduction_buggy' has 1 data-race.
~~~~ Data-race 1 (CIDI) ~~~~
40 |             s_data[tid] += s_data[tid + stride];
Globals: s_data[] = 65  |  blockIdx: x=0, y=0, z=0
Locals:
  stride=128, threadIdx x=65  vs  stride=64, threadIdx x=1
True alarm detected!
```

#### How Faial Found This Race
Faial identified that on line 40 (`s_data[tid] += s_data[tid + stride]`),
Thread 65 with `stride=128` reads `s_data[65 + 128] = s_data[193]` while
Thread 1 with `stride=64` writes `s_data[1]` — with no `__syncthreads()`
between butterfly iterations, these accesses from different stride stages
overlap and conflict.

#### Race Count Comparison
Faial reports **1 race** while GPUVerify reports **2 errors** and
compute-sanitizer reports **2 hazards**. This is consistent with the
pattern seen in issue #4736 — Faial reports each conflicting pair once,
while GPUVerify counts each direction of the conflict separately
(read-write AND write-read). All three tools identify the same underlying
missing `__syncthreads()` bug.

---

## Tool Comparison Summary

| Tool | Result | Classification | Races Found | Notes |
|------|--------|----------------|-------------|-------|
| compute-sanitizer | 2 hazards | ✅ True Positive | 2 | Runtime detection |
| GPUVerify | RACE DETECTED | ✅ True Positive | 2 errors | Counts each direction separately |
| Faial | RACE DETECTED | ✅ True Positive | 1 race | Reports each conflict once |