# Issue #7264 — Write-Write Data Race in Reduction (Butterfly Shuffle)

## Source
- **GitHub Issue:** https://github.com/triton-lang/triton/issues/7264
- **Repo:** triton-lang/triton
- **Reproduced on:** Triton 3.0.0 + PyTorch 2.4.0 + NVIDIA GeForce MX450 (sm75)

## What is the Bug?

When lowering a `tl.sum` reduction followed by a layout conversion, Triton
generates a butterfly shuffle where threads write their accumulated result
to shared memory without `__syncthreads()` between stages. Threads in one
stage read locations being written by threads in the previous stage — a
write-write (WAW) hazard.

The output is often numerically close to correct, but the result is
**non-deterministic** across runs and `compute-sanitizer` reports the
hazards explicitly.

## Results

### Plain run (non-deterministic result):
```
Numerically CORRECT (race is silent)
```

### Under compute-sanitizer:
```
RACECHECK SUMMARY: 2 hazards displayed (2 errors, 0 warnings)
```

## Requirements

- Python 3.10+
- CUDA 12.x
- Any NVIDIA GPU
- Triton 3.0.0
- compute-sanitizer (included with CUDA toolkit)

## How to Run

```bash
# Plain run
python reproduce.py

# Racecheck
compute-sanitizer --tool racecheck python reproduce.py
```

---

## Static Analysis Results

### GPUVerify

| Property | Value |
|----------|-------|
| Result | **RACE DETECTED** |
| Classification | ✅ True Positive |
| Race Type | WAW/RAW race in butterfly shuffle sum reduction |
| Errors | 2 (write-read and read-write on `s_data`) |

GPUVerify output:
```
error: possible write-read race on s_data[513]:
  thread 1 reads s_data[tid+stride] while thread 513 writes s_data[tid]
error: possible read-write race on s_data[1]:
  thread 1 writes s_data[tid] while thread 129 reads s_data[tid+stride]
GPUVerify kernel analyser finished with 0 verified, 2 errors
```

### Faial

| Property | Value |
|----------|-------|
| Result | **RACE DETECTED** |
| Classification | ✅ True Positive |
| Race Type | WAW/RAW race in butterfly shuffle |
| Races Found | 1 |
| Command | `faial-drf --cu-to-json=/usr/local/bin/cu-to-json kernel_clean.cu` |

Faial output:
```
Kernel 'reduction_buggy' has 1 data-race.
~~~~ Data-race 1 (CIDI) ~~~~
40 |     s_data[tid] += s_data[tid + stride];
  stride=128, threadIdx x=65 reads; stride=64, threadIdx x=1 writes
True alarm detected!
```

### Weft

| Property | Value |
|----------|-------|
| Result | **No races detected** |
| Classification | ❌ False Negative |
| Race Type Missed | Cross-iteration WAW/RAW race in butterfly shuffle |
| Weft Version | Built from source (github.com/lightsighter/Weft) |
| Command | `weft -f kernel_clean.ptx -t 4 -i -d` |
| PTX Compiled With | `nvcc -ptx kernel_weft.cu` (added `__launch_bounds__(1024)`) |

Weft output:
```
WEFT INFO: No deadlocks detected in kernel reduction_buggy!
WEFT INFO: No races detected in kernel reduction_buggy!
WEFT STATISTICS for Kernel reduction_buggy
  CTA Thread Count:                     1024
  Shared Memory Locations:              1025
  Physical Named Barriers;                 1
  Dynamic Barrier Instances:               1
  Weft Statements:                      5118
  Total Race Tests:                    10170
```

#### Why Weft Missed This Bug

Weft performed 10,170 race tests across 1,025 shared memory locations —
this is not a scope issue. The miss is the same **cross-iteration reasoning
limitation** seen in issues #96 and #4736.

The butterfly reduction loop is:
```c
for (int stride = N/2; stride > 0; stride >>= 1) {
    if (tid < stride) {
        s_data[tid] += s_data[tid + stride];
        // MISSING __syncthreads() here
    }
}
```

Weft unrolls this loop statically (stride = 512 → 256 → ... → 1). The race
occurs across iterations: the write to `s_data[tid]` in iteration `stride=512`
overlaps with the read of `s_data[tid+stride]` in a later stage where a
different thread's address aliases into the previously written location.

Without a `__syncthreads()` between stages, threads from the next stage begin
reading before all threads in the current stage have finished writing. Weft
sees the single initial `__syncthreads()` (before the loop), finds the
barrier ordering correct within what it can reason about, and reports no
races. The address aliasing across unrolled loop stages — that
`s_data[tid]` written at `stride=512` becomes `s_data[tid+stride]` read by
another thread at `stride=256` — is outside Weft's reasoning scope.

This is the same root cause as issues #96 and #4736, confirming a consistent
pattern: Weft cannot detect races in iterative algorithms where shared memory
slots are reused across loop stages without barriers between stages.

---

## Tool Comparison Summary

| Tool | Result | Classification | Notes |
|------|--------|----------------|-------|
| GPUVerify | RACE DETECTED | ✅ True Positive | 2 errors (both directions on `s_data`) |
| Faial | RACE DETECTED | ✅ True Positive | 1 race (cross-stage alias) |
| Weft | No races detected | ❌ False Negative | Cross-iteration address aliasing — same limitation as #96 and #4736 |