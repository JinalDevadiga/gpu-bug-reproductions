# Issue #4736 — Racecheck Bug when `tl.min` Used with `tl.sum`

## Source
- **GitHub Issue:** https://github.com/triton-lang/triton/issues/4736
- **Repo:** triton-lang/triton
- **Reproduced on:** Triton 3.6.0 + PyTorch 2.4.0 + NVIDIA GeForce MX450 (sm75)
- **Status:** Open

## What is the Bug?

When `tl.min` is combined with `tl.sum` in a kernel, the `tl.min` reduction
lowering generates a butterfly shuffle where all threads write to shared
memory without `__syncthreads()` between butterfly stages. Threads in one
stage read shared memory locations that threads in the previous stage are
still writing — a write-write (WAW) hazard.

## Results

### Plain run (Triton 3.6.0):
```
Numerically correct: False
```

### Under compute-sanitizer:
```
RACECHECK SUMMARY: 2 hazards displayed (2 errors, 0 warnings)
```

## Requirements

- Python 3.10+
- CUDA 12.x
- Any NVIDIA GPU
- Triton 3.6.0
- compute-sanitizer (included with CUDA toolkit)

## How to Run

```bash
# Plain run
python reproduce.py

# Racecheck
compute-sanitizer --tool=racecheck python reproduce.py
```

---

## Static Analysis Results

### GPUVerify

| Property | Value |
|----------|-------|
| Result | **RACE DETECTED** |
| Classification | ✅ True Positive |
| Race Type | WAW/RAW race in butterfly shuffle (missing `__syncthreads`) |
| Errors | 4 (write-read and read-write on `s_val` and `s_idx`) |

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
| Result | **RACE DETECTED** |
| Classification | ✅ True Positive |
| Race Type | WAW/RAW race in butterfly shuffle |
| Races Found | 2 (one for `s_val`, one for `s_idx`) |
| Command | `faial-drf --cu-to-json=/usr/local/bin/cu-to-json kernel_clean.cu` |

Faial output:
```
Kernel 'min_reduction_buggy' has 2 data-races.
~~~~ Data-race 1 (CIDI) ~~~~
50 |   float other_val = s_val[tid + stride];
62 |   s_val[tid] = keep_left ? s_val[tid] : other_val;
  stride=16, threadIdx x=1 reads; stride=1, threadIdx x=0 writes
True alarm detected!
~~~~ Data-race 2 (CIDI) ~~~~
51 |   int other_idx = s_idx[tid + stride];
63 |   s_idx[tid] = keep_left ? s_idx[tid] : other_idx;
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
| PTX Compiled With | `nvcc -ptx kernel_weft.cu` (added `__launch_bounds__(32)`) |

Weft output:
```
WEFT INFO: No deadlocks detected in kernel min_reduction_buggy!
WEFT INFO: No races detected in kernel min_reduction_buggy!
WEFT STATISTICS for Kernel min_reduction_buggy
  CTA Thread Count:                       32
  Shared Memory Locations:                34
  Physical Named Barriers;                 1
  Dynamic Barrier Instances:               1
  Weft Statements:                       315
  Total Race Tests:                     1717
```

#### Why Weft Missed This Bug

Unlike issue #4233, Weft did fully analyze this kernel — 34 shared memory
locations, 1,717 race tests performed. The miss is a **reasoning limitation**,
not a scope limitation.

The butterfly reduction loop is:
```c
for (int stride = N_COORDS/2; stride > 0; stride >>= 1) {
    if (tid < stride) {
        float other_val = s_val[tid + stride];   // read from tid+stride
        s_val[tid] = ...;                         // write to tid
        // MISSING __syncthreads() here
    }
}
```

Weft fully unrolls this loop statically (stride = 16 → 8 → 4 → 2 → 1).
The race occurs **across iterations**: the write to `s_val[tid]` in
iteration `stride=16` overlaps with the read of `s_val[tid+stride]` in
a later iteration where a different thread's `tid` equals the write target.

This is the same **cross-iteration reasoning limitation** seen in issue #96.
Weft's happens-before analysis flattens the unrolled loop into a single
instruction trace and checks that every conflicting pair is separated by a
barrier. Within each unrolled iteration there is a `__syncthreads()` only
at the start (the one initial barrier), but the critical observation — that
the write target `s_val[tid]` in one butterfly stage becomes the read source
`s_val[tid+stride]` for a thread in the next stage — requires reasoning about
address aliasing across loop iterations that Weft's barrier dependence graph
cannot encode.

The root cause is identical to issue #96: Weft reasons correctly within a
single barrier-separated phase but cannot track how shared memory slots
alias across different stages of an unrolled loop.

---

## Tool Comparison Summary

| Tool | Result | Classification | Notes |
|------|--------|----------------|-------|
| GPUVerify | RACE DETECTED | ✅ True Positive | 4 errors (both arrays, both directions) |
| Faial | RACE DETECTED | ✅ True Positive | 2 races (one per array) |
| Weft | No races detected | ❌ False Negative | Cross-iteration address aliasing — same limitation as issue #96 |