# Issue #8311 — Incorrect Results from `warp_specialize=True`

## Source
- **GitHub Issue:** https://github.com/triton-lang/triton/issues/8311
- **Repo:** triton-lang/triton
- **Reported on:** Triton commit 8ee5840 (between v3.2 and v3.3, Sep 2025)
- **GPU required:** RTX 5090 or H100 (sm90+) to observe wrong numerical output
- **Status:** Closed

## What is the Bug?

`warp_specialize=True` splits warps into producer/consumer teams:
- **Producer warps** load data via TMA into shared memory
- **Consumer warps** compute (`tl.dot`)

The bug: consumer warps proceed to `tl.dot` before the TMA load for the
current tile has completed — reading stale or partially-written shared
memory. A classic producer-consumer race with a missing barrier.

Result on RTX 5090 (as reported):
```
warp_specialize=True  -->  99.3% of output values are WRONG
```

## Requirements

- Python 3.10+
- CUDA 12.x
- Any NVIDIA GPU (to compile and inspect IR)
- NVIDIA H100 or RTX 5090 (sm90+) to observe wrong output at runtime

## How to Run

```bash
python reproduce.py
```

---

## Static Analysis Results

### GPUVerify

| Property | Value |
|----------|-------|
| Result | **RACE DETECTED** |
| Classification | ✅ True Positive |
| Race Type | Producer-consumer race on shared memory (missing barrier) |
| Errors | 2 (write-read on `s_x` and `s_y`) |

GPUVerify output:
```
error: possible write-read race on s_x[1]:
  Write by thread 1  (producer): s_x[tid] = x_tile[tid];
  Read  by thread 17 (consumer): out[idx] = s_x[idx] * s_y[idx];
error: possible write-read race on s_y[1]:
  Write by thread 1  (producer): s_y[tid] = y_tile[tid];
  Read  by thread 17 (consumer): out[idx] = s_x[idx] * s_y[idx];
GPUVerify kernel analyser finished with 0 verified, 2 errors
```

### Faial

| Property | Value |
|----------|-------|
| Result | **RACE DETECTED** |
| Classification | ✅ True Positive |
| Race Type | Producer-consumer race on shared memory |
| Races Found | 2 (one for `s_x`, one for `s_y`) |
| Command | `faial-drf --cu-to-json=/usr/local/bin/cu-to-json kernel_clean.cu` |

Faial output:
```
Kernel 'warp_specialize_buggy' has 2 data-races.
~~~~ Data-race 1 (CIDI) ~~~~
35 |   s_x[tid] = x_tile[tid];
45 |   out[idx] = s_x[idx] * s_y[idx];
  threadIdx x=16 writes, threadIdx x=0 reads — missing barrier
True alarm detected!
~~~~ Data-race 2 (CIDI) ~~~~
36 |   s_y[tid] = y_tile[tid];
45 |   out[idx] = s_x[idx] * s_y[idx];
True alarm detected!
```

### Weft

| Property | Value |
|----------|-------|
| Result | **No races detected** |
| Classification | ❌ False Negative |
| Race Type Missed | Producer-consumer shared memory race (missing barrier) |
| Weft Version | Built from source (github.com/lightsighter/Weft) |
| Command | `weft -f kernel_clean.ptx -t 4 -i -d` |
| PTX Compiled With | `nvcc -ptx kernel_weft.cu` (added `__launch_bounds__(32)`) |

Weft output:
```
WEFT INFO: No deadlocks detected in kernel warp_specialize_buggy!
WEFT INFO: No races detected in kernel warp_specialize_buggy!
WEFT STATISTICS for Kernel warp_specialize_buggy
  CTA Thread Count:                       32
  Shared Memory Locations:                16
  Physical Named Barriers;                 1
  Dynamic Barrier Instances:               0
  Weft Statements:                        64
  Total Race Tests:                       96
```

#### Why Weft Missed This Bug

This is a **different failure mode** from the cross-iteration issues (#96,
#4736, #7264). Weft did analyze the shared memory — 16 locations, 96 race
tests performed. The key detail is `Dynamic Barrier Instances: 0`.

The kernel has producer threads (0-15) writing to `s_x` and `s_y`, and
consumer threads (16-31) reading from them, with **no barrier at all**
between the two groups:

```c
// Producer warps (threads 0-15)
if (tid < TILE) {
    s_x[tid] = x_tile[tid];
    s_y[tid] = y_tile[tid];
    // BUG: no __syncthreads() after store
}

// Consumer warps (threads 16-31)
if (tid >= TILE) {
    // MISSING __syncthreads() here
    out[idx] = s_x[idx] * s_y[idx];
}
```

Weft builds its happens-before relationships using barrier instructions
(`bar.sync`/`bar.arrive` in PTX). With zero dynamic barrier instances,
the barrier dependence graph is empty — there are no synchronization edges
to reason about at all. Weft appears to treat threads with no common
barrier as unrelated (not racing) rather than as potentially conflicting.

This reveals a subtle gap in Weft's design: it was built primarily for
**warp-specialized kernels that use named barriers** (`bar.sync`/`bar.arrive`)
for explicit producer-consumer coordination. For kernels where the
synchronization is entirely missing (no barriers at all between conflicting
accesses), Weft's analysis produces a vacuously safe result — it finds no
barrier violations because there are no barriers to violate.

In contrast, GPUVerify and Faial both correctly identify that the absence
of any ordering between the producer writes and consumer reads constitutes
a race.

---

## Tool Comparison Summary

| Tool | Result | Classification | Notes |
|------|--------|----------------|-------|
| GPUVerify | RACE DETECTED | ✅ True Positive | 2 errors — producer/consumer with no barrier |
| Faial | RACE DETECTED | ✅ True Positive | 2 races — correctly identified missing barrier |
| Weft | No races detected | ❌ False Negative | 0 dynamic barriers — vacuously safe result; designed for named-barrier kernels |

## PTX Analysis (-arch=sm_90)

We compiled `kernel_clean.cu` with `-arch=sm_90` to inspect whether
targeting H100/RTX 5090 architecture changes the generated instructions:

```bash
nvcc -arch=sm_90 -ptx kernel_clean.cu -o kernel_sm90.ptx
```

The PTX shows only standard instructions even when targeting sm_90:
```ptx
.target sm_90              <- compiled for H100 architecture
st.shared.f32              <- standard shared memory store
ld.global.nc.f32           <- standard global load — NOT TMA
ld.shared.f32              <- standard shared memory load
```

**No TMA instructions appear** — no `cp.async.bulk.tensor`, no `mbarrier.arrive`,
no `mbarrier.wait`. This is because our simplified kernel uses standard C++
loads — compiling for sm_90 does not add TMA semantics.

Additionally the compiler **demoted** `s_x` and `s_y` from shared memory
to registers in some paths:
```ptx
// _ZZ21warp_specialize_buggyPKfS0_PfE3s_x has been demoted
// _ZZ21warp_specialize_buggyPKfS0_PfE3s_y has been demoted
```
This means the simplified kernel does not fully represent the real shared
memory layout of the original Triton warp-specialized kernel.

The real Triton kernel on H100/RTX 5090 would use:
```ptx
cp.async.bulk.tensor.2d.shared::cluster.global ...  // TMA async load
mbarrier.arrive.expect_tx.shared::cta.b64 ...       // H100 barrier arrive
mbarrier.wait.parity.shared::cta.b64 ...            // H100 barrier wait
```

### Key Distinction

| Feature | Our simplified kernel | Real Triton kernel on H100 |
|---------|----------------------|---------------------------|
| Load type | Standard `ld.global` | TMA (`cp.async.bulk.tensor`) |
| Sync needed | `__syncthreads()` | `mbarrier.arrive/wait` |
| Race type GPUVerify found | Write-read between producer/consumer threads | Missing barrier before TMA completes |
| Architecture flag helps? | ❌ No — standard PTX generated | N/A — needs real H100 hardware |
| s_x/s_y location | Demoted to registers by compiler | Shared memory (warp-specialized) |

The producer-consumer race GPUVerify detected in our simplified kernel
and the TMA barrier race in the real kernel both stem from the same root
cause — **missing synchronization between producer and consumer warps**.
The True Positive is valid but the simplified kernel does not capture
the TMA async semantics of the original warp-specialized Triton kernel.