# Issue #666 — Incorrect Results When Clearing Shared Memory Before Pipelined Loops

## Source
- **GitHub Issue:** https://github.com/tile-ai/tilelang/issues/666
- **Repo:** TileLang
- **Status:** Closed

## What is the Bug?

When `T.clear()` is applied to a shared memory buffer (e.g., `B_shared`)
before a pipelined loop (`T.Pipelined` with `num_stages > 1`), TileLang
generates CUDA code where the clear operation is not properly synchronized
with the asynchronous pipeline stages on NVIDIA H100 GPUs.

The pipelined loop performs staged, asynchronous loads into shared memory
using **TMA (Tensor Memory Accelerator)** — an H100-specific hardware unit
that performs bulk memory transfers independently of CUDA threads. If the
shared buffer is cleared just before the pipeline begins, the zeroing
operation and the TMA loads can overlap in time because `__syncthreads()`
does **not** wait for TMA transfers to complete. You need `mbarrier` instead.

As a result, the shared memory may contain a mix of:
- newly loaded data
- partially cleared (zeroed) values
- stale data

This causes the GEMM computation to use incorrect inputs, leading to
significant output errors (reported ~79.3% mismatched elements on H100).

This issue does not appear on GPUs without TMA hardware (MX450, RTX 4090)
because their async loads are synchronous enough that `__syncthreads()` is
sufficient.

---

## What is TMA and Why Does It Cause This Race?

On **normal GPUs (MX450, RTX 4090)**:
```
Thread 0: zeros buf_dyn_shmem[0]
Thread 1: zeros buf_dyn_shmem[1]
__syncthreads()   <- ALL threads wait — zeroing is guaranteed complete
Thread 0: loads A[0] into buf_dyn_shmem[0]   <- safe
Thread 1: loads A[1] into buf_dyn_shmem[1]   <- safe
```
`__syncthreads()` is sufficient — zeroing completes before any loading starts.

On **H100 (sm90)**:
```
Thread 0: zeros buf_dyn_shmem[0]
Thread 1: zeros buf_dyn_shmem[1]
__syncthreads()   <- threads wait, BUT TMA hardware keeps running!
TMA engine: loads A[0] into buf_dyn_shmem[0]  <- async, overlaps with clear!
            buf_dyn_shmem[0] = mix of zeroed and loaded values -> WRONG
```
`__syncthreads()` only synchronizes **CUDA threads** — it does NOT stop the
**TMA hardware engine**. The correct fix requires:
```cuda
mbarrier.arrive.expect_tx.shared::cta.b64 ...  // tell TMA barrier
mbarrier.wait.parity.shared::cta.b64 ...        // wait for TMA completion
```

---

## Key Pattern in Generated CUDA Code
```c
// Step 1: T.clear(B_shared) — zero shared memory
*(uint4*)(buf_dyn_shmem + ...) = make_uint4(0, 0, 0, 0);

__syncthreads();  // <- INSUFFICIENT on H100 — does not wait for TMA!

// Step 2: Async TMA loads into SAME shared memory — overlaps with clear!
*(uint4*)(buf_dyn_shmem + ...) = *(uint4*)(B + ...);

// Pipeline loop
for (int ko = 0; ko < 30; ++ko) { ... }
```

---

## Real Generated CUDA Kernel

Running `reproduce.py` generates the actual TileLang CUDA kernel which uses:
- `tl::ptx_ldmatrix_x4()` — warp-level matrix load (TileLang-specific)
- `tl::mma_sync()` — warp-level MMA instruction (TileLang-specific)
- `*(uint4*)` — 128-bit vectorized shared memory loads
- Complex swizzled shared memory addressing for H100 memory layout
- TileLang-specific headers (`tl_templates/cuda/instruction/mma.h` etc.)

This real kernel **cannot be parsed by GPUVerify, Faial, or Weft** because
they do not understand TileLang-specific headers and primitives.

---

## Requirements

- Python 3.10+
- CUDA 12.3+
- TileLang 0.1.8
- Any NVIDIA GPU (for code generation and inspection)
- NVIDIA H100 (sm90+) to observe incorrect numerical output at runtime

## System Configuration (Tested On)

- GPU: NVIDIA GeForce MX450 (sm75, 2GB VRAM)
- CUDA Version: 12.3 (WSL)
- OS: Ubuntu (WSL on Windows)
- Python Version: 3.10
- PyTorch Version: 2.4.0

Note: The incorrect numerical output could not be observed on this setup
(no H100), but the buggy CUDA code pattern is confirmed via reproduce.py.

## Setup
```bash
conda create -n tilelang-bugs python=3.12 -y
conda activate tilelang-bugs
pip install torch==2.4.0
pip install tilelang==0.1.8
```

## How to Run
```bash
python reproduce.py
```

---

## Why We Cannot Run Static Analysis Tools on the Real Kernel

The real TileLang-generated kernel uses TileLang-specific headers:
```cuda
#include <tl_templates/cuda/instruction/mma.h>
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
tl::ptx_ldmatrix_x4(...)
tl::mma_sync(...)
```

GPUVerify, Faial, and Weft **cannot parse these headers or primitives** —
they are TileLang-internal and not standard CUDA. Therefore we wrote a
simplified `kernel_clean.cu` that:
- Captures the same synchronization structure (clear → sync → pipeline)
- Uses standard CUDA (no TileLang headers)
- Is compilable by all three tools

---

## Why the Simplified Kernel is Genuinely Race-Free

Our `kernel_clean.cu` uses **standard loads**:
```cuda
buf_dyn_shmem[i * STAGE_SIZE + tid] = 0.0f;     // standard store
__syncthreads();                                   // sufficient for standard loads
buf_dyn_shmem[0 * STAGE_SIZE + tid] = A[tid];   // standard load
```

For standard loads, `__syncthreads()` IS sufficient — threads wait for all
stores to complete before any loads begin. There is genuinely no race in our
simplified kernel. All three tools correctly report no races.

The race only happens when TMA is used — which our simplified kernel does
not use.

---

## PTX Analysis (-arch=sm_90)

We compiled `kernel_clean.cu` with `-arch=sm_90` to inspect the PTX:
```bash
nvcc -arch=sm_90 -ptx kernel_clean.cu -o kernel_sm90.ptx
```

The PTX shows only standard instructions even when targeting sm_90:
```ptx
.target sm_90          <- compiled for H100 architecture
bar.sync 0             <- standard __syncthreads() — NOT mbarrier
st.shared.u32          <- standard shared memory store
ld.global.nc.f32       <- standard global load — NOT cp.async.bulk.tensor
st.shared.f32          <- standard shared memory store
ld.shared.f32          <- standard shared memory load
```

**No TMA instructions appear** — no `cp.async.bulk.tensor`, no `mbarrier`.
This confirms our simplified kernel cannot model the H100 race regardless
of compilation target. TMA instructions only appear when explicitly used in
code (as TileLang does internally).

The real TileLang kernel compiled for sm_90 would show:
```ptx
cp.async.bulk.tensor.2d.shared::cluster.global ...  // TMA async load
mbarrier.arrive.expect_tx.shared::cta.b64 ...       // H100 barrier arrive
mbarrier.wait.parity.shared::cta.b64 ...            // H100 barrier wait
```
These are the instructions that cause the race — and that no existing
static analysis tool understands.

---

## Static Analysis Results

### GPUVerify

| Property | Value |
|----------|-------|
| Result | **Verified (no races found)** |
| Classification | ❌ False Negative |
| Race Type Missed | H100 TMA async pipeline race |
| Command | `python GPUVerify.py --cuda --blockDim=4 --gridDim=1 kernel_clean.cu` |

GPUVerify output:
```
GPUVerify kernel analyser finished with 1 verified, 0 errors
Verified: kernel_clean.cu
- no data races within thread blocks
- no data races between thread blocks
- no barrier divergence
- no assertion failures
```

#### Why GPUVerify Missed This Bug

Three compounding reasons:

1. **Cannot parse real kernel** — TileLang headers are incompatible with
   GPUVerify. We had to use a simplified kernel instead.

2. **Simplified kernel is genuinely race-free** — standard loads with
   `__syncthreads()` are correctly synchronized. GPUVerify correctly
   verifies no race exists in our simplified version.

3. **No TMA model** — GPUVerify (2018) has no knowledge of H100's TMA
   hardware (`cp.async.bulk.tensor`), `mbarrier`, or sm90+ async pipeline
   semantics. Even if it could parse the real kernel, it could not detect
   this class of race.

GPUVerify also has no `--arch` flag — it targets sm_35 internally and
cannot be directed to model sm_90 behavior.

### Faial

| Property | Value |
|----------|-------|
| Result | **DRF (Data-Race Free)** |
| Classification | ❌ False Negative |
| Race Type Missed | H100 TMA async pipeline race |
| Command | `faial-drf --cu-to-json=/usr/local/bin/cu-to-json kernel_clean.cu` |

Faial output:
```
Kernel 'main_kernel' is DRF!
```

#### Why Faial Missed This Bug

Same three reasons as GPUVerify — cannot parse real kernel, simplified
kernel is genuinely race-free, and no TMA async model. Faial performs
barrier-phase analysis at the thread level and has no visibility into
H100 TMA hardware behavior.

### Weft

| Property | Value |
|----------|-------|
| Result | **No races detected** |
| Classification | ❌ False Negative |
| Race Type Missed | H100 TMA async pipeline race |
| Weft Version | Built from source (github.com/lightsighter/Weft) |
| Command | `weft -f kernel_clean.ptx -t 4 -i -d` |
| PTX Compiled With | `nvcc -ptx kernel_weft.cu` (added `__launch_bounds__(128)`) |

Weft output:
```
WEFT INFO: No deadlocks detected in kernel main_kernel!
WEFT INFO: Barriers properly recycled in kernel main_kernel!
WEFT INFO: No races detected in kernel main_kernel!
WEFT STATISTICS for Kernel main_kernel
  CTA Thread Count:                      128
  Shared Memory Locations:               512
  Physical Named Barriers;                 1
  Dynamic Barrier Instances:              61
  Total Race Tests:                   432512
```

#### Why Weft Missed This Bug

Weft was designed for **named barrier** verification (`bar.sync N` /
`bar.arrive N` in PTX) in warp-specialized kernels. The H100 `mbarrier`
is a **fundamentally different** synchronization primitive:

| Barrier Type | Weft Can Detect? | Purpose |
|-------------|-----------------|---------|
| `__syncthreads()` (`bar.sync 0`) | ✅ Yes | Thread synchronization |
| Named barriers (`bar.sync N`) | ✅ Yes | Warp group sync — Weft's primary purpose |
| `mbarrier` (TMA async) | ❌ No | TMA hardware sync — H100-specific, introduced 2022 |

Weft was built in 2015 — `mbarrier` did not exist until H100 (2022).
Weft has no model for TMA async operations and cannot detect missing or
incorrect `mbarrier` usage.

Additionally, same three reasons apply: cannot parse real kernel,
simplified kernel is genuinely race-free, no TMA model.

---

## Tool Comparison Summary

| Tool | Result | Classification | Notes |
|------|--------|----------------|-------|
| GPUVerify | Verified | ❌ False Negative | Cannot parse real kernel; simplified kernel genuinely race-free; no TMA model |
| Faial | DRF | ❌ False Negative | Cannot parse real kernel; simplified kernel genuinely race-free; no TMA model |
| Weft | No races detected | ❌ False Negative | Cannot parse real kernel; simplified kernel genuinely race-free; no mbarrier model |

## Key Takeaway

All three tools give False Negative for the same fundamental reason: **the
race in #666 is an H100 TMA async pipeline race that operates at the hardware
level, below the thread synchronization level that all three tools reason
about.** Runtime confirmation requires actual H100 hardware where ~79.3%
of output values are wrong. A future static analysis tool with sm_90+ TMA
async semantics would be needed to detect this class of race statically.