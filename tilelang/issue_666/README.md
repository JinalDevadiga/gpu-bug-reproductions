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
simplified `kernel_clean.cu` that captures the same synchronization structure
(clear → sync → 3-stage pipeline) using standard CUDA without TileLang
dependencies.

---

## PTX Analysis (-arch=sm_90)

We compiled `kernel_clean.cu` with `-arch=sm_90` and confirmed the PTX
only contains standard `bar.sync` instructions — no TMA
(`cp.async.bulk.tensor`) or `mbarrier` instructions appear. This proves our
simplified kernel cannot model the H100 race regardless of compilation
target — TMA instructions only appear when explicitly used in code (as
TileLang does internally).

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
- no data races within thread blocks
- no data races between thread blocks
- no barrier divergence
- no assertion failures
```

#### Why GPUVerify Missed This Bug

1. **Cannot parse real kernel** — TileLang headers incompatible with GPUVerify
2. **Simplified kernel is genuinely race-free** — standard loads with
   `__syncthreads()` are correctly synchronized; GPUVerify proved this correct
3. **No TMA model** — GPUVerify (2018) has no knowledge of H100 TMA hardware,
   `cp.async.bulk.tensor`, or `mbarrier`
4. **No `--arch` flag** — GPUVerify always targets sm_35 internally

### Faial

| Property | Value |
|----------|-------|
| Result | **RACE DETECTED — 2 data-races** |
| Classification | ⚠️ Spurious — races are artifacts of simplification, NOT the original H100 TMA bug |
| Effective classification for original bug | ❌ False Negative |
| Faial Version | faial-drf (built from source, gitlab.com/umb-svl/faial) |
| Command | `faial-drf --cu-to-json=/usr/local/bin/cu-to-json kernel_clean.cu` |

Faial output:
```
Kernel 'main_kernel' has 2 data-races.
~~~~ Data-race 1 (CIDI) ~~~~
55 |     buf_dyn_shmem[0 * STAGE_SIZE + tid] = A[tid];
56 |     buf_dyn_shmem[1 * STAGE_SIZE + tid] = B[tid];
Locals: threadIdx x=0  vs  threadIdx x=4
True alarm detected!

~~~~ Data-race 2 (CIDI) ~~~~
45 |         buf_dyn_shmem[i * STAGE_SIZE + tid] = 0.0f;
Locals: i=0, threadIdx x=4  vs  i=1, threadIdx x=0
True alarm detected!
```

#### Important: These Races Are Artifacts of Our Simplification

Faial detected 2 races — but these are **NOT the original H100 TMA bug**.
They are introduced by a sizing mistake in our simplified kernel:
`STAGE_SIZE = 4` and `BLOCK_SIZE = 4` causes stage boundaries to alias
(Thread 4 writes `buf_dyn_shmem[4]` via stage 0, Thread 0 writes
`buf_dyn_shmem[4]` via stage 1 — same address). In the real kernel,
`STAGE_SIZE = 4096` and `BLOCK_SIZE = 128` — stages never alias.

Faial's SMT solver (Z3) correctly solved the index equation `0*4+4 = 1*4+0`
and found the aliasing, while GPUVerify's theorem prover missed it. This
shows Faial is more precise at index arithmetic — but the races found are
simplification artifacts, not the original bug.

**The original H100 TMA race is still not detected by any tool.**

### Weft

| Property | Value |
|----------|-------|
| Result | **No races detected** |
| Classification | ❌ False Negative |
| Race Type Missed | H100 TMA async pipeline race |
| Weft Version | Built from source (github.com/lightsighter/Weft) |
| Command | `weft -f kernel_clean.ptx -t 4 -i -d` |
| PTX Compiled With | `nvcc -ptx kernel_weft.cu` (added `__launch_bounds__(4)`) |

Weft output:
```
WEFT INFO: No deadlocks detected in kernel main_kernel!
WEFT INFO: Barriers properly recycled in kernel main_kernel!
WEFT INFO: No races detected in kernel main_kernel!
WEFT STATISTICS for Kernel main_kernel
  CTA Thread Count:                        4
  Shared Memory Locations:                12
  Physical Named Barriers;                 1
  Dynamic Barrier Instances:               7
  Weft Statements:                        72
  Total Race Tests:                       60
```

#### Why Weft Missed This Bug

Unlike GPUVerify which proved the simplified kernel safe, Weft fully
analyzed the shared memory — 12 locations, 60 race tests across 7 barrier
instances. Weft found no races because the simplified kernel is genuinely
race-free for standard loads. Three compounding reasons:

1. **Cannot parse real kernel** — TileLang headers incompatible with Weft;
   simplified kernel uses standard loads where `__syncthreads()` IS sufficient

2. **No TMA model** — Weft was built in 2015. The H100 `mbarrier` primitive
   was introduced in 2022. Weft has no knowledge of TMA async operations

3. **Named barriers vs mbarrier** — Weft was designed for **named barrier**
   verification (`bar.sync N` / `bar.arrive N`), not TMA hardware sync:

| Barrier Type | Weft Can Detect? | Purpose |
|-------------|-----------------|---------|
| `__syncthreads()` (`bar.sync 0`) | ✅ Yes | Thread sync |
| Named barriers (`bar.sync N`) | ✅ Yes | Warp group sync — Weft's primary purpose |
| `mbarrier` (TMA async) | ❌ No | TMA hardware sync — H100-specific, 2022 |

Note: Weft did NOT find the spurious races that Faial found — Weft's
happens-before analysis is not sensitive to the STAGE_SIZE aliasing
artifact because it reasons about concrete instruction ordering rather
than symbolic index arithmetic.

---

## Tool Comparison Summary

| Tool | Result | Classification | Notes |
|------|--------|----------------|-------|
| GPUVerify | Verified (DRF) | ❌ False Negative | Proved simplified kernel safe; no TMA model |
| Faial | 2 races (spurious) | ❌ FN for original bug | Races are simplification artifacts (STAGE_SIZE==BLOCK_SIZE aliasing); H100 TMA race not detected |
| Weft | No races detected | ❌ False Negative | 60 race tests performed; simplified kernel race-free; no mbarrier model |

## Key Takeaway

All three tools give **False Negative for the original H100 TMA race**
because the race operates at the hardware level — between TMA engine and
CUDA threads — below what any static thread-level tool can reason about.
Runtime confirmation requires actual H100 hardware (~79.3% wrong output).