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

The pipelined loop performs staged, asynchronous loads into shared memory. 
If the shared buffer is cleared just before the pipeline begins, the zeroing 
operation and the pipeline loads can overlap in time.

As a result, the shared memory may contain a mix of:
- newly loaded data
- partially cleared (zeroed) values
- stale data

This causes the GEMM computation to use incorrect inputs, leading to 
significant output errors (reported ~79.3% mismatched elements on H100).

This issue does not typically appear on GPUs like RTX 4090 due to differences 
in memory consistency and async execution behavior.

---

## Reproduction Strategy

This reproduction focuses only on the buggy pattern described in the issue.

It demonstrates how placing `T.clear(B_shared)` before a pipelined loop leads 
to unsafe code generation. The goal is to inspect the generated CUDA kernel 
and identify the problematic ordering between shared memory clearing and 
pipeline execution.

---

## Key Pattern in Generated CUDA Code

In the generated CUDA kernel, look for:

1. A loop that zeroes shared memory (from `T.clear(B_shared)`)
2. A single `__syncthreads()` after the clear
3. The beginning of pipelined async loads into the same shared memory region

Example pattern:

```c
// Zeroing shared memory
for (...) {
    *(uint4*)(buf_dyn_shmem + ...) = make_uint4(0, 0, 0, 0);
}

__syncthreads();  // insufficient synchronization

// Async loads into same shared memory
*(uint4*)(buf_dyn_shmem + ...) = *(uint4*)(B + ...);

// Pipeline loop starts
for (int ko = 0; ko < ...; ++ko) { ... }
```

On H100, the async pipeline may overlap with the clearing operation,
leading to inconsistent shared memory contents.

### Requirements

* Python 3.10+
* CUDA 12.3+
* TileLang 0.1.8
* Any NVIDIA GPU (for code generation and inspection)
* NVIDIA H100 (to observe incorrect numerical output at runtime)

### System Configuration (Tested On)
* GPU: NVIDIA GeForce MX450 (Laptop GPU, 2GB VRAM)
* CUDA Version: 12.3 (WSL)
* OS: Ubuntu (WSL on Windows)
* Python Version: 3.10
* PyTorch Version: 2.4.0

Note: The incorrect numerical output could not be observed on this setup,
but the problematic CUDA code pattern is reproducible.

Setup
```bash
conda create -n tilelang-bugs python=3.12 -y
conda activate tilelang-bugs
pip install torch==2.4.0
pip install tilelang==0.1.8
```

### How to Run
```bash
cd issue_666
python reproduce.py
```

### What to Look For

Inspect the generated CUDA kernel and identify:
* A shared memory zeroing loop (T.clear)
* Only one __syncthreads() after clearing
* Immediate reuse of the same memory by pipelined async loads

This pattern indicates a potential race between:
* shared memory clearing
* asynchronous pipeline loads

### Note on Hardware Behavior

This bug is hardware-sensitive:
* On H100: leads to incorrect numerical output due to async pipeline behavior
* On other GPUs (e.g., RTX series): may not produce visible errors