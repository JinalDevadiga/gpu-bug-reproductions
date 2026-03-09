# Issue #1257 — Missing `__syncthreads()` after `AtomicAdd` in Generated CUDA Kernel

## Source
- **GitHub Issue:** https://github.com/tile-ai/tilelang/issues/1257
- **Repo:** TileLang
- **Status:** Fixed in v0.1.8

## Environment Used

- **GPU:** NVIDIA GeForce MX450 (Laptop GPU, 2GB VRAM)
- **CUDA Toolkit Version:** 12.3
- **OS:** Ubuntu (WSL on Windows)
- **Python Version:** 3.10
- **PyTorch Version:** 2.4.0

> Note: This bug is related to code generation and should reproduce on any CUDA-capable NVIDIA GPU.



## What is the Bug?

TileLang version 0.1.6 generated CUDA kernels that were missing a
`__syncthreads()` barrier after an `AtomicAdd` operation on shared memory.
This is a data race: some threads would read from shared memory before
other threads had finished writing to it, potentially producing wrong results.

In CUDA, when multiple threads write to shared memory, you must call
`__syncthreads()` to make sure ALL threads have finished writing before
ANY thread reads. Without this barrier, reads and writes from different
threads overlap — this is the data race.

## Buggy vs Fixed Generated Code

### Version 0.1.6 (BUGGY):
```c
shared[...] = 0;
__syncthreads();
AtomicAdd((&(shared[((int)threadIdx.x)])), 1);
// ← missing __syncthreads() here!
a[((int)threadIdx.x)] = shared[...];  // reads stale data!
```

### Version 0.1.8 (FIXED):
```c
shared[...] = 0;
__syncthreads();
AtomicAdd((&(shared[((int)threadIdx.x)])), 1);
__syncthreads();  // ← correctly added
a[((int)threadIdx.x)] = shared[...];
```

## Requirements

- Python 3.10+
- CUDA 12.3+
- Any NVIDIA GPU
- To reproduce the bug: `tilelang==0.1.6`
- To see the fix: `tilelang==0.1.8`

## How to Reproduce

### Step 1 — Clone the repository
```bash
git clone https://github.com/JinalDevadiga/tilelang-bug-reproductions.git
cd tilelang-bug-reproductions
```

### Step 2 — Navigate to this issue folder
```bash
cd issue_1257
```

### Step 3 — Install the buggy version
```bash
pip install tilelang==0.1.6
```

### Step 4 — Run the script
```bash
python reproduce.py
```

### What to look for
Look at the "Generated CUDA kernel source" section in the output.
In version 0.1.6, there is NO `__syncthreads()` after the `AtomicAdd` line.
This confirms the data race bug in the generated code.

### Verify the Fix
```bash
pip install tilelang==0.1.8
python reproduce.py
```

You should now see `__syncthreads()` after `AtomicAdd`.

## Expected Output (Buggy — v0.1.6)
The generated CUDA kernel will show:
```
AtomicAdd((&(shared[((int)threadIdx.x)])), 1);
a[((int)threadIdx.x)] = shared[...];   <-- no __syncthreads() between these!
```

## Expected Output (Fixed — v0.1.8)
The generated CUDA kernel will show:
```
AtomicAdd((&(shared[((int)threadIdx.x)])), 1);
__syncthreads();                         <-- correctly inserted
a[((int)threadIdx.x)] = shared[...];
```

## Note on Non-Determinism
The numerical output may still show the correct sum (1024) even with the
bug present, because race conditions are non-deterministic — they do not
always produce wrong values on every run. The bug is confirmed by inspecting
the generated CUDA source code directly, not just the output values.
