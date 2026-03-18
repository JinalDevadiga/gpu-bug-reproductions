# Issue #17439 — `ThreadStorageSync` Pass Must Run After `MergeSharedMemoryAllocations`

## Source
- **GitHub Issue:** https://github.com/apache/tvm/issues/17439
- **Reported by:** @LeiWang1999, Oct 4 2024
- **Related PR:** https://github.com/apache/tvm/pull/17441
- **Reproduced on:** apache-tvm 0.11.1, Python 3.10, Ubuntu/WSL2
- **Status:** Open

## What is the Bug?

In TVM's lowering pipeline (`src/driver/driver_api.cc#L585-L613`), the
`ThreadSync` pass runs **before** `MergeSharedMemoryAllocations`:

```cpp
mixed_pass_list.push_back(tir::transform::ThreadSync("shared"));            // line ~590
...
mixed_pass_list.push_back(tir::transform::MergeSharedMemoryAllocations());  // line ~613
```

`ThreadSync` inserts `tvm_storage_sync` barriers based on the assumption
that `A_shared`, `B_shared`, and `C_shared` are **separate memory regions**.
`MergeSharedMemoryAllocations` then reuses the same memory space for all
three buffers. After the merge, `Store C_shared` writes to the same address
as `Load A_shared`, but there is no barrier between them.

```
// After merge — C_shared reuses A_shared/B_shared address space
Store A_shared
Store B_shared
tvm_storage_sync          <- correctly placed (ThreadSync saw separate buffers)
Load A_shared
Load B_shared
Store C_shared            <- NOW ALIASES A_shared — missing barrier here!
tvm_storage_sync
Load C_shared
```

## Requirements

- Python 3.9 – 3.11
- `apache-tvm==0.11.1`
- No GPU required (TIR inspection)

## Setup

```bash
conda activate tvm-bugs    # apache-tvm 0.11.1
```

## How to Run

```bash
python reproduce.py
```

## Root Cause

`src/driver/driver_api.cc#L585-L613` — `ThreadSync("shared")` is scheduled
before `MergeSharedMemoryAllocations` in the mixed pass pipeline. Sync
barrier placement is invalidated when shared memory buffers are merged into
a single allocation after the fact.

---

## Static Analysis Results

### GPUVerify

| Property | Value |
|----------|-------|
| Result | **RACE DETECTED — 1 error** |
| Classification | ✅ True Positive |
| Race Type | Read-Write race on `C_shared[1]` between threads in same block |

GPUVerify output:
```
error: possible read-write race on C_shared[1]:
  Write by thread (0,0) in block (1,1): C_shared[cse_var_1] = C_shared[cse_var_1] + ...
  Read  by thread (8,8) in block (1,1): C_shared[cse_var_1] = C_shared[cse_var_1] + ...
GPUVerify kernel analyser finished with 0 verified, 1 error
```

### Faial

| Property | Value |
|----------|-------|
| Result | **RACE DETECTED — 12 data-races** |
| Classification | ✅ True Positive |
| Race Type | Read-Write races on `C_shared` across multiple loop iteration combinations |
| Faial Version | faial-drf (built from source, gitlab.com/umb-svl/faial) |
| Command | `faial-drf --cu-to-json=/usr/local/bin/cu-to-json kernel_clean.cu` |

Faial detected 12 races — all true positives — spread across two lines:
- **Line 66** (8 races): `C_shared[cse_var_1] = C_shared[cse_var_1] + A_shared[0] * B_shared[0]`
  across different combinations of loop variables `ic`, `jc`, `k`
- **Line 38** (4 races): `C_shared[ic * 16 + jc] = 0.0f` in the initialisation loop
  across different combinations of `ic`, `jc`

In all 12 cases, Thread 0 and Thread 1 access the same `C_shared` index
simultaneously with no `__syncthreads()` protecting the access.

#### Why Faial Found More Races Than GPUVerify
GPUVerify reports **1 error** while Faial reports **12 races**. Both are
correct — they differ in reporting granularity. Faial enumerates races
across all distinct loop variable combinations that produce conflicts,
while GPUVerify reports a single representative instance of the race.
All 12 of Faial's races stem from the same root cause: the missing
barrier after shared memory aliasing caused by the wrong pass ordering.

---

## Tool Comparison Summary

| Tool | Result | Classification | Races Found | Notes |
|------|--------|----------------|-------------|-------|
| GPUVerify | RACE DETECTED | ✅ True Positive | 1 error | Representative instance |
| Faial | RACE DETECTED | ✅ True Positive | 12 races | All loop iteration combinations |