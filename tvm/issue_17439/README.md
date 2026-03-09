# Issue #17439 — `ThreadStorageSync` Pass Must Run After `MergeSharedMemoryAllocations`

## Source
- **GitHub Issue:** https://github.com/apache/tvm/issues/17439
- **Reported by:** @LeiWang1999, Oct 4 2024
- **Related PR:** https://github.com/apache/tvm/pull/17441
- **Reproduced on:** apache-tvm 0.11.1, Python 3.10, Ubuntu/WSL2
- **Status:** Open

## Approach

CUDA codegen is not available in the `apache-tvm` pip wheel build.
TIR inspection approach used — same as Triton #8311 and TVM #7246.
The three-stage TIR dump (before sync, after sync, after merge) directly
shows the faulty pass ordering.

## What is the Bug?

In TVM's lowering pipeline (`src/driver/driver_api.cc#L585-L613`), the
`ThreadSync` pass runs **before** `MergeSharedMemoryAllocations`:

```cpp
mixed_pass_list.push_back(tir::transform::ThreadSync("shared"));   // line ~590
...
mixed_pass_list.push_back(tir::transform::MergeSharedMemoryAllocations()); // line ~613
```

`ThreadSync` inserts `tvm_storage_sync` barriers based on the assumption
that `A_shared`, `B_shared`, and `C_shared` are **separate memory regions**.
`MergeSharedMemoryAllocations` then reuses the same memory space for all
three buffers. After the merge, `Store C_shared` writes to the same address
as `Load A_shared`, but there is no barrier between them:

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

Multiple threads can now read stale `A_shared` values while another thread
has already overwritten that memory with `C_shared` data → **silent wrong
result on CUDA**.

## Observed TIR Evidence

The three-stage TIR dump confirms the pass ordering:

- **Before ThreadSync**: three separate `allocate()` for `C.shared`,
  `A.shared`, `B.shared` — no sync barriers present.
- **After ThreadSync**: `tvm_storage_sync` barriers inserted correctly
  for the separate-buffer view.
- **After MergeSharedMemory**: allocations may be collapsed to one base
  pointer, but the sync barriers remain in the positions computed for
  the pre-merge layout — now incorrect.

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