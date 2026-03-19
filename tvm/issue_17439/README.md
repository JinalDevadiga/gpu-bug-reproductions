# Issue #17439 — `ThreadStorageSync` Pass Must Run After `MergeSharedMemoryAllocations`

## Source
- **GitHub Issue:** https://github.com/apache/tvm/issues/17439
- **Reported by:** @LeiWang1999, Oct 4 2024
- **Related PR:** https://github.com/apache/tvm/pull/17441
- **Reproduced on:** apache-tvm 0.11.1, Python 3.10, Ubuntu/WSL2
- **Status:** Open

## What is the Bug?

In TVM's lowering pipeline, the `ThreadSync` pass runs **before**
`MergeSharedMemoryAllocations`. `ThreadSync` inserts `tvm_storage_sync`
barriers based on separate `A_shared`, `B_shared`, and `C_shared` buffers.
After `MergeSharedMemoryAllocations` reuses the same address space for all
three, `Store C_shared` writes to the same memory as `Load A_shared` with
no barrier between them — a read-write race causing silent wrong results.

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
| Race Type | Read-write race on `C_shared` (aliased with `A_shared` after merge) |
| Errors | 1 |

GPUVerify output:
```
error: possible read-write race on C_shared[1]:
  Write by thread (0,0): C_shared[cse_var_1] = C_shared[cse_var_1] + ...
  Read  by thread (8,8): C_shared[cse_var_1] = C_shared[cse_var_1] + ...
GPUVerify kernel analyser finished with 0 verified, 1 error
```

### Faial

| Property | Value |
|----------|-------|
| Result | **RACE DETECTED** |
| Classification | ✅ True Positive |
| Race Type | Read-write race on aliased shared memory |
| Races Found | 12 |
| Command | `faial-drf --cu-to-json=/usr/local/bin/cu-to-json kernel_clean.cu` |

Faial found 12 data-races — all stemming from the same aliased
`C_shared`/`A_shared` region, reported across different loop iteration
combinations of `ic`, `jc`, and `k`.

### Weft

| Property | Value |
|----------|-------|
| Result | **OOM — process killed during emulation** |
| Classification | ⚠️ Inconclusive (scalability limit) |
| Weft Version | Built from source (github.com/lightsighter/Weft) |
| Command | `weft -f kernel_clean.ptx -t 1 -v` |
| PTX Compiled With | `nvcc -ptx kernel_weft.cu` (added `__launch_bounds__(256)`) |
| System | 7.6 GB RAM, 2 GB swap |

Weft output:
```
WEFT WARNING: No line information found! Line numbers from PTX will be used!
                Try re-running nvcc with the '-lineinfo' flag!
WEFT INFO: No deadlocks detected in kernel tvm_matmul_buggy!
WEFT INFO: Barriers properly recycled in kernel tvm_matmul_buggy!
Killed
```

Both `-t 4` (4 threads) and `-t 1` (single thread) were tried — both
resulted in the process being killed by the OS during the emulation phase.

#### Why Weft Was Killed

The kernel has:
- 256 threads (16×16 block)
- 49 barrier instructions
- Triple-nested loops (`ic`, `jc` up to 16, `k` up to 64)

Weft emulates every thread's instruction trace in full before building the
happens-before graph. For a kernel with deeply nested loops unrolled across
256 threads, the dynamic instruction count grows rapidly — the
`Compute Happens-Before/After Relationships` phase in particular scales
quadratically with the number of weft statements, requiring bitsets of size
`O(statements²)` stored in memory.

The previous issue #96 (1024 threads, simpler loop) used ~4.9 GB of RAM and
completed in 29 seconds. The TVM matmul kernel, with 256 threads but a much
deeper loop structure (16×16×64 iterations with 49 barriers), exhausts
available memory before the happens-before computation can complete.

This is a **known scalability limitation** of Weft. The Weft paper
(PLDI 2015) notes that memory usage is the primary constraint for large
kernels, and recommends the tool for kernels with bounded, statically
analyzable loops. The TVM matmul kernel's deeply nested loops produce a
dynamic instruction count that exceeds what Weft can handle on a 7.6 GB
system.

#### Significance

This result is itself meaningful: it demonstrates that Weft's exhaustive
emulation-based approach has practical memory limits that prevent analysis
of larger production GPU kernels. The TVM matmul kernel is not especially
large by modern standards — GPUVerify and Faial both analyzed it successfully.
This highlights a scalability gap between Weft's 2015-era design and the
complexity of modern framework-generated kernels.

---

## Tool Comparison Summary

| Tool | Result | Classification | Notes |
|------|--------|----------------|-------|
| GPUVerify | RACE DETECTED | ✅ True Positive | 1 error — aliased shared memory race |
| Faial | RACE DETECTED | ✅ True Positive | 12 races — all same root cause |
| Weft | OOM / Killed | ⚠️ Inconclusive | Process killed during emulation — scalability limit exceeded |