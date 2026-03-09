# Issue #10210 — Silent Wrong Results When `parallel()` Applied to Reduction Axis

## Source
- **GitHub Issue:** https://github.com/apache/tvm/issues/10210
- **Reproduced on:** apache-tvm 0.11.1, Python 3.10, Ubuntu/WSL2 (CPU only)
- **Status:** Open

## What is the Bug?

TVM's TE scheduling API allows `s[op].parallel(axis)` to be called on any
loop axis — including **reduction axes**. When a reduction axis is
parallelised, multiple OS threads each compute a partial sum and write back
to the **same output element** `C[i][j]` without any atomic operation or
barrier:

```
parallel for k in range(K):       // <- race: multiple threads
    C[i][j] += A[i][k] * B[k][j] // write same accumulator
```

This is a **WAW (write-write) data race** on the accumulator. Partial
products from competing threads are silently overwritten and lost, producing
a numerically wrong result. TVM compiles and runs the kernel to completion
with **no warning or error** of any kind.

## Results

```
Sequential GEMM  — max error vs numpy: 9.54e-06  (PASS)
Parallel-reduction GEMM (10 runs) — max error: 29.68,  min error: 27.05

-> BUG CONFIRMED: parallel reduction race detected.
   max_error=29.68 >> 1.0 tolerance.
   Multiple threads wrote to the same C[i][j] accumulator without
   synchronisation -> lost updates -> wrong result.
   TVM emitted no warning during compilation.
```

## Requirements

- Python 3.9 – 3.11
- `apache-tvm==0.11.1`
- Multi-core CPU
- No GPU required

## Setup

```bash
conda activate tvm-bugs    # apache-tvm 0.11.1
```

## How to Run

```bash
python reproduce.py
```

## Root Cause

`src/te/schedule/schedule_lang.cc` — `Stage::parallel()` does not check
whether the target axis is a spatial or reduction axis before marking it
`ForKind::kParallel`. The threadpool in `src/runtime/thread_pool.cc` then
launches independent threads for each k-iteration, all performing
non-atomic read-modify-write on the same output element.