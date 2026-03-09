# Issue #17072 — Race Condition in TIR `ComputationCache` (CSE Pass)

## Source
- **GitHub Issue:** https://github.com/apache/tvm/issues/17072
- **Reported by:** @guillon, Jun 7 2024
- **Reproduced on:** apache-tvm 0.11.1, Python 3.10, Ubuntu/WSL2
- **Status:** Closed (Feb 8, 2025)

## Approach

This bug requires a high core-count machine to trigger reliably at runtime
(reported on a 52-core Intel Xeon Gold 6230R). It is not reproducible at
runtime on low core-count hardware (e.g. MX450 laptop). Source inspection
approach used — same as Triton #8311.

## What is the Bug?

The CSE (Common Subexpression Elimination) TIR pass uses a `static` cache
declared in `src/tir/transforms/common_subexpr_elim_tools.h#L115`:

```cpp
// static cache shared across ALL threads — no synchronisation
static ComputationCache cache_;
```

When multiple Python threads each call `tvm.build()` concurrently, they
all enter the CSE pass and race on read-modify-write operations to this
single shared `cache_`:

- Thread A calls `cache_.find(expr)` (reading)
- Thread B calls `cache_.insert(expr, ...)` simultaneously (writing)
- HashTable is corrupted → **segmentation fault**

TVM provides no warning that concurrent builds share this state.

## Symptom

```
Segmentation fault
```

Observed when launching 100 parallel threads each compiling a matmul
operator, for a total of 100,000 compilations on a 52-core machine.
The bug is flaky — probability scales with core count and task volume.

## Requirements

- Python 3.9 – 3.11
- `apache-tvm==0.11.1`
- 50+ core CPU for reliable runtime reproduction
- No GPU required

## Setup

```bash
conda activate tvm-bugs    # apache-tvm 0.11.1
```

## How to Run

```bash
# Default (32 threads, 500 tasks) — confirms faulty code location
python reproduce.py

# High load — more likely to trigger on many-core machines
python reproduce.py 100 100000
```

## Root Cause

`src/tir/transforms/common_subexpr_elim_tools.h#L115` — the
`ComputationCache` is declared `static`, making it a single instance
shared across all threads in the process. The CSE pass was not designed
with concurrent compilation in mind.