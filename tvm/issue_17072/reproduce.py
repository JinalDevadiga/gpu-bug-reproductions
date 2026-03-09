"""
Reproduction of apache/tvm#17072:
  Race condition in TIR ComputationCache (CSE pass).
  Requires 50+ core machine to trigger reliably.
  Source inspection approach used — same as Triton #8311.

Environment:
  conda activate tvm-bugs   (apache-tvm==0.11.1)
"""
import sys
import threading
import tvm
from tvm import te

print(f"TVM version : {tvm.__version__}")

print("""
=== Faulty Code (src/tir/transforms/common_subexpr_elim_tools.h#L115) ===

  static ComputationCache cache_;

  Thread A: cache_.find(expr)          // reading
  Thread B: cache_.insert(expr, ...)   // writing simultaneously
  -> HashTable corruption -> Segfault
""")

NUM_THREADS = int(sys.argv[1]) if len(sys.argv) > 1 else 32
TOTAL_TASKS = int(sys.argv[2]) if len(sys.argv) > 2 else 500

print(f"NUM_THREADS={NUM_THREADS}  TOTAL_TASKS={TOTAL_TASKS}\n")

sys.setswitchinterval(0.00001)

errors  = []
lock    = threading.Lock()
counter = {"n": 0}

def build_matmul(idx):
    try:
        M = K = N = 64
        A = te.placeholder((M, K), name="A", dtype="float32")
        B = te.placeholder((K, N), name="B", dtype="float32")
        k = te.reduce_axis((0, K), name="k")
        C = te.compute(
            (M, N),
            lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
            name="C",
        )
        s = te.create_schedule(C.op)
        tvm.build(s, [A, B, C], target="llvm")
        with lock:
            counter["n"] += 1
    except Exception as exc:
        with lock:
            errors.append((idx, str(exc)))

task_idx, active = 0, []
while task_idx < TOTAL_TASKS or active:
    while task_idx < TOTAL_TASKS and len(active) < NUM_THREADS:
        t = threading.Thread(target=build_matmul, args=(task_idx,))
        t.start()
        active.append(t)
        task_idx += 1
    active = [t for t in active if t.is_alive()]

print(f"Builds completed: {counter['n']} / {TOTAL_TASKS}")

if errors:
    print(f"\n-> BUG CONFIRMED: {len(errors)} thread(s) crashed.")
    for idx, msg in errors[:5]:
        print(f"   Task {idx}: {msg[:120]}")
else:
    print(
        "\n-> Runtime race not observed on this hardware.\n"
        "   Reported segfault on 52-core Xeon with 100 threads / 100k tasks."
    )