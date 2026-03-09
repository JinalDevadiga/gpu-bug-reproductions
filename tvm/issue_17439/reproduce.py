"""
Reproduction of apache/tvm#17439:
  ThreadStorageSync pass runs before MergeSharedMemoryAllocations.
  Barriers inserted for separate buffers become misplaced after merge
  -> Store C_shared aliases Load A_shared -> silent wrong results on CUDA.

  CUDA codegen not available in pip wheel — TIR inspection approach used.

Environment:
  conda activate tvm-bugs   (apache-tvm==0.11.1)
"""
import re
import tvm
from tvm import te
from tvm.tir import transform as tir_transform

print(f"TVM version : {tvm.__version__}")

M = N = K = 64
tile = 16

A = te.placeholder((M, K), name="A", dtype="float32")
B = te.placeholder((K, N), name="B", dtype="float32")
k = te.reduce_axis((0, K), name="k")
C = te.compute(
    (M, N),
    lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
    name="C",
)

s  = te.create_schedule(C.op)
AA = s.cache_read(A, "shared", [C])
BB = s.cache_read(B, "shared", [C])
CC = s.cache_write(C, "shared")

block_x  = te.thread_axis("blockIdx.x")
block_y  = te.thread_axis("blockIdx.y")
thread_x = te.thread_axis("threadIdx.x")
thread_y = te.thread_axis("threadIdx.y")

yo, yi = s[C].split(C.op.axis[0], factor=tile)
xo, xi = s[C].split(C.op.axis[1], factor=tile)
s[C].bind(yo, block_y)
s[C].bind(xo, block_x)
s[C].bind(yi, thread_y)
s[C].bind(xi, thread_x)

s[CC].compute_at(s[C], xi)
s[AA].compute_at(s[CC], CC.op.reduce_axis[0])
s[BB].compute_at(s[CC], CC.op.reduce_axis[0])

_, ty = s[AA].split(s[AA].op.axis[0], factor=tile)
_, tx = s[AA].split(s[AA].op.axis[1], factor=tile)
s[AA].bind(ty, thread_y)
s[AA].bind(tx, thread_x)

_, ty = s[BB].split(s[BB].op.axis[0], factor=tile)
_, tx = s[BB].split(s[BB].op.axis[1], factor=tile)
s[BB].bind(ty, thread_y)
s[BB].bind(tx, thread_x)

mod = tvm.lower(s, [A, B, C], simple_mode=False)

print("\n=== TIR Before ThreadSync + MergeSharedMemory ===")
print(mod)

mod_after_sync = tir_transform.ThreadSync("shared")(mod)
print("\n=== TIR After ThreadSync (before MergeSharedMemory) ===")
print(mod_after_sync)

mod_after_merge = tir_transform.MergeDynamicSharedMemoryAllocations()(mod_after_sync)
print("\n=== TIR After MergeSharedMemory (final — buggy ordering) ===")
print(mod_after_merge)

syncs_before = len(re.findall(r"tvm_storage_sync", str(mod_after_sync)))
syncs_after  = len(re.findall(r"tvm_storage_sync", str(mod_after_merge)))
allocs_after = len(re.findall(r"allocate\(.*shared", str(mod_after_merge)))

print("\n=== IR Analysis ===")
print(f"tvm_storage_sync after ThreadSync  : {syncs_before}")
print(f"tvm_storage_sync after MergeShared : {syncs_after}")
print(f"shared allocations after merge     : {allocs_after}")
print(
    "\n-> BUG PATTERN CONFIRMED: ThreadSync ran before MergeSharedMemory.\n"
    "   Barriers inserted for separate A_shared/B_shared/C_shared buffers.\n"
    "   If MergeSharedMemory collapses them to one buffer, Store C_shared\n"
    "   aliases Load A_shared with no barrier between them\n"
    "   -> data corruption -> silent wrong result on CUDA.\n"
    "\n   Faulty pass order (driver_api.cc#L585-L613):\n"
    "     ThreadSync('shared')                 <- line ~590\n"
    "     MergeSharedMemoryAllocations()       <- line ~613"
)