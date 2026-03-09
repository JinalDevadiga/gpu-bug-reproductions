import torch
import tilelang
from tilelang import language as T

tilelang.disable_cache()

@tilelang.jit
def get_kernel(m: int):
    @T.prim_func
    def test_kernel(
        a: T.Tensor[(m, ), "int32"],
    ):
        with T.Kernel(1, threads=1024) as (bx):
            shared = T.alloc_shared((1024, ), "int32")
            tx = T.get_thread_binding(0)
            shared[tx ^ 1] = 0
            T.atomic_add(shared[tx], 1)
            # BUG: missing __syncthreads() here before reading shared memory
            a[tx] = shared[tx ^ 32]
    return test_kernel


print("=== Issue #1257: Missing __syncthreads() after AtomicAdd ===")
print()

m = 1024
kernel = get_kernel(m)

print("--- Generated CUDA kernel source ---")
print(kernel.get_kernel_source())
print()

print("--- Running kernel ---")
a = torch.zeros((m, ), device="cuda", dtype=torch.int32)
kernel(a)

print(f"Sum of output: {a.sum().item()}")
print(f"Expected sum:  {m}")
print()

if a.sum().item() != m:
    print("BUG CONFIRMED: Output is wrong due to missing __syncthreads()!")
else:
    print("Output happened to be correct this run (race condition is non-deterministic)")
