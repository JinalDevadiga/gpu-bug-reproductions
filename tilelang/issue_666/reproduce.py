import tilelang
import tilelang.language as T


def matmul_buggy(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):
    """Buggy version - T.clear on shared memory BEFORE pipelined loop"""
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local  = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            T.clear(B_shared)  # BUG - clearing shared memory before pipeline
                                # causes sync issue with async pipeline stages on H100

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                for k, j in T.Parallel(block_K, block_N):
                    B_shared[k, j] = B[ko * block_K + k, bx * block_N + j]
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])
    return main


M, N, K = 1024, 1024, 1024
block_M, block_N, block_K = 128, 128, 32

print("=== Issue #666: Incorrect Results When Clearing Shared Memory")
print("NOTE: The incorrect OUTPUT only manifests on H100 GPUs.")
print()
print("--- Compiling BUGGY version (T.clear on shared memory before pipeline) ---")
func_buggy = matmul_buggy(M, N, K, block_M, block_N, block_K)
kernel_buggy = tilelang.compile(func_buggy, out_idx=[2], target="cuda")
print("Generated CUDA source (BUGGY):")
print(kernel_buggy.get_kernel_source())
