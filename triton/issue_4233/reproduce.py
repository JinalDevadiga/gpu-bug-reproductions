import torch
import triton
import triton.language as tl

print(f"Triton version : {triton.__version__}")
print(f"PyTorch version: {torch.__version__}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    sm    = props.major * 10 + props.minor
    print(f"GPU            : {props.name}  (sm{sm})")
print()


# BUGGY: uses tl.load + tl.store for scatter_add.
# When multiple threads map to the same output index, each thread does a
# non-atomic read-modify-write. The last writer wins and all other
# accumulations are lost — a classic write-write (WAW) data race.
@triton.jit
def scatter_add_racy(index_ptr, src_ptr, out_ptr,
                     N: tl.constexpr, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex  = xoffset + tl.arange(0, XBLOCK)
    xmask   = xindex < N

    idx = tl.load(index_ptr + xindex, xmask)
    val = tl.load(src_ptr   + xindex, xmask)

    # non-atomic: read current value, add, write back — race when idx has duplicates
    cur = tl.load(out_ptr + idx, xmask)
    tl.store(out_ptr + idx, cur + val, xmask)


# CORRECT: uses tl.atomic_add so every update is applied atomically.
@triton.jit
def scatter_add_atomic(index_ptr, src_ptr, out_ptr,
                       N: tl.constexpr, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex  = xoffset + tl.arange(0, XBLOCK)
    xmask   = xindex < N

    idx = tl.load(index_ptr + xindex, xmask)
    val = tl.load(src_ptr   + xindex, xmask).to(tl.float32)

    tl.atomic_add(out_ptr + idx, val, xmask, sem='relaxed')


# --- inputs from the original issue ---
index_tensor  = torch.tensor([0, 1, 2, 2, 1, 0], device='cuda', dtype=torch.int32)
to_add_tensor = torch.tensor([1, 1, 1, 1, 1, 1], device='cuda', dtype=torch.float32)
N             = len(index_tensor)
XBLOCK        = 8   # power of 2 >= N
expected      = torch.tensor([2.0, 2.0, 2.0], device='cuda')

grid = (1,)

print("=== scatter_add_racy (buggy — non-atomic read-modify-write) ===")
out_racy = torch.zeros(3, device='cuda', dtype=torch.float32)
scatter_add_racy[grid](index_tensor, to_add_tensor, out_racy, N, XBLOCK)
torch.cuda.synchronize()
print(f"  result  : {out_racy.tolist()}")
print(f"  expected: {expected.tolist()}")
if torch.allclose(out_racy, expected):
    print("  -> PASSED (race did not manifest this run — try again or increase N)")
else:
    print("  -> BUG CONFIRMED: duplicate indices cause lost updates due to WAW race")

print()
print("=== scatter_add_atomic (correct — tl.atomic_add) ===")
out_atomic = torch.zeros(3, device='cuda', dtype=torch.float32)
scatter_add_atomic[grid](index_tensor, to_add_tensor, out_atomic, N, XBLOCK)
torch.cuda.synchronize()
print(f"  result  : {out_atomic.tolist()}")
print(f"  expected: {expected.tolist()}")
print(f"  -> {'PASSED' if torch.allclose(out_atomic, expected) else 'FAILED'}")