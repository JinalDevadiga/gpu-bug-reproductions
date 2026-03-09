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


@triton.jit
def my_kernel(index_ptr, out_ptr):
    write_index = tl.atomic_add(index_ptr + tl.arange(0, 1), val=1, sem="relaxed")
    tl.store(
        out_ptr + write_index[:, None] * 8 + tl.arange(0, 8)[None, :],
        tl.full([1, 8], 2, dtype=tl.int64),
    )


index = torch.zeros([1], device="cuda", dtype=torch.int32)
index[0] = 1
out = torch.zeros([2, 8], device="cuda", dtype=torch.int64)

compiled = my_kernel[(1,)](index, out)

print("Result  :", out[:, :8].tolist())
print("Expected: [[0, 0, 0, 0, 0, 0, 0, 0], [2, 2, 2, 2, 2, 2, 2, 2]]")
print()

expected = torch.zeros([2, 8], dtype=torch.int64)
expected[1, :] = 2

if torch.equal(out.cpu(), expected):
    print("PASSED (bug may be fixed in this Triton version)")
else:
    print("BUG CONFIRMED: tl.atomic_add return value is wrong across threads")
    print("  Only thread 0 gets the correct atomic_add return value;")
    print("  all other threads in the block receive 0 instead.")

print()
print("--- TTGIR (shows how atomic_add return value is broadcast to threads) ---")
print()
print(compiled.asm['ttgir'])