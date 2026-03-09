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
def op(fl, xl, fr, xr):
    """First order linear recurrence combine function."""
    f = fr * fl
    x = fr * xl + xr
    return f, x


@triton.jit
def kernel(exp_ref, vals_ref, out_exp_ref, out_vals_ref,
           BS: tl.constexpr, reverse: tl.constexpr):
    idx  = tl.arange(0, BS)
    exp  = tl.load(exp_ref  + idx)
    vals = tl.load(vals_ref + idx)
    exp, vals = tl.associative_scan(
        (exp, vals), axis=0, combine_fn=op, reverse=reverse
    )
    tl.store(out_exp_ref  + idx, exp)
    tl.store(out_vals_ref + idx, vals)


BS     = 4
device = torch.device("cuda")

exp_init  = torch.tensor([1.0,  1.5, 0.8, 2.0], device=device)
vals_init = torch.tensor([1.0, -1.0, 0.5, 2.0], device=device)

# Ground truth from JAX lax.associative_scan(..., reverse=True)
exp_expected  = torch.tensor([2.4,      2.4,      1.6, 2.0], device=device)
vals_expected = torch.tensor([3.14999,  2.14999,  2.1, 2.0], device=device)

print("=== reverse=False (should be correct) ===")
out_exp_fwd  = torch.empty(BS, device=device)
out_vals_fwd = torch.empty(BS, device=device)
kernel[(1,)](exp_init, vals_init, out_exp_fwd, out_vals_fwd, BS, False)
print(f"  exp  : {out_exp_fwd.tolist()}")
print(f"  vals : {out_vals_fwd.tolist()}")

print()
print("=== reverse=True (buggy) ===")
out_exp  = torch.empty(BS, device=device)
out_vals = torch.empty(BS, device=device)
kernel[(1,)](exp_init, vals_init, out_exp, out_vals, BS, True)
print(f"  exp  result  : {out_exp.tolist()}")
print(f"  exp  expected: {exp_expected.tolist()}")
print(f"  vals result  : {out_vals.tolist()}")
print(f"  vals expected: {vals_expected.tolist()}")
print()

correct = (torch.allclose(out_exp, exp_expected, atol=1e-3) and
           torch.allclose(out_vals, vals_expected, atol=1e-3))

if correct:
    print("PASSED (bug may be fixed in this Triton version)")
else:
    print("BUG CONFIRMED: tl.associative_scan reverse=True produces wrong results")
    print(f"  exp  diff: {(out_exp  - exp_expected).tolist()}")
    print(f"  vals diff: {(out_vals - vals_expected).tolist()}")