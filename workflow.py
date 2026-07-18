"""
op-gpu workflow: kvlang code → kvload → op-gpu read /func/ → TileLang compile → test.

Usage:
  # Step 1: load kvlang code into kvspace (one-time)
  ../kvlang/kvlang load test/fusion_cases/

  # Step 2: run workflow on a specific function
  python3 workflow.py /func/fusion_cases/linear_relu
  python3 workflow.py --all                          # run all compilable patterns

Pipeline:
  1. read func from kvspace via kvspace-py + kvfunc
  2. classify op sequence → fusion pattern
  3. auto-generate TileLang kernel for the pattern
  4. compile with tilelang.jit
  5. generate random test data → run → compare with PyTorch reference
"""
import sys, time

import torch
import tilelang
import tilelang.language as T

from kvspace import connect, KVSpace
from kvfunc import read_func, list_funcs

# ═══════════════════════════════════════════════════════════════════════════
# 1. Fusion pattern database
# ═══════════════════════════════════════════════════════════════════════════

# (opcode_sequence) → (pattern_name, template_key, status)
FUSION_DB = {
    ("tensor.matmul", "tensor.add"):                  ("linear",       "linear",    "compilable"),
    ("tensor.matmul", "tensor.add", "tensor.relu"):   ("linear_relu",  "linear_activation", "compilable"),
    ("tensor.matmul", "tensor.add", "tensor.gelu"):   ("linear_gelu",  "linear_activation", "compilable"),
    ("tensor.matmul", "tensor.add", "tensor.silu"):   ("linear_silu",  "linear_activation", "compilable"),
    ("tensor.matmul", "tensor.add", "tensor.swish"):  ("linear_swish", "linear_activation", "compilable"),
}

ACTIVATION_PT = {
    "relu":  lambda t: torch.relu(t),
    "gelu":  lambda t: torch.nn.functional.gelu(t, approximate='tanh'),
    "silu":  lambda t: torch.nn.functional.silu(t),
    "swish": lambda t: torch.nn.functional.silu(t),
}


def classify(ops: list) -> tuple:
    """Match op sequence to fusion pattern. Returns (pattern_name, template_key, activation)."""
    tensor_ops = [o for o in ops if o["op"] != "return"]
    opcodes = tuple(o["op"] for o in tensor_ops)
    result = FUSION_DB.get(opcodes)
    if result is None:
        return None, None, None
    pattern, template, status = result
    if status != "compilable":
        return pattern, template, None
    last_op = tensor_ops[-1]["op"]
    activation = last_op.replace("tensor.", "") if last_op.startswith("tensor.") else None
    if activation not in ("relu", "gelu", "silu", "swish"):
        activation = None
    return pattern, template, activation


# ═══════════════════════════════════════════════════════════════════════════
# 2. TileLang kernel generators
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_SHAPE = {"M": 512, "N": 512, "K": 256}
DEFAULT_BLOCK = {"BM": 128, "BN": 128, "BK": 32}


def _activation_expr(name: str, x, dtype: str):
    """Return TileLang expression for activation applied to x."""
    half = T.cast(0.5, dtype)
    one  = T.cast(1, dtype)
    zero = T.cast(0, dtype)
    if name == "relu":
        return T.max(x, zero)
    if name == "gelu":
        sqrt_half = T.cast(0.7071067811865475, dtype)
        return half * x * (T.erf(x * sqrt_half) + one)
    if name in ("silu", "swish"):
        return x * T.sigmoid(x)
    return x


def make_linear_kernel(activation: str = None):
    """Generate TileLang kernel for matmul + bias [+ activation].

    Activation dispatch via Python closure — TileLang traces the resulting
    primitives at kernel definition time.
    """

    @tilelang.jit(out_idx=[-1])
    def kernel(M, N, K, BM, BN, BK, dtype="float16", accum_dtype="float32"):
        @T.prim_func
        def main(
            A:    T.Tensor((M, K), dtype),
            B:    T.Tensor((K, N), dtype),
            bias: T.Tensor((N,), dtype),
            C:    T.Tensor((M, N), dtype),
        ):
            with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
                As = T.alloc_shared((BM, BK), dtype)
                Bs = T.alloc_shared((BK, BN), dtype)
                Cl = T.alloc_fragment((BM, BN), accum_dtype)
                T.clear(Cl)
                for k in T.Pipelined(T.ceildiv(K, BK), num_stages=3):
                    T.copy(A[by * BM, k * BK], As)
                    T.copy(B[k * BK, bx * BN], Bs)
                    T.gemm(As, Bs, Cl)

                if activation:
                    for i, j in T.Parallel(BM, BN):
                        x = Cl[i, j] + bias[bx * BN + j]
                        Cl[i, j] = _activation_expr(activation, x, dtype)
                else:
                    for i, j in T.Parallel(BM, BN):
                        Cl[i, j] = Cl[i, j] + bias[bx * BN + j]

                T.copy(Cl, C[by * BM, bx * BN])
        return main
    return kernel


# ═══════════════════════════════════════════════════════════════════════════
# 3. Compile + test
# ═══════════════════════════════════════════════════════════════════════════

class CompileResult:
    def __init__(self, success: bool, diff: float, tl_ms: float, pt_ms: float,
                 error: str = None):
        self.success = success
        self.diff = diff
        self.tl_ms = tl_ms
        self.pt_ms = pt_ms
        self.error = error


def compile_and_test(kv: KVSpace, func_path: str, shape: dict = None,
                     block: dict = None) -> CompileResult:
    """Read func from kvspace, auto-generate TileLang kernel, compile, test."""
    s = shape or DEFAULT_SHAPE
    b = block or DEFAULT_BLOCK
    M, N, K = s["M"], s["N"], s["K"]
    BM, BN, BK = b["BM"], b["BN"], b["BK"]

    func = read_func(kv, func_path)
    tensor_ops = [o for o in func["ops"] if o["op"] != "return"]

    pattern, template, activation = classify(func["ops"])
    if pattern is None:
        opcodes = " → ".join(o["op"] for o in tensor_ops)
        return CompileResult(False, 0, 0, 0, f"no fusion pattern for: {opcodes}")
    if template is None:
        return CompileResult(False, 0, 0, 0, f"pattern '{pattern}' not compilable")

    kernel_fn = {"linear": make_linear_kernel, "linear_activation": make_linear_kernel}[template](activation)
    mod = kernel_fn(M, N, K, BM, BN, BK, "float16", "float32")
    torch.cuda.synchronize()

    a    = torch.randn((M, K), dtype=torch.float16, device='cuda')
    b    = torch.randn((K, N), dtype=torch.float16, device='cuda')
    bias = torch.randn((N,),  dtype=torch.float16, device='cuda')

    out_tl = mod(a, b, bias)
    torch.cuda.synchronize()

    ref = a @ b + bias
    if activation and activation in ACTIVATION_PT:
        ref = ACTIVATION_PT[activation](ref)
    diff = (out_tl - ref).abs().max().item()

    # benchmark
    for _ in range(20):
        mod(a, b, bias)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(200):
        mod(a, b, bias)
    torch.cuda.synchronize()
    tl_ms = (time.perf_counter() - t0) / 200 * 1000

    t0 = time.perf_counter()
    for _ in range(200):
        r = a @ b + bias
        if activation and activation in ACTIVATION_PT:
            r = ACTIVATION_PT[activation](r)
        _ = r
    torch.cuda.synchronize()
    pt_ms = (time.perf_counter() - t0) / 200 * 1000

    return CompileResult(diff < 0.5, diff, tl_ms, pt_ms)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    kv = connect()

    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        funcs = list_funcs(kv)
        if not funcs:
            print("no functions found in /func/fusion_cases/")
            sys.exit(1)

        print(f"{'func':<28} {'pattern':<18} {'diff':>9} {'status':>6} "
              f"{'TileLang':>9} {'PyTorch':>9} {'speedup':>7}")
        print("-" * 100)

        compiled = 0
        for name in funcs:
            path = f"/func/fusion_cases/{name}"
            result = compile_and_test(kv, path)

            if result.error and "no fusion pattern" in result.error:
                func = read_func(kv, path)
                ops = "→".join(o["op"].replace("tensor.", "") for o in func["ops"]
                               if o["op"] != "return")
                print(f"{name:<28} {'—':<18} {'—':>9} {'⚪':>6}  ({ops})")
                continue

            if result.error:
                print(f"{name:<28} {'—':<18} {'—':>9} {'🔴':>6}  {result.error}")
                continue

            pattern_name, _, _ = classify(read_func(kv, path)["ops"])
            sp = result.pt_ms / result.tl_ms if result.tl_ms > 0 else 0
            print(f"{name:<28} {pattern_name or '—':<18} {result.diff:>9.6f} "
                  f"{'✅' if result.success else '❌':>6} "
                  f"{result.tl_ms:>8.4f}ms {result.pt_ms:>8.4f}ms {sp:>6.2f}×")
            compiled += 1

        print(f"\ncompilable: {compiled}/{len(funcs)}")

    elif len(sys.argv) > 1:
        path = sys.argv[1]
        func = read_func(kv, path)
        print(f"read {path}")
        for o in func["ops"]:
            rw = ",".join(o.get("writes", []))
            rr = ",".join(o.get("reads", []))
            print(f"  {o['op']}({rr}) → {rw}")

        pattern, template, activation = classify(func["ops"])
        if pattern is None:
            opcodes = tuple(o["op"] for o in func["ops"] if o["op"] != "return")
            print(f"\nno fusion pattern for: {opcodes}")
            sys.exit(1)

        print(f"\nfuse: {pattern}  activation: {activation or 'none'}  "
              f"template: {template}")
        result = compile_and_test(kv, path)

        if result.error:
            print(f"❌ {result.error}")
            sys.exit(1)

        print(f"  max diff: {result.diff:.6f}  "
              f"{'✅' if result.success else '❌'}")
        print(f"  TileLang: {result.tl_ms:.4f} ms")
        print(f"  PyTorch:  {result.pt_ms:.4f} ms")
        print(f"  speedup:  {result.pt_ms/result.tl_ms:.2f}×")

    else:
        print("usage: python3 workflow.py <func_path> | --all")
        print("  python3 workflow.py /func/fusion_cases/linear_relu")
        print("  python3 workflow.py --all")
        sys.exit(1)
