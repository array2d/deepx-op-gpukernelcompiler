"""
op-gpu workflow: kvlang code → kvload → op-gpu read /func/ → TileLang compile → test.

Usage:
  # Step 1: load kvlang code into kvspace (one-time)
  ../kvlang/kvlang load test/fusion_cases/

  # Step 2: run workflow on a specific function
  python3 workflow.py /func/fusion_cases/linear_relu
  python3 workflow.py /func/fusion_cases/ffn_swiglu
  python3 workflow.py --all                          # run all compilable patterns

Pipeline:
  1. read func from kvspace /func/<path>  (TLV decode)
  2. classify op sequence → fusion pattern
  3. auto-generate TileLang kernel for the pattern
  4. compile with tilelang.jit
  5. generate random test data → run → compare with PyTorch reference
"""
import sys, struct, time, os
import redis
import torch
import tilelang
import tilelang.language as T

# ═══════════════════════════════════════════════════════════════════════════
# 1. TLV codec
# ═══════════════════════════════════════════════════════════════════════════

def tlv_decode(data: bytes) -> str:
    if not data: return ""
    kl = data[0]
    raw_len = struct.unpack_from('<I', data, 1 + kl)[0]
    return data[1+kl+4 : 1+kl+4+raw_len].decode()


# ═══════════════════════════════════════════════════════════════════════════
# 2. Read kvlang func from kvspace
# ═══════════════════════════════════════════════════════════════════════════

def read_func(r: redis.Redis, path: str) -> dict:
    """Read a kvlang function from kvspace. Returns {signature, ops}."""
    sig = tlv_decode(r.get(path))
    ops = []
    i = 0
    while True:
        op_raw = r.get(f"{path}/[{i},0]")
        if op_raw is None: break
        opcode = tlv_decode(op_raw)
        reads = []; j = 1
        while True:
            v = r.get(f"{path}/[{i},-{j}]")
            if v is None: break
            reads.append(tlv_decode(v)); j += 1
        writes = []; j = 1
        while True:
            v = r.get(f"{path}/[{i},{j}]")
            if v is None: break
            writes.append(tlv_decode(v)); j += 1
        ops.append({"op": opcode, "reads": reads, "writes": writes})
        i += 1
    return {"signature": sig, "ops": ops}


# ═══════════════════════════════════════════════════════════════════════════
# 3. Fusion pattern database
# ═══════════════════════════════════════════════════════════════════════════

# (opcode_sequence) → (pattern_name, template_key, status)
#   template_key: which TileLang kernel to generate
#   status: "compilable" | "needs_template" | "not_matched"
FUSION_DB = {
    ("tensor.matmul", "tensor.add"):                  ("linear",       "linear",    "compilable"),
    ("tensor.matmul", "tensor.add", "tensor.relu"):   ("linear_relu",  "linear_activation", "compilable"),
    ("tensor.matmul", "tensor.add", "tensor.gelu"):   ("linear_gelu",  "linear_activation", "compilable"),
    ("tensor.matmul", "tensor.add", "tensor.silu"):   ("linear_silu",  "linear_activation", "compilable"),
    ("tensor.matmul", "tensor.add", "tensor.swish"):  ("linear_swish", "linear_activation", "compilable"),
}

# activation → PyTorch reference
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
    # extract activation from last op
    last_op = tensor_ops[-1]["op"]
    activation = last_op.replace("tensor.", "") if last_op.startswith("tensor.") else None
    if activation not in ("relu", "gelu", "silu", "swish"):
        activation = None  # e.g. plain linear: last op is "add", not an activation
    return pattern, template, activation


# ═══════════════════════════════════════════════════════════════════════════
# 4. TileLang kernel generators
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_SHAPE = {"M": 512, "N": 512, "K": 256}
DEFAULT_BLOCK = {"BM": 128, "BN": 128, "BK": 32}


def _activation_expr(name: str, x: str, dtype: str):
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
    return x  # identity (no activation)


def make_linear_kernel(activation: str = None):
    """Generate a TileLang kernel for matmul + bias [+ activation].

    Returns a @tilelang.jit callable that accepts (M,N,K,BM,BN,BK,dtype,accum_dtype).
    TileLang traces the prim_func body — so activation dispatch happens at trace time
    via Python closure, embedding the correct TileLang primitives directly.
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
# 5. Compile + test
# ═══════════════════════════════════════════════════════════════════════════

class CompileResult:
    def __init__(self, success: bool, diff: float, tl_ms: float, pt_ms: float, error: str = None):
        self.success = success
        self.diff = diff
        self.tl_ms = tl_ms
        self.pt_ms = pt_ms
        self.error = error


def compile_and_test(r: redis.Redis, func_path: str, shape: dict = None, block: dict = None) -> CompileResult:
    """Read func from kvspace, auto-generate TileLang kernel, compile, test.

    Returns CompileResult with correctness diff and benchmark timing.
    """
    s = shape or DEFAULT_SHAPE
    b = block or DEFAULT_BLOCK
    M, N, K = s["M"], s["N"], s["K"]
    BM, BN, BK = b["BM"], b["BN"], b["BK"]

    # 1. Read func
    func = read_func(r, func_path)
    func_name = func_path.rsplit("/", 1)[-1]
    tensor_ops = [o for o in func["ops"] if o["op"] != "return"]

    # 2. Classify
    pattern, template, activation = classify(func["ops"])
    if pattern is None:
        opcodes = " → ".join(o["op"] for o in tensor_ops)
        return CompileResult(False, 0, 0, 0, f"no fusion pattern for: {opcodes}")
    if template is None:
        return CompileResult(False, 0, 0, 0, f"pattern '{pattern}' not compilable")

    opcodes = "→".join(o["op"].replace("tensor.", "") for o in tensor_ops)
    act_label = f" ({activation})" if activation else ""

    # 3. Generate TileLang kernel
    kernel_factory = {"linear": make_linear_kernel, "linear_activation": make_linear_kernel}[template]
    kernel_fn = kernel_factory(activation)

    # 4. Compile
    dtype_str = "float16"
    accum_str = "float32"
    mod = kernel_fn(M, N, K, BM, BN, BK, dtype_str, accum_str)
    torch.cuda.synchronize()

    # 5. Generate test data
    a    = torch.randn((M, K), dtype=torch.float16, device='cuda')
    b    = torch.randn((K, N), dtype=torch.float16, device='cuda')
    bias = torch.randn((N,),  dtype=torch.float16, device='cuda')

    # 6. Run
    out_tl = mod(a, b, bias)
    torch.cuda.synchronize()

    # 7. PyTorch reference
    ref = a @ b + bias
    if activation and activation in ACTIVATION_PT:
        ref = ACTIVATION_PT[activation](ref)
    diff = (out_tl - ref).abs().max().item()

    # 8. Benchmark
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
        if activation and activation in ACTIVATION_PT:
            _ = ACTIVATION_PT[activation](a @ b + bias)
        else:
            _ = a @ b + bias
    torch.cuda.synchronize()
    pt_ms = (time.perf_counter() - t0) / 200 * 1000

    return CompileResult(diff < 0.5, diff, tl_ms, pt_ms)


# ═══════════════════════════════════════════════════════════════════════════
# 6. Main
# ═══════════════════════════════════════════════════════════════════════════

def scan_fusion_cases(r: redis.Redis) -> list:
    """Scan kvspace for all functions under /func/fusion_cases/."""
    import subprocess
    kv_bin = os.path.join(os.path.dirname(__file__), "..", "kvlang", "kvlang")
    result = subprocess.run([kv_bin, "kvspace", "list", "/func/fusion_cases"], capture_output=True, text=True)
    names = []
    for line in result.stdout.strip().split('\n'):
        name = line.strip()
        if name and not name.startswith('['):
            names.append(name)
    # fallback: scan from redis keys
    if not names:
        all_keys = r.keys("/func/fusion_cases/*")
        seen = set()
        for k in all_keys:
            k = k.decode() if isinstance(k, bytes) else k
            # extract func name: /func/fusion_cases/<name>/... or /func/fusion_cases/<name>
            parts = k.removeprefix("/func/fusion_cases/").split("/")
            if parts[0] and not parts[0].startswith('['):
                seen.add(parts[0])
        names = sorted(seen)
    return names


if __name__ == "__main__":
    r = redis.Redis(host='127.0.0.1', port=6379)

    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        funcs = scan_fusion_cases(r)
        if not funcs:
            print("no functions found in /func/fusion_cases/")
            sys.exit(1)

        print(f"{'func':<28} {'pattern':<18} {'diff':>9} {'status':>6} {'TileLang':>9} {'PyTorch':>9} {'speedup':>7}")
        print("-" * 100)

        compiled = 0
        for name in funcs:
            path = f"/func/fusion_cases/{name}"
            result = compile_and_test(r, path)

            if result.error and "no fusion pattern" in result.error:
                func = read_func(r, path)
                ops = "→".join(o["op"].replace("tensor.", "") for o in func["ops"] if o["op"] != "return")
                print(f"{name:<28} {'—':<18} {'—':>9} {'⚪':>6}  ({ops})")
                continue

            if result.error:
                print(f"{name:<28} {'—':<18} {'—':>9} {'🔴':>6}  {result.error}")
                continue

            pattern, _, activation = classify(read_func(r, path)["ops"])
            label = f"{pattern}"
            icon = "✅" if result.success else "❌"
            sp = result.pt_ms / result.tl_ms if result.tl_ms > 0 else 0
            print(f"{name:<28} {label:<18} {result.diff:>9.6f} {icon:>6} {result.tl_ms:>8.4f}ms {result.pt_ms:>8.4f}ms {sp:>6.2f}×")
            compiled += 1

        # Summary: pending patterns
        print(f"\ncompilable: {compiled}/{len(funcs)}")

    elif len(sys.argv) > 1:
        path = sys.argv[1]
        func = read_func(r, path)
        ops = func["ops"]
        print(f"read {path}")
        for o in ops:
            rw = ",".join(o.get("writes", []))
            rr = ",".join(o.get("reads", []))
            print(f"  {o['op']}({rr}) → {rw}")

        pattern, template, activation = classify(ops)
        if pattern is None:
            opcodes = tuple(o["op"] for o in ops if o["op"] != "return")
            print(f"\nno fusion pattern for: {opcodes}")
            sys.exit(1)

        print(f"\nfuse: {pattern}  activation: {activation or 'none'}  template: {template}")
        result = compile_and_test(r, path)

        if result.error:
            print(f"❌ {result.error}")
            sys.exit(1)

        print(f"  max diff: {result.diff:.6f}  {'✅' if result.success else '❌'}")
        print(f"  TileLang: {result.tl_ms:.4f} ms")
        print(f"  PyTorch:  {result.pt_ms:.4f} ms")
        print(f"  speedup:  {result.pt_ms/result.tl_ms:.2f}×")

    else:
        print("usage: python3 workflow.py <func_path> | --all")
        print("example: python3 workflow.py /func/fusion_cases/linear_relu")
        print("         python3 workflow.py --all")
        sys.exit(1)
