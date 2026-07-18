"""
GPU kernel dispatcher — selects between TileLang / Triton backends.
"""
import torch
from kvspace import KVSpace
from .kvfunc import read_func
from .fusion import classify, ACTIVATION_PT
from .kernel_tilelang import make_linear_kernel as _make_tilelang
from .kernel_triton  import make_linear_kernel as _make_triton

DEFAULT_SHAPE = {"M": 512, "N": 512, "K": 256}
DEFAULT_BLOCK = {"BM": 128, "BN": 128, "BK": 32}

ENGINES = {
    "tilelang": _make_tilelang,
    "triton":   _make_triton,
}


class CompileResult:
    def __init__(self, success: bool, diff: float, tl_ms: float, pt_ms: float,
                 error: str = None):
        self.success = success
        self.diff = diff
        self.tl_ms = tl_ms
        self.pt_ms = pt_ms
        self.error = error


def compile_and_test(kv: KVSpace, func_path: str, fusion: str = "tilelang",
                     shape: dict = None, block: dict = None) -> CompileResult:
    """Read func from kvspace, generate kernel, compile, test."""
    s = shape or DEFAULT_SHAPE
    b = block or DEFAULT_BLOCK
    M, N, K = s["M"], s["N"], s["K"]
    BM, BN, BK = b["BM"], b["BN"], b["BK"]

    func = read_func(kv, func_path)
    tensor_ops = [o for o in func["ops"] if o["op"] != "return"]

    pattern, template, activation, status = classify(func["ops"])
    if status == "not_matched":
        opcodes = " → ".join(o["op"] for o in tensor_ops)
        return CompileResult(False, 0, 0, 0, f"no fusion pattern for: {opcodes}")
    if status != "compilable":
        return CompileResult(False, 0, 0, 0, f"{status}: {pattern}")

    kernel_factory = ENGINES[fusion]
    kernel_fn = kernel_factory(activation)

    if fusion == "tilelang":
        mod = kernel_fn(M, N, K, BM, BN, BK, "float16", "float32")
        torch.cuda.synchronize()
    else:
        mod = kernel_fn

    a    = torch.randn((M, K), dtype=torch.float16, device='cuda')
    b    = torch.randn((K, N), dtype=torch.float16, device='cuda')
    bias = torch.randn((N,),  dtype=torch.float16, device='cuda')

    out_tl = mod(a, b, bias)
    torch.cuda.synchronize()

    ref = a @ b + bias
    if activation and activation in ACTIVATION_PT:
        ref = ACTIVATION_PT[activation](ref)
    diff = (out_tl - ref).abs().max().item()

    for _ in range(20):
        mod(a, b, bias)
    torch.cuda.synchronize()

    N_ITER = 200
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(N_ITER):
        mod(a, b, bias)
    end.record()
    torch.cuda.synchronize()
    tl_ms = start.elapsed_time(end) / N_ITER

    start.record()
    for _ in range(N_ITER):
        r = a @ b + bias
        if activation and activation in ACTIVATION_PT:
            r = ACTIVATION_PT[activation](r)
    end.record()
    torch.cuda.synchronize()
    pt_ms = start.elapsed_time(end) / N_ITER

    return CompileResult(diff < 0.5, diff, tl_ms, pt_ms)
