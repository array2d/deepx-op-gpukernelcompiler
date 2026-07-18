"""
op-gpu compile worker: read kvspace func → TileLang compile → .so → kvspace → verify.

Pipe: python3 read_func.py /func/tmp/inference | python3 compile_worker.py
"""
import sys, json, base64, hashlib, struct, os
import redis
import torch
import tilelang
import tilelang.language as T

# ── TLV codec ──────────────────────────────────────────────────────────

def tlv_decode(data: bytes) -> str:
    if not data: return ""
    kl = data[0]
    raw_len = struct.unpack_from('<I', data, 1 + kl)[0]
    return data[1+kl+4 : 1+kl+4+raw_len].decode()

def tlv_encode(kind: str, raw: bytes) -> bytes:
    kl = len(kind)
    buf = bytearray(1 + kl + 4 + len(raw))
    buf[0] = kl
    buf[1:1+kl] = kind.encode()
    struct.pack_into('<I', buf, 1+kl, len(raw))
    buf[1+kl+4:] = raw
    return bytes(buf)


# ── Compile: linear_relu ───────────────────────────────────────────────

def compile_linear_relu(spec: dict) -> str:
    """Build + compile fused matmul+bias+relu kernel. Returns .so path."""
    inputs  = spec["inputs"]
    M, K = inputs[0]["shape"]    # x: [M, K]
    N    = inputs[1]["shape"][1] # W: [K, N]
    dtype = inputs[0].get("dtype", "float16")
    BM, BN, BK = 128, 128, 32

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
                    T.copy(A[by*BM, k*BK], As)
                    T.copy(B[k*BK, bx*BN], Bs)
                    T.gemm(As, Bs, Cl)
                for i, j in T.Parallel(BM, BN):
                    Cl[i, j] = T.max(Cl[i, j] + bias[bx*BN + j], 0)
                T.copy(Cl, C[by*BM, bx*BN])
        return main

    print(f"  M={M} N={N} K={K} dtype={dtype} blocks={BM}x{BN}x{BK}", file=sys.stderr)
    mod = kernel(M, N, K, BM, BN, BK, dtype, "float32")
    torch.cuda.synchronize()

    # Verify correctness
    a = torch.randn((M, K), dtype=getattr(torch, dtype), device='cuda')
    b = torch.randn((K, N), dtype=getattr(torch, dtype), device='cuda')
    bias = torch.randn((N,), dtype=getattr(torch, dtype), device='cuda')
    out_tl = mod(a, b, bias); torch.cuda.synchronize()
    out_ref = torch.relu(a @ b + bias)
    diff = (out_tl - out_ref).abs().max().item()
    ok = diff < 0.5
    print(f"  verify: max_diff={diff:.6f} {'OK' if ok else 'FAIL'}", file=sys.stderr)

    # Find compiled .so in TileLang cache
    cache_dir = os.path.expanduser("~/.cache/tilelang")
    newest = None
    for root, dirs, files in os.walk(cache_dir):
        for f in files:
            if f.endswith('.so'):
                p = os.path.join(root, f)
                if newest is None or os.path.getmtime(p) > os.path.getmtime(newest):
                    newest = p
    if not newest:
        raise RuntimeError("compiled .so not found in cache")
    return newest


# ── kvspace storage ────────────────────────────────────────────────────

def store_to_kvspace(r: redis.Redis, func_name: str, spec: dict, so_path: str):
    base = f"/func/{func_name}_triton"
    with open(so_path, 'rb') as f:
        so_bytes = f.read()
    so_hash = hashlib.sha256(so_bytes).hexdigest()[:12]
    r.set(base, tlv_encode("string", f"compiled:{so_hash}".encode()))
    r.set(f"{base}/so", tlv_encode("bytes", so_bytes))
    r.set(f"{base}/meta", tlv_encode("string", json.dumps({
        "pattern": spec["pattern"],
        "inputs":  spec["inputs"],
        "hash":    so_hash,
        "so_size": len(so_bytes),
    }).encode()))
    print(f"  stored {len(so_bytes)}B → {base}", file=sys.stderr)


# ── C++ caller ─────────────────────────────────────────────────────────

def gen_cpp_caller(spec: dict, func_name: str):
    """Generate a C++ file that dlopen's the .so and calls the kernel."""
    inputs  = spec["inputs"]
    M, K = inputs[0]["shape"]
    N    = inputs[1]["shape"][1]
    dtype = inputs[0].get("dtype", "float16")
    ctype = {"float16": "half", "float32": "float"}.get(dtype, "float")

    return f'''// dlopen + call compiled kvlang kernel
#include <dlfcn.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>

int main() {{
    void* h = dlopen("kernel_lib.so", RTLD_LAZY);
    if (!h) {{ std::cerr << "dlopen: " << dlerror() << "\\n"; return 1; }}

    using KernelFn = void(*)({ctype}*, {ctype}*, {ctype}*, {ctype}*, cudaStream_t);
    auto fn = (KernelFn)dlsym(h, "call");
    if (!fn) {{ std::cerr << "dlsym: " << dlerror() << "\\n"; return 1; }}

    {ctype} *A, *B, *bias, *C;
    cudaMalloc(&A,    {M}*{K}*sizeof({ctype}));
    cudaMalloc(&B,    {K}*{N}*sizeof({ctype}));
    cudaMalloc(&bias, {N}*sizeof({ctype}));
    cudaMalloc(&C,    {M}*{N}*sizeof({ctype}));

    cudaStream_t s; cudaStreamCreate(&s);
    fn(A, B, bias, C, s);
    cudaStreamSynchronize(s);
    std::cout << "kernel executed\\n";
    return 0;
}}
'''


# ── Main ───────────────────────────────────────────────────────────────

COMPILERS = {"linear_relu": compile_linear_relu}

if __name__ == "__main__":
    spec = json.load(sys.stdin)
    r = redis.Redis(host='127.0.0.1', port=6379)

    pattern   = spec["pattern"]
    func_name = spec["func"]

    if pattern not in COMPILERS:
        print(f"ERROR: unknown pattern {pattern}", file=sys.stderr)
        sys.exit(1)

    print(f"compile {func_name} pattern={pattern}", file=sys.stderr)
    so_path = COMPILERS[pattern](spec)
    store_to_kvspace(r, func_name, spec, so_path)

    # Also write the generated C++ caller for reference
    cpp = gen_cpp_caller(spec, func_name)
    print(f"\n// ── C++ caller (dlopen) ──", file=sys.stderr)
    print(cpp, file=sys.stderr)
