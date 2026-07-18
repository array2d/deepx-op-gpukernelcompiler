"""
op-gpu: read kvspace /func/ → TileLang compile → make data → test run.

Usage: python3 run_fusion.py /func/tmp/inference
"""
import sys, struct, redis, torch, tilelang, tilelang.language as T

# ── TLV decode ──────────────────────────────────────────────────────

def tlv_decode(data: bytes) -> str:
    if not data: return ""
    kl = data[0]
    raw_len = struct.unpack_from('<I', data, 1 + kl)[0]
    return data[1+kl+4 : 1+kl+4+raw_len].decode()

# ── Read func from kvspace ──────────────────────────────────────────

def read_func(r: redis.Redis, path: str) -> list:
    ops = []
    i = 0
    while True:
        op_raw = r.get(f"{path}/[{i},0]")
        if op_raw is None: break
        opcode = tlv_decode(op_raw)
        reads, writes = [], []
        j = 1
        while True:
            v = r.get(f"{path}/[{i},-{j}]")
            if v is None: break
            reads.append(tlv_decode(v)); j += 1
        j = 1
        while True:
            v = r.get(f"{path}/[{i},{j}]")
            if v is None: break
            writes.append(tlv_decode(v)); j += 1
        ops.append({"op": opcode, "reads": reads, "writes": writes})
        i += 1
    return ops

# ── Fuse: matmul+add+relu → TileLang kernel ─────────────────────────

def compile_and_run(ops: list):
    # Hardcoded test sizes (real impl reads from heap-plat meta)
    M, N, K = 512, 512, 256
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

    print(f"  M={M} N={N} K={K} blocks={BM}x{BN}x{BK}")
    mod = kernel(M, N, K, BM, BN, BK, "float16", "float32")
    torch.cuda.synchronize()

    # Make random test data, run, compare with PyTorch reference
    a    = torch.randn((M, K), dtype=torch.float16, device='cuda')
    b    = torch.randn((K, N), dtype=torch.float16, device='cuda')
    bias = torch.randn((N,),  dtype=torch.float16, device='cuda')

    out_tl  = mod(a, b, bias); torch.cuda.synchronize()
    out_ref = torch.relu(a @ b + bias)

    diff = (out_tl - out_ref).abs().max().item()
    print(f"  max diff: {diff:.6f}  {'✅' if diff < 0.5 else '❌'}")

    # Quick bench
    import time
    for _ in range(20): mod(a, b, bias)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(200): mod(a, b, bias)
    torch.cuda.synchronize()
    tl_ms = (time.perf_counter() - t0) / 200 * 1000

    t0 = time.perf_counter()
    for _ in range(200): _ = torch.relu(a @ b + bias)
    torch.cuda.synchronize()
    pt_ms = (time.perf_counter() - t0) / 200 * 1000

    print(f"  PyTorch {pt_ms:.4f}ms  TileLang {tl_ms:.4f}ms  speedup {pt_ms/tl_ms:.2f}×")

# ── Main ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/func/tmp/inference"
    r = redis.Redis(host='127.0.0.1', port=6379)

    print(f"read {path}")
    ops = read_func(r, path)
    for o in ops:
        print(f"  {o['op']}({','.join(o['reads'])}) -> {','.join(o['writes'])}" if o['writes'] else
              f"  {o['op']}({','.join(o['reads'])})")

    opcodes = [o['op'] for o in ops if o['op'] != 'return']
    if opcodes == ['tensor.matmul', 'tensor.add', 'tensor.relu']:
        print(f"\nfuse: linear_relu ← {opcodes}")
        compile_and_run(ops)
    else:
        print(f"\nno fusion for {opcodes}")
