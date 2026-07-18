"""
deepx-op-gpukernelcompiler:
  Read kvlang func from kvspace → TileLang compile fused kernel → GPU execute.
"""
import struct
import redis
import torch
import tilelang
import tilelang.language as T

# ── 0. TLV decoder (kvspace XValue wire format) ──────────────────────────

def decode_tlv(data: bytes) -> str:
    """kvspace XValue TLV: [1B kind_len][N B kind][4B raw_len LE][raw]"""
    if not data: return ""
    kl = data[0]
    raw_len = struct.unpack_from('<I', data, 1 + kl)[0]
    start = 1 + kl + 4
    return data[start:start + raw_len].decode()

# ── 1. Read function from kvspace ────────────────────────────────────────

r = redis.Redis(host='127.0.0.1', port=6379)
sig = decode_tlv(r.get("/func/inference"))
ops = [decode_tlv(r.get(f"/func/inference/{i}")) for i in range(3)]

print("=== kvspace func ===")
print(sig)
for o in ops:
    print(f"  {o}")

# ── 2. TileLang kernel: fused matmul + add + relu ────────────────────────

M, N, K = 512, 512, 512
BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32

@T.prim_func
def fused_linear_relu(
    A: T.Buffer((M, K), "float16"),
    B: T.Buffer((K, N), "float16"),
    bias: T.Buffer((1, N), "float16"),
    out: T.Buffer((M, N), "float16"),
):
    T.func_attr({"global_symbol": "fused_linear_relu", "tir.noalias": True})
    # grid: CEIL(M/BLOCK_M) × CEIL(N/BLOCK_N), threads=256
    for bx in T.parallel(T.ceildiv(M, BLOCK_M)):
        for by in T.parallel(T.ceildiv(N, BLOCK_N)):
            # Allocate shared + local
            a_shared = T.alloc_shared((BLOCK_M, BLOCK_K), "float16")
            b_shared = T.alloc_shared((BLOCK_K, BLOCK_N), "float16")
            c_local = T.alloc_fragment((BLOCK_M, BLOCK_N), "float16")

            T.clear(c_local)
            for k in T.serial(T.ceildiv(K, BLOCK_K)):
                T.copy(A[bx*BLOCK_M:(bx+1)*BLOCK_M, k*BLOCK_K:(k+1)*BLOCK_K], a_shared)
                T.copy(B[k*BLOCK_K:(k+1)*BLOCK_K, by*BLOCK_N:(by+1)*BLOCK_N], b_shared)
                T.gemm(a_shared, b_shared, c_local)

            # bias add + relu
            for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                c_local[i, j] = c_local[i, j] + bias[0, j]
            for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                c_local[i, j] = T.max(c_local[i, j], T.cast(0, "float16"))

            T.copy(c_local, out[bx*BLOCK_M:(bx+1)*BLOCK_M, by*BLOCK_N:(by+1)*BLOCK_N])

# ── 3. Compile ───────────────────────────────────────────────────────────

print(f"\n=== TileLang compile: {M}×{N}×{K}, fp16 ===")
mod = tilelang.compile(fused_linear_relu, target="cuda")

# ── 4. Test data & run ───────────────────────────────────────────────────

a = torch.randn((M, K), dtype=torch.float16, device='cuda')
b = torch.randn((K, N), dtype=torch.float16, device='cuda')
bias = torch.randn((1, N), dtype=torch.float16, device='cuda')
out_tl = torch.empty((M, N), dtype=torch.float16, device='cuda')

# Reference
out_ref = torch.relu(a @ b + bias)

# Run
mod(a, b, bias, out_tl)
torch.cuda.synchronize()

# ── 5. Verify ────────────────────────────────────────────────────────────

diff = (out_tl - out_ref).abs().max().item()
print(f"max diff: {diff:.6f}")
print("✅ PASS" if diff < 0.5 else f"❌ FAIL")

# ── 6. Benchmark ─────────────────────────────────────────────────────────

import time

# warmup
for _ in range(20):
    mod(a, b, bias, out_tl)
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(200):
    mod(a, b, bias, out_tl)
torch.cuda.synchronize()
tl_ms = (time.perf_counter() - t0) / 200 * 1000

t0 = time.perf_counter()
for _ in range(200):
    _ = torch.relu(a @ b + bias)
torch.cuda.synchronize()
pt_ms = (time.perf_counter() - t0) / 200 * 1000

print(f"\n=== Benchmark ({M}×{N}×{K} fp16) ===")
print(f"  PyTorch:  {pt_ms:.4f} ms")
print(f"  TileLang: {tl_ms:.4f} ms")
print(f"  Speedup:  {pt_ms/tl_ms:.2f}×")
