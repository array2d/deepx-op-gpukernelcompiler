"""TileLang fused kernel: matmul + bias + relu — from kvspace func."""
import struct
import redis
import torch
import tilelang
import tilelang.language as T

# ── TLV decoder ────────────────────────────────────────────────────────

def decode_tlv(data: bytes) -> str:
    if not data: return ""
    kl = data[0]
    raw_len = struct.unpack_from('<I', data, 1 + kl)[0]
    start = 1 + kl + 4
    return data[start:start + raw_len].decode()

# ── Read from kvspace ──────────────────────────────────────────────────

r = redis.Redis(host='127.0.0.1', port=6379)
sig = decode_tlv(r.get("/func/inference"))
ops = [decode_tlv(r.get(f"/func/inference/{i}")) for i in range(3)]
print("=== kvspace func ===")
print(sig)
for o in ops: print(f"  {o}")

# ── TileLang kernel: matmul + bias + ReLU ──────────────────────────────

M, N, K = 512, 512, 512
BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32

@tilelang.jit(out_idx=[-1])
def fused_linear_relu(M, N, K, block_M, block_N, block_K,
                       dtype="float16", accum_dtype="float32"):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        bias: T.Tensor((N,), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = T.max(C_local[i, j] + bias[bx * block_N + j], 0)

            T.copy(C_local, C[by * block_M, bx * block_N])
    return main

# ── Compile & run ──────────────────────────────────────────────────────

print(f"\n=== TileLang compile: {M}×{N}×{K} fp16 ===")
mod = fused_linear_relu(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K)

a = torch.randn((M, K), dtype=torch.float16, device='cuda')
b = torch.randn((K, N), dtype=torch.float16, device='cuda')
bias = torch.randn((N,), dtype=torch.float16, device='cuda')
out_tl = mod(a, b, bias)
torch.cuda.synchronize()

out_ref = torch.relu(a @ b + bias)
diff = (out_tl - out_ref).abs().max().item()
print(f"max diff: {diff:.6f}")
print("✅ PASS" if diff < 0.5 else "❌ FAIL")

# ── Benchmark ──────────────────────────────────────────────────────────

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

print(f"\n=== Benchmark ({M}×{N}×{K}) ===")
print(f"  PyTorch:  {pt_ms:.4f} ms")
print(f"  TileLang: {tl_ms:.4f} ms")
print(f"  Speedup:  {pt_ms/tl_ms:.2f}×")
