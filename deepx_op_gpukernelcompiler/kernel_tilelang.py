"""
TileLang kernel: fused matmul + bias + activation.
"""
import torch
import tilelang
import tilelang.language as T


def _activation(name: str, x, dtype: str):
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
    """Return a @tilelang.jit factory: fn(M,N,K,BM,BN,BK,dtype,accum) → callable(A,B,bias)→C."""

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
                        Cl[i, j] = _activation(activation, x, dtype)
                else:
                    for i, j in T.Parallel(BM, BN):
                        Cl[i, j] = Cl[i, j] + bias[bx * BN + j]

                T.copy(Cl, C[by * BM, bx * BN])
        return main
    return kernel
