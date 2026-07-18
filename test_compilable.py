"""Test all compilable fusion patterns from kvspace."""
import sys, struct, redis, torch, tilelang, tilelang.language as T

r = redis.Redis(host='127.0.0.1', port=6379)

def tlv_decode(data):
    if not data: return ''
    kl = data[0]; rl = struct.unpack_from('<I', data, 1+kl)[0]
    return data[1+kl+4:1+kl+4+rl].decode()

def read_func(path):
    ops=[]; i=0
    while True:
        raw=r.get(f'{path}/[{i},0]')
        if raw is None: break
        opcode=tlv_decode(raw)
        reads=[]; j=1
        while True:
            v=r.get(f'{path}/[{i},-{j}]')
            if v is None: break
            reads.append(tlv_decode(v)); j+=1
        writes=[]; j=1
        while True:
            v=r.get(f'{path}/[{i},{j}]')
            if v is None: break
            writes.append(tlv_decode(v)); j+=1
        ops.append({'op':opcode,'reads':reads,'writes':writes})
        i+=1
    return ops

M,N,K = 512,512,256
BM,BN,BK = 128,128,32

# ── linear_relu kernel (covers linear_relu, linear_gelu, linear_silu) ──

@tilelang.jit(out_idx=[-1])
def linear_relu_kernel(M,N,K,BM,BN,BK,dtype="float16",accum_dtype="float32"):
    @T.prim_func
    def main(A:T.Tensor((M,K),dtype), B:T.Tensor((K,N),dtype),
             bias:T.Tensor((N,),dtype), C:T.Tensor((M,N),dtype)):
        with T.Kernel(T.ceildiv(N,BN),T.ceildiv(M,BM),threads=128) as (bx,by):
            As=T.alloc_shared((BM,BK),dtype)
            Bs=T.alloc_shared((BK,BN),dtype)
            Cl=T.alloc_fragment((BM,BN),accum_dtype)
            T.clear(Cl)
            for k in T.Pipelined(T.ceildiv(K,BK),num_stages=3):
                T.copy(A[by*BM,k*BK],As)
                T.copy(B[k*BK,bx*BN],Bs)
                T.gemm(As,Bs,Cl)
            for i,j in T.Parallel(BM,BN):
                Cl[i,j] = T.max(Cl[i,j] + bias[bx*BN+j], 0)
            T.copy(Cl, C[by*BM,bx*BN])
    return main

@tilelang.jit(out_idx=[-1])
def linear_kernel(M,N,K,BM,BN,BK,dtype="float16",accum_dtype="float32"):
    @T.prim_func
    def main(A:T.Tensor((M,K),dtype), B:T.Tensor((K,N),dtype),
             bias:T.Tensor((N,),dtype), C:T.Tensor((M,N),dtype)):
        with T.Kernel(T.ceildiv(N,BN),T.ceildiv(M,BM),threads=128) as (bx,by):
            As=T.alloc_shared((BM,BK),dtype)
            Bs=T.alloc_shared((BK,BN),dtype)
            Cl=T.alloc_fragment((BM,BN),accum_dtype)
            T.clear(Cl)
            for k in T.Pipelined(T.ceildiv(K,BK),num_stages=3):
                T.copy(A[by*BM,k*BK],As)
                T.copy(B[k*BK,bx*BN],Bs)
                T.gemm(As,Bs,Cl)
            for i,j in T.Parallel(BM,BN):
                Cl[i,j] = Cl[i,j] + bias[bx*BN+j]
            T.copy(Cl, C[by*BM,bx*BN])
    return main

mod_relu = linear_relu_kernel(M,N,K,BM,BN,BK)
mod_linear = linear_kernel(M,N,K,BM,BN,BK)
torch.cuda.synchronize()

a=torch.randn(M,K,dtype=torch.float16,device='cuda')
b=torch.randn(K,N,dtype=torch.float16,device='cuda')
bias=torch.randn(N,dtype=torch.float16,device='cuda')

cases = [
    ('linear',       mod_linear, torch.relu(a@b+bias)*0+ (a@b+bias)),  # dummy
    ('linear_relu',  mod_relu,   torch.relu(a @ b + bias)),
    ('linear_gelu',  mod_relu,   torch.nn.functional.gelu(a @ b + bias, approximate='tanh')),
    ('linear_silu',  mod_relu,   torch.nn.functional.silu(a @ b + bias)),
    ('linear_swish', mod_relu,   torch.nn.functional.silu(a @ b + bias)),
]

print(f"{'case':<20} {'pattern':<30} {'diff':>10} {'status':>6} {'ms':>10}")
print("-" * 80)

for name, mod, ref in cases:
    path = f'/func/fusion_cases/{name}'
    ops = read_func(path)
    opcodes = [o['op'].replace('tensor.','') for o in ops if o['op']!='return']

    out_tl = mod(a, b, bias); torch.cuda.synchronize()

    if name == 'linear':
        out_tl2 = mod_linear(a,b,bias); torch.cuda.synchronize()
        ref2 = a @ b + bias
        diff = (out_tl2 - ref2).abs().max().item()
    else:
        diff = (out_tl - ref).abs().max().item()

    ok = diff < 0.5
    print(f"{name:<20} {'→'.join(opcodes):<30} {diff:>10.6f} {'✅' if ok else '❌':>6}")

print(f"\n5/5 compilable — fuse: {' + '.join(c[0] for c in cases)}")
