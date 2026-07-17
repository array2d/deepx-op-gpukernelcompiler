# GPU Kernel Compiler 方案分析

> deepx-op-gpukernelcompiler — 为 kvlang 的 tensor 计算后端选择 GPU kernel 编译器。

## 一、需求约束

| 约束 | 说明 |
|------|------|
| **C++ 独立部署** | 运行时零 Python 依赖，kernel 编译产物可 dlopen 加载 |
| **算子融合** | 连续 tensor op（matmul→add→relu）自动融合为单个 GPU kernel |
| **AOT 编译** | kvlang compile 模块将子函数 → 编译产物 → 缓存到 kvspace |
| **多后端** | NVIDIA A800 优先，预留 AMD/昇腾扩展 |
| **代码量小** | 不在编译器本身花太多代码 |

---

## 二、候选方案对比

### 2.1 全景对比

| 维度 | **TileLang** | **Triton** | **Apache TVM** | **Halide** |
|------|-------------|-----------|---------------|-----------|
| C++ 独立部署 | ✅ AOT → .so + dlopen | ❌ libtriton.so 仅 Python API | ✅ 成熟 C++ runtime (300K) | ✅ 纯 C++ |
| 算子融合 | ✅ 编译器自动 | ✅ JIT 融合 | ✅ 图级+算子级双层 | ✅ |
| GEMM H100 FP16 | **1,890 TFLOPS (92%)** | 1,720 TFLOPS | — | — |
| Attention/MLA vs Triton | **5.56×** (MLA) | 1× (baseline) | — | — |
| 代码量 (MLA) | **~80行 Python** | — | — | 不适合 |
| 跨平台 | CUDA/ROCm/CPU/Ascend | CUDA 主/AMD 次 | CUDA/ROCm/Metal/Vulkan/CPU | CUDA/Metal/OpenCL |
| 部署方式 | AOT: Python→.so→dlopen | JIT: Python→PTX | AOT: Python/C++→.so | AOT: C++→.so |
| 生产验证 | DeepSeek-V3.2, Qwen3.5 | torch.compile, vLLM | 工业界多年 | Google, Adobe |
| 安装 | `pip install tilelang` | `pip install triton` | 源码编译较复杂 | 源码编译 |

### 2.2 TileLang vs Triton — 性能数据

| 场景 | TileLang | Triton | 优势 |
|------|----------|--------|------|
| GEMM H100 FP16 1K×1K×1K | 1,890 TFLOPS | 1,720 TFLOPS | +9.9% |
| GEMM A100 FP16 | 940 TFLOPS | 890 TFLOPS | +5.6% |
| GEMM MI300X FP16 | 1,420 TFLOPS | 1,380 TFLOPS | +2.9% |
| DeepSeek MLA H100 | **5.56× Triton** | 1× | — |
| FP8 MHA (L≥4K) | **1.48× Triton** | 1× | — |
| Sparse Attention | 12.3ms | 14.8ms | +16.9% |

来源: ICLR 2026, Tawa 2025, DeepSeek 技术报告

---

## 三、TileLang 架构

```
┌─ Python DSL 前端 ───────────────────────────────────────┐
│ @T.prim_func                                             │
│ def fused_kernel(...):                                   │
│     T.copy(src, dst)     # tile 级数据搬运               │
│     T.gemm(A, B, C)      # 矩阵乘                       │
│     T.Parallel(...)      # 并行标注                      │
└──────────────────────────┬──────────────────────────────┘
                           │
┌─ 编译器 (TVM-based) ─────▼──────────────────────────────┐
│ Phase 1: LowerAndLegalize                                │
│   LayoutInference → LowerTileOp → LegalizeMemoryAccess   │
│                                                          │
│ Phase 2: OptimizeForTarget (CUDA)                        │
│   WarpSpecialized → InjectTMA → WGMMA lowering           │
│   cp.async injection → PTX codegen                       │
└──────────────────────────┬──────────────────────────────┘
                           │
┌─ AOT 部署 ───────────────▼──────────────────────────────┐
│ generate_source() → kernel.cu → nvcc → kernel_lib.so     │
│                                                          │
│ // C++ 加载                                              │
│ void* h = dlopen("kernel_lib.so", RTLD_LAZY);            │
│ auto fn = dlsym(h, "call");                              │
│ fn(input_ptr, weight_ptr, output_ptr, stream);           │
└─────────────────────────────────────────────────────────┘
```

### 关键设计点

1. **TVM-based, 非 MLIR** — 复用 TVM 的 TIR、pass pipeline、多后端，无 MLIR 依赖
2. **tile 级原语** — `T.copy`/`T.gemm`/`T.Parallel` 等高级抽象，编译器自动推导线程绑定、共享内存布局
3. **声明式 + 编译器自动优化** — 程序员写"做什么"，编译器决定"怎么并行"
4. **Hopper 特性** — TMA、WGMMA、warp specialization、mbarrier 全支持

---

## 四、kvlang 集成方案

### 4.1 编译期（kvlang compile 模块）

```
kvlang AST (lower 后读写码 IR):
  def __fusion_0(x, W, b) -> (out) {
      tensor.matmul(x, W)   -> tmp1
      tensor.add(tmp1, b)   -> tmp2
      tensor.relu(tmp2)     -> out
  }
        │
        ▼  compile 模块识别连续 tensor op，切子函数
        │
        ▼  生成 TileLang Python 脚本
  @T.prim_func
  def __fusion_0(...):
      ...
        │
        ▼  subprocess: python3 tilelang_compile.py __fusion_0
        │
        ▼  产物:
  /func/__fusion_0_triton       = kernel_lib.so (二进制 blob)
  /func/__fusion_0_triton/meta  = {signature, grid, threads, ...}
```

### 4.2 运行时（deepx-op-gpukernelcompiler 进程）

```cpp
// C++ 进程，不依赖 Python
void* handle = dlopen(kernel_path, RTLD_LAZY);
auto kernel_fn = (KernelFunc)dlsym(handle, "call");

// kvlang dispatch 时:
// 1. 从 kvspace 读 kernel_lib.so + meta
// 2. dlopen → dlsym("call")
// 3. kernel_fn(shm_ptr_in, shm_ptr_w, shm_ptr_out, cudaStream)
// 4. SET kvspace → Notify done
```

### 4.3 编译进程（仅编译时存在）

TileLang 的 Python DSL 只在**编译期**运行。编译产物是 `.so` + meta，运行时是纯 C++：

```
编译时:  Python → TileLang DSL → .so       (一次性，缓存)
运行时:  C++  → dlopen(.so) → CUDA launch   (常驻进程)
```

---

## 五、选型结论

**推荐 TileLang。**

| 理由 | 说明 |
|------|------|
| **C++ AOT 部署** | `.so` + dlopen 已在 xLLM + Qwen3.5 生产验证 |
| **性能领先** | MLA 5.56× Triton，GEMM +10%，Attention +48% |
| **代码量** | 80 行 Python 实现 FlashMLA 级别 kernel |
| **多后端** | A800(CUDA) + 未来 AMD/昇腾 |
| **TVM 基础** | 复用 TVM 成熟 C++ 部署管线 |
| **DeepSeek 验证** | DeepSeek-V3.2 已采用 |
| **安装简单** | `pip install tilelang` |

Triton 作为备选保留——如果 TileLang 的 TVM 依赖链在某些环节不兼容，可以退回到 Triton subprocess 编译 + C++ 加载 PTX/CUBIN 的模式。

---

## 六、参考资料

- [TileLang GitHub](https://github.com/tile-ai/tilelang)
- [TileLang ICLR 2026 Paper](https://openreview.net/pdf?id=Jb1WkNSfUB)
- [TileLang vs Triton Benchmark](https://huggingface.co/blog/AtlasCloud-AI/writing-high-performance-kernels-in-tilelang)
- [TileLang 编译器架构](https://zread.ai/tile-ai/tilelang/8-architecture-overview)
- [Apache TVM](https://tvm.apache.org/)
