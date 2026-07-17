# kvlang → TileLang 接口设计方案

> kvlang 是唯一编程界面。TileLang 代码永远由 compile 模块自动生成。
> 本文定义 TileLang 需要 kvlang 提供什么输入，以及 kvlang 如何提供。

## 一、职责边界

```
┌─ kvlang ────────────────────────────────────────┐
│                                                  │
│  kv 代码 (人/Agent 手写):                         │
│    def inference(x, W, b) -> (out) {              │
│        tensor.matmul(x, W) -> tmp1               │
│        tensor.add(tmp1, b) -> tmp2               │
│        tensor.relu(tmp2)  -> out                 │
│    }                                             │
│                                                  │
│  掌控一切:                                        │
│  ├── 模型结构: kv 代码 = DAG 定义                 │
│  ├── 输入参数: tensor.new 时确定 shape/dtype      │
│  ├── tensor 生命周期: heap-plat 管理              │
│  └── 调度决策: 何时编译、用哪个后端               │
│                                                  │
│  compile 模块 ──→ 识别融合组 ──→ 生成 TileLang 代码│
│                                                  │
└──────────────────────┬───────────────────────────┘
                       │ 编译请求
                       ▼
┌─ TileLang ──────────────────────────────────────┐
│                                                  │
│  只做一件事: 给定 op 序列 + shape/dtype           │
│            → 编译为 GPU kernel (.so)              │
│                                                  │
│  不需要知道:                                      │
│  ├── kvspace 路径                                │
│  ├── vthread / PC / 调度                         │
│  ├── 控制流 (if/while)                           │
│  └── tensor 生命周期 (谁分配的、何时释放)          │
│                                                  │
└──────────────────────────────────────────────────┘
```

## 二、TileLang 需要什么输入

TileLang 编译器是纯函数：**输入 op 序列 + shape/dtype → 输出 .so**。

| 输入 | 来源 | 说明 |
|------|------|------|
| **op 序列** | kvlang AST（lower 后读写码 IR） | 如 `[matmul, add, relu]` |
| **tensor shapes** | heap-plat 写入 kvspace 的 meta | `{x: [M,K], W: [K,N], b: [1,N]}` |
| **tensor dtypes** | heap-plat meta | `float16` / `bfloat16` / `float32` |
| **accum dtype** | kvlang 标注或默认规则 | matmul 累积通常 `float32` |
| **内存指针** | heap-plat 返回的 GPU 地址 | **仅运行时传入，编译时不需要** |

### 2.1 编译时输入（确定 kernel 结构）

```
kvlang compile 模块从 kvspace 读取:

  /heap/tensor/x/meta    → {shape: [512, 256], dtype: "float16", address: {shm_name: "x_shm"}}
  /heap/tensor/W/meta    → {shape: [256, 512], dtype: "float16", address: {shm_name: "W_shm"}}
  /heap/tensor/b/meta    → {shape: [512],     dtype: "float16", address: {shm_name: "b_shm"}}

提取 shape + dtype → 填入 TileLang 模板:

  M, K = 512, 256   ← x.shape
  K, N = 256, 512   ← W.shape
  dtype = "float16"

生成 kernel 签名:
  A: T.Tensor((M, K), dtype)     ← x
  B: T.Tensor((K, N), dtype)     ← W
  bias: T.Tensor((N,), dtype)    ← b
  C: T.Tensor((M, N), dtype)     ← out
```

### 2.2 运行时输入（执行 kernel）

```
kvlang dispatch 时，从 heap-plat meta 读取 GPU 地址:

  x_shm_ptr  = 0x7f...   ← /heap/tensor/x/meta.address.shm_ptr
  W_shm_ptr  = 0x7f...   ← /heap/tensor/W/meta.address.shm_ptr
  b_shm_ptr  = 0x7f...   ← /heap/tensor/b/meta.address.shm_ptr
  out_shm_ptr = 0x7f...  ← heap-plat 新分配的

传给 kernel:
  mod(x_shm_ptr, W_shm_ptr, b_shm_ptr, out_shm_ptr, cuda_stream)
```

Triton/TileLang 完全不感知 kvspace 路径。

## 三、kvlang 如何提供这些输入

### 3.1 shape/dtype 的来源: heap-plat meta

```
tensor.new("f32", "[512,256]") -> /data/x
        │
        ▼  heap-plat 进程:
  cudaMalloc → 分配 GPU 显存
  SET /heap/tensor/x/meta = {
    shape: [512, 256],
    dtype: "float16",
    byte_size: 262144,
    device: "gpu0",
    address: {shm_name: "x_shm_0", ptr: 0x7f1234000000, node: "n1"}
  }
```

kvlang VM 不管理显存，compile 模块只需读 `/heap/tensor/<path>/meta` 即可获得 shape/dtype/ptr。

### 3.2 op 序列的来源: AST 模式匹配

compile 模块扫描已 lower 的 AST，识别连续 tensor op 段：

```go
// internal/compile/pattern.go

func matchFusion(stmts []ast.Stmt) (*FusionGroup, int) {
    // 贪心匹配: 从当前 stmt 开始，尽可能长的 tensor op 序列
    for i, stmt := range stmts {
        if !isTensorOp(stmt.Opcode) {
            return nil, i  // 遇到非 tensor op → 切断
        }
    }
    ops := extractOps(stmts)
    pattern := patternDB.Match(ops)  // [matmul, add, relu] → "linear_relu"
    if pattern == nil {
        return nil, 0  // 无匹配 → 逐 op fallback
    }
    return &FusionGroup{Pattern: pattern, Ops: ops}, len(ops)
}
```

### 3.3 融合决策时机: 首次执行

kvlang 不在 parse 时编译，在**首次 CALL 时 lazy compile**：

```
Execute: CALL inference(x, W, b)
  │
  ├─ 检查 /func/inference_triton 是否存在?
  │   ├─ 存在 → 直接 CALL 编译版本 ✅
  │   └─ 不存在 → 进入编译流程
  │
  └─ compile 模块:
      1. 读 AST → 识别融合组
      2. 读 heap-plat meta → 获取 shape/dtype
      3. 生成 TileLang Python → subprocess 编译
      4. 缓存 kernel_lib.so → kvspace
      5. 改写 AST: tensor op 组 → call inference_triton
      6. 执行编译版本
```

缓存命中的后续调用零编译开销。

## 四、输入格式总结

TileLang 编译子进程接收的是一个简单 JSON，不是 kvlang 代码：

```json
// kvlang → subprocess stdin
{
  "func": "__fusion_0",
  "pattern": "linear_relu",
  "inputs": [
    {"name": "x", "shape": [512, 256], "dtype": "float16"},
    {"name": "W", "shape": [256, 512], "dtype": "float16"},
    {"name": "b", "shape": [512],      "dtype": "float16"}
  ],
  "outputs": [
    {"name": "out", "shape": [512, 512], "dtype": "float16"}
  ]
}
```

TileLang 编译进程：
1. 读 JSON → 选模板 → 填 shape/dtype → 生成 `@T.prim_func`
2. `tilelang.compile()` → `.so`
3. 输出二进制到 stdout，kvlang 捕获后写入 kvspace

kvlang 端只需要维护约 10-20 个模式模板，每个模板是一个带占位符的 Python 字符串。
