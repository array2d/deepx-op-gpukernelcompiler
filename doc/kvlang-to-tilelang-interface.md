# kvlang → TileLang 输入接口

> kvlang 是唯一编程界面。TileLang 代码永远由 compile 模块自动生成。

## 一、定位

```
kvlang:  计算平台的中央调度执行器
         由 kv 代码定义并掌控一切:
           - 模型结构 (DAG)
           - 输入参数 (shape / dtype)
           - tensor 生命周期 (heap-plat)
           - 跨进程显存管理 (shm ptr)
           - 分布式拓扑 (node IP / GPU ID)
           - 调度决策 (何时编译、选哪个后端)

TileLang: GPU kernel 编译器
          输入: op 序列 + shape + dtype → 输出: kernel_lib.so
          完全被动，不感知 kvspace、调度、分布式
```

## 二、kvlang 掌控的全部信息

kv 代码描述了完整的模型计算图。以一段 kv 代码为例：

```kv
// 模型结构 — kv 代码即 DAG 定义
def inference(x, W, b) -> (out) {
    tensor.matmul(x, W) -> tmp1
    tensor.add(tmp1, b) -> tmp2
    tensor.relu(tmp2)  -> out
}
```

tensor.new 由 heap-plat 消费，执行后在 kvspace 留下完整 meta：

```
kvlang 代码:  tensor.new("f16", "[512,256]") -> /data/x, gpu=0

         │  heap-plat 进程: cudaMalloc → 分配 GPU 0 显存
         ▼
kvspace /heap/tensor/data/x:
  {
    "shape":    [512, 256],
    "dtype":    "float16",
    "byte_size": 262144,
    "device":   "gpu",
    "gpu_id":   0,
    "node_ip":  "10.0.0.1",
    "address": {
      "shm_name": "/deepx_shm_x_0",
      "ptr":      0x7f1234000000,
      "offset":   0
    }
  }
```

compile 模块从 kvspace 读这些 meta，提取 TileLang 需要的部分。

## 三、TileLang 需要什么

### 3.1 编译时输入：shape + dtype + op 序列

这三者完全确定 kernel 的结构。TileLang 不需要 kvspace 路径、节点 IP、GPU ID、ptr。

```
从 kvspace /heap/tensor/<path>/meta 提取:

  路径         → compile 模块内部映射
  shape        → TileLang 模板占位符 M, N, K
  dtype        → "float16" / "bfloat16"
  accum_dtype  → 由规则推导 (matmul 积分为 float32)

从 kvlang AST 提取:

  op 序列      → 模式匹配 [matmul, add, relu] → "linear_relu"
```

### 3.2 运行时输入：GPU 显存裸指针

编译完成后的 kernel 只接受裸指针 + CUDA stream：

```
运行时 kvlang dispatch:

  /heap/tensor/data/x/meta.address.ptr   → 0x7f1234000000
  /heap/tensor/data/W/meta.address.ptr   → 0x7f1235000000
  /heap/tensor/data/b/meta.address.ptr   → 0x7f1236000000
  /heap/tensor/data/out/meta.address.ptr → 0x7f1237000000  (新分配)

  dlopen(kernel_lib.so)
  kernel_fn(x_ptr, W_ptr, b_ptr, out_ptr, cuda_stream)

TileLang 编译产物永远不知道:
  - 这些指针来自哪个节点
  - kvspace 路径是什么
  - tensor 的生命周期由谁管理
```

## 四、kvlang → TileLang 编译请求格式

compile 模块组装 JSON，通过 subprocess stdin 传给 TileLang 编译进程：

```json
{
  "func": "__fusion_0",
  "pattern": "linear_relu",
  "ops": ["matmul", "add", "relu"],
  "inputs": [
    {"shape": [512, 256], "dtype": "float16"},
    {"shape": [256, 512], "dtype": "float16"},
    {"shape": [512],      "dtype": "float16"}
  ],
  "outputs": [
    {"shape": [512, 512], "dtype": "float16"}
  ],
  "accum_dtype": "float32"
}
```

TileLang 编译进程：

```
1. 读 JSON → 选模式模板
2. 填入 shape/dtype → 生成 @T.prim_func Python 代码
3. tilelang.compile() → kernel_lib.so
4. base64(.so) → stdout
5. kvlang 捕获 → SET kvspace /func/__fusion_0_triton
```

## 五、kvspace 中的 tensor meta 规范

heap-plat 写入的 meta 是单一真源。compile 模块和 dispatch 模块都读它：

```
/heap/tensor/<path>/meta:
  shape       []int         必然存在，编译时必须
  dtype       string        必然存在，编译时必须
  byte_size   int64         运行时校验
  device      string        "gpu" / "cpu"
  gpu_id      int           多 GPU 调度时选择
  node_ip     string        分布式时跨节点路由
  address:
    shm_name  string        跨进程共享内存名
    ptr       uint64        本地进程可直接 cast 的 GPU 虚拟地址
    offset    int64         大 tensor 分片的偏移
```

编译时只用 shape + dtype。运行时用 ptr + gpu_id。

## 六、compile 模块在 kvlang 中的位置

```
kvlang/
├── internal/compile/
│   ├── DESIGN.md
│   ├── compile.go         # 入口: Fuse(*ast.File) → *ast.File
│   ├── pattern.go         # 融合模式库 (10-20 个), 匹配 + 注册
│   ├── codegen.go         # 根据模式 + shape/dtype 生成 TileLang Python
│   └── invoke.go          # subprocess: python compile_worker.py < JSON
│
compile 在 lower 之后、WriteBody 之前运行:
  AST (lower 后) → compile.Fuse() → AST (tensor 组替换为 call)
                                     │
                                     ▼
                               WriteBody → kvspace
```

### 模式库示例

```go
var patterns = []FusionPattern{
    {
        Name:    "linear_relu",
        Ops:     []string{"tensor.matmul", "tensor.add", "tensor.relu"},
        MinOps:  3,
    },
    {
        Name:    "linear",
        Ops:     []string{"tensor.matmul", "tensor.add"},
        MinOps:  2,
    },
    {
        Name:    "matmul",
        Ops:     []string{"tensor.matmul"},
        MinOps:  1,
    },
    // ... 持续扩展
}
```

TileLang 的 Python 模板代码存放在 `internal/compile/templates/` 目录，每模式一个 `.py.tmpl` 文件，compile 模块用 Go `text/template` 渲染 shape/dtype 占位符。

## 七、生命周期：lazy compile + cache

```
首次 CALL inference(x, W, b):
  /func/inference_triton 不存在?
    → compile.Fuse() → 生成 TileLang Python → subprocess 编译
    → kernel_lib.so → /func/inference_triton
    → 改写 AST: tensor op 组 → call inference_triton
    → 执行

后续 CALL:
  /func/inference_triton 存在 → 直接 dispatch
    读 heap-plat meta → 取 ptr
    dlopen(.so) → kernel_fn(ptr...) → done
```

shape 变化时（同一函数、不同 batch size）：

```
inference(x_512, W, b)  → kernel(M=512, K=256, N=512) → cache hit
inference(x_256, W, b)  → kernel(M=256, K=256, N=512) → recompile (shape 不同)
inference(x_512, W, b)  → kernel(M=512, K=256, N=512) → cache hit
```

shape 是编译时的参数，只要 shape 变化就需要重新编译。kvspace 中以 shape hash 为 key 缓存多个编译版本。
