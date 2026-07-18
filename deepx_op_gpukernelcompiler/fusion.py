"""
Fusion pattern database — map kvlang opcode sequences to TileLang kernel templates.
"""
import torch

# (opcode_sequence) → (pattern_name, template_key, status)
#
# status 语义:
#   compilable      — 单 kernel 融合，已有模板，可编译测试
#   needs_template  — 单 kernel 融合可行，TileLang 支持原语，缺模板
#   multi_kernel    — 需拆成多个 fusable 子 kernel（核心原因：≥2 次 GEMM）
#   not_fusable     — TileLang 无对应原语或访存模式不兼容

FUSION_DB = {
    # ── compilable: 单 GEMM + 逐元素 ──────────────────────────────────
    ("tensor.matmul", "tensor.add"):                  ("linear",       "linear",    "compilable"),
    ("tensor.matmul", "tensor.add", "tensor.relu"):   ("linear_relu",  "linear_activation", "compilable"),
    ("tensor.matmul", "tensor.add", "tensor.gelu"):   ("linear_gelu",  "linear_activation", "compilable"),
    ("tensor.matmul", "tensor.add", "tensor.silu"):   ("linear_silu",  "linear_activation", "compilable"),
    ("tensor.matmul", "tensor.add", "tensor.swish"):  ("linear_swish", "linear_activation", "compilable"),

    # ── needs_template: 单 kernel，TileLang 支持原语，缺模板 ──────────
    # attention: GEMM + 逐元素 / softmax
    ("tensor.matmul", "tensor.mul"):                  ("attention_score",   "gemm_elem",   "needs_template"),
    ("tensor.matmul",):                               ("attention_output",  "gemm",        "needs_template"),
    ("tensor.add", "tensor.softmax"):                 ("attention_softmax", "softmax",     "needs_template"),
    # normalization: 逐元素 + 规约链
    ("tensor.mean", "tensor.sub", "tensor.square", "tensor.mean", "tensor.add",
     "tensor.sqrt", "tensor.div", "tensor.mul", "tensor.add"):   ("layernorm", "norm", "needs_template"),
    ("tensor.square", "tensor.mean", "tensor.add", "tensor.rsqrt",
     "tensor.mul", "tensor.mul"):                                 ("rmsnorm",   "norm", "needs_template"),
    ("tensor.sub", "tensor.add", "tensor.sqrt", "tensor.div",
     "tensor.mul", "tensor.add"):                                 ("batchnorm", "norm", "needs_template"),
    ("tensor.reshape", "tensor.mean", "tensor.sub", "tensor.square",
     "tensor.mean", "tensor.add", "tensor.sqrt", "tensor.div",
     "tensor.mul", "tensor.add"):                                 ("groupnorm", "norm", "needs_template"),
    # residual: 逐元素
    ("tensor.add", "tensor.square", "tensor.mean", "tensor.add",
     "tensor.rsqrt", "tensor.mul", "tensor.mul"):                 ("residual_add_norm", "norm", "needs_template"),

    # ── multi_kernel: ≥2 次 GEMM，需拆成多 kernel ────────────────────
    ("tensor.matmul", "tensor.add", "tensor.relu",
     "tensor.matmul", "tensor.add"):                  ("ffn_two_layer",  None, "multi_kernel"),
    ("tensor.matmul", "tensor.add", "tensor.silu",
     "tensor.matmul", "tensor.add", "tensor.mul",
     "tensor.matmul", "tensor.add"):                  ("ffn_gated",      None, "multi_kernel"),
    ("tensor.matmul", "tensor.add", "tensor.gelu",
     "tensor.matmul", "tensor.add", "tensor.gelu",
     "tensor.matmul", "tensor.add"):                  ("ffn_sequential", None, "multi_kernel"),
    ("tensor.matmul", "tensor.add", "tensor.softmax",
     "tensor.matmul", "tensor.add", "tensor.mul",
     "tensor.matmul", "tensor.add"):                  ("moe_ffn",        None, "multi_kernel"),
    # attention: multi-GEMM
    ("tensor.matmul", "tensor.add", "tensor.matmul", "tensor.add",
     "tensor.matmul", "tensor.add"):                  ("qkv_projection", None, "multi_kernel"),
    ("tensor.matmul", "tensor.add", "tensor.matmul", "tensor.add",
     "tensor.matmul", "tensor.softmax", "tensor.matmul", "tensor.add",
     "tensor.matmul", "tensor.add"):                  ("mha_forward",    None, "multi_kernel"),
    # lora: 多 GEMM (A, B) + add
    ("tensor.matmul", "tensor.add", "tensor.matmul",
     "tensor.matmul", "tensor.mul", "tensor.add"):    ("lora_forward",   None, "multi_kernel"),
    # resnet: 2 conv + skip connection
    ("tensor.conv2d", "tensor.add", "tensor.relu",
     "tensor.conv2d", "tensor.add", "tensor.add",
     "tensor.relu"):                                  ("resnet_block",   None, "multi_kernel"),
    # kv_cache: 2 GEMM + scatter
    ("tensor.matmul", "tensor.add", "tensor.matmul", "tensor.add",
     "tensor.scatter", "tensor.scatter"):             ("kv_cache_update", None, "multi_kernel"),
    # norm + GEMM: 2 kernel (norm kernel + linear kernel)
    ("tensor.mean", "tensor.sub", "tensor.square", "tensor.mean",
     "tensor.add", "tensor.rsqrt", "tensor.mul", "tensor.mul",
     "tensor.add", "tensor.matmul", "tensor.add"):    ("pre_norm_linear", None, "multi_kernel"),

    # ── not_fusable: TileLang 缺原语 ──────────────────────────────────
    ("tensor.conv2d", "tensor.add"):                  ("conv2d",       None, "not_fusable"),
    ("tensor.conv2d", "tensor.add", "tensor.relu"):   ("conv2d_relu",  None, "not_fusable"),
    ("tensor.conv2d", "tensor.sub", "tensor.add", "tensor.rsqrt",
     "tensor.mul", "tensor.mul", "tensor.add",
     "tensor.relu"):                                  ("conv_bn_relu", None, "not_fusable"),
    ("tensor.conv2d_dw", "tensor.add", "tensor.relu",
     "tensor.conv2d", "tensor.add"):                  ("depthwise_conv", None, "not_fusable"),
    # embedding/cross_entropy: gather/scatter，非 GEMM 计算模式
    ("tensor.embedding",):                            ("embedding_lookup", None, "not_fusable"),
    ("tensor.log_softmax", "tensor.nll_loss"):        ("cross_entropy",    None, "not_fusable"),
}

ACTIVATION_PT = {
    "relu":  lambda t: torch.relu(t),
    "gelu":  lambda t: torch.nn.functional.gelu(t, approximate='tanh'),
    "silu":  lambda t: torch.nn.functional.silu(t),
    "swish": lambda t: torch.nn.functional.silu(t),
}


def classify(ops: list) -> tuple:
    """Match op sequence → (pattern_name, template_key, activation, status)."""
    tensor_ops = [o for o in ops if o["op"] != "return"]
    opcodes = tuple(o["op"] for o in tensor_ops)
    result = FUSION_DB.get(opcodes)
    if result is None:
        return None, None, None, "not_matched"
    pattern, template, status = result
    if status != "compilable":
        return pattern, template, None, status
    last_op = tensor_ops[-1]["op"]
    activation = last_op.replace("tensor.", "") if last_op.startswith("tensor.") else None
    if activation not in ("relu", "gelu", "silu", "swish"):
        activation = None
    return pattern, template, activation, status
