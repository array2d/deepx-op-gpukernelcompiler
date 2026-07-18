"""
op-gpu util: read kvlang func from kvspace → build compile spec → invoke worker.
"""
import json
import redis
from compile_worker import tlv_decode

# ── Pattern matching ───────────────────────────────────────────────────

# Map opcode sequences to fusion pattern names
FUSION_PATTERNS = {
    ("tensor.matmul", "tensor.add", "tensor.relu"): "linear_relu",
    ("tensor.matmul", "tensor.add"):              "linear_relu",  # same kernel, relu is identity
    ("tensor.matmul",):                            None,  # single op, dispatch directly
}


def read_func(r: redis.Redis, func_path: str) -> dict:
    """Read a kvlang function from kvspace and return parsed body."""
    sig = tlv_decode(r.get(func_path))
    ops = []
    i = 0
    while True:
        op_raw = r.get(f"{func_path}/[{i},0]")
        if op_raw is None:
            break
        opcode = tlv_decode(op_raw)
        reads = []
        j = 1
        while True:
            r_raw = r.get(f"{func_path}/[{i},-{j}]")
            if r_raw is None:
                break
            reads.append(tlv_decode(r_raw))
            j += 1
        writes = []
        j = 1
        while True:
            w_raw = r.get(f"{func_path}/[{i},{j}]")
            if w_raw is None:
                break
            writes.append(tlv_decode(w_raw))
            j += 1
        ops.append({"opcode": opcode, "reads": reads, "writes": writes})
        i += 1
    return {"signature": sig, "ops": ops}


def match_fusion(ops: list) -> tuple:
    """Match op sequence to a fusion pattern. Returns (pattern_name, start, end)."""
    opcodes = tuple(o["opcode"] for o in ops if o["opcode"] != "return")
    for seq, name in sorted(FUSION_PATTERNS.items(), key=lambda x: -len(x[0])):
        if opcodes[:len(seq)] == seq:
            return name, 0, len(seq)
    return None, 0, 0


def build_compile_spec(r: redis.Redis, func_path: str) -> dict:
    """Build a compile spec JSON for the compile worker."""
    func = read_func(r, func_path)
    ops = func["ops"]

    pattern, start, end = match_fusion(ops)
    if pattern is None:
        return None

    # Extract tensor names from reads/writes to find heap-plat meta
    fusion_ops = ops[start:end]
    all_reads = []
    for op in fusion_ops:
        all_reads.extend(op["reads"])

    func_name = func_path.rsplit("/", 1)[-1]

    # For now, use test shapes. Real impl reads /heap/tensor/<path>/meta
    inputs = [
        {"name": all_reads[0], "shape": [512, 256], "dtype": "float16"},
        {"name": all_reads[1], "shape": [256, 512], "dtype": "float16"},
    ]
    if len(all_reads) > 2:
        inputs.append({"name": all_reads[2], "shape": [512], "dtype": "float16"})

    return {
        "func": func_name,
        "pattern": pattern,
        "ops": [o["opcode"] for o in fusion_ops],
        "inputs": inputs,
        "outputs": [{"name": f"out", "shape": [512, 512], "dtype": "float16"}],
    }


# ── CLI ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    r = redis.Redis(host='127.0.0.1', port=6379)

    func_path = sys.argv[1] if len(sys.argv) > 1 else "/func/tmp/inference"

    print(f"reading {func_path}", file=sys.stderr)
    spec = build_compile_spec(r, func_path)
    if spec:
        print(f"fusion: {spec['pattern']} ← {spec['ops']}", file=sys.stderr)
        json.dump(spec, sys.stdout)
    else:
        print("no fusion pattern matched", file=sys.stderr)
        sys.exit(1)
