"""Analyze all 30 kvlang funcs in kvspace — classify fusability, run compilable ones."""
import sys, struct, redis

r = redis.Redis(host='127.0.0.1', port=6379)

def tlv_decode(data):
    if not data: return ""
    kl = data[0]; rl = struct.unpack_from('<I', data, 1+kl)[0]
    return data[1+kl+4:1+kl+4+rl].decode()

def read_func(path):
    ops = []; i = 0
    while True:
        raw = r.get(f"{path}/[{i},0]")
        if raw is None: break
        opcode = tlv_decode(raw)
        reads = []; j = 1
        while True:
            v = r.get(f"{path}/[{i},-{j}]")
            if v is None: break
            reads.append(tlv_decode(v)); j += 1
        writes = []; j = 1
        while True:
            v = r.get(f"{path}/[{i},{j}]")
            if v is None: break
            writes.append(tlv_decode(v)); j += 1
        ops.append({"op": opcode, "reads": reads, "writes": writes})
        i += 1
    return ops

# ── Fusion pattern DB ──────────────────────────────────────────────

FUSION_DB = {
    ("tensor.matmul", "tensor.add", "tensor.relu"):  ("linear_relu",     "compilable"),
    ("tensor.matmul", "tensor.add", "tensor.gelu"):  ("linear_gelu",     "compilable"),
    ("tensor.matmul", "tensor.add", "tensor.silu"):  ("linear_silu",     "compilable"),
    ("tensor.matmul", "tensor.add", "tensor.swish"): ("linear_swish",    "compilable"),
    ("tensor.matmul", "tensor.add"):                  ("linear",          "compilable"),
    ("tensor.conv2d", "tensor.add", "tensor.relu"):  ("conv2d_relu",     "needs_template"),
    ("tensor.conv2d", "tensor.add"):                  ("conv2d",          "needs_template"),
}

def classify(ops):
    opcodes = tuple(o["op"] for o in ops if o["op"] != "return")
    return FUSION_DB.get(opcodes, (None, "not_matched"))

# ── Scan all funcs ──────────────────────────────────────────────────

funcs = sorted(r.keys("/func/tmp/*"))
# funcs are like /func/tmp/01_linear, /func/tmp/02_linear_relu, etc.
# Actually keys() returns all keys, not just pattern. Let me use scan.

results = {"compilable": [], "needs_template": [], "not_matched": [], "errors": []}

for i in range(1, 31):
    num = f"{i:02d}"
    # Find the func name by listing /func/tmp/
    pass

# Scan: list all funcs under /func/fusion_cases/
import subprocess
kv = "/home/peng.li24/github.com/array2d/kvlang/kvlang"
out = subprocess.run([kv, "kvspace", "tree", "/func/fusion_cases"], capture_output=True, text=True)
func_names = []
for line in out.stdout.strip().split('\n'):
    s = line.strip()
    if (s.startswith('├──') or s.startswith('└──')) and not s[4:].startswith('['):
        func_names.append(s[4:].strip())

print(f"Found {len(func_names)} functions in /func/fusion_cases/\n")
print(f"{'#':<4} {'func':<30} {'ops':<6} {'pattern':<20} {'status'}")
print("-" * 80)

for name in sorted(func_names):
    path = f"/func/fusion_cases/{name}"
    ops = read_func(path)
    tensor_ops = [o for o in ops if o["op"] != "return"]
    opcodes = tuple(o["op"] for o in tensor_ops)
    pattern, status = classify(tensor_ops)

    icon = {"compilable": "🟢", "needs_template": "🟡", "not_matched": "⚪"}.get(status, "🔴")
    short_ops = "→".join(o["op"].replace("tensor.","") for o in tensor_ops[:4])
    if len(tensor_ops) > 4: short_ops += f"…+{len(tensor_ops)-4}"

    results[status].append({"name": name, "path": path, "ops": ops, "opcodes": opcodes, "pattern": pattern})
    print(f"{icon:4} {name:<30} {len(tensor_ops):<6} {pattern or short_ops:<20} {status}")

print(f"\n─ Totals ─")
print(f"  🟢 compilable:     {len(results['compilable'])}")
print(f"  🟡 needs_template: {len(results['needs_template'])}")
print(f"  ⚪ not_matched:    {len(results['not_matched'])}")

# Show which opcode signatures need templates
if results["needs_template"] or results["not_matched"]:
    print(f"\n─ Opcode signatures needing templates ─")
    seen = set()
    for r in results["needs_template"] + results["not_matched"]:
        sig = str(r["opcodes"])
        if sig not in seen:
            seen.add(sig)
            print(f"  {r['name']}: {r['opcodes']}")
