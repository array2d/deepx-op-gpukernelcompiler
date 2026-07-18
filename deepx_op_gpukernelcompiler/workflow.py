"""
op-gpu workflow CLI: kvlang code → kvload → read /func/ → compile → test.

Usage:
  ../kvlang/kvlang load test/fusion_cases/        # step 1: load .kv into kvspace
  python -m deepx_op_gpukernelcompiler.workflow --all
  python -m deepx_op_gpukernelcompiler.workflow --fusion triton --all
  python -m deepx_op_gpukernelcompiler.workflow /func/fusion_cases/linear_relu
"""
import sys
from kvspace import connect
from .kvfunc import read_func, list_funcs
from .fusion import classify
from .kernel import compile_and_test

STATUS_ICON = {
    "compilable":      "✅",
    "needs_template":  "🟡",
    "multi_kernel":    "🔷",
    "not_fusable":     "⚪",
    "not_matched":     "⚪",
}


def main():
    fusion = "tilelang"
    args = sys.argv[1:]

    if "--fusion" in args:
        idx = args.index("--fusion")
        fusion = args[idx + 1]
        del args[idx:idx + 2]

    if fusion not in ("tilelang", "triton"):
        print(f"unknown fusion engine: {fusion}")
        sys.exit(1)

    kv = connect()

    if len(args) >= 1 and args[0] == "--all":
        _run_all(kv, fusion)
    elif len(args) >= 1:
        _run_one(kv, args[0], fusion)
    else:
        print("usage: python -m deepx_op_gpukernelcompiler.workflow [--fusion tilelang|triton] <path> | --all")
        print("  python -m deepx_op_gpukernelcompiler.workflow /func/fusion_cases/linear_relu")
        print("  python -m deepx_op_gpukernelcompiler.workflow --fusion triton --all")
        sys.exit(1)


def _run_one(kv, path: str, fusion: str):
    func = read_func(kv, path)
    print(f"read {path}")
    for o in func["ops"]:
        rw = ",".join(o.get("writes", []))
        rr = ",".join(o.get("reads", []))
        print(f"  {o['op']}({rr}) → {rw}")

    pattern, template, activation, status = classify(func["ops"])
    print(f"\nfuse: {pattern or '—'}  status: {status}  engine: {fusion}"
          f"  activation: {activation or 'none'}")

    if status != "compilable":
        icon = STATUS_ICON.get(status, "⚪")
        print(f"  {icon} {status}")
        sys.exit(1)

    result = compile_and_test(kv, path, fusion=fusion)
    if result.error:
        print(f"❌ {result.error}")
        sys.exit(1)

    print(f"  max diff: {result.diff:.6f}  {'✅' if result.success else '❌'}")
    print(f"  {fusion}: {result.tl_ms:.4f} ms")
    print(f"  PyTorch: {result.pt_ms:.4f} ms")
    print(f"  speedup: {result.pt_ms/result.tl_ms:.2f}×")


def _run_all(kv, fusion: str):
    funcs = list_funcs(kv)
    if not funcs:
        print("no functions found in /func/fusion_cases/")
        sys.exit(1)

    print(f"engine: {fusion}")
    print(f"{'func':<28} {'pattern':<20} {'status':<16} {'diff':>9} "
          f"{fusion:>9} {'PyTorch':>9} {'sp':>6}")
    print("-" * 102)

    counts = {}
    for name in funcs:
        path = f"/func/fusion_cases/{name}"
        func = read_func(kv, path)
        pattern, _, _, status = classify(func["ops"])
        counts[status] = counts.get(status, 0) + 1

        if status == "compilable":
            result = compile_and_test(kv, path, fusion=fusion)
            sp = result.pt_ms / result.tl_ms if result.tl_ms > 0 else 0
            print(f"{name:<28} {pattern:<20} {status:<16} {result.diff:>9.6f} "
                  f"{result.tl_ms:>8.4f}ms {result.pt_ms:>8.4f}ms {sp:>5.2f}×")
        else:
            ops = "→".join(o["op"].replace("tensor.", "") for o in func["ops"]
                           if o["op"] != "return")
            icon = STATUS_ICON.get(status, "⚪")
            print(f"{name:<28} {pattern or '—':<20} {icon} {status:<14} {'—':>9}  ({ops})")

    print(f"\n─ summary ─")
    for s in ["compilable", "needs_template", "multi_kernel", "not_fusable", "not_matched"]:
        if counts.get(s):
            print(f"  {STATUS_ICON.get(s, '⚪')} {s}: {counts[s]}")


if __name__ == "__main__":
    main()
