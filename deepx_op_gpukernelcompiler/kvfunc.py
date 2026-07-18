"""
kvfunc: read kvlang compiled functions from kvspace via kvspace-py.

A kvlang function is stored as instruction-addressable slots:
  /func/<pkg>/<name>          = signature string
  /func/<pkg>/<name>/[i,0]   = opcode (string)
  /func/<pkg>/<name>/[i,-j]  = j-th read operand path
  /func/<pkg>/<name>/[i,j]   = j-th write slot path
"""
from kvspace import KVSpace, ErrNotFound


def read_func(kv: KVSpace, path: str) -> dict:
    """Read a kvlang function from kvspace.

    Returns {"signature": str, "ops": [{"op": str, "reads": [str], "writes": [str]}, ...]}
    """
    sig = kv.get(path).as_str()
    ops = []
    i = 0
    while True:
        opcode = _get_str(kv, f"{path}/[{i},0]")
        if opcode is None:
            break
        reads, writes = [], []
        j = 1
        while (r := _get_str(kv, f"{path}/[{i},-{j}]")) is not None:
            reads.append(r); j += 1
        j = 1
        while (w := _get_str(kv, f"{path}/[{i},{j}]")) is not None:
            writes.append(w); j += 1
        ops.append({"op": opcode, "reads": reads, "writes": writes})
        i += 1
    return {"signature": sig, "ops": ops}


def list_funcs(kv: KVSpace, prefix: str = "/func/fusion_cases") -> list:
    """List function names under a kvspace prefix (filtering instruction slots)."""
    try:
        children = kv.list(prefix)
    except ErrNotFound:
        return []
    funcs = []
    for c in children:
        if c.startswith("["):
            continue
        try:
            if kv.get(f"{prefix}/{c}").kind == "string":
                funcs.append(c)
        except ErrNotFound:
            pass
    return sorted(funcs)


def _get_str(kv: KVSpace, key: str) -> str | None:
    try:
        v = kv.get(key)
        return v.as_str() if v.kind == "string" else None
    except ErrNotFound:
        return None
