"""Numerical sandbox for agent-written analysis code.

The agent may run NumPy code over copies of the weights (and over activation
arrays it has requested). The sandbox

* never exposes torch or any autograd / optimizer library,
* only allows ``import numpy`` / ``import math``,
* scans the code (AST) for *fitting* operations (least squares, inverses,
  regression, generic optimizers) and records audit flags rather than
  forbidding them, unless ``forbid_fitting=True`` (condition H),
* charges the weight values the code reads.

This is an *audit* boundary, not a security boundary: isolation from the host
is provided by the Docker container.
"""

from __future__ import annotations

import ast
import contextlib
import io
import math

import numpy as np

FIT_NAMES = {
    "lstsq": "fit:least_squares", "pinv": "fit:pseudo_inverse",
    "solve": "fit:linear_solve", "inv": "fit:inverse",
    "polyfit": "fit:polyfit", "curve_fit": "fit:curve_fit",
    "minimize": "fit:optimizer", "LogisticRegression": "fit:regression",
    "Ridge": "fit:regression", "LinearRegression": "fit:regression",
    "grad": "fit:gradient", "backward": "fit:gradient",
}
DECOMP_NAMES = {"svd": "analysis:svd", "eig": "analysis:eig", "eigh": "analysis:eig"}
ALLOWED_IMPORTS = {"numpy", "math"}


class SandboxViolation(RuntimeError):
    pass


def scan_code(code: str) -> list[str]:
    """Return audit flags found in ``code``."""
    flags: list[str] = []
    tree = ast.parse(code)
    for node in ast.walk(tree):
        name = None
        if isinstance(node, ast.Attribute):
            name = node.attr
        elif isinstance(node, ast.Name):
            name = node.id
        if name in FIT_NAMES:
            flags.append(FIT_NAMES[name])
        if name in DECOMP_NAMES:
            flags.append(DECOMP_NAMES[name])
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) \
                else [node.module or ""]
            for m in mods:
                if m.split(".")[0] not in ALLOWED_IMPORTS:
                    flags.append(f"blocked_import:{m}")
        # iterative update loops (e.g. hand-written gradient descent) leave
        # a characteristic "x -= lr * ..." pattern inside a loop
        if isinstance(node, (ast.For, ast.While)):
            for sub in ast.walk(node):
                if isinstance(sub, ast.AugAssign) and isinstance(sub.op, (ast.Sub, ast.Add)):
                    flags.append("suspect:iterative_update_loop")
                    break
    return sorted(set(flags))


def _safe_import(name, *args, **kwargs):
    if name.split(".")[0] not in ALLOWED_IMPORTS:
        raise SandboxViolation(f"import of '{name}' is not allowed in the sandbox")
    return __import__(name, *args, **kwargs)


SAFE_BUILTINS = {
    k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
    for k in ["abs", "all", "any", "bool", "dict", "enumerate", "float", "int",
              "len", "list", "max", "min", "print", "range", "reversed", "round",
              "set", "sorted", "str", "sum", "tuple", "zip", "isinstance",
              "ValueError", "Exception", "map", "filter"]
}
SAFE_BUILTINS["__import__"] = _safe_import


def run_code(code: str, weights_getter, arrays: dict[str, np.ndarray],
             forbid_fitting: bool = False, max_output_chars: int = 6000):
    """Execute ``code``.

    Inside the code:
      ``W(name)``   -> numpy copy of a parameter tensor (charged on read)
      ``A[key]``    -> arrays previously captured with ``capture_activations``
      ``out``       -> dict; arrays put here can later be applied with
                       ``apply_sandbox_output`` (a separate, charged edit tool)
    Returns (stdout, out_dict, flags).
    """
    flags = scan_code(code)
    if any(f.startswith("blocked_import") for f in flags):
        raise SandboxViolation(f"blocked imports: {flags}")
    if forbid_fitting and any(f.startswith(("fit:", "suspect:")) for f in flags):
        raise SandboxViolation(f"fitting operations are forbidden in this condition: {flags}")
    out: dict[str, np.ndarray] = {}
    ns = {"np": np, "math": math, "W": weights_getter, "A": arrays, "out": out,
          "__builtins__": SAFE_BUILTINS}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(code, "<agent_sandbox>", "exec"), ns)  # noqa: S102
    text = buf.getvalue()
    if len(text) > max_output_chars:
        text = text[:max_output_chars] + f"\n...[truncated {len(text) - max_output_chars} chars]"
    return text, out, flags
