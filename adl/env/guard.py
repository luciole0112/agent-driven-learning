"""Guardrails and the fixed research protocol for agent edits.

Two kinds of protection, both enforced by :class:`~adl.env.environment.ADLEnvironment`:

* **Argument validation** (always on): every tool call is checked against its
  spec before it runs: unknown or missing arguments, wrong types, channel
  indices out of range, head connections with more than one value. A rejected
  call changes nothing and costs no budget.
* **Protocol and rollback** (when a :class:`Guard` is given): each persistent
  edit must belong to one cycle

      observe -> hypothesis -> one edit -> evaluate -> decide (accept | reject)

  An edit that drops base-class accuracy more than ``max_base_drop`` below the
  initial accuracy is rolled back at once, before the agent sees the result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from adl.models.cnn import CONV_LAYERS

EDIT_TOOLS = {"set_weights", "add_to_weights", "scale_channel", "lowrank_update",
              "add_channel", "remove_channel", "connect"}
HEADS = {"head_concept": 1, "head_base": 3}


@dataclass
class Guard:
    max_base_drop: float = 0.02        # absolute drop below the initial base accuracy
    require_hypothesis: bool = True    # an edit needs a recorded hypothesis
    require_contrast: bool = True      # a hypothesis needs label-1 and label-0 examples seen
    require_eval_after_edit: bool = True
    max_scale_factor: float = 2.0      # scale_channel factor must lie in [1/f, f] (by magnitude)

    def as_dict(self) -> dict:
        return asdict(self)


class ArgError(ValueError):
    pass


def _check_type(key: str, value, schema: dict):
    t = schema.get("type")
    if t == "integer":
        if isinstance(value, bool):
            raise ArgError(f"'{key}' must be an integer, got {value!r}")
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if not isinstance(value, int):
            raise ArgError(f"'{key}' must be an integer, got {value!r}")
    elif t == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ArgError(f"'{key}' must be a number, got {value!r}")
    elif t == "string":
        if not isinstance(value, str):
            raise ArgError(f"'{key}' must be a string, got {value!r}")
    elif t == "boolean":
        if not isinstance(value, bool):
            raise ArgError(f"'{key}' must be true or false, got {value!r}")
    elif t == "array":
        # a bare number is accepted where one value is enough (e.g. a head weight)
        if isinstance(value, bool) or not isinstance(value, (list, int, float)):
            raise ArgError(f"'{key}' must be an array of numbers, got {type(value).__name__}")
        items = schema.get("items")
        if items and isinstance(value, list):
            value = [_check_type(f"{key}[{i}]", v, items) for i, v in enumerate(value)]
    if "enum" in schema and value not in schema["enum"]:
        raise ArgError(f"'{key}' must be one of {schema['enum']}, got {value!r}")
    return value


def _channel(env, layer: str, ch: int, key: str) -> None:
    n = env.model.channels(layer)
    if not 0 <= ch < n:
        raise ArgError(f"'{key}'={ch} is out of range: {layer} has channels 0..{n - 1}")


def _count(v) -> int:
    if isinstance(v, list):
        return sum(_count(x) for x in v)
    return 1


def validate_args(env, name: str, args: dict, spec: dict) -> dict:
    """Return cleaned args or raise :class:`ArgError` with a message for the agent."""
    params = spec.get("params", {})
    unknown = sorted(set(args) - set(params))
    if unknown:
        raise ArgError(f"unknown argument(s) {unknown} for '{name}'; allowed: {sorted(params)}")
    missing = [k for k in spec.get("required", []) if k not in args]
    if missing:
        raise ArgError(f"missing required argument(s) {missing} for '{name}'")
    if name == "remove_channel" and args.get("layer") in HEADS:
        raise ArgError("heads cannot be removed; to cut a head weight use connect with values=[0]")
    out ={k: _check_type(k, v, params[k]) for k, v in args.items()}

    # tool-specific checks
    if "layer" in out and "channel" in out and out["layer"] in CONV_LAYERS:
        _channel(env, out["layer"], out["channel"], "channel")
    if name == "render_kernels" and out.get("out_channel") is not None:
        _channel(env, out["layer"], out["out_channel"], "out_channel")
    if name == "probe_ablate":
        for c in out["channels"]:
            _channel(env, out["layer"], c, "channels")
    if name == "connect":
        _channel(env, out["src_layer"], out["src_channel"], "src_channel")
        dst = out["dst"]
        if dst in HEADS:
            if out["src_layer"] != "conv3":
                raise ArgError("only conv3 connects to the heads")
            if not 0 <= out["dst_channel"] < HEADS[dst]:
                raise ArgError(f"'dst_channel' must be 0..{HEADS[dst] - 1} for {dst}")
            if "values" in out and _count(out["values"]) != 1:
                raise ArgError(f"a connection into {dst} is ONE scalar weight; give values=[w] "
                               f"(got {_count(out['values'])} numbers)")
        elif dst in CONV_LAYERS:
            _channel(env, dst, out["dst_channel"], "dst_channel")
        else:
            raise ArgError(f"'dst' must be the next conv layer or one of {sorted(HEADS)}, got {dst!r}")
    if name == "scale_channel" and env.guard is not None:
        f, lim = abs(out["factor"]), env.guard.max_scale_factor
        if not (1 / lim <= f <= lim):
            raise ArgError(f"factor magnitude must be within [{1 / lim:.3g}, {lim:g}]; "
                           f"make smaller steps and verify each one")
    return out
