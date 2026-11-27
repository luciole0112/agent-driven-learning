"""Observe - reason - act - verify loop driving an LLM/VLM over the ADL tools."""

from __future__ import annotations

import json
from typing import Protocol

from adl.agent.llm import ChatClient, image_part
from adl.env.environment import TOOL_SPECS, ADLEnvironment

SYSTEM_PROMPT = """You are the learning mechanism of a small convolutional neural network.
There is no optimizer and no gradient: the network changes only when you edit it.

Goal: make the network's `head_concept` output score HIGH for images that contain
the target concept (label 1) and LOW for images that do not (label 0). You are not
told what the concept is: discover it from the training images and from the
network's internal responses. Keep the base-class accuracy (head_base) intact.

Work like a scientist:
1. Observe: look at examples and at what existing channels respond to.
2. Hypothesise: what feature is missing, where in the network it should live,
   and how it could be built from existing features.
3. Act: edit kernels, add or connect channels, change biases. Prefer small,
   explainable edits. Before each edit, write a short rationale and a
   prediction of what `evaluate` will show.
4. Verify: evaluate, compare with your prediction, roll back failed edits.
5. Keep notes with the `note` tool.

Rules: never try to train the network or write optimisation loops in the
sandbox. Budgets are limited; check them with `inspect_model`. Call `submit`
when you are done. The final model is scored on data you never see."""


class LLM(Protocol):
    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict: ...


def tool_schemas(env: ADLEnvironment) -> list[dict]:
    out = []
    for name in env.available_tools():
        s = TOOL_SPECS[name]
        props = dict(s.get("params", {}))
        props["rationale"] = {"type": "string", "description": "why you call this tool now"}
        out.append({"type": "function", "function": {
            "name": name, "description": s["desc"],
            "parameters": {"type": "object", "properties": props,
                           "required": s.get("required", [])}}})
    return out


def context_window(messages: list[dict], env: ADLEnvironment, keep_last: int = 30,
                   keep_images: int = 2) -> list[dict]:
    """What the model sees each turn.

    Local VLMs have short contexts, so the full history is kept on disk but
    the prompt holds: system + task, the agent's notes, and the most recent
    ``keep_last`` messages (starting at an assistant turn so tool results stay
    paired with their calls). Only the newest ``keep_images`` image messages
    keep their pixels; older ones become a text placeholder.
    """
    head, tail = messages[:2], messages[2:]
    if len(tail) > keep_last:
        start = len(tail) - keep_last
        while start < len(tail) and tail[start].get("role") != "assistant":
            start += 1
        tail = tail[start:]
    out, seen = [], 0
    for m in reversed(tail):
        if isinstance(m.get("content"), list):
            seen += 1
            if seen > keep_images:
                m = {"role": m["role"], "content": "[older images omitted to save context]"}
        out.append(m)
    out.reverse()
    extra = []
    if env.notes:
        extra = [{"role": "user", "content": "Your notes so far:\n- " + "\n- ".join(env.notes[-20:])}]
    return head + extra + out


def run_agent(env: ADLEnvironment, llm: LLM | None = None, max_turns: int = 150,
              transcript_path: str | None = None, log=print) -> dict:
    llm = llm or ChatClient()
    tools = tool_schemas(env)
    task = ("Task: teach the network the hidden concept. Start with `inspect_model`."
            + ("\nNote: in this condition images are withheld; you only get numbers."
               if env.condition == "no_semantics" else "")
            + ("\nNote: in this condition you cannot observe internals; only evaluate and edit."
               if env.condition == "no_observation" else ""))
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": task}]
    for turn in range(max_turns):
        msg = llm.chat(context_window(messages, env), tools)
        messages.append({k: v for k, v in msg.items() if k in ("role", "content", "tool_calls")})
        calls = msg.get("tool_calls") or []
        if not calls:
            messages.append({"role": "user", "content":
                             "Continue by calling a tool, or call `submit` if you are done."})
            continue
        pending_images = []
        for c in calls:
            name = c["function"]["name"]
            try:
                args = json.loads(c["function"].get("arguments") or "{}")
            except json.JSONDecodeError as e:
                args, name = {}, f"__bad_json__:{e}"
            rationale = args.pop("rationale", "")
            if name.startswith("__bad_json__"):
                text, images = f"ERROR: could not parse arguments ({name})", []
            else:
                res = env.call(name, args, rationale)
                text, images = res.text, res.images
            log(f"[turn {turn}] {name}({json.dumps(args)[:200]}) -> {text[:200]}")
            messages.append({"role": "tool", "tool_call_id": c.get("id", name), "content": text})
            pending_images += [(name, im) for im in images]
        if pending_images:
            parts = [{"type": "text", "text": "Images returned by: "
                      + ", ".join(n for n, _ in pending_images)}]
            parts += [image_part(im) for _, im in pending_images]
            messages.append({"role": "user", "content": parts})
        if env.submitted:
            break
    if transcript_path:
        with open(transcript_path, "w") as f:
            json.dump([_strip_images(m) for m in messages], f, ensure_ascii=False, indent=1)
    return env.final_report()


def _strip_images(m: dict) -> dict:
    if isinstance(m.get("content"), list):
        return {**m, "content": [p if p.get("type") == "text" else {"type": "image", "omitted": True}
                                 for p in m["content"]]}
    return m
