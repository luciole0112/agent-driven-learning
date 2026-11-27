"""Check that the configured VLM endpoint works for the agent loop.

Tests, in order: plain chat, image input, tool calling. Prints PASS/FAIL for
each so a model that lacks a capability is caught before a long run.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np

from adl.agent.llm import ChatClient, LLMConfig, image_part
from adl.env.environment import _png


def main() -> int:
    cfg = LLMConfig()
    cli = ChatClient(cfg)
    print(f"endpoint={cfg.base_url} model={cfg.model}")
    ok = True

    t = time.time()
    try:
        r = cli.chat([{"role": "user", "content": "Reply with the single word: ready"}])
        print(f"[PASS] chat ({time.time() - t:.1f}s): {str(r.get('content'))[:80]!r}")
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] chat: {e}")
        return 1

    img = np.zeros((16, 16, 3))
    img[4:12, 4:12, 0] = 1.0  # red square on black
    t = time.time()
    try:
        r = cli.chat([{"role": "user", "content": [
            {"type": "text", "text": "What colour is the square in this image? One word."},
            image_part(_png(img, 8))]}])
        ans = str(r.get("content")).lower()
        status = "PASS" if "red" in ans else "WARN"
        ok &= status == "PASS"
        print(f"[{status}] image input ({time.time() - t:.1f}s): {ans[:80]!r}")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"[FAIL] image input: {e}")

    tool = [{"type": "function", "function": {
        "name": "evaluate", "description": "Evaluate the model on a split.",
        "parameters": {"type": "object", "properties": {"split": {"type": "string", "enum": ["train", "dev"]}},
                       "required": ["split"]}}}]
    t = time.time()
    try:
        r = cli.chat([{"role": "user", "content": "Call the evaluate tool on the dev split."}], tool)
        calls = r.get("tool_calls") or []
        if calls and calls[0]["function"]["name"] == "evaluate":
            args = json.loads(calls[0]["function"].get("arguments") or "{}")
            print(f"[PASS] tool calling ({time.time() - t:.1f}s): evaluate({args})")
        else:
            ok = False
            print(f"[FAIL] tool calling: no tool call returned; content={str(r.get('content'))[:120]!r}")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"[FAIL] tool calling: {e}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
