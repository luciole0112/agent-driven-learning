"""Minimal client for OpenAI-compatible chat endpoints (default: local Ollama).

No API key is needed for local Ollama. The endpoint and model are taken from
environment variables so they can be swapped without code changes:

    ADL_LLM_BASE_URL  (default http://localhost:11434/v1)
    ADL_LLM_MODEL     (default qwen3.8:27b)
    ADL_LLM_API_KEY   (optional; sent as Bearer token if set)
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from dataclasses import dataclass


@dataclass
class LLMConfig:
    base_url: str = os.environ.get("ADL_LLM_BASE_URL", "http://localhost:11434/v1")
    model: str = os.environ.get("ADL_LLM_MODEL", "qwen3.8:27b")
    api_key: str | None = os.environ.get("ADL_LLM_API_KEY")
    temperature: float = 0.6
    max_tokens: int = 4096
    timeout_s: int = 600


def image_part(png: bytes) -> dict:
    return {"type": "image_url",
            "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}}


class ChatClient:
    def __init__(self, cfg: LLMConfig | None = None):
        self.cfg = cfg or LLMConfig()

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        body = {"model": self.cfg.model, "messages": messages,
                "temperature": self.cfg.temperature, "max_tokens": self.cfg.max_tokens}
        if tools:
            body["tools"] = tools
        req = urllib.request.Request(
            self.cfg.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode(), method="POST",
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {self.cfg.api_key}"} if self.cfg.api_key else {})})
        with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as r:
            data = json.loads(r.read())
        return data["choices"][0]["message"]
