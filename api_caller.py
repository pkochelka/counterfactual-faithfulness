#!/usr/bin/env python3
from __future__ import annotations

"""Minimal client for an OpenAI-compatible chat completions endpoint."""
import os
from pathlib import Path
import requests


def load_env_defaults(path: str | Path = ".env.local") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env_defaults()

BASE_URL = os.environ["BASE_URL"].rstrip("/")

REQUEST_TIMEOUT = 540
NO_AUTH_TOKEN_NAME = "__NO_AUTH__"


def is_openrouter_base_url(base_url: str = BASE_URL) -> bool:
    return "openrouter.ai" in base_url.lower()

def call_api(
    user_message: str,
    model_id: str,
    *,
    token_name: str = "",
    max_tokens: int = 2048,
    temperature: float = 0.0,
    reasoning_effort: str | None = None,
    disable_thinking: bool = False,
) -> dict:
    """Send a single user message to the chat completions endpoint."""
    token = None if token_name == NO_AUTH_TOKEN_NAME else os.getenv(token_name if token_name else "AUTH_TOKEN")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": user_message}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if reasoning_effort:
        if is_openrouter_base_url():
            payload["reasoning"] = {"effort": reasoning_effort}
        else:
            payload["reasoning_effort"] = reasoning_effort
    if disable_thinking:
        if is_openrouter_base_url():
            payload["reasoning"] = {"effort": "none", "exclude": True}
        else:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

    resp = requests.post(
        f"{BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()

    return resp.json()
