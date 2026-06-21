#!/usr/bin/env python3
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

def call_api(
    user_message: str,
    model_id: str,
    *,
    token_name: str = "",
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> dict:
    """Send a single user message to the chat completions endpoint."""
    token = os.getenv(token_name if token_name else "AUTH_TOKEN")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": user_message}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    resp = requests.post(
        f"{BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()

    return resp.json()
