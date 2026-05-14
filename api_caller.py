#!/usr/bin/env python3
"""Minimal client for an OpenAI-compatible chat completions endpoint."""
import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv(".env.local")

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

    try:
        resp = requests.post(
            f"{BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
    except requests.HTTPError as err:
        sys.exit(f"HTTP {resp.status_code}:\n{resp}")
    except requests.RequestException as err:
        sys.exit(f"Request failed: {err}")

    return resp.json()