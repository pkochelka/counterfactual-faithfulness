#!/usr/bin/env python3
"""Minimal client for an OpenAI-compatible chat completions endpoint."""
import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv(".env.local")

BASE_URL = os.environ["BASE_URL"].rstrip("/")
AUTH_TOKEN = os.getenv("AUTH_TOKEN")

REQUEST_TIMEOUT = 540

HEADERS = {"Content-Type": "application/json"}
if AUTH_TOKEN:
    HEADERS["Authorization"] = f"Bearer {AUTH_TOKEN}"

def call_api(
    user_message: str,
    model_id: str,
    *,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> dict:
    """Send a single user message to the chat completions endpoint."""
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": user_message}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        resp = requests.post(
            f"{BASE_URL}/chat/completions",
            headers=HEADERS,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
    except requests.HTTPError as err:
        sys.exit(f"HTTP {resp.status_code}:\n{resp}")
    except requests.RequestException as err:
        sys.exit(f"Request failed: {err}")

    return resp.json()