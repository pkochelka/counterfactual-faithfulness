from __future__ import annotations

import threading
from typing import Any


BATCH_RUNTIME: dict[str, Any] = {
    "jobs": {},
    "lock": threading.Lock(),
}
