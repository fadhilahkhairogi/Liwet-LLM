"""Torch backend: re-exports the original `liwet_models` implementation.

This keeps `liwet-space-cpp/app.py` byte-for-byte identical to the original UI
while letting the same app run against GGUF. Select it with
`LIWET_BACKEND=torch`.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SPACE_DIR = Path(__file__).resolve().parent.parent.parent / "liwet-space"
if str(_SPACE_DIR) not in sys.path:
    sys.path.insert(0, str(_SPACE_DIR))

from liwet_models import (  # noqa: E402,F401  (single source)
    DEFAULT_SYSTEM_PROMPT,
    LIWET_CATALOG,
    ModelSpec,
    ModelUnavailableError,
    build_prompt_from_messages,
    check_repo_availability,
    get_model_spec,
    load_liwet_model,
    stream_generate,
)

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "LIWET_CATALOG",
    "ModelSpec",
    "ModelUnavailableError",
    "build_prompt_from_messages",
    "check_repo_availability",
    "get_model_spec",
    "load_liwet_model",
    "stream_generate",
]
