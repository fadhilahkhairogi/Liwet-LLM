"""Backend selector for `liwet-space-cpp`.

`app.py` imports exactly the names it imported from `liwet_models` in the
original Space; this module picks the implementation based on
`LIWET_BACKEND` (``llamacpp`` | ``server`` | ``torch``).

Defaults:
  * ``LIWET_SERVER_URL`` set  -> ``server``
  * otherwise                 -> ``llamacpp`` (in-process, CPU or GPU)
Set ``LIWET_BACKEND=torch`` to run the original PyTorch path unchanged.
"""
from __future__ import annotations

import importlib
import os

_VALID = ("llamacpp", "server", "torch")


def _select_backend() -> str:
    requested = os.environ.get("LIWET_BACKEND", "").strip().lower()
    if requested:
        if requested not in _VALID:
            raise ValueError(
                f"LIWET_BACKEND={requested!r} tidak dikenal; pilih {_VALID}."
            )
        return requested
    if os.environ.get("LIWET_SERVER_URL", "").strip():
        return "server"
    return "llamacpp"


_MODULES = {
    "llamacpp": "backends.llamacpp_inproc",
    "server": "backends.llamacpp_server",
    "torch": "backends.torch_backend",
}

ACTIVE_BACKEND = _select_backend()
_mod = importlib.import_module(_MODULES[ACTIVE_BACKEND])

DEFAULT_SYSTEM_PROMPT = _mod.DEFAULT_SYSTEM_PROMPT
LIWET_CATALOG = _mod.LIWET_CATALOG
ModelSpec = getattr(_mod, "ModelSpec", None)
ModelUnavailableError = _mod.ModelUnavailableError
build_prompt_from_messages = _mod.build_prompt_from_messages
check_repo_availability = _mod.check_repo_availability
get_model_spec = _mod.get_model_spec
load_liwet_model = _mod.load_liwet_model
stream_generate = _mod.stream_generate

__all__ = [
    "ACTIVE_BACKEND",
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
