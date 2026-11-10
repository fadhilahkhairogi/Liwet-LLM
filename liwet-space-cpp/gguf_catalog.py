"""Locate Liwet GGUF files for the llama.cpp backend.

Naming follows the staging repos `fadhilahkhairogi/<Model>-GGUF`:
`<Model>-BF16.gguf`, `<Model>-F16.gguf`, `<Model>-Q8_0.gguf`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

_THIS_DIR = Path(__file__).resolve().parent
_SPACE_DIR = _THIS_DIR.parent / "liwet-space"
if str(_SPACE_DIR) not in sys.path:
    sys.path.insert(0, str(_SPACE_DIR))

from liwet_models import LIWET_CATALOG, ModelSpec, get_model_spec  # noqa: E402  (single source)

DEFAULT_GGUF_DIR = Path(os.environ.get("LIWET_GGUF_DIR", r"E:\liwet_gguf_build\gguf"))
QUANTS = ("BF16", "F16", "Q8_0")
DEFAULT_QUANT = os.environ.get("LIWET_GGUF_QUANT", "Q8_0")


def repo_short_name(repo_id: str) -> str:
    return repo_id.rsplit("/", 1)[-1]


def gguf_filename(repo_id: str, quant: Optional[str] = None) -> str:
    return f"{repo_short_name(repo_id)}-{quant or DEFAULT_QUANT}.gguf"


def local_gguf_path(repo_id: str, quant: Optional[str] = None) -> Optional[Path]:
    """Return the local GGUF path if present, else None.

    Search order: `LIWET_GGUF_DIR` -> sibling `GGUF/` folder next to the
    safetensors checkpoints (legacy layout).
    """
    filename = gguf_filename(repo_id, quant)
    candidates = [DEFAULT_GGUF_DIR / filename]
    local_root = Path(
        os.environ.get(
            "LIWET_LOCAL_MODELS_DIR",
            str(_SPACE_DIR.parent / "TUBES_1_SAFETENSOR"),
        )
    )
    candidates.append(local_root.parent / "GGUF" / filename)
    for path in candidates:
        if path.is_file():
            return path
    return None


def gguf_repo_id(repo_id: str) -> str:
    return f"fadhilahkhairogi/{repo_short_name(repo_id)}-GGUF"


def resolve_gguf(repo_id: str, quant: Optional[str] = None) -> Path:
    """Local file first; otherwise download from the matching `-GGUF` repo."""
    from liwet_models import ModelUnavailableError

    local = local_gguf_path(repo_id, quant)
    if local is not None:
        return local
    try:
        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(
                repo_id=gguf_repo_id(repo_id),
                filename=gguf_filename(repo_id, quant),
            )
        )
    except Exception as err:  # network / missing repo
        raise ModelUnavailableError(
            f"GGUF untuk {repo_id} ({quant or DEFAULT_QUANT}) tidak ditemukan di "
            f"{DEFAULT_GGUF_DIR} dan tidak dapat diunduh dari {gguf_repo_id(repo_id)}."
        ) from err


def catalog_repo_ids() -> list[str]:
    return [spec.repo_id for spec in LIWET_CATALOG]


__all__ = [
    "DEFAULT_GGUF_DIR",
    "DEFAULT_QUANT",
    "QUANTS",
    "LIWET_CATALOG",
    "ModelSpec",
    "get_model_spec",
    "gguf_filename",
    "gguf_repo_id",
    "local_gguf_path",
    "resolve_gguf",
    "repo_short_name",
]
