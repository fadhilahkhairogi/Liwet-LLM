"""HTTP backend that talks to an already-running `llama-server`.

Use this when one server process should be shared by several clients, or when
the model runs on another machine. It returns the same bundle attributes and
generator contract as the in-process backend, but generation uses the server's
native sampler (`LIWET_CPP_SAMPLER` has no effect here). The in-process backend
remains the parity reference; this one is documented as "fast mode".
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

_THIS_DIR = Path(__file__).resolve().parent
_SPACE_DIR = _THIS_DIR.parent.parent / "liwet-space"
if str(_SPACE_DIR) not in sys.path:
    sys.path.insert(0, str(_SPACE_DIR))

from liwet_models import (  # noqa: E402  (single source)
    DEFAULT_SYSTEM_PROMPT,
    LIWET_CATALOG,
    ModelSpec,
    ModelUnavailableError,
    _CHAT_ROLE_MARKERS,
    _clean_reply,
    _encode_prompt,
    _stop_token_ids,
    build_prompt_from_messages,
    get_model_spec,
)
from gguf_catalog import repo_short_name  # noqa: E402

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


def _server_url() -> str:
    url = os.environ.get("LIWET_SERVER_URL", "").strip().rstrip("/")
    if not url:
        raise ModelUnavailableError(
            "LIWET_SERVER_URL belum diatur; backend server tidak dapat dipakai."
        )
    return url


def _post(path: str, payload: dict, stream: bool = False, timeout: float = 600.0):
    import requests

    return requests.post(
        _server_url() + path, json=payload, stream=stream, timeout=timeout
    )


def _parse_label_params(label: str) -> int:
    text = label.strip().upper().replace("_", "")
    try:
        if text.endswith("M"):
            return int(float(text[:-1]) * 1_000_000)
        if text.endswith("B"):
            return int(float(text[:-1]) * 1_000_000_000)
        return int(float(text))
    except ValueError:
        return 0


class _Ids:
    __slots__ = ("ids",)

    def __init__(self, ids: List[int]) -> None:
        self.ids = list(ids)


class ServerTokenizer:
    """Tokenizer proxy over `POST /tokenize` (llama-server)."""

    def __init__(self) -> None:
        self._special: Dict[str, Optional[int]] = {}
        self._bos: Optional[int] = None

    def _tokenize(self, text: str, add_special: bool) -> List[int]:
        resp = _post(
            "/tokenize",
            {"content": text, "add_special": bool(add_special)},
            timeout=60.0,
        )
        resp.raise_for_status()
        data = resp.json()
        ids = data.get("tokens")
        if ids is None:
            raise ModelUnavailableError("Respons /tokenize tidak memuat 'tokens'.")
        return [int(t) for t in ids]

    def encode(self, text: str, add_special_tokens: bool = False) -> _Ids:
        ids = self._tokenize(text, add_special=add_special_tokens)
        bos = self.bos_id()
        if not add_special_tokens and bos is not None and ids and ids[0] == bos:
            ids = ids[1:]
        elif add_special_tokens and bos is not None and (not ids or ids[0] != bos):
            ids = [bos] + ids
        return _Ids(ids)

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        # llama-server has no public detokenize endpoint in all builds; the
        # streamed `content` already is detokenized text, so this is only used
        # by helpers that we bypass for server mode.
        return " ".join(str(i) for i in ids)

    def bos_id(self) -> Optional[int]:
        if self._bos is not None:
            return self._bos
        try:
            ids = self._tokenize("<bos>", add_special=False)
        except Exception:
            ids = []
        self._bos = int(ids[0]) if len(ids) == 1 else None
        return self._bos

    def token_to_id(self, token: str) -> Optional[int]:
        if token == "<bos>":
            return self.bos_id()
        if token in self._special:
            return self._special[token]
        tid: Optional[int] = None
        try:
            ids = self._tokenize(token, add_special=False)
            if len(ids) == 1:
                tid = int(ids[0])
        except Exception:
            tid = None
        self._special[token] = tid
        return tid


@dataclass
class LiwetServerBundle:
    repo_id: str
    spec: ModelSpec
    llm: None
    tokenizer: ServerTokenizer
    device: types.SimpleNamespace
    dtype: str
    total_params: int
    active_params: int
    context_len: int
    lock: threading.Lock
    base_url: str = field(default_factory=_server_url)


def _model_name(repo_id: str) -> str:
    return repo_short_name(repo_id)


def check_repo_availability(repo_id: str, timeout_s: float = 3.0) -> bool:
    try:
        resp = _post("/health", {}, timeout=timeout_s)
        return resp.status_code == 200
    except Exception:
        return False


def load_liwet_model(
    repo_id: str,
    device_name: str = "auto",
    dtype_name: Optional[str] = None,
    token: Optional[str] = None,
) -> LiwetServerBundle:
    base = _server_url()
    try:
        health = _post("/health", {}, timeout=5.0)
        if health.status_code != 200:
            raise ModelUnavailableError(
                f"llama-server di {base} tidak sehat (HTTP {health.status_code})."
            )
    except ModelUnavailableError:
        raise
    except Exception as err:
        raise ModelUnavailableError(
            f"Tidak dapat menghubungi llama-server di {base}: {err}"
        ) from err

    spec = get_model_spec(repo_id)
    return LiwetServerBundle(
        repo_id=repo_id,
        spec=spec,
        llm=None,
        tokenizer=ServerTokenizer(),
        device=types.SimpleNamespace(type="server"),
        dtype=(dtype_name or os.environ.get("LIWET_GGUF_QUANT", "Q8_0")).lower(),
        total_params=_parse_label_params(spec.total_params_label),
        active_params=_parse_label_params(spec.active_params_label),
        context_len=int(spec.context_len),
        lock=threading.Lock(),
    )


def _stop_strings(spec: ModelSpec) -> List[str]:
    stops = ["<|end|>", "<|user|>", "<|endoftext|>"]
    if spec.prompt_mode == "chat_role":
        stops.extend(_CHAT_ROLE_MARKERS)
    return stops


def stream_generate(
    bundle: LiwetServerBundle,
    prompt_text: str,
    max_new_tokens: int = 160,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    do_sample: Optional[bool] = None,
    seed: Optional[int] = None,
    stats: Optional[dict] = None,
) -> Iterator[str]:
    if not prompt_text.strip():
        raise ValueError("Prompt tidak boleh kosong.")

    ids, truncated = _encode_prompt(bundle, prompt_text, max_new_tokens)
    sampling = do_sample if do_sample is not None else float(temperature) > 0.0
    payload = {
        "prompt": ids,
        "n_predict": max(0, int(max_new_tokens)),
        "stream": True,
        "temperature": 0.0 if not sampling else float(temperature),
        "top_k": int(top_k),
        "top_p": float(top_p),
        "repeat_penalty": float(repetition_penalty),
        "stop": _stop_strings(bundle.spec),
        "cache_prompt": False,
    }
    if seed is not None:
        payload["seed"] = int(seed)

    wall_started = time.perf_counter()
    generated = 0
    accum = ""
    previous_text = ""

    with bundle.lock:
        with _post("/completion", payload, stream=True) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                piece = chunk.get("content") or ""
                if piece:
                    accum += piece
                    generated += 1
                    clean = _clean_reply(accum, bundle.spec.prompt_mode)
                    if clean != previous_text:
                        previous_text = clean
                        yield clean

                if chunk.get("stop"):
                    break
                if bundle.spec.prompt_mode == "chat_role" and any(
                    marker in accum for marker in _CHAT_ROLE_MARKERS
                ):
                    break

    wall_seconds = max(time.perf_counter() - wall_started, 1e-6)
    if stats is not None:
        stats.update(
            {
                "text": previous_text,
                "prompt_tokens": len(ids),
                "generated_tokens": generated,
                "seconds": wall_seconds,
                "wall_seconds": wall_seconds,
                "tokens_per_second": generated / wall_seconds,
                "truncated": truncated,
            }
        )
