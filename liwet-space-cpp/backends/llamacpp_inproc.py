"""In-process llama.cpp backend (GGUF on CPU or GPU offload).

Implements the exact UI contract of `liwet-space/liwet_models.py` while
running the quantized GGUF through `llama-cpp-python`. Sampling is replayed in
NumPy in the same order as the PyTorch sampler (repetition penalty over all
seen ids -> temperature -> top-k -> top-p -> multinomial) so the "parity"
mode is deterministic and independent from llama.cpp's native sampler.

The prompt builder, tokenizer helpers, stop ids and reply cleanup are imported
from `liwet-space/liwet_models.py` (single source of truth).
"""
from __future__ import annotations

import os
import sys
import threading
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

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
from gguf_catalog import (  # noqa: E402
    DEFAULT_QUANT,
    QUANTS,
    local_gguf_path,
    resolve_gguf,
    repo_short_name,
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

_QUANT_ENV = "LIWET_GGUF_QUANT"
_NGL_ENV = "LIWET_GGUF_NGL"
_SAMPLER_ENV = "LIWET_CPP_SAMPLER"  # "parity" (default) | "native"

_MAX_CACHED = 2
_CACHE: "Dict[Tuple[str, str], LiwetCppBundle]" = {}
_CACHE_LOCK = threading.Lock()


def _physical_cpu_count() -> int:
    env_val = os.environ.get("LIWET_CPU_THREADS", "").strip()
    if env_val.isdigit() and int(env_val) > 0:
        return int(env_val)
    try:
        import psutil

        return max(1, psutil.cpu_count(logical=False) or 1)
    except Exception:
        return max(1, (os.cpu_count() or 2) // 2)


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


class LlamaTokenizerAdapter:
    """Minimal `tokenizers.Tokenizer` look-alike backed by llama.cpp."""

    def __init__(self, llm) -> None:
        self._llm = llm
        self._special: Dict[str, Optional[int]] = {}

    def encode(self, text: str, add_special_tokens: bool = False) -> _Ids:
        ids = self._llm.tokenize(
            text.encode("utf-8"), add_bos=bool(add_special_tokens), special=True
        )
        return _Ids(ids)

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        raw = self._llm.detokenize(list(ids), special=not skip_special_tokens)
        return raw.decode("utf-8", errors="replace")

    def token_to_id(self, token: str) -> Optional[int]:
        if token in self._special:
            return self._special[token]
        tid: Optional[int] = None
        if token == "<bos>":
            tid = int(self._llm.token_bos())
        elif token == "<eos>":
            tid = int(self._llm.token_eos())
        if tid is None and token.startswith("<") and token.endswith(">"):
            ids = self._llm.tokenize(
                token.encode("utf-8"), add_bos=False, special=True
            )
            if len(ids) == 1:
                tid = int(ids[0])
        self._special[token] = tid
        return tid


@dataclass
class LiwetCppBundle:
    repo_id: str
    spec: ModelSpec
    llm: object
    tokenizer: LlamaTokenizerAdapter
    device: types.SimpleNamespace
    dtype: str
    total_params: int
    active_params: int
    context_len: int
    lock: threading.Lock
    quant: str
    model_path: str

    # Compatibility attributes referenced by the shared helpers.
    @property
    def model(self):
        return self.llm


def _gpu_layers() -> int:
    raw = os.environ.get(_NGL_ENV, "").strip()
    if raw:
        try:
            return int(raw)
        except ValueError:
            pass
    try:
        import llama_cpp

        if llama_cpp.llama_supports_gpu_offload():
            return -1
    except Exception:
        pass
    return 0


def _backend_available() -> bool:
    try:
        import llama_cpp  # noqa: F401

        return True
    except Exception:
        return False


def check_repo_availability(repo_id: str, timeout_s: float = 3.0) -> bool:
    """A model is offered when its GGUF exists locally (or via HF download)."""
    quant = os.environ.get(_QUANT_ENV, DEFAULT_QUANT)
    if local_gguf_path(repo_id, quant) is not None:
        return True
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        return bool(api.repo_exists(f"fadhilahkhairogi/{repo_short_name(repo_id)}-GGUF"))
    except Exception:
        return False


def _build_llama(model_path: Path, spec: ModelSpec) -> object:
    from llama_cpp import Llama

    ngl = _gpu_layers()
    threads = _physical_cpu_count()
    n_ctx = int(spec.context_len)
    return Llama(
        model_path=str(model_path),
        n_gpu_layers=ngl,
        n_ctx=n_ctx,
        n_batch=512,
        n_ubatch=512,
        n_threads=threads,
        n_threads_batch=threads,
        logits_all=True,
        verbose=False,
        seed=0,
    )


def load_liwet_model(
    repo_id: str,
    device_name: str = "auto",
    dtype_name: Optional[str] = None,
    token: Optional[str] = None,
) -> LiwetCppBundle:
    if not _backend_available():
        raise ModelUnavailableError(
            "llama-cpp-python belum terpasang di lingkungan ini."
        )
    spec = get_model_spec(repo_id)
    quant = (dtype_name or os.environ.get(_QUANT_ENV, DEFAULT_QUANT)).upper()
    if quant not in QUANTS:
        quant = DEFAULT_QUANT

    key = (repo_id, quant)
    with _CACHE_LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            _CACHE.pop(key)
            _CACHE[key] = cached
            return cached

    path = resolve_gguf(repo_id, quant)
    llm = _build_llama(path, spec)

    ngl = _gpu_layers()
    device = types.SimpleNamespace(type="cuda" if ngl != 0 else "cpu")
    bundle = LiwetCppBundle(
        repo_id=repo_id,
        spec=spec,
        llm=llm,
        tokenizer=LlamaTokenizerAdapter(llm),
        device=device,
        dtype=quant.lower(),
        total_params=_parse_label_params(spec.total_params_label),
        active_params=_parse_label_params(spec.active_params_label),
        context_len=int(spec.context_len),
        lock=threading.Lock(),
        quant=quant,
        model_path=str(path),
    )

    with _CACHE_LOCK:
        _CACHE[key] = bundle
        while len(_CACHE) > _MAX_CACHED:
            _CACHE.pop(next(iter(_CACHE)))
    return bundle


# ---------------------------------------------------------------------------
# Sampler (mirrors liwet_models._sample_next_token)
# ---------------------------------------------------------------------------


def _apply_repetition_penalty(
    logits: np.ndarray, seen_ids: List[int], penalty: float
) -> None:
    if penalty == 1.0 or not seen_ids:
        return
    uniq = np.unique(np.asarray(seen_ids, dtype=np.int64))
    vals = logits[uniq]
    logits[uniq] = np.where(vals > 0, vals / penalty, vals * penalty)


def _softmax_f32(x: np.ndarray) -> np.ndarray:
    m = np.max(x)
    e = np.exp((x - m).astype(np.float32))
    total = e.sum(dtype=np.float32)
    return e / total


def _sample_next_token_np(
    logits: np.ndarray,
    sampling: bool,
    temperature: float,
    top_k: int,
    top_p: float,
    rng: np.random.Generator,
) -> int:
    if not sampling:
        return int(np.argmax(logits))
    temp = max(float(temperature), 1e-5)
    logits = logits.astype(np.float32) / np.float32(temp)
    if top_k is not None and int(top_k) > 0:
        k = min(int(top_k), logits.size)
        kth = np.partition(logits, -k)[-k]
        logits = np.where(logits < kth, np.float32(-np.inf), logits).astype(
            np.float32
        )
    if top_p is not None and 0.0 < float(top_p) < 1.0:
        order = np.argsort(-logits, kind="stable")
        sorted_logits = logits[order]
        probs = _softmax_f32(sorted_logits)
        remove = (np.cumsum(probs, dtype=np.float32) - probs) > float(top_p)
        sorted_logits = np.where(
            remove, np.float32(-np.inf), sorted_logits
        ).astype(np.float32)
        logits = np.full_like(logits, np.float32(-np.inf))
        logits[order] = sorted_logits
    probs = _softmax_f32(logits)
    probs = probs.astype(np.float64)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        return int(np.argmax(logits))
    probs /= total
    return int(rng.choice(logits.size, p=probs))


def _logits_for_last_token(llm, token_id: int) -> np.ndarray:
    llm.eval([token_id])
    row = llm.n_tokens - 1
    return np.asarray(llm.scores[row], dtype=np.float32).copy()


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def stream_generate(
    bundle: LiwetCppBundle,
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
    prompt_tokens = len(ids)
    max_new = max(0, int(max_new_tokens))
    stop_ids = _stop_token_ids(bundle.tokenizer)
    sampling = do_sample if do_sample is not None else float(temperature) > 0.0
    use_parity = os.environ.get(_SAMPLER_ENV, "parity").strip().lower() != "native"
    rng = np.random.default_rng(seed) if use_parity else None

    generated_ids: List[int] = []
    previous_text = ""
    wall_started = time.perf_counter()
    model_seconds = 0.0

    with bundle.lock:
        llm = bundle.llm
        llm.reset()
        started = time.perf_counter()
        llm.eval(ids)
        logits = np.asarray(llm.scores[llm.n_tokens - 1], dtype=np.float32).copy()
        model_seconds += time.perf_counter() - started

        for _ in range(max_new):
            if use_parity:
                adj = logits.copy()
                _apply_repetition_penalty(
                    adj, ids + generated_ids, float(repetition_penalty)
                )
                token_id = _sample_next_token_np(
                    adj, sampling, temperature, top_k, top_p, rng
                )
            else:
                token_id = _native_next_token(
                    llm,
                    logits,
                    ids + generated_ids,
                    sampling,
                    temperature,
                    top_k,
                    top_p,
                    repetition_penalty,
                    rng,
                )
            if token_id in stop_ids:
                break

            generated_ids.append(int(token_id))
            started = time.perf_counter()
            logits = _logits_for_last_token(llm, int(token_id))
            model_seconds += time.perf_counter() - started

            raw_text = bundle.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )
            clean_text = _clean_reply(raw_text, bundle.spec.prompt_mode)
            if clean_text != previous_text:
                previous_text = clean_text
                yield clean_text

            if bundle.spec.prompt_mode == "chat_role" and any(
                marker in raw_text for marker in _CHAT_ROLE_MARKERS
            ):
                break

    wall_seconds = max(time.perf_counter() - wall_started, 1e-6)
    seconds = max(model_seconds, 1e-6)
    if stats is not None:
        stats.update(
            {
                "text": previous_text,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": len(generated_ids),
                "seconds": seconds,
                "wall_seconds": wall_seconds,
                "tokens_per_second": len(generated_ids) / seconds,
                "truncated": truncated,
            }
        )


def _native_next_token(
    llm,
    logits: np.ndarray,
    seen_ids: List[int],
    sampling: bool,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    rng: Optional[np.random.Generator],
) -> int:
    """Approximate native path: same math, shared RNG generator.

    llama-cpp-python 0.3.x samples inside `eval`; exposing those internals
    would couple us to private APIs, so the native mode keeps the NumPy
    sampler but is intended for speed tests only (documented in README).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    adj = logits.copy()
    _apply_repetition_penalty(adj, seen_ids, float(repetition_penalty))
    return _sample_next_token_np(adj, sampling, temperature, top_k, top_p, rng)
