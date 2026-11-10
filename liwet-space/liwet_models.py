from __future__ import annotations

import gc
import json
import math
import os
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn.functional as F
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError
from safetensors.torch import load_file
from tokenizers import Tokenizer
from torch import nn

OWNER = "fadhilahkhairogi"
DEFAULT_SYSTEM_PROMPT = (
    "Anda adalah asisten AI berbahasa Indonesia yang membantu, ramah, dan ringkas."
)

torch.set_float32_matmul_precision("high")


def _use_fastpath() -> bool:
    return os.environ.get("LIWET_FASTPATH", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _supports_sdpa_enable_gqa() -> bool:
    try:
        ver_parts = torch.__version__.split("+")[0].split(".")
        return (int(ver_parts[0]), int(ver_parts[1])) >= (2, 5)
    except Exception:
        return False


# Parity note: SDPA's enable_gqa=True dispatches to a different kernel than the
# reference path (which expands KV heads with repeat_interleave). The kernel swap
# changes bf16 rounding enough to flip near-tie router/argmax decisions, so the
# fast path keeps repeat_interleave for bitwise-identical output. Opt back in with
# LIWET_SDPA_GQA=1 only for non-parity experiments.
_SDPA_ENABLE_GQA = _supports_sdpa_enable_gqa() and os.environ.get(
    "LIWET_SDPA_GQA", "0"
).strip().lower() in {"1", "true", "yes", "on"}


def _sdpa_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    num_kv_groups: int = 1,
) -> torch.Tensor:
    if num_kv_groups > 1:
        if _SDPA_ENABLE_GQA:
            return F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=is_causal, enable_gqa=True
            )
        k = k.repeat_interleave(num_kv_groups, dim=1)
        v = v.repeat_interleave(num_kv_groups, dim=1)
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=is_causal
    )


class StaticKVCache:
    """Pre-allocated static KV-cache and mask buffers for zero-allocation / CUDA Graph decode."""

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        max_seq_len: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.max_seq_len = max_seq_len
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        self.k_cache = torch.zeros(
            (num_layers, 1, num_kv_heads, max_seq_len, head_dim),
            dtype=dtype,
            device=device,
        )
        self.v_cache = torch.zeros(
            (num_layers, 1, num_kv_heads, max_seq_len, head_dim),
            dtype=dtype,
            device=device,
        )
        self.attn_mask = torch.full(
            (1, 1, 1, max_seq_len),
            float("-inf"),
            dtype=dtype,
            device=device,
        )
        self.pos_arange = torch.arange(max_seq_len, dtype=torch.long, device=device)

    def reset_for_prefill(self, prompt_len: int) -> None:
        self.attn_mask.fill_(float("-inf"))
        if prompt_len > 0:
            self.attn_mask[..., :prompt_len].zero_()

    def open_position(self, pos: int) -> torch.Tensor:
        self.attn_mask[..., pos : pos + 1].zero_()
        return self.pos_arange[pos : pos + 1]


class CudaGraphDecodeRunner:
    """Captures single-token decode into a CUDA Graph to eliminate Python launch overhead."""

    def __init__(self, model: nn.Module, cache: StaticKVCache, vocab_size: int) -> None:
        self.model = model
        self.cache = cache
        self.device = cache.device
        self.token_buf = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        self.pos_buf = torch.zeros((1,), dtype=torch.long, device=self.device)
        self.logits_buf = torch.empty((1, vocab_size), dtype=torch.float32, device=self.device)
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self._capture()

    def _capture(self) -> None:
        if hasattr(self.model, "prepare_static_decode"):
            self.model.prepare_static_decode(self.device, self.cache.dtype)
        # Ensure at least position 0 is unmasked during warmup/capture so softmax is valid
        self.cache.reset_for_prefill(1)
        self.pos_buf.zero_()
        self.token_buf.zero_()

        side_stream = torch.cuda.Stream(device=self.device)
        side_stream.wait_stream(torch.cuda.current_stream(device=self.device))
        with torch.cuda.stream(side_stream), torch.inference_mode():
            for _ in range(3):
                _ = self.model.forward_decode_static(self.token_buf, self.pos_buf, self.cache)
        torch.cuda.current_stream(device=self.device).wait_stream(side_stream)

        g = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(g):
            out = self.model.forward_decode_static(self.token_buf, self.pos_buf, self.cache)
            self.logits_buf.copy_(out)
        self.graph = g

    def step(self, token_1x1: torch.Tensor, pos: int) -> torch.Tensor:
        self.token_buf.copy_(token_1x1)
        self.pos_buf.copy_(self.cache.pos_arange[pos : pos + 1])
        self.cache.attn_mask[..., pos : pos + 1].zero_()
        self.graph.replay()
        return self.logits_buf


class ModelUnavailableError(RuntimeError):
    """Raised when a model repository has not yet been published on Hugging Face Hub."""


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str
    label: str
    scale: str
    arch: str
    stage: str
    prompt_mode: str
    total_params_label: str
    active_params_label: str
    context_len: int
    recommended_params: dict
    description: str


LIWET_CATALOG: List[ModelSpec] = [
    ModelSpec(
        repo_id=f"{OWNER}/Liwet-163M-id-chat",
        label="Liwet-163M Chat (SimPO · Terbaik)",
        scale="163M Dense GQA",
        arch="dense_163m",
        stage="SimPO Chat",
        prompt_mode="chat_role",
        total_params_label="163.0M",
        active_params_label="163.0M",
        context_len=2048,
        recommended_params={
            "max_new_tokens": 192,
            "temperature": 0.55,
            "top_k": 40,
            "top_p": 0.9,
            "repetition_penalty": 1.15,
            "do_sample": True,
        },
        description="Model utama 163M Dense GQA hasil alignment SimPO dari checkpoint SFT.",
    ),
    ModelSpec(
        repo_id=f"{OWNER}/Liwet-163M-id-instruct",
        label="Liwet-163M Instruct (SFT)",
        scale="163M Dense GQA",
        arch="dense_163m",
        stage="SFT Instruct",
        prompt_mode="chat_role",
        total_params_label="163.0M",
        active_params_label="163.0M",
        context_len=2048,
        recommended_params={
            "max_new_tokens": 192,
            "temperature": 0.65,
            "top_k": 40,
            "top_p": 0.9,
            "repetition_penalty": 1.12,
            "do_sample": True,
        },
        description="Model 163M Dense GQA hasil Supervised Fine-Tuning (165.6k percakapan).",
    ),
    ModelSpec(
        repo_id=f"{OWNER}/Liwet-163M-id-base",
        label="Liwet-163M Base (Pretrain)",
        scale="163M Dense GQA",
        arch="dense_163m",
        stage="Pretrain Base",
        prompt_mode="raw_bos",
        total_params_label="163.0M",
        active_params_label="163.0M",
        context_len=2048,
        recommended_params={
            "max_new_tokens": 128,
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "do_sample": True,
        },
        description="Model fondasi 163M Dense GQA hasil pretraining 2.10B token Bahasa Indonesia.",
    ),
    ModelSpec(
        repo_id=f"{OWNER}/Liwet-66M-id-chat",
        label="Liwet-66M Chat (Conversational SFT)",
        scale="66.7M MiniTransformer",
        arch="mini_66m",
        stage="Chat SFT",
        prompt_mode="chat_tags",
        total_params_label="66.7M",
        active_params_label="66.7M",
        context_len=1024,
        recommended_params={
            "max_new_tokens": 160,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.9,
            "repetition_penalty": 1.15,
            "do_sample": True,
        },
        description="Model chat ringan 66.7M dengan token <|user|>/<|assistant|>/<|end|>.",
    ),
    ModelSpec(
        repo_id=f"{OWNER}/Liwet-66M-id-instruct-v2",
        label="Liwet-66M Instruct v2 (2-Epoch Base)",
        scale="66.7M MiniTransformer",
        arch="mini_66m",
        stage="SFT Instruct (v2)",
        prompt_mode="instruct_plain",
        total_params_label="66.7M",
        active_params_label="66.7M",
        context_len=1024,
        recommended_params={
            "max_new_tokens": 160,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "do_sample": True,
        },
        description="Model SFT instruksi 66.7M dari pretraining 2 epoch (~852M token).",
    ),
    ModelSpec(
        repo_id=f"{OWNER}/Liwet-66M-id-instruct-v1",
        label="Liwet-66M Instruct v1 (1-Epoch Base)",
        scale="66.7M MiniTransformer",
        arch="mini_66m",
        stage="SFT Instruct (v1)",
        prompt_mode="instruct_plain",
        total_params_label="66.7M",
        active_params_label="66.7M",
        context_len=1024,
        recommended_params={
            "max_new_tokens": 160,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "do_sample": True,
        },
        description="Model SFT instruksi 66.7M dari pretraining 1 epoch (~428M token).",
    ),
    ModelSpec(
        repo_id=f"{OWNER}/Liwet-66M-id-base-v2",
        label="Liwet-66M Base v2 (Pretrain 2-Epoch)",
        scale="66.7M MiniTransformer",
        arch="mini_66m",
        stage="Pretrain Base (v2)",
        prompt_mode="raw",
        total_params_label="66.7M",
        active_params_label="66.7M",
        context_len=1024,
        recommended_params={
            "max_new_tokens": 128,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "do_sample": True,
        },
        description="Base model 66.7M hasil pretraining 2 epoch (~852M token).",
    ),
    ModelSpec(
        repo_id=f"{OWNER}/Liwet-66M-id-base-v1",
        label="Liwet-66M Base v1 (Pretrain 1-Epoch)",
        scale="66.7M MiniTransformer",
        arch="mini_66m",
        stage="Pretrain Base (v1)",
        prompt_mode="raw",
        total_params_label="66.7M",
        active_params_label="66.7M",
        context_len=1024,
        recommended_params={
            "max_new_tokens": 128,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "do_sample": True,
        },
        description="Base model 66.7M hasil pretraining 1 epoch (~428M token).",
    ),
    ModelSpec(
        repo_id=f"{OWNER}/Liwet-400M-A188M-id-chat",
        label="Liwet-400M-A188M MoE Chat (SimPO)",
        scale="400.1M MoE (187.8M Active)",
        arch="moe_400m",
        stage="SimPO Chat",
        prompt_mode="chat_role",
        total_params_label="400.1M",
        active_params_label="187.8M",
        context_len=2048,
        recommended_params={
            "max_new_tokens": 192,
            "temperature": 0.65,
            "top_k": 50,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "do_sample": True,
        },
        description="Model sparse Hybrid MoE (32L, 8 expert top-2) setelah alignment SimPO.",
    ),
    ModelSpec(
        repo_id=f"{OWNER}/Liwet-400M-A188M-id-instruct",
        label="Liwet-400M-A188M MoE Instruct (SFT)",
        scale="400.1M MoE (187.8M Active)",
        arch="moe_400m",
        stage="SFT Instruct",
        prompt_mode="chat_role",
        total_params_label="400.1M",
        active_params_label="187.8M",
        context_len=2048,
        recommended_params={
            "max_new_tokens": 192,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "do_sample": True,
        },
        description="Model sparse Hybrid MoE setelah Supervised Fine-Tuning.",
    ),
    ModelSpec(
        repo_id=f"{OWNER}/Liwet-400M-A188M-id-base",
        label="Liwet-400M-A188M MoE Base (Pretrain)",
        scale="400.1M MoE (187.8M Active)",
        arch="moe_400m",
        stage="Pretrain Base",
        prompt_mode="raw_bos",
        total_params_label="400.1M",
        active_params_label="187.8M",
        context_len=2048,
        recommended_params={
            "max_new_tokens": 128,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "do_sample": True,
        },
        description="Model fondasi sparse Hybrid MoE (400.1M total / 187.8M aktif per token).",
    ),
]

SPEC_BY_REPO: Dict[str, ModelSpec] = {spec.repo_id: spec for spec in LIWET_CATALOG}
LIST_REPOS: List[str] = [spec.repo_id for spec in LIWET_CATALOG]


def get_model_spec(repo_id: str) -> ModelSpec:
    if repo_id in SPEC_BY_REPO:
        return SPEC_BY_REPO[repo_id]
    return ModelSpec(
        repo_id=repo_id,
        label=repo_id,
        scale="Custom",
        arch="auto",
        stage="Custom",
        prompt_mode="chat_role",
        total_params_label="-",
        active_params_label="-",
        context_len=2048,
        recommended_params={
            "max_new_tokens": 160,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "do_sample": True,
        },
        description=f"Custom Hugging Face Hub repository: {repo_id}",
    )


_THIS_DIR = Path(__file__).resolve().parent
_WORKSPACE_ROOT = (
    _THIS_DIR.parent.parent if _THIS_DIR.name == "web" else _THIS_DIR.parent
)
_STAGE_CHECKPOINT_SUBDIRS: Dict[str, Tuple[str, str]] = {
    "Liwet-400M-A188M-id-chat": ("simpo", "final"),
    "Liwet-400M-A188M-id-instruct": ("sft", "final"),
    "Liwet-400M-A188M-id-base": ("pretrain", "final"),
}


def _local_candidate_dirs(repo_id: str) -> List[Path]:
    repo_name = repo_id.split("/")[-1]
    candidates: List[Path] = []
    env_root = os.environ.get("LIWET_LOCAL_MODELS_DIR", "").strip()
    if env_root:
        candidates.append(Path(env_root) / repo_name)
    stage_info = _STAGE_CHECKPOINT_SUBDIRS.get(repo_name)
    if stage_info is not None:
        stage, sub = stage_info
        candidates.append(
            _WORKSPACE_ROOT / "TUBES_1_400M_MOE" / "checkpoints" / stage / sub
        )
    candidates.append(_WORKSPACE_ROOT / "TUBES_1_SAFETENSOR" / repo_name)
    return candidates


def _find_local_model_files(repo_id: str) -> Optional[Tuple[Path, Path, Path]]:
    fallback_tok = (
        _WORKSPACE_ROOT
        / "TUBES_1_400M_MOE"
        / "data"
        / "tokenizer"
        / "tokenizer.json"
    )
    for cdir in _local_candidate_dirs(repo_id):
        if not cdir.is_dir():
            continue
        cfg_file = cdir / "config.json"
        if not cfg_file.exists():
            continue
        weights_file = cdir / "model.safetensors"
        if not weights_file.exists():
            weights_file = cdir / "pytorch_model.bin"
        if not weights_file.exists():
            continue
        tok_file = cdir / "tokenizer.json"
        if not tok_file.exists() and fallback_tok.exists():
            tok_file = fallback_tok
        if tok_file.exists():
            return cfg_file, tok_file, weights_file
    return None


def _is_repo_on_hub(repo_id: str, timeout_s: float = 3.0) -> bool:
    api = HfApi()
    try:
        info = api.repo_info(repo_id=repo_id, repo_type="model", timeout=timeout_s)
        siblings = {s.rfilename for s in (info.siblings or [])}
        has_cfg = "config.json" in siblings
        has_weights = any(
            name.endswith(".safetensors") or name == "pytorch_model.bin"
            for name in siblings
        )
        return has_cfg and has_weights
    except Exception:
        return False


def check_repo_availability(repo_id: str, timeout_s: float = 3.0) -> bool:
    if _is_repo_on_hub(repo_id, timeout_s=timeout_s):
        return True
    return _find_local_model_files(repo_id) is not None


def _rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    if hasattr(F, "rms_norm"):
        try:
            return F.rms_norm(hidden_states, (weight.numel(),), weight, eps)
        except (RuntimeError, NotImplementedError):
            pass
    normed = hidden_states.float()
    variance = normed.pow(2).mean(-1, keepdim=True)
    normed = normed * torch.rsqrt(variance + eps)
    return (weight.float() * normed).to(input_dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# ---------------------------------------------------------------------------
# Family 1: Liwet-66M (MiniTransformer with KV-cache decode support)
# ---------------------------------------------------------------------------


@dataclass
class MiniGPTConfig:
    vocab_size: int = 32000
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 8
    n_embd: int = 512
    mlp_hidden: int = 2048
    dropout: float = 0.0
    bias: bool = False
    rope_base: float = 10000.0
    tie_weights: bool = True

    @property
    def max_position_embeddings(self) -> int:
        return self.block_size

    @classmethod
    def from_dict(cls, data: dict) -> "MiniGPTConfig":
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid})


class MiniRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_f = x.float()
        x_f = x_f * torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (self.weight.float() * x_f).to(dtype)


def _build_mini_rope(seq_len: int, head_dim: int, base: float, device, dtype):
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    freqs = torch.outer(torch.arange(seq_len, device=device, dtype=torch.float32), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


class MiniCausalSelfAttention(nn.Module):
    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=2)
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)

        past_len = past_key_value[0].shape[2] if past_key_value is not None else 0
        cos_slice = cos[past_len : past_len + t].unsqueeze(0).unsqueeze(0).to(q.dtype)
        sin_slice = sin[past_len : past_len + t].unsqueeze(0).unsqueeze(0).to(q.dtype)
        q = (q * cos_slice) + (_rotate_half(q) * sin_slice)
        k = (k * cos_slice) + (_rotate_half(k) * sin_slice)

        if past_key_value is not None:
            k = torch.cat((past_key_value[0], k), dim=2)
            v = torch.cat((past_key_value[1], v), dim=2)

        present = (k, v) if use_cache else None
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=(t > 1),
        )
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(y), present


class MiniSwiGLU(nn.Module):
    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        hidden = config.mlp_hidden
        self.w1 = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.w3 = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.w2 = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class MiniBlock(nn.Module):
    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        self.ln1 = MiniRMSNorm(config.n_embd)
        self.attn = MiniCausalSelfAttention(config)
        self.ln2 = MiniRMSNorm(config.n_embd)
        self.mlp = MiniSwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        attn_out, present = self.attn(
            self.ln1(x), cos, sin, past_key_value=past_key_value, use_cache=use_cache
        )
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, present


class MiniTransformer(nn.Module):
    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([MiniBlock(config) for _ in range(config.n_layer)])
        self.ln_f = MiniRMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        cos, sin = _build_mini_rope(
            config.block_size,
            config.n_embd // config.n_head,
            config.rope_base,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        if config.tie_weights:
            self.lm_head.weight = self.tok_emb.weight
        self._rope_device = None

    def _get_rope(self, device, dtype):
        if self._rope_device != (device, dtype):
            self.rope_cos = self.rope_cos.to(device=device, dtype=dtype)
            self.rope_sin = self.rope_sin.to(device=device, dtype=dtype)
            self._rope_device = (device, dtype)
        return self.rope_cos, self.rope_sin

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        use_cache: bool = False,
    ):
        if past_key_values is None and input_ids.shape[1] > self.config.block_size:
            input_ids = input_ids[:, -self.config.block_size :]
        x = self.drop(self.tok_emb(input_ids))
        cos, sin = self._get_rope(input_ids.device, x.dtype)
        presents = [] if use_cache else None
        for idx, block in enumerate(self.blocks):
            layer_past = past_key_values[idx] if past_key_values is not None else None
            x, present = block(x, cos, sin, past_key_value=layer_past, use_cache=use_cache)
            if use_cache:
                presents.append(present)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return CausalLMOutput(
            loss=None,
            logits=logits,
            past_key_values=tuple(presents) if use_cache else None,
            aux_loss=None,
        )

    def create_static_cache(self, device: torch.device, dtype: torch.dtype) -> StaticKVCache:
        return StaticKVCache(
            num_layers=self.config.n_layer,
            num_kv_heads=self.config.n_head,
            max_seq_len=self.config.block_size,
            head_dim=self.config.n_embd // self.config.n_head,
            device=device,
            dtype=dtype,
        )

    def prepare_static_decode(self, device: torch.device, dtype: torch.dtype) -> None:
        self._get_rope(device, dtype)

    def forward_prefill_static(self, input_ids: torch.Tensor, cache: StaticKVCache) -> torch.Tensor:
        if input_ids.shape[1] > self.config.block_size:
            input_ids = input_ids[:, -self.config.block_size :]
        b, seq_len = input_ids.shape
        x = self.drop(self.tok_emb(input_ids))
        cos, sin = self._get_rope(input_ids.device, x.dtype)
        cos_s = cos[:seq_len].unsqueeze(0).unsqueeze(0)
        sin_s = sin[:seq_len].unsqueeze(0).unsqueeze(0)
        cache.reset_for_prefill(seq_len)
        for idx, block in enumerate(self.blocks):
            h = block.ln1(x)
            attn = block.attn
            c = attn.n_head * attn.head_dim
            q, k, v = attn.qkv(h).split(c, dim=2)
            q = q.view(b, seq_len, attn.n_head, attn.head_dim).transpose(1, 2)
            k = k.view(b, seq_len, attn.n_head, attn.head_dim).transpose(1, 2)
            v = v.view(b, seq_len, attn.n_head, attn.head_dim).transpose(1, 2)
            q = (q * cos_s) + (_rotate_half(q) * sin_s)
            k = (k * cos_s) + (_rotate_half(k) * sin_s)
            cache.k_cache[idx, :, :, :seq_len, :].copy_(k)
            cache.v_cache[idx, :, :, :seq_len, :].copy_(v)
            y = _sdpa_attn(q, k, v, attn_mask=None, is_causal=(seq_len > 1), num_kv_groups=1)
            x = x + attn.proj(y.transpose(1, 2).contiguous().view(b, seq_len, c))
            x = x + block.mlp(block.ln2(x))
        return self.lm_head(self.ln_f(x[:, -1:, :]))[:, 0, :].float()

    def forward_decode_static(
        self, input_ids: torch.Tensor, pos_buf: torch.Tensor, cache: StaticKVCache
    ) -> torch.Tensor:
        b = input_ids.shape[0]
        x = self.drop(self.tok_emb(input_ids))
        cos, sin = self._get_rope(input_ids.device, x.dtype)
        cos_s = cos[pos_buf].unsqueeze(0).unsqueeze(0)
        sin_s = sin[pos_buf].unsqueeze(0).unsqueeze(0)
        for idx, block in enumerate(self.blocks):
            h = block.ln1(x)
            attn = block.attn
            c = attn.n_head * attn.head_dim
            q, k, v = attn.qkv(h).split(c, dim=2)
            q = q.view(b, 1, attn.n_head, attn.head_dim).transpose(1, 2)
            k = k.view(b, 1, attn.n_head, attn.head_dim).transpose(1, 2)
            v = v.view(b, 1, attn.n_head, attn.head_dim).transpose(1, 2)
            q = (q * cos_s) + (_rotate_half(q) * sin_s)
            k = (k * cos_s) + (_rotate_half(k) * sin_s)
            cache.k_cache[idx].index_copy_(2, pos_buf, k)
            cache.v_cache[idx].index_copy_(2, pos_buf, v)
            y = _sdpa_attn(
                q,
                cache.k_cache[idx],
                cache.v_cache[idx],
                attn_mask=cache.attn_mask,
                is_causal=False,
                num_kv_groups=1,
            )
            x = x + attn.proj(y.transpose(1, 2).contiguous().view(b, 1, c))
            x = x + block.mlp(block.ln2(x))
        return self.lm_head(self.ln_f(x))[:, 0, :].float()


# ---------------------------------------------------------------------------
# Family 2 & 3: Dense 163M (GQA + qkv_proj) & MoE 400M-A188M (GQA + q/k/v_proj)
# ---------------------------------------------------------------------------


@dataclass
class CausalLMOutput:
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None
    aux_loss: Optional[torch.Tensor] = None


@dataclass
class LiwetLMConfig:
    vocab_size: int = 32000
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 16
    num_attention_heads: int = 12
    num_key_value_heads: int = 4
    head_dim: int = 0
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    hidden_act: str = "silu"
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True
    use_moe: bool = False
    num_experts: int = 8
    num_experts_per_tok: int = 2
    moe_intermediate_size: int = 360
    moe_capacity_factor: float = 1.25
    router_aux_loss_coef: float = 0.01
    router_z_loss_coef: float = 0.001
    router_jitter_noise: float = 0.0
    router_bias_update_speed: float = 0.001
    router_load_decay: float = 0.01
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0
    unk_token_id: int = 3
    model_type: str = "indonesian_chat_lm"
    architectures: list = field(default_factory=lambda: ["IndonesianChatLM"])

    def __post_init__(self) -> None:
        if not self.head_dim or self.head_dim <= 0:
            self.head_dim = self.hidden_size // self.num_attention_heads

    @classmethod
    def from_dict(cls, data: dict) -> "LiwetLMConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return _rms_norm(hidden_states, self.weight, self.eps)


class DenseRotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_position_embeddings: int, theta: float) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.register_buffer(
            "inv_freq",
            1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim)),
            persistent=False,
        )
        self._build_cache(max_position_embeddings, torch.device("cpu"))

    def _build_cache(self, seq_len: int, device: torch.device) -> None:
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq.to(device=device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self.max_seq_len_cached = seq_len

    def ensure_size(self, size: int, device: torch.device) -> None:
        if size > self.max_seq_len_cached:
            self._build_cache(int(size), device)

    def forward(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[position_ids], self.sin_cached[position_ids]


class DenseAttention163M(nn.Module):
    def __init__(self, config: LiwetLMConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.q_dim = self.num_heads * self.head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.qkv_proj = nn.Linear(self.hidden_size, self.q_dim + 2 * self.kv_dim, bias=False)
        self.o_proj = nn.Linear(self.q_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        is_causal: bool = False,
    ):
        b, seq_len, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split((self.q_dim, self.kv_dim, self.kv_dim), dim=-1)
        q = q.view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos_d = cos.to(q.dtype).unsqueeze(1)
        sin_d = sin.to(q.dtype).unsqueeze(1)
        q = (q * cos_d) + (_rotate_half(q) * sin_d)
        k = (k * cos_d) + (_rotate_half(k) * sin_d)

        if past_key_value is not None:
            k = torch.cat((past_key_value[0], k), dim=2)
            v = torch.cat((past_key_value[1], v), dim=2)
        present = (k, v) if use_cache else None

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=is_causal
        )
        out = out.transpose(1, 2).reshape(b, seq_len, -1)
        return self.o_proj(out), present


class DenseMLP163M(nn.Module):
    def __init__(self, config: LiwetLMConfig) -> None:
        super().__init__()
        self.gate_up_proj = nn.Linear(
            config.hidden_size, 2 * config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(hidden_states).chunk(2, dim=-1)
        return self.down_proj(self.act_fn(gate) * up)


class DenseDecoderLayer163M(nn.Module):
    def __init__(self, config: LiwetLMConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = DenseAttention163M(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = DenseMLP163M(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        is_causal: bool = False,
    ):
        residual = hidden_states
        h_attn, present = self.self_attn(
            self.input_layernorm(hidden_states),
            cos,
            sin,
            past_key_value=past_key_value,
            use_cache=use_cache,
            is_causal=is_causal,
        )
        hidden_states = residual + h_attn
        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present


class DenseIndonesianChatLM(nn.Module):
    def __init__(self, config: LiwetLMConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = DenseRotaryEmbedding(
            config.head_dim, config.max_position_embeddings, config.rope_theta
        )
        self.layers = nn.ModuleList(
            [DenseDecoderLayer163M(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        use_cache: bool = False,
    ) -> CausalLMOutput:
        batch_size, seq_len = input_ids.shape
        past_len = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        position_ids = (
            torch.arange(past_len, past_len + seq_len, device=input_ids.device, dtype=torch.long)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        self.rotary_emb.ensure_size(past_len + seq_len, input_ids.device)
        cos, sin = self.rotary_emb(position_ids)
        is_causal = past_len == 0 and seq_len > 1
        hidden_states = self.embed_tokens(input_ids)
        presents = [] if use_cache else None
        for idx, layer in enumerate(self.layers):
            layer_past = past_key_values[idx] if past_key_values is not None else None
            hidden_states, present = layer(
                hidden_states, cos, sin, past_key_value=layer_past, use_cache=use_cache, is_causal=is_causal
            )
            if use_cache:
                presents.append(present)
        logits = self.lm_head(self.norm(hidden_states))
        return CausalLMOutput(
            loss=None,
            logits=logits,
            past_key_values=tuple(presents) if use_cache else None,
            aux_loss=None,
        )

    def create_static_cache(self, device: torch.device, dtype: torch.dtype) -> StaticKVCache:
        return StaticKVCache(
            num_layers=self.config.num_hidden_layers,
            num_kv_heads=self.config.num_key_value_heads,
            max_seq_len=self.config.max_position_embeddings,
            head_dim=self.config.head_dim,
            device=device,
            dtype=dtype,
        )

    def prepare_static_decode(self, device: torch.device, dtype: torch.dtype) -> None:
        self.rotary_emb.ensure_size(self.config.max_position_embeddings, device)
        self._rope_d_cache = (
            self.rotary_emb.cos_cached.to(device=device, dtype=dtype),
            self.rotary_emb.sin_cached.to(device=device, dtype=dtype),
        )

    def _get_rope_d(self, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        pair = getattr(self, "_rope_d_cache", None)
        if pair is None or pair[0].device != device or pair[0].dtype != dtype:
            self.prepare_static_decode(device, dtype)
            pair = self._rope_d_cache
        return pair

    def forward_prefill_static(self, input_ids: torch.Tensor, cache: StaticKVCache) -> torch.Tensor:
        b, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)
        cos_t, sin_t = self._get_rope_d(input_ids.device, hidden_states.dtype)
        cos_s = cos_t[:seq_len].unsqueeze(0).unsqueeze(0)
        sin_s = sin_t[:seq_len].unsqueeze(0).unsqueeze(0)
        cache.reset_for_prefill(seq_len)
        is_causal = seq_len > 1
        for idx, layer in enumerate(self.layers):
            residual = hidden_states
            h = layer.input_layernorm(hidden_states)
            attn = layer.self_attn
            q, k, v = attn.qkv_proj(h).split((attn.q_dim, attn.kv_dim, attn.kv_dim), dim=-1)
            q = q.view(b, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)
            k = k.view(b, seq_len, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
            v = v.view(b, seq_len, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
            q = (q * cos_s) + (_rotate_half(q) * sin_s)
            k = (k * cos_s) + (_rotate_half(k) * sin_s)
            cache.k_cache[idx, :, :, :seq_len, :].copy_(k)
            cache.v_cache[idx, :, :, :seq_len, :].copy_(v)
            out = _sdpa_attn(q, k, v, attn_mask=None, is_causal=is_causal, num_kv_groups=attn.num_kv_groups)
            hidden_states = residual + attn.o_proj(out.transpose(1, 2).reshape(b, seq_len, -1))
            hidden_states = hidden_states + layer.mlp(layer.post_attention_layernorm(hidden_states))
        return self.lm_head(self.norm(hidden_states[:, -1:, :]))[:, 0, :].float()

    def forward_decode_static(
        self, input_ids: torch.Tensor, pos_buf: torch.Tensor, cache: StaticKVCache
    ) -> torch.Tensor:
        b = input_ids.shape[0]
        hidden_states = self.embed_tokens(input_ids)
        cos_t, sin_t = self._get_rope_d(input_ids.device, hidden_states.dtype)
        cos_s = cos_t[pos_buf].unsqueeze(0).unsqueeze(0)
        sin_s = sin_t[pos_buf].unsqueeze(0).unsqueeze(0)
        for idx, layer in enumerate(self.layers):
            residual = hidden_states
            h = layer.input_layernorm(hidden_states)
            attn = layer.self_attn
            q, k, v = attn.qkv_proj(h).split((attn.q_dim, attn.kv_dim, attn.kv_dim), dim=-1)
            q = q.view(b, 1, attn.num_heads, attn.head_dim).transpose(1, 2)
            k = k.view(b, 1, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
            v = v.view(b, 1, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
            q = (q * cos_s) + (_rotate_half(q) * sin_s)
            k = (k * cos_s) + (_rotate_half(k) * sin_s)
            cache.k_cache[idx].index_copy_(2, pos_buf, k)
            cache.v_cache[idx].index_copy_(2, pos_buf, v)
            out = _sdpa_attn(
                q,
                cache.k_cache[idx],
                cache.v_cache[idx],
                attn_mask=cache.attn_mask,
                is_causal=False,
                num_kv_groups=attn.num_kv_groups,
            )
            hidden_states = residual + attn.o_proj(out.transpose(1, 2).reshape(b, 1, -1))
            hidden_states = hidden_states + layer.mlp(layer.post_attention_layernorm(hidden_states))
        return self.lm_head(self.norm(hidden_states))[:, 0, :].float()


class MoE(nn.Module):
    def __init__(self, config: LiwetLMConfig) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.capacity_factor = config.moe_capacity_factor
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.register_buffer("expert_bias", torch.zeros(config.num_experts), persistent=True)
        self.register_buffer(
            "load_ema", torch.full((config.num_experts,), 1.0 / config.num_experts), persistent=True
        )
        self.gate_up_proj = nn.Parameter(
            torch.empty(config.num_experts, config.hidden_size, 2 * config.moe_intermediate_size)
        )
        self.down_proj = nn.Parameter(
            torch.empty(config.num_experts, config.moe_intermediate_size, config.hidden_size)
        )

    def _apply(self, fn, *args, **kwargs):
        super()._apply(fn, *args, **kwargs)
        self.expert_bias = self.expert_bias.float()
        self.load_ema = self.load_ema.float()
        return self

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        num_tokens = batch_size * sequence_length
        flat_states = hidden_states.reshape(num_tokens, self.hidden_size)
        router_logits = F.linear(flat_states, self.router.weight).float()
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        selection_logits = router_logits + self.expert_bias.float()
        top_indices = torch.topk(selection_logits, self.top_k, dim=-1).indices
        top_weights = torch.gather(router_probs, dim=-1, index=top_indices)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        if num_tokens == 1 and not getattr(self, "_disable_fast_moe", False):
            experts = top_indices[0]
            weights = top_weights[0].to(flat_states.dtype)
            selected_gate_up = self.gate_up_proj[experts]
            selected_down = self.down_proj[experts]
            expert_input = flat_states.repeat(self.top_k, 1).unsqueeze(1)
            gate_up = torch.bmm(expert_input, selected_gate_up)
            gate, up = gate_up.chunk(2, dim=-1)
            expert_output = torch.bmm(F.silu(gate) * up, selected_down).reshape(
                self.top_k, self.hidden_size
            )
            output = (expert_output * weights.unsqueeze(-1)).sum(dim=0, keepdim=True)
            return output.view(batch_size, sequence_length, self.hidden_size)

        capacity = max(
            1, math.ceil(num_tokens * self.top_k / self.num_experts * self.capacity_factor)
        )
        num_assignments = num_tokens * self.top_k
        flat_indices = top_indices.reshape(-1)
        flat_weights = top_weights.reshape(-1).to(flat_states.dtype)
        order = torch.argsort(flat_indices)
        flat_weights = flat_weights[order]
        sorted_indices = flat_indices[order]
        expert_ids = torch.arange(self.num_experts, device=flat_states.device)
        offsets_end = torch.searchsorted(sorted_indices, expert_ids, right=True)
        offsets = torch.cat([offsets_end.new_zeros(1), offsets_end[:-1]])
        positions = torch.arange(num_assignments, device=flat_states.device) - offsets[sorted_indices]
        valid = positions < capacity
        expert_slots = torch.where(
            valid,
            sorted_indices * capacity + positions,
            sorted_indices.new_full((), self.num_experts * capacity),
        )
        token_indices = order // self.top_k
        expert_input = flat_states.new_zeros(
            self.num_experts * capacity + 1, self.hidden_size
        )
        expert_input[expert_slots] = flat_states[token_indices]
        expert_input = expert_input[: self.num_experts * capacity].view(
            self.num_experts, capacity, self.hidden_size
        )
        gate_up = torch.bmm(expert_input, self.gate_up_proj)
        gate, up = gate_up.chunk(2, dim=-1)
        expert_output = torch.bmm(F.silu(gate) * up, self.down_proj).reshape(
            self.num_experts * capacity, self.hidden_size
        )
        gather_slots = torch.where(valid, expert_slots, expert_slots.new_zeros(()))
        combined = expert_output[gather_slots] * (flat_weights * valid.to(flat_weights.dtype)).unsqueeze(-1)
        output = flat_states.new_zeros(num_tokens, self.hidden_size)
        output = output.index_add_(0, token_indices, combined)
        return output.view(batch_size, sequence_length, self.hidden_size)


class MoEAttention400M(nn.Module):
    def __init__(self, config: LiwetLMConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def _fused_qkv_weight(self) -> torch.Tensor:
        w = self.q_proj.weight
        versions = (w._version, self.k_proj.weight._version, self.v_proj.weight._version)
        cache = getattr(self, "_qkv_weight_cache", None)
        if (
            cache is not None
            and cache[0] == versions
            and cache[1].device == w.device
            and cache[1].dtype == w.dtype
        ):
            return cache[1]
        fused = torch.cat([self.q_proj.weight, self.k_proj.weight, self.v_proj.weight], dim=0)
        self._qkv_weight_cache = (versions, fused)
        return fused

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        b, q_len, _ = hidden_states.shape
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_key_value_heads * self.head_dim
        query, key, value = torch.split(
            F.linear(hidden_states, self._fused_qkv_weight()), [q_dim, kv_dim, kv_dim], dim=-1
        )
        query = query.view(b, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(b, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value = value.view(b, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos_d = cos.to(query.dtype)
        sin_d = sin.to(query.dtype)
        query = (query * cos_d) + (_rotate_half(query) * sin_d)
        key = (key * cos_d) + (_rotate_half(key) * sin_d)

        if past_key_value is not None:
            key = torch.cat((past_key_value[0], key), dim=2)
            value = torch.cat((past_key_value[1], value), dim=2)
        present = (key, value) if use_cache else None

        if self.num_key_value_heads != self.num_heads:
            groups = self.num_heads // self.num_key_value_heads
            key = key.repeat_interleave(groups, dim=1)
            value = value.repeat_interleave(groups, dim=1)

        out = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=(q_len == key.shape[2]),
        )
        out = out.transpose(1, 2).reshape(b, q_len, self.hidden_size)
        return self.o_proj(out), present


class MoESwiGLU(nn.Module):
    def __init__(self, config: LiwetLMConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class MoEDecoderLayer400M(nn.Module):
    def __init__(self, config: LiwetLMConfig) -> None:
        super().__init__()
        self.self_attn = MoEAttention400M(config)
        self.mlp = MoE(config) if config.use_moe else MoESwiGLU(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        residual = hidden_states
        h_attn, present = self.self_attn(
            self.input_layernorm(hidden_states),
            cos,
            sin,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + h_attn
        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present


class MoEIndonesianChatLM(nn.Module):
    def __init__(self, config: LiwetLMConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [MoEDecoderLayer400M(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.register_buffer(
            "_inv_freq",
            1.0
            / (
                config.rope_theta
                ** (torch.arange(0, config.head_dim, 2, dtype=torch.float32) / config.head_dim)
            ),
            persistent=False,
        )

    def _rotary_embeddings(
        self, position_ids: torch.Tensor, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        table = getattr(self, "_rope_table", None)
        size = self.config.max_position_embeddings
        if (
            table is None
            or table["device"] != position_ids.device
            or table["dtype"] != dtype
            or table["size"] < size
        ):
            positions = torch.arange(size, device=position_ids.device, dtype=torch.float32)
            inv_freq = self._inv_freq.to(device=position_ids.device, dtype=torch.float32)
            freqs = torch.cat((positions.unsqueeze(-1) * inv_freq,) * 2, dim=-1)
            table = {
                "device": position_ids.device,
                "dtype": dtype,
                "size": size,
                "cos": freqs.cos().to(dtype),
                "sin": freqs.sin().to(dtype),
            }
            self._rope_table = table
        return table["cos"][position_ids].unsqueeze(1), table["sin"][position_ids].unsqueeze(1)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        use_cache: bool = False,
    ) -> CausalLMOutput:
        batch_size, query_len = input_ids.shape
        past_len = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        position_ids = (
            torch.arange(past_len, past_len + query_len, device=input_ids.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        hidden_states = self.embed_tokens(input_ids)
        cos, sin = self._rotary_embeddings(position_ids, hidden_states.dtype)
        presents = [] if use_cache else None
        for idx, layer in enumerate(self.layers):
            layer_past = past_key_values[idx] if past_key_values is not None else None
            hidden_states, present = layer(
                hidden_states, cos, sin, past_key_value=layer_past, use_cache=use_cache
            )
            if use_cache:
                presents.append(present)
        logits = self.lm_head(self.norm(hidden_states))
        return CausalLMOutput(
            loss=None,
            logits=logits,
            past_key_values=tuple(presents) if use_cache else None,
            aux_loss=None,
        )

    def create_static_cache(self, device: torch.device, dtype: torch.dtype) -> StaticKVCache:
        return StaticKVCache(
            num_layers=self.config.num_hidden_layers,
            num_kv_heads=self.config.num_key_value_heads,
            max_seq_len=self.config.max_position_embeddings,
            head_dim=self.config.head_dim,
            device=device,
            dtype=dtype,
        )

    def prepare_static_decode(self, device: torch.device, dtype: torch.dtype) -> None:
        self._rotary_embeddings(torch.zeros((1, 1), dtype=torch.long, device=device), dtype)
        for layer in self.layers:
            layer.self_attn._static_qkv_w = layer.self_attn._fused_qkv_weight()

    def forward_prefill_static(self, input_ids: torch.Tensor, cache: StaticKVCache) -> torch.Tensor:
        if input_ids.shape[1] > self.config.max_position_embeddings:
            input_ids = input_ids[:, -self.config.max_position_embeddings :]
        b, q_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)
        self._rotary_embeddings(
            torch.zeros((1, 1), dtype=torch.long, device=input_ids.device), hidden_states.dtype
        )
        table = self._rope_table
        cos_d = table["cos"][:q_len].unsqueeze(0).unsqueeze(0)
        sin_d = table["sin"][:q_len].unsqueeze(0).unsqueeze(0)
        cache.reset_for_prefill(q_len)
        is_causal = q_len > 1
        for idx, layer in enumerate(self.layers):
            residual = hidden_states
            h = layer.input_layernorm(hidden_states)
            attn = layer.self_attn
            q_dim = attn.num_heads * attn.head_dim
            kv_dim = attn.num_key_value_heads * attn.head_dim
            qkv_w = getattr(attn, "_static_qkv_w", None)
            if qkv_w is None:
                qkv_w = attn._fused_qkv_weight()
            q, k, v = torch.split(F.linear(h, qkv_w), [q_dim, kv_dim, kv_dim], dim=-1)
            q = q.view(b, q_len, attn.num_heads, attn.head_dim).transpose(1, 2)
            k = k.view(b, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
            v = v.view(b, q_len, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
            q = (q * cos_d) + (_rotate_half(q) * sin_d)
            k = (k * cos_d) + (_rotate_half(k) * sin_d)
            cache.k_cache[idx, :, :, :q_len, :].copy_(k)
            cache.v_cache[idx, :, :, :q_len, :].copy_(v)
            groups = attn.num_heads // attn.num_key_value_heads
            out = _sdpa_attn(q, k, v, attn_mask=None, is_causal=is_causal, num_kv_groups=groups)
            hidden_states = residual + attn.o_proj(out.transpose(1, 2).reshape(b, q_len, attn.hidden_size))
            hidden_states = hidden_states + layer.mlp(layer.post_attention_layernorm(hidden_states))
        return self.lm_head(self.norm(hidden_states[:, -1:, :]))[:, 0, :].float()

    def forward_decode_static(
        self, input_ids: torch.Tensor, pos_buf: torch.Tensor, cache: StaticKVCache
    ) -> torch.Tensor:
        b = input_ids.shape[0]
        hidden_states = self.embed_tokens(input_ids)
        table = self._rope_table
        cos_d = table["cos"][pos_buf].unsqueeze(0).unsqueeze(0)
        sin_d = table["sin"][pos_buf].unsqueeze(0).unsqueeze(0)
        for idx, layer in enumerate(self.layers):
            residual = hidden_states
            h = layer.input_layernorm(hidden_states)
            attn = layer.self_attn
            q_dim = attn.num_heads * attn.head_dim
            kv_dim = attn.num_key_value_heads * attn.head_dim
            qkv_w = attn._static_qkv_w
            q, k, v = torch.split(F.linear(h, qkv_w), [q_dim, kv_dim, kv_dim], dim=-1)
            q = q.view(b, 1, attn.num_heads, attn.head_dim).transpose(1, 2)
            k = k.view(b, 1, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
            v = v.view(b, 1, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
            q = (q * cos_d) + (_rotate_half(q) * sin_d)
            k = (k * cos_d) + (_rotate_half(k) * sin_d)
            cache.k_cache[idx].index_copy_(2, pos_buf, k)
            cache.v_cache[idx].index_copy_(2, pos_buf, v)
            groups = attn.num_heads // attn.num_key_value_heads
            out = _sdpa_attn(
                q,
                cache.k_cache[idx],
                cache.v_cache[idx],
                attn_mask=cache.attn_mask,
                is_causal=False,
                num_kv_groups=groups,
            )
            hidden_states = residual + attn.o_proj(out.transpose(1, 2).reshape(b, 1, attn.hidden_size))
            hidden_states = hidden_states + layer.mlp(layer.post_attention_layernorm(hidden_states))
        return self.lm_head(self.norm(hidden_states))[:, 0, :].float()


# ---------------------------------------------------------------------------
# Unified Loading, Caching, Formatting & Streaming
# ---------------------------------------------------------------------------


@dataclass
class LiwetBundle:
    repo_id: str
    spec: ModelSpec
    model: nn.Module
    tokenizer: Tokenizer
    device: torch.device
    dtype: torch.dtype
    total_params: int
    active_params: int
    context_len: int
    lock: threading.Lock
    static_cache: Optional[StaticKVCache] = None
    decode_runner: Optional[CudaGraphDecodeRunner] = None


_BUNDLE_CACHE: "OrderedDict[Tuple[str, str, str], LiwetBundle]" = OrderedDict()
_CACHE_LOCK = threading.Lock()
MAX_CACHED_BUNDLES = 2
_CPU_THREADS_CONFIGURED = False


def _physical_cpu_count() -> int:
    env_val = os.environ.get("LIWET_CPU_THREADS", "").strip()
    if env_val.isdigit() and int(env_val) > 0:
        return int(env_val)
    try:
        import psutil  # type: ignore

        phys = psutil.cpu_count(logical=False)
        if phys and phys > 0:
            return int(phys)
    except Exception:
        pass
    try:
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            pairs = set()
            phys_id = "0"
            for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith("physical id"):
                    phys_id = line.split(":", 1)[1].strip()
                elif line.startswith("core id"):
                    pairs.add((phys_id, line.split(":", 1)[1].strip()))
            if pairs:
                return len(pairs)
    except Exception:
        pass
    logical = os.cpu_count() or 4
    return max(1, min(8, logical if logical <= 4 else max(4, (logical * 2) // 3)))


def _configure_cpu_threads() -> None:
    global _CPU_THREADS_CONFIGURED
    if _CPU_THREADS_CONFIGURED or not _use_fastpath():
        return
    threads = _physical_cpu_count()
    if torch.get_num_threads() != threads:
        torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    _CPU_THREADS_CONFIGURED = True


def resolve_device(choice: str = "auto") -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def resolve_dtype(choice: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if choice == "float32":
        return torch.float32
    if choice == "float16":
        return torch.float16
    if choice == "bfloat16":
        return torch.bfloat16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _count_params(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    expert_total = 0
    expert_active = 0
    for mod in model.modules():
        if isinstance(mod, MoE):
            per_expert = (mod.gate_up_proj.numel() + mod.down_proj.numel()) // mod.num_experts
            expert_total += per_expert * mod.num_experts
            expert_active += per_expert * mod.top_k
    return total, total - expert_total + expert_active


def _download_model_files(repo_id: str, token: Optional[str] = None) -> Tuple[Path, Path, Path]:
    try:
        cfg_file = Path(
            hf_hub_download(repo_id=repo_id, filename="config.json", repo_type="model", token=token)
        )
        tok_file = Path(
            hf_hub_download(
                repo_id=repo_id, filename="tokenizer.json", repo_type="model", token=token
            )
        )
        try:
            weights_file = Path(
                hf_hub_download(
                    repo_id=repo_id, filename="model.safetensors", repo_type="model", token=token
                )
            )
        except EntryNotFoundError:
            weights_file = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename="pytorch_model.bin",
                    repo_type="model",
                    token=token,
                )
            )
        return cfg_file, tok_file, weights_file
    except Exception as err:
        local_files = _find_local_model_files(repo_id)
        if local_files is not None:
            return local_files
        if isinstance(err, (RepositoryNotFoundError, EntryNotFoundError)):
            raise ModelUnavailableError(
                f"Repositori '{repo_id}' belum tersedia di Hugging Face Hub maupun folder lokal."
            ) from err
        raise ModelUnavailableError(
            f"Gagal mengunduh berkas model '{repo_id}' ({type(err).__name__}: {err})."
        ) from err


def _load_state_dict(weights_file: Path) -> dict:
    if weights_file.suffix == ".safetensors":
        return load_file(str(weights_file), device="cpu")
    sd = torch.load(str(weights_file), map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    return sd


def _instantiate_model(cfg_dict: dict, state_dict: dict) -> Tuple[nn.Module, int]:
    arch = cfg_dict.get("architectures", [])
    is_mini = cfg_dict.get("model_type") == "mini_transformer" or "MiniTransformer" in arch
    if is_mini:
        cfg = MiniGPTConfig.from_dict(cfg_dict)
        model = MiniTransformer(cfg)
        if cfg.tie_weights and "lm_head.weight" not in state_dict and "tok_emb.weight" in state_dict:
            state_dict["lm_head.weight"] = state_dict["tok_emb.weight"]
        model.load_state_dict(state_dict, strict=True)
        return model, cfg.block_size

    cfg = LiwetLMConfig.from_dict(cfg_dict)
    has_qkv = any("qkv_proj.weight" in k for k in state_dict)
    if state_dict and all(k.startswith("model.") for k in state_dict):
        state_dict = {k[len("model.") :]: v for k, v in state_dict.items()}
    if (
        cfg.tie_word_embeddings
        and "lm_head.weight" not in state_dict
        and "embed_tokens.weight" in state_dict
    ):
        state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"]

    if has_qkv and not cfg.use_moe:
        model = DenseIndonesianChatLM(cfg)
    else:
        model = MoEIndonesianChatLM(cfg)
    model.load_state_dict(state_dict, strict=True)
    return model, cfg.max_position_embeddings


def load_liwet_model(
    repo_id: str,
    device_name: str = "auto",
    dtype_name: str = "auto",
    token: Optional[str] = None,
) -> LiwetBundle:
    device = resolve_device(device_name)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        _configure_cpu_threads()
    dtype = resolve_dtype(dtype_name, device)
    cache_key = (repo_id, str(device), str(dtype))

    with _CACHE_LOCK:
        if cache_key in _BUNDLE_CACHE:
            _BUNDLE_CACHE.move_to_end(cache_key)
            return _BUNDLE_CACHE[cache_key]

    cfg_file, tok_file, weights_file = _download_model_files(repo_id, token=token)
    cfg_dict = json.loads(cfg_file.read_text(encoding="utf-8-sig"))
    state_dict = _load_state_dict(weights_file)
    model, context_len = _instantiate_model(cfg_dict, state_dict)
    del state_dict
    model = model.to(device=device, dtype=dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    tokenizer = Tokenizer.from_file(str(tok_file))
    total_params, active_params = _count_params(model)
    bundle = LiwetBundle(
        repo_id=repo_id,
        spec=get_model_spec(repo_id),
        model=model,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        total_params=total_params,
        active_params=active_params,
        context_len=context_len,
        lock=threading.Lock(),
    )

    with _CACHE_LOCK:
        _BUNDLE_CACHE[cache_key] = bundle
        _BUNDLE_CACHE.move_to_end(cache_key)
        while len(_BUNDLE_CACHE) > MAX_CACHED_BUNDLES:
            _, evicted = _BUNDLE_CACHE.popitem(last=False)
            evicted.decode_runner = None
            evicted.static_cache = None
            del evicted
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return bundle


def clear_model_cache() -> None:
    with _CACHE_LOCK:
        for b in _BUNDLE_CACHE.values():
            b.decode_runner = None
            b.static_cache = None
        _BUNDLE_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_prompt_from_messages(
    spec: ModelSpec,
    messages: List[dict],
    system_prompt: str = "",
    use_chat_template: bool = True,
) -> str:
    clean_msgs = [
        {"role": m["role"], "content": str(m["content"]).strip()}
        for m in messages
        if str(m.get("content", "")).strip()
    ]
    if not clean_msgs:
        return ""

    if not use_chat_template or spec.prompt_mode in {"raw", "raw_bos"}:
        return clean_msgs[-1]["content"]

    if spec.prompt_mode == "instruct_plain":
        return f"{clean_msgs[-1]['content']}\n\n"

    if spec.prompt_mode == "chat_tags":
        parts = []
        sys_text = system_prompt.strip()
        first_user_done = False
        for m in clean_msgs:
            role = m["role"]
            content = m["content"]
            if role == "user":
                if sys_text and not first_user_done:
                    content = f"{sys_text}\n\n{content}"
                first_user_done = True
                parts.append(f"<|user|>\n{content}\n")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}\n<|end|>\n")
        parts.append("<|assistant|>\n")
        return "".join(parts)

    # Default: role-based format (163M & 400M)
    parts = []
    if system_prompt.strip():
        parts.append(f"System: {system_prompt.strip()}")
    role_map = {"system": "System", "user": "User", "assistant": "Assistant"}
    for m in clean_msgs:
        parts.append(f"{role_map.get(m['role'], 'User')}: {m['content']}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


def _encode_prompt(bundle: LiwetBundle, prompt: str, max_new_tokens: int) -> Tuple[List[int], bool]:
    mode = bundle.spec.prompt_mode
    ids = list(bundle.tokenizer.encode(prompt, add_special_tokens=False).ids)
    bos_id = bundle.tokenizer.token_to_id("<bos>")
    if bos_id is not None and mode in {"raw_bos", "chat_role"}:
        ids = [bos_id] + ids
    budget = bundle.context_len - int(max_new_tokens)
    truncated = False
    if budget > 0 and len(ids) > budget:
        ids = ids[-budget:]
        truncated = True
    if not ids:
        raise ValueError("Prompt menghasilkan 0 token.")
    return ids, truncated


def _stop_token_ids(tokenizer: Tokenizer) -> set:
    stops = set()
    for name in ("<eos>", "<|endoftext|>", "<|end|>", "<|user|>"):
        tid = tokenizer.token_to_id(name)
        if tid is not None:
            stops.add(tid)
    return stops


def _clean_reply(text: str, prompt_mode: str) -> str:
    text = text.strip()
    if prompt_mode == "chat_role":
        if text.startswith("Assistant:"):
            text = text[len("Assistant:") :].lstrip()
        for marker in ("\n\nUser:", "\n\nSystem:", "\nUser:", "\nSystem:"):
            idx = text.find(marker)
            if idx != -1:
                text = text[:idx]
    elif prompt_mode == "chat_tags":
        for marker in ("<|end|>", "<|user|>", "<|endoftext|>"):
            idx = text.find(marker)
            if idx != -1:
                text = text[:idx]
    return text.strip()


def _get_static_decode_artifacts(
    bundle: LiwetBundle, want_cuda_graph: bool = True
) -> Tuple[StaticKVCache, Optional[CudaGraphDecodeRunner]]:
    try:
        param_dev = next(bundle.model.parameters()).device
        if param_dev != bundle.device:
            bundle.device = param_dev
            bundle.decode_runner = None
            bundle.static_cache = None
    except StopIteration:
        pass

    cache = bundle.static_cache
    if cache is None or cache.device != bundle.device or cache.dtype != bundle.dtype:
        cache = bundle.model.create_static_cache(bundle.device, bundle.dtype)
        bundle.static_cache = cache
        bundle.decode_runner = None

    if hasattr(bundle.model, "prepare_static_decode"):
        bundle.model.prepare_static_decode(bundle.device, bundle.dtype)

    runner = bundle.decode_runner
    if (
        want_cuda_graph
        and bundle.device.type == "cuda"
        and os.environ.get("LIWET_DISABLE_CUDA_GRAPH", "0").strip() != "1"
    ):
        if runner is None or runner.device != bundle.device:
            try:
                vocab_size = int(bundle.model.config.vocab_size)
                runner = CudaGraphDecodeRunner(bundle.model, cache, vocab_size)
                bundle.decode_runner = runner
            except Exception as err:
                print(
                    f"[liwet] Peringatan: CUDA Graph capture gagal untuk {bundle.repo_id} "
                    f"({type(err).__name__}: {err}); beralih ke static eager decode."
                )
                runner = None
                bundle.decode_runner = None
    else:
        runner = None
    return cache, runner


def _apply_rep_penalty_mask(
    logits: torch.Tensor, seen_mask: Optional[torch.Tensor], repetition_penalty: float
) -> torch.Tensor:
    if seen_mask is None or repetition_penalty == 1.0:
        return logits
    row = logits[0]
    penalized = torch.where(row > 0, row / repetition_penalty, row * repetition_penalty)
    return torch.where(seen_mask, penalized, row).unsqueeze(0)


def _sample_next_token(
    logits: torch.Tensor,
    sampling: bool,
    temperature: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    if not sampling:
        return torch.argmax(logits, dim=-1, keepdim=True)
    temp = max(float(temperature), 1e-5)
    logits = logits / temp
    if top_k is not None and int(top_k) > 0:
        k = min(int(top_k), logits.size(-1))
        kth = torch.topk(logits, k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and 0.0 < float(top_p) < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        remove = sorted_probs.cumsum(dim=-1) - sorted_probs > float(top_p)
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(
            dim=-1, index=sorted_idx, src=sorted_logits
        )
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def prefill_once(bundle: LiwetBundle, input_ids: torch.Tensor) -> torch.Tensor:
    with bundle.lock, torch.inference_mode():
        if _use_fastpath() and hasattr(bundle.model, "forward_prefill_static"):
            cache, _ = _get_static_decode_artifacts(bundle, want_cuda_graph=False)
            return bundle.model.forward_prefill_static(input_ids, cache)
        out = bundle.model.forward(input_ids, past_key_values=None, use_cache=False)
        return out.logits[:, -1, :].float()


def greedy_token_ids(
    bundle: LiwetBundle,
    prompt_ids: List[int],
    steps: int = 128,
    repetition_penalty: float = 1.0,
) -> List[int]:
    if steps <= 0:
        return []
    toks: List[int] = []
    if not _use_fastpath() or not hasattr(bundle.model, "forward_prefill_static"):
        with bundle.lock, torch.inference_mode():
            inp = torch.tensor([prompt_ids], dtype=torch.long, device=bundle.device)
            out = bundle.model.forward(inp, past_key_values=None, use_cache=True)
            pkv = out.past_key_values
            logits = out.logits[:, -1, :].float()
            seen = inp.view(-1).clone()
            for step_idx in range(steps):
                adj = logits.clone()
                if repetition_penalty != 1.0 and seen.numel() > 0:
                    uniq = torch.unique(seen)
                    vals = adj[0, uniq]
                    adj[0, uniq] = torch.where(
                        vals > 0, vals / repetition_penalty, vals * repetition_penalty
                    )
                nxt = torch.argmax(adj, dim=-1, keepdim=True)
                tok = int(nxt.item())
                toks.append(tok)
                if step_idx + 1 < steps:
                    seen = torch.cat([seen, nxt.view(-1)])
                    out = bundle.model.forward(nxt, past_key_values=pkv, use_cache=True)
                    pkv = out.past_key_values
                    logits = out.logits[:, -1, :].float()
        return toks

    with bundle.lock, torch.inference_mode():
        cache, runner = _get_static_decode_artifacts(bundle, want_cuda_graph=True)
        if len(prompt_ids) > cache.max_seq_len:
            prompt_ids = prompt_ids[-cache.max_seq_len :]
        inp = torch.tensor([prompt_ids], dtype=torch.long, device=bundle.device)
        logits = bundle.model.forward_prefill_static(inp, cache)
        seen_mask: Optional[torch.Tensor] = None
        if repetition_penalty != 1.0:
            seen_mask = torch.zeros(
                int(bundle.model.config.vocab_size), dtype=torch.bool, device=bundle.device
            )
            seen_mask[inp[0]] = True
        pos = len(prompt_ids)
        for step_idx in range(steps):
            adj = _apply_rep_penalty_mask(logits, seen_mask, repetition_penalty)
            nxt = torch.argmax(adj, dim=-1, keepdim=True)
            tok = int(nxt.item())
            toks.append(tok)
            if step_idx + 1 >= steps or pos >= cache.max_seq_len:
                break
            if seen_mask is not None:
                seen_mask[tok] = True
            if runner is not None:
                try:
                    logits = runner.step(nxt, pos)
                except Exception:
                    bundle.decode_runner = None
                    runner = None
                    pos_buf = cache.open_position(pos)
                    logits = bundle.model.forward_decode_static(nxt, pos_buf, cache)
            else:
                pos_buf = cache.open_position(pos)
                logits = bundle.model.forward_decode_static(nxt, pos_buf, cache)
            pos += 1
    return toks


_CHAT_ROLE_MARKERS = ("\n\nUser:", "\n\nSystem:", "\nUser:", "\nSystem:")


def stream_generate(
    bundle: LiwetBundle,
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
    generated_ids: List[int] = []
    previous_text = ""
    sampling = do_sample if do_sample is not None else float(temperature) > 0.0

    if not _use_fastpath() or not hasattr(bundle.model, "forward_prefill_static"):
        max_total = prompt_tokens + max_new
        token_buffer = torch.empty(
            (1, max_total), dtype=torch.long, device=bundle.device
        )
        token_buffer[0, :prompt_tokens] = torch.tensor(
            ids, dtype=torch.long, device=bundle.device
        )
        length = prompt_tokens
        past_key_values = None

        if bundle.device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()

        with bundle.lock, torch.inference_mode():
            if seed is not None:
                torch.manual_seed(int(seed))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(seed))

            for _ in range(max_new):
                current_ids = (
                    token_buffer[:, :length]
                    if past_key_values is None
                    else token_buffer[:, length - 1 : length]
                )
                output = bundle.model.forward(
                    current_ids, past_key_values=past_key_values, use_cache=True
                )
                past_key_values = output.past_key_values
                logits = output.logits[:, -1, :].float()

                if repetition_penalty != 1.0:
                    token_ids = torch.unique(token_buffer[0, :length])
                    token_logits = logits[0, token_ids]
                    logits[0, token_ids] = torch.where(
                        token_logits > 0,
                        token_logits / repetition_penalty,
                        token_logits * repetition_penalty,
                    )

                next_token = _sample_next_token(
                    logits, sampling, temperature, top_k, top_p
                )
                token_id = int(next_token.item())
                if token_id in stop_ids:
                    break

                token_buffer[0, length] = next_token[0, 0]
                length += 1
                generated_ids.append(token_id)

                raw_text = bundle.tokenizer.decode(generated_ids, skip_special_tokens=True)
                clean_text = _clean_reply(raw_text, bundle.spec.prompt_mode)
                if clean_text != previous_text:
                    previous_text = clean_text
                    yield clean_text

                if bundle.spec.prompt_mode == "chat_role" and any(
                    m in raw_text for m in _CHAT_ROLE_MARKERS
                ):
                    break

        if bundle.device.type == "cuda":
            torch.cuda.synchronize()
        seconds = max(time.perf_counter() - started, 1e-6)
        if stats is not None:
            stats.update(
                {
                    "text": previous_text,
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": len(generated_ids),
                    "seconds": seconds,
                    "wall_seconds": seconds,
                    "tokens_per_second": len(generated_ids) / seconds,
                    "truncated": truncated,
                }
            )
        return

    wall_started = time.perf_counter()
    model_seconds = 0.0
    debug_decode = os.environ.get("LIWET_DEBUG_DECODE", "0").strip() == "1"

    with bundle.lock, torch.inference_mode():
        cache, runner = _get_static_decode_artifacts(bundle, want_cuda_graph=True)
        if prompt_tokens > cache.max_seq_len:
            ids = ids[-cache.max_seq_len :]
            prompt_tokens = len(ids)
            truncated = True

        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        if bundle.device.type == "cuda":
            torch.cuda.synchronize()
        t_slice = time.perf_counter()

        inp = torch.tensor([ids], dtype=torch.long, device=bundle.device)
        seen_mask: Optional[torch.Tensor] = None
        if repetition_penalty != 1.0:
            seen_mask = torch.zeros(
                int(bundle.model.config.vocab_size), dtype=torch.bool, device=bundle.device
            )
            seen_mask[inp[0]] = True

        stable_idx = 0
        stable_prefix = ""

        if max_new > 0:
            logits = bundle.model.forward_prefill_static(inp, cache)
            pos = prompt_tokens

            for step_idx in range(max_new):
                adj = _apply_rep_penalty_mask(logits, seen_mask, repetition_penalty)
                next_token = _sample_next_token(
                    adj, sampling, temperature, top_k, top_p
                )
                token_id = int(next_token.item())
                if token_id in stop_ids:
                    break

                generated_ids.append(token_id)
                if seen_mask is not None:
                    seen_mask[token_id] = True

                # Incremental detokenization on clean UTF-8 / non-marker boundaries
                tail = bundle.tokenizer.decode(
                    generated_ids[stable_idx:], skip_special_tokens=True
                )
                raw_text = stable_prefix + tail
                if len(generated_ids) - stable_idx >= 32:
                    probe = len(generated_ids) - 12
                    seg = bundle.tokenizer.decode(
                        generated_ids[stable_idx:probe], skip_special_tokens=True
                    )
                    tail_cand = bundle.tokenizer.decode(
                        generated_ids[probe:], skip_special_tokens=True
                    )
                    can_commit = (
                        "\ufffd" not in seg
                        and "\ufffd" not in tail_cand
                        and bool(stable_prefix or seg.strip())
                        and len(seg) >= 16
                        and ("\n" not in seg[-16:])
                        and ("<" not in seg[-16:])
                    )
                    if can_commit:
                        if debug_decode:
                            full_chk = bundle.tokenizer.decode(
                                generated_ids, skip_special_tokens=True
                            )
                            assert stable_prefix + seg + tail_cand == full_chk, (
                                "Incremental decode mismatch"
                            )
                        stable_prefix = stable_prefix + seg
                        stable_idx = probe
                        raw_text = stable_prefix + tail_cand

                clean_text = _clean_reply(raw_text, bundle.spec.prompt_mode)
                if clean_text != previous_text:
                    previous_text = clean_text
                    model_seconds += time.perf_counter() - t_slice
                    yield clean_text
                    t_slice = time.perf_counter()

                if bundle.spec.prompt_mode == "chat_role":
                    recent = raw_text[-64:]
                    if any(m in recent for m in _CHAT_ROLE_MARKERS):
                        break

                if step_idx + 1 >= max_new or pos >= cache.max_seq_len:
                    break

                if runner is not None:
                    try:
                        logits = runner.step(next_token, pos)
                    except Exception as err:
                        print(
                            f"[liwet] Peringatan: CUDA Graph replay gagal "
                            f"({type(err).__name__}); fallback static eager."
                        )
                        bundle.decode_runner = None
                        runner = None
                        pos_buf = cache.open_position(pos)
                        logits = bundle.model.forward_decode_static(next_token, pos_buf, cache)
                else:
                    pos_buf = cache.open_position(pos)
                    logits = bundle.model.forward_decode_static(next_token, pos_buf, cache)
                pos += 1

        # Guarantee exact full decode on final answer
        if generated_ids:
            final_raw = bundle.tokenizer.decode(generated_ids, skip_special_tokens=True)
            final_clean = _clean_reply(final_raw, bundle.spec.prompt_mode)
            if final_clean != previous_text:
                previous_text = final_clean
                model_seconds += time.perf_counter() - t_slice
                yield final_clean
                t_slice = time.perf_counter()

        if bundle.device.type == "cuda":
            torch.cuda.synchronize()
        model_seconds += time.perf_counter() - t_slice

    seconds = max(model_seconds, 1e-6)
    wall_seconds = max(time.perf_counter() - wall_started, 1e-6)
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


def generate_text(
    bundle: LiwetBundle,
    prompt_text: str,
    max_new_tokens: int = 160,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    do_sample: Optional[bool] = None,
    seed: Optional[int] = None,
) -> Tuple[str, dict]:
    stats: dict = {}
    final_text = ""
    for partial in stream_generate(
        bundle,
        prompt_text,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
        seed=seed,
        stats=stats,
    ):
        final_text = partial
    return final_text, stats

def discover_local_checkpoints(project_root: Path) -> List[dict]:
    found: List[dict] = []
    project_root = Path(project_root)
    ckpt_root = project_root / "checkpoints"
    stage_meta = [
        ("simpo", "400M SimPO Aligned", "chat_role", "Liwet-400M-A188M-id-chat"),
        ("sft", "400M SFT Instruct", "chat_role", "Liwet-400M-A188M-id-instruct"),
        ("pretrain", "400M Pretrain Base", "raw_bos", "Liwet-400M-A188M-id-base"),
    ]
    st_root = project_root.parent / "TUBES_1_SAFETENSOR"
    for stage, stage_label, prompt_mode, repo_name in stage_meta:
        candidates = []
        if ckpt_root.exists():
            s_dir = ckpt_root / stage
            if s_dir.exists():
                final_dir = s_dir / "final"
                if final_dir.exists():
                    candidates.append(("final", final_dir))
        if st_root.exists():
            st_dir = st_root / repo_name
            if st_dir.exists():
                candidates.append(("SafeTensors", st_dir))
        if ckpt_root.exists():
            s_dir = ckpt_root / stage
            if s_dir.exists():
                for step_dir in sorted(s_dir.glob("step_*"), key=lambda p: p.name, reverse=True):
                    candidates.append((step_dir.name, step_dir))
        for name, cdir in candidates:
            has_cfg = (cdir / "config.json").exists()
            has_w = (cdir / "model.safetensors").exists() or (cdir / "pytorch_model.bin").exists()
            if has_cfg and has_w:
                found.append(
                    {
                        "label": f"Lokal · {stage_label} ({name})",
                        "path": str(cdir),
                        "stage": stage_label,
                        "prompt_mode": prompt_mode,
                    }
                )
    return found


def load_liwet_local(
    ckpt_dir: str | Path,
    device_name: str = "auto",
    dtype_name: str = "auto",
    fallback_tokenizer: Optional[Path] = None,
    prompt_mode: Optional[str] = None,
) -> LiwetBundle:
    ckpt_path = Path(ckpt_dir).resolve()
    cfg_file = ckpt_path / "config.json"
    if not cfg_file.exists():
        raise FileNotFoundError(f"config.json tidak ditemukan di {ckpt_path}")
    tok_file = ckpt_path / "tokenizer.json"
    if not tok_file.exists() and fallback_tokenizer is not None:
        tok_file = Path(fallback_tokenizer)
    if not tok_file.exists():
        raise FileNotFoundError(f"tokenizer.json tidak ditemukan di {ckpt_path}")
    weights_file = ckpt_path / "model.safetensors"
    if not weights_file.exists():
        weights_file = ckpt_path / "pytorch_model.bin"
    if not weights_file.exists():
        raise FileNotFoundError(f"Bobot model tidak ditemukan di {ckpt_path}")

    device = resolve_device(device_name)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        _configure_cpu_threads()
    dtype = resolve_dtype(dtype_name, device)
    cache_key = (f"local::{ckpt_path}", str(device), str(dtype))
    with _CACHE_LOCK:
        if cache_key in _BUNDLE_CACHE:
            _BUNDLE_CACHE.move_to_end(cache_key)
            return _BUNDLE_CACHE[cache_key]

    cfg_dict = json.loads(cfg_file.read_text(encoding="utf-8-sig"))
    state_dict = _load_state_dict(weights_file)
    model, context_len = _instantiate_model(cfg_dict, state_dict)
    del state_dict
    model = model.to(device=device, dtype=dtype)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    tokenizer = Tokenizer.from_file(str(tok_file))
    total_params, active_params = _count_params(model)

    inferred_mode = prompt_mode
    if inferred_mode is None:
        parts = {part.lower() for part in ckpt_path.parts}
        if "pretrain" in parts:
            inferred_mode = "raw_bos"
        else:
            inferred_mode = "chat_role"

    spec = ModelSpec(
        repo_id=f"local:{ckpt_path.parent.name}/{ckpt_path.name}",
        label=f"Lokal 400M ({ckpt_path.parent.name}/{ckpt_path.name})",
        scale="400.1M MoE (187.8M Active)" if total_params > 250_000_000 else f"{total_params/1e6:.1f}M",
        arch="moe_400m" if cfg_dict.get("use_moe") else "dense",
        stage=ckpt_path.parent.name,
        prompt_mode=inferred_mode,
        total_params_label=f"{total_params/1e6:.1f}M",
        active_params_label=f"{active_params/1e6:.1f}M",
        context_len=context_len,
        recommended_params={
            "max_new_tokens": 192,
            "temperature": 0.65,
            "top_k": 50,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "do_sample": True,
        },
        description=f"Checkpoint lokal dari {ckpt_path.parent.name}/{ckpt_path.name}.",
    )

    bundle = LiwetBundle(
        repo_id=spec.repo_id,
        spec=spec,
        model=model,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        total_params=total_params,
        active_params=active_params,
        context_len=context_len,
        lock=threading.Lock(),
    )

    with _CACHE_LOCK:
        _BUNDLE_CACHE[cache_key] = bundle
        _BUNDLE_CACHE.move_to_end(cache_key)
        while len(_BUNDLE_CACHE) > MAX_CACHED_BUNDLES:
            _, evicted = _BUNDLE_CACHE.popitem(last=False)
            evicted.decode_runner = None
            evicted.static_cache = None
            del evicted
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return bundle
