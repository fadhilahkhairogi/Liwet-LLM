"""Export Liwet checkpoints (66M, 163M, 400M MoE) to HuggingFace LlamaForCausalLM / MixtralForCausalLM layout."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import torch
from safetensors.torch import load_file, save_file

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCAL_MODELS = REPO_ROOT.parent / "TUBES_1_SAFETENSOR"
DEFAULT_OUT_ROOT = Path(r"E:\liwet_gguf_build\hf_llama_tmp")

PRIORITY_MODELS = [
    "fadhilahkhairogi/Liwet-400M-A188M-id-chat",
    "fadhilahkhairogi/Liwet-163M-id-chat",
    "fadhilahkhairogi/Liwet-66M-id-chat",
]


def _all_catalog_models() -> List[str]:
    space_dir = REPO_ROOT / "liwet-space"
    if str(space_dir) not in sys.path:
        sys.path.insert(0, str(space_dir))
    import liwet_models as lm  # type: ignore

    return [spec.repo_id for spec in lm.LIWET_CATALOG]


def _resolve_repo_names(models: List[str] | None, export_all: bool) -> List[str]:
    all_repos = _all_catalog_models()
    if export_all:
        return [r.split("/")[-1] for r in all_repos]
    chosen = models if models else PRIORITY_MODELS
    short_map = {r.split("/")[-1]: r.split("/")[-1] for r in all_repos}
    resolved: List[str] = []
    for item in chosen:
        short = item.split("/")[-1]
        resolved.append(short_map.get(short, short))
    return resolved


def _load_source_state_dict(model_dir: Path) -> Dict[str, torch.Tensor]:
    st_path = model_dir / "model.safetensors"
    if st_path.exists():
        return load_file(str(st_path), device="cpu")
    bin_path = model_dir / "pytorch_model.bin"
    if bin_path.exists():
        sd = torch.load(str(bin_path), map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "model" in sd:
            sd = sd["model"]
        return sd
    raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin in {model_dir}")


def _detect_family(src_cfg: dict, src_sd: Dict[str, torch.Tensor]) -> str:
    arch = src_cfg.get("architectures", [])
    if src_cfg.get("model_type") == "mini_transformer" or "MiniTransformer" in arch:
        return "mini_66m"
    if bool(src_cfg.get("use_moe", False)):
        return "moe_400m"
    if any("qkv_proj.weight" in k for k in src_sd):
        return "dense_163m"
    raise ValueError(f"Unrecognized Liwet checkpoint architecture: {src_cfg.get('architectures')} / keys sample={list(src_sd.keys())[:5]}")


def _check_tied_weight(
    src_sd: Dict[str, torch.Tensor],
    emb_key: str,
    head_key: str,
    consumed: Set[str],
) -> None:
    if head_key in src_sd:
        if not torch.equal(src_sd[head_key], src_sd[emb_key]):
            max_diff = float((src_sd[head_key].float() - src_sd[emb_key].float()).abs().max().item())
            raise AssertionError(
                f"Expected tied weights {head_key} == {emb_key}, but max_diff={max_diff}"
            )
        consumed.add(head_key)


def _export_mini_66m(
    src_cfg: dict, src_sd: Dict[str, torch.Tensor], tok_json: dict
) -> Tuple[dict, Dict[str, torch.Tensor], dict]:
    vocab_size = int(src_cfg["vocab_size"])
    hidden_size = int(src_cfg["n_embd"])
    intermediate_size = int(src_cfg["mlp_hidden"])
    n_layer = int(src_cfg["n_layer"])
    n_head = int(src_cfg["n_head"])
    head_dim = hidden_size // n_head
    block_size = int(src_cfg["block_size"])
    rope_theta = float(src_cfg.get("rope_base", 10000.0))
    tie_weights = bool(src_cfg.get("tie_weights", True))
    if not tie_weights:
        raise AssertionError("Expected MiniTransformer tie_weights=True")
    if bool(src_cfg.get("bias", False)):
        raise AssertionError("Expected MiniTransformer bias=False")

    consumed: Set[str] = set()
    out_sd: Dict[str, torch.Tensor] = {}

    out_sd["model.embed_tokens.weight"] = src_sd["tok_emb.weight"].contiguous()
    consumed.add("tok_emb.weight")
    _check_tied_weight(src_sd, "tok_emb.weight", "lm_head.weight", consumed)

    out_sd["model.norm.weight"] = src_sd["ln_f.weight"].contiguous()
    consumed.add("ln_f.weight")

    for i in range(n_layer):
        k_ln1 = f"blocks.{i}.ln1.weight"
        k_qkv = f"blocks.{i}.attn.qkv.weight"
        k_proj = f"blocks.{i}.attn.proj.weight"
        k_ln2 = f"blocks.{i}.ln2.weight"
        k_w1 = f"blocks.{i}.mlp.w1.weight"
        k_w3 = f"blocks.{i}.mlp.w3.weight"
        k_w2 = f"blocks.{i}.mlp.w2.weight"

        out_sd[f"model.layers.{i}.input_layernorm.weight"] = src_sd[k_ln1].contiguous()
        qkv = src_sd[k_qkv]
        if tuple(qkv.shape) != (3 * hidden_size, hidden_size):
            raise AssertionError(f"{k_qkv} shape {tuple(qkv.shape)} != {(3 * hidden_size, hidden_size)}")
        q, k, v = qkv.split(hidden_size, dim=0)
        out_sd[f"model.layers.{i}.self_attn.q_proj.weight"] = q.contiguous()
        out_sd[f"model.layers.{i}.self_attn.k_proj.weight"] = k.contiguous()
        out_sd[f"model.layers.{i}.self_attn.v_proj.weight"] = v.contiguous()
        out_sd[f"model.layers.{i}.self_attn.o_proj.weight"] = src_sd[k_proj].contiguous()
        out_sd[f"model.layers.{i}.post_attention_layernorm.weight"] = src_sd[k_ln2].contiguous()
        out_sd[f"model.layers.{i}.mlp.gate_proj.weight"] = src_sd[k_w1].contiguous()
        out_sd[f"model.layers.{i}.mlp.up_proj.weight"] = src_sd[k_w3].contiguous()
        out_sd[f"model.layers.{i}.mlp.down_proj.weight"] = src_sd[k_w2].contiguous()
        consumed.update({k_ln1, k_qkv, k_proj, k_ln2, k_w1, k_w3, k_w2})

    extra_keys = set(src_sd.keys()) - consumed
    if extra_keys:
        raise AssertionError(f"Unmapped source tensors in mini_66m: {sorted(extra_keys)}")

    # Verify token IDs from tokenizer.json for special tokens
    added_by_content = {item["content"]: item["id"] for item in tok_json.get("added_tokens", [])}
    unk_id = int(added_by_content.get("<|unk|>", 0))
    pad_id = int(added_by_content.get("<|pad|>", 1))
    eos_id = int(added_by_content.get("<|endoftext|>", 2))

    out_cfg = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": n_layer,
        "num_attention_heads": n_head,
        "num_key_value_heads": n_head,
        "head_dim": head_dim,
        "hidden_act": "silu",
        "max_position_embeddings": block_size,
        "initializer_range": 0.02,
        "rms_norm_eps": 1e-6,  # Hard-coded in MiniRMSNorm
        "tie_word_embeddings": True,
        "attention_bias": False,
        "mlp_bias": False,
        "rope_theta": rope_theta,
        "bos_token_id": None,
        "eos_token_id": eos_id,
        "pad_token_id": pad_id,
        "unk_token_id": unk_id,
        "torch_dtype": src_cfg.get("torch_dtype", "bfloat16"),
    }

    out_tok_cfg = {
        "backend": "tokenizers",
        "tokenizer_class": "TokenizersBackend",
        "clean_up_tokenization_spaces": False,
        "model_max_length": block_size,
        "bos_token": None,
        "eos_token": "<|endoftext|>",
        "pad_token": "<|pad|>",
        "unk_token": "<|unk|>",
        "add_bos_token": False,
        "add_eos_token": False,
    }
    return out_cfg, out_sd, out_tok_cfg


def _strip_model_prefix(src_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if src_sd and all(k.startswith("model.") for k in src_sd):
        return {k[len("model.") :]: v for k, v in src_sd.items()}
    return dict(src_sd)


def _export_dense_163m(
    src_cfg: dict, src_sd_raw: Dict[str, torch.Tensor], src_tok_cfg: dict | None
) -> Tuple[dict, Dict[str, torch.Tensor], dict]:
    src_sd = _strip_model_prefix(src_sd_raw)
    vocab_size = int(src_cfg["vocab_size"])
    hidden_size = int(src_cfg["hidden_size"])
    intermediate_size = int(src_cfg["intermediate_size"])
    n_layer = int(src_cfg["num_hidden_layers"])
    n_head = int(src_cfg["num_attention_heads"])
    n_kv_head = int(src_cfg["num_key_value_heads"])
    head_dim = int(src_cfg.get("head_dim") or (hidden_size // n_head))
    max_pos = int(src_cfg["max_position_embeddings"])
    rope_theta = float(src_cfg.get("rope_theta", 10000.0))
    rms_eps = float(src_cfg.get("rms_norm_eps", 1e-5))
    tie_weights = bool(src_cfg.get("tie_word_embeddings", True))
    if not tie_weights:
        raise AssertionError("Expected DenseIndonesianChatLM tie_word_embeddings=True")

    q_dim = n_head * head_dim
    kv_dim = n_kv_head * head_dim

    consumed: Set[str] = set()
    out_sd: Dict[str, torch.Tensor] = {}

    out_sd["model.embed_tokens.weight"] = src_sd["embed_tokens.weight"].contiguous()
    consumed.add("embed_tokens.weight")
    _check_tied_weight(src_sd, "embed_tokens.weight", "lm_head.weight", consumed)

    out_sd["model.norm.weight"] = src_sd["norm.weight"].contiguous()
    consumed.add("norm.weight")

    for i in range(n_layer):
        k_in_ln = f"layers.{i}.input_layernorm.weight"
        k_qkv = f"layers.{i}.self_attn.qkv_proj.weight"
        k_o = f"layers.{i}.self_attn.o_proj.weight"
        k_post_ln = f"layers.{i}.post_attention_layernorm.weight"
        k_gate_up = f"layers.{i}.mlp.gate_up_proj.weight"
        k_down = f"layers.{i}.mlp.down_proj.weight"

        out_sd[f"model.layers.{i}.input_layernorm.weight"] = src_sd[k_in_ln].contiguous()

        qkv = src_sd[k_qkv]
        if tuple(qkv.shape) != (q_dim + 2 * kv_dim, hidden_size):
            raise AssertionError(
                f"{k_qkv} shape {tuple(qkv.shape)} != {(q_dim + 2 * kv_dim, hidden_size)}"
            )
        q, k, v = qkv.split((q_dim, kv_dim, kv_dim), dim=0)
        out_sd[f"model.layers.{i}.self_attn.q_proj.weight"] = q.contiguous()
        out_sd[f"model.layers.{i}.self_attn.k_proj.weight"] = k.contiguous()
        out_sd[f"model.layers.{i}.self_attn.v_proj.weight"] = v.contiguous()
        out_sd[f"model.layers.{i}.self_attn.o_proj.weight"] = src_sd[k_o].contiguous()

        out_sd[f"model.layers.{i}.post_attention_layernorm.weight"] = src_sd[k_post_ln].contiguous()

        gate_up = src_sd[k_gate_up]
        if tuple(gate_up.shape) != (2 * intermediate_size, hidden_size):
            raise AssertionError(
                f"{k_gate_up} shape {tuple(gate_up.shape)} != {(2 * intermediate_size, hidden_size)}"
            )
        gate, up = gate_up.chunk(2, dim=0)
        out_sd[f"model.layers.{i}.mlp.gate_proj.weight"] = gate.contiguous()
        out_sd[f"model.layers.{i}.mlp.up_proj.weight"] = up.contiguous()
        out_sd[f"model.layers.{i}.mlp.down_proj.weight"] = src_sd[k_down].contiguous()
        consumed.update({k_in_ln, k_qkv, k_o, k_post_ln, k_gate_up, k_down})

    extra_keys = set(src_sd.keys()) - consumed
    if extra_keys:
        raise AssertionError(f"Unmapped source tensors in dense_163m: {sorted(extra_keys)}")

    out_cfg = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_hidden_layers": n_layer,
        "num_attention_heads": n_head,
        "num_key_value_heads": n_kv_head,
        "head_dim": head_dim,
        "hidden_act": src_cfg.get("hidden_act", "silu"),
        "max_position_embeddings": max_pos,
        "initializer_range": float(src_cfg.get("initializer_range", 0.02)),
        "rms_norm_eps": rms_eps,
        "tie_word_embeddings": True,
        "attention_bias": False,
        "mlp_bias": False,
        "rope_theta": rope_theta,
        "bos_token_id": src_cfg.get("bos_token_id", 1),
        "eos_token_id": src_cfg.get("eos_token_id", 2),
        "pad_token_id": src_cfg.get("pad_token_id", 0),
        "unk_token_id": src_cfg.get("unk_token_id", 3),
        "torch_dtype": src_cfg.get("torch_dtype", "bfloat16"),
    }

    out_tok_cfg = dict(src_tok_cfg) if src_tok_cfg else {
        "backend": "tokenizers",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "clean_up_tokenization_spaces": False,
        "model_max_length": max_pos,
        "tokenizer_class": "TokenizersBackend",
    }
    out_tok_cfg["add_bos_token"] = False
    out_tok_cfg["add_eos_token"] = False
    return out_cfg, out_sd, out_tok_cfg


def _export_moe_400m(
    src_cfg: dict, src_sd_raw: Dict[str, torch.Tensor], src_tok_cfg: dict | None
) -> Tuple[dict, Dict[str, torch.Tensor], dict]:
    src_sd = _strip_model_prefix(src_sd_raw)
    vocab_size = int(src_cfg["vocab_size"])
    hidden_size = int(src_cfg["hidden_size"])
    n_experts = int(src_cfg["num_experts"])
    top_k = int(src_cfg["num_experts_per_tok"])
    moe_inter = int(src_cfg["moe_intermediate_size"])
    n_layer = int(src_cfg["num_hidden_layers"])
    n_head = int(src_cfg["num_attention_heads"])
    n_kv_head = int(src_cfg["num_key_value_heads"])
    head_dim = int(src_cfg.get("head_dim") or (hidden_size // n_head))
    max_pos = int(src_cfg["max_position_embeddings"])
    rope_theta = float(src_cfg.get("rope_theta", 10000.0))
    rms_eps = float(src_cfg.get("rms_norm_eps", 1e-5))
    tie_weights = bool(src_cfg.get("tie_word_embeddings", True))
    if not tie_weights:
        raise AssertionError("Expected MoEIndonesianChatLM tie_word_embeddings=True")

    consumed: Set[str] = set()
    out_sd: Dict[str, torch.Tensor] = {}

    out_sd["model.embed_tokens.weight"] = src_sd["embed_tokens.weight"].contiguous()
    consumed.add("embed_tokens.weight")
    _check_tied_weight(src_sd, "embed_tokens.weight", "lm_head.weight", consumed)

    out_sd["model.norm.weight"] = src_sd["norm.weight"].contiguous()
    consumed.add("norm.weight")

    for i in range(n_layer):
        k_in_ln = f"layers.{i}.input_layernorm.weight"
        k_q = f"layers.{i}.self_attn.q_proj.weight"
        k_k = f"layers.{i}.self_attn.k_proj.weight"
        k_v = f"layers.{i}.self_attn.v_proj.weight"
        k_o = f"layers.{i}.self_attn.o_proj.weight"
        k_post_ln = f"layers.{i}.post_attention_layernorm.weight"
        k_router = f"layers.{i}.mlp.router.weight"
        k_bias = f"layers.{i}.mlp.expert_bias"
        k_ema = f"layers.{i}.mlp.load_ema"
        k_gate_up = f"layers.{i}.mlp.gate_up_proj"
        k_down = f"layers.{i}.mlp.down_proj"

        out_sd[f"model.layers.{i}.input_layernorm.weight"] = src_sd[k_in_ln].contiguous()
        out_sd[f"model.layers.{i}.self_attn.q_proj.weight"] = src_sd[k_q].contiguous()
        out_sd[f"model.layers.{i}.self_attn.k_proj.weight"] = src_sd[k_k].contiguous()
        out_sd[f"model.layers.{i}.self_attn.v_proj.weight"] = src_sd[k_v].contiguous()
        out_sd[f"model.layers.{i}.self_attn.o_proj.weight"] = src_sd[k_o].contiguous()
        out_sd[f"model.layers.{i}.post_attention_layernorm.weight"] = src_sd[k_post_ln].contiguous()

        out_sd[f"model.layers.{i}.block_sparse_moe.gate.weight"] = src_sd[k_router].contiguous()
        # Keep selection bias F32 to match MoE.forward (.float()) and GGUF F32 rule
        out_sd[f"model.layers.{i}.block_sparse_moe.gate.e_score_correction.bias"] = (
            src_sd[k_bias].to(torch.float32).contiguous()
        )

        gate_up = src_sd[k_gate_up]
        if tuple(gate_up.shape) != (n_experts, hidden_size, 2 * moe_inter):
            raise AssertionError(
                f"{k_gate_up} shape {tuple(gate_up.shape)} != {(n_experts, hidden_size, 2 * moe_inter)}"
            )
        down = src_sd[k_down]
        if tuple(down.shape) != (n_experts, moe_inter, hidden_size):
            raise AssertionError(
                f"{k_down} shape {tuple(down.shape)} != {(n_experts, moe_inter, hidden_size)}"
            )

        for e in range(n_experts):
            # Liwet bmm: [1, hidden] @ [hidden, 2*moe_inter] -> Linear weight [out, in] is transposed
            w_gate = gate_up[e, :, :moe_inter].transpose(0, 1).contiguous()
            w_up = gate_up[e, :, moe_inter:].transpose(0, 1).contiguous()
            w_down = down[e].transpose(0, 1).contiguous()
            out_sd[f"model.layers.{i}.block_sparse_moe.experts.{e}.w1.weight"] = w_gate
            out_sd[f"model.layers.{i}.block_sparse_moe.experts.{e}.w3.weight"] = w_up
            out_sd[f"model.layers.{i}.block_sparse_moe.experts.{e}.w2.weight"] = w_down

        consumed.update({k_in_ln, k_q, k_k, k_v, k_o, k_post_ln, k_router, k_bias, k_gate_up, k_down})
        if k_ema in src_sd:
            consumed.add(k_ema)

    extra_keys = set(src_sd.keys()) - consumed
    if extra_keys:
        raise AssertionError(f"Unmapped source tensors in moe_400m: {sorted(extra_keys)}")

    out_cfg = {
        "architectures": ["MixtralForCausalLM"],
        "model_type": "mixtral",
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        # Mixtral expects expert intermediate size in intermediate_size (ffn length = 360)
        "intermediate_size": moe_inter,
        "num_hidden_layers": n_layer,
        "num_attention_heads": n_head,
        "num_key_value_heads": n_kv_head,
        "head_dim": head_dim,
        "hidden_act": src_cfg.get("hidden_act", "silu"),
        "max_position_embeddings": max_pos,
        "initializer_range": float(src_cfg.get("initializer_range", 0.02)),
        "rms_norm_eps": rms_eps,
        "tie_word_embeddings": True,
        "rope_theta": rope_theta,
        "num_local_experts": n_experts,
        "num_experts_per_tok": top_k,
        "router_aux_loss_coef": float(src_cfg.get("router_aux_loss_coef", 0.01)),
        "expert_selection_fn": "softmax_weight",
        "expert_weights_norm": False,
        "liwet_selection_bias": True,
        "bos_token_id": src_cfg.get("bos_token_id", 1),
        "eos_token_id": src_cfg.get("eos_token_id", 2),
        "pad_token_id": src_cfg.get("pad_token_id", 0),
        "unk_token_id": src_cfg.get("unk_token_id", 3),
        "torch_dtype": src_cfg.get("torch_dtype", "bfloat16"),
    }

    out_tok_cfg = dict(src_tok_cfg) if src_tok_cfg else {
        "backend": "tokenizers",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "clean_up_tokenization_spaces": False,
        "model_max_length": max_pos,
        "tokenizer_class": "TokenizersBackend",
    }
    out_tok_cfg["add_bos_token"] = False
    out_tok_cfg["add_eos_token"] = False
    return out_cfg, out_sd, out_tok_cfg


def _check_f16_safety(out_sd: Dict[str, torch.Tensor]) -> float:
    max_abs = 0.0
    worst_name = ""
    for name, tensor in out_sd.items():
        val = float(tensor.detach().float().abs().max().item())
        if not torch.isfinite(tensor.float()).all():
            raise ValueError(f"Non-finite values in exported tensor {name}")
        if val > max_abs:
            max_abs = val
            worst_name = name
    if max_abs >= 65504.0:
        raise ValueError(f"Tensor {worst_name} overflows F16 range: max_abs={max_abs}")
    return max_abs


def export_one(model_name: str, local_root: Path, out_root: Path) -> dict:
    src_dir = local_root / model_name
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    src_cfg = json.loads((src_dir / "config.json").read_text(encoding="utf-8-sig"))
    src_tok_path = src_dir / "tokenizer.json"
    if not src_tok_path.exists():
        raise FileNotFoundError(f"tokenizer.json missing in {src_dir}")
    tok_json = json.loads(src_tok_path.read_text(encoding="utf-8-sig"))

    src_tok_cfg_path = src_dir / "tokenizer_config.json"
    src_tok_cfg = (
        json.loads(src_tok_cfg_path.read_text(encoding="utf-8-sig"))
        if src_tok_cfg_path.exists()
        else None
    )

    src_sd = _load_source_state_dict(src_dir)
    family = _detect_family(src_cfg, src_sd)

    if family == "mini_66m":
        out_cfg, out_sd, out_tok_cfg = _export_mini_66m(src_cfg, src_sd, tok_json)
    elif family == "dense_163m":
        out_cfg, out_sd, out_tok_cfg = _export_dense_163m(src_cfg, src_sd, src_tok_cfg)
    elif family == "moe_400m":
        out_cfg, out_sd, out_tok_cfg = _export_moe_400m(src_cfg, src_sd, src_tok_cfg)
    else:
        raise AssertionError(family)

    max_abs = _check_f16_safety(out_sd)

    dst_dir = out_root / model_name
    dst_dir.mkdir(parents=True, exist_ok=True)
    (dst_dir / "config.json").write_text(
        json.dumps(out_cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (dst_dir / "tokenizer_config.json").write_text(
        json.dumps(out_tok_cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    shutil.copy2(src_tok_path, dst_dir / "tokenizer.json")
    save_file(out_sd, str(dst_dir / "model.safetensors"))

    return {
        "model": model_name,
        "family": family,
        "hf_arch": out_cfg["architectures"][0],
        "num_tensors": len(out_sd),
        "max_abs_weight": round(max_abs, 6),
        "f16_safe": max_abs < 65504.0,
        "out_dir": str(dst_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=None, help="Short or full model names")
    parser.add_argument("--all", action="store_true", help="Export all 11 catalog models")
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL_MODELS)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    args = parser.parse_args()

    model_names = _resolve_repo_names(args.models, args.all)
    results = []
    for name in model_names:
        info = export_one(name, args.local_dir, args.out_root)
        results.append(info)
        print(
            f"[export] {info['model']} -> {info['hf_arch']} "
            f"({info['num_tensors']} tensors, max|w|={info['max_abs_weight']}, dir={info['out_dir']})",
            flush=True,
        )

    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "export_summary.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
