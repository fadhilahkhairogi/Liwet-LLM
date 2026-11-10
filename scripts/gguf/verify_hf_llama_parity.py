"""Verify exported HF LlamaForCausalLM / MixtralForCausalLM folders against Liwet CPU fp32 reference (Gate B2.2)."""
from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM, MixtralForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPORT_ROOT = Path(r"E:\liwet_gguf_build\hf_llama_tmp")
DEFAULT_REF_ROOT = Path(r"E:\liwet_gguf_build\parity_ref")
DEFAULT_MODELS = [
    "Liwet-400M-A188M-id-chat",
    "Liwet-163M-id-chat",
    "Liwet-66M-id-chat",
]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _check_autoconfig(exp_dir: Path, short_name: str) -> dict:
    cfg = AutoConfig.from_pretrained(exp_dir, trust_remote_code=False)
    d = cfg.to_dict()

    rope_theta = d.get("rope_theta")
    if rope_theta is None and isinstance(d.get("rope_parameters"), dict):
        rope_theta = d["rope_parameters"].get("rope_theta")
    assert rope_theta == 10000.0, f"{short_name}: expected rope_theta=10000.0, got {rope_theta}"
    assert d.get("tie_word_embeddings") is True, (
        f"{short_name}: expected tie_word_embeddings=True, got {d.get('tie_word_embeddings')}"
    )

    expected_eps = 1e-6 if "66M" in short_name else 1e-5
    actual_eps = float(d.get("rms_norm_eps"))
    assert abs(actual_eps - expected_eps) < 1e-9, (
        f"{short_name}: expected rms_norm_eps={expected_eps}, got {actual_eps}"
    )

    if d.get("model_type") == "mixtral":
        assert d.get("expert_selection_fn") == "softmax_weight", d.get("expert_selection_fn")
        assert d.get("expert_weights_norm") is False, d.get("expert_weights_norm")
        assert d.get("liwet_selection_bias") is True, d.get("liwet_selection_bias")
        assert d.get("num_local_experts") == 8, d.get("num_local_experts")
        assert d.get("num_experts_per_tok") == 2, d.get("num_experts_per_tok")
        assert d.get("intermediate_size") == 360, d.get("intermediate_size")
    return d


def _check_tokenizer(exp_dir: Path, sample_prompt: str, expected_ids: List[int]) -> dict:
    tok = AutoTokenizer.from_pretrained(exp_dir)
    encoded = tok.encode(sample_prompt, add_special_tokens=False)
    return {
        "class": type(tok).__name__,
        "bos_token": tok.bos_token,
        "eos_token": tok.eos_token,
        "exact_encode_match": encoded == expected_ids,
        "len_encoded": len(encoded),
        "len_expected": len(expected_ids),
    }


def _load_mixtral_with_liwet_router(exp_dir: Path) -> MixtralForCausalLM:
    cfg = AutoConfig.from_pretrained(exp_dir, trust_remote_code=False)
    model = MixtralForCausalLM(cfg).to(dtype=torch.float32)
    raw_sd = load_file(str(exp_dir / "model.safetensors"), device="cpu")
    consumed = set()

    model_sd = {}
    for k in ["model.embed_tokens.weight", "model.norm.weight"]:
        model_sd[k] = raw_sd[k].to(torch.float32)
        consumed.add(k)
    model_sd["lm_head.weight"] = raw_sd["model.embed_tokens.weight"].to(torch.float32)

    n_layers = int(cfg.num_hidden_layers)
    n_experts = int(cfg.num_local_experts)
    inter = int(cfg.intermediate_size)
    hidden = int(cfg.hidden_size)

    biases: Dict[int, torch.Tensor] = {}
    for i in range(n_layers):
        for suffix in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
        ]:
            key = f"model.layers.{i}.{suffix}"
            model_sd[key] = raw_sd[key].to(torch.float32)
            consumed.add(key)

        gate_k = f"model.layers.{i}.block_sparse_moe.gate.weight"
        bias_k = f"model.layers.{i}.block_sparse_moe.gate.e_score_correction.bias"
        model_sd[f"model.layers.{i}.mlp.gate.weight"] = raw_sd[gate_k].to(torch.float32)
        biases[i] = raw_sd[bias_k].to(torch.float32)
        consumed.update({gate_k, bias_k})

        w1_list = []
        w3_list = []
        w2_list = []
        for e in range(n_experts):
            k1 = f"model.layers.{i}.block_sparse_moe.experts.{e}.w1.weight"
            k3 = f"model.layers.{i}.block_sparse_moe.experts.{e}.w3.weight"
            k2 = f"model.layers.{i}.block_sparse_moe.experts.{e}.w2.weight"
            w1_list.append(raw_sd[k1].to(torch.float32))
            w3_list.append(raw_sd[k3].to(torch.float32))
            w2_list.append(raw_sd[k2].to(torch.float32))
            consumed.update({k1, k3, k2})

        gate_up = torch.cat([torch.stack(w1_list, dim=0), torch.stack(w3_list, dim=0)], dim=1)
        down = torch.stack(w2_list, dim=0)
        assert tuple(gate_up.shape) == (n_experts, 2 * inter, hidden)
        assert tuple(down.shape) == (n_experts, hidden, inter)
        model_sd[f"model.layers.{i}.mlp.experts.gate_up_proj"] = gate_up
        model_sd[f"model.layers.{i}.mlp.experts.down_proj"] = down

    unused = set(raw_sd.keys()) - consumed
    assert not unused, f"Unconsumed safetensors keys: {sorted(unused)}"

    missing, unexpected = model.load_state_dict(model_sd, strict=True)
    assert not missing and not unexpected, f"missing={missing}, unexpected={unexpected}"
    assert torch.equal(model.lm_head.weight, model.model.embed_tokens.weight)

    # Patch each MixtralTopKRouter to replicate Liwet MoE selection bias + softmax_weight routing
    for i, layer in enumerate(model.model.layers):
        gate = layer.mlp.gate
        gate.register_buffer("expert_bias", biases[i].clone(), persistent=False)

        def _make_forward(mod):
            def _liwet_router_forward(hidden_states: torch.Tensor):
                flat = hidden_states.reshape(-1, mod.hidden_dim)
                router_logits = F.linear(flat, mod.weight).float()
                router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
                sel = router_logits + mod.expert_bias.float()
                router_indices = torch.topk(sel, mod.top_k, dim=-1).indices
                router_scores = torch.gather(router_probs, dim=-1, index=router_indices)
                router_scores = router_scores / router_scores.sum(dim=-1, keepdim=True).clamp_min(1e-9)
                mod._last_indices = router_indices.detach().cpu().numpy().astype(np.int32)
                return router_logits, router_scores, router_indices

            return _liwet_router_forward

        gate.forward = _make_forward(gate)  # type: ignore[method-assign]

    model.eval()
    return model


def _load_ref_model_cpu(short_name: str):
    ref_dir = REPO_ROOT / "scripts" / "parity" / "reference"
    if str(ref_dir) not in sys.path:
        sys.path.insert(0, str(ref_dir))
    import liwet_models_ref as lm  # type: ignore

    orig_dl = lm._download_model_files
    lm._download_model_files = lambda repo_id, token=None: lm._find_local_model_files(repo_id) or orig_dl(repo_id, token=token)
    bundle = lm.load_liwet_model(f"fadhilahkhairogi/{short_name}", device_name="cpu", dtype_name="float32")
    return lm, bundle


def verify_model(
    short_name: str,
    export_root: Path,
    ref_root: Path,
    greedy_steps: int = 32,
    greedy_limit: int = 2,
) -> dict:
    exp_dir = export_root / short_name
    ref_dir = ref_root / short_name
    cfg_dict = _check_autoconfig(exp_dir, short_name)

    meta = json.loads((ref_dir / "meta.json").read_text(encoding="utf-8"))
    ref_fp32 = np.load(ref_dir / "prefill_fp32_cpu.npz")
    prompt_by_id = {p["id"]: p for p in meta["prompts"]}

    # Compare tokenizer against raw encoded prompt (note chat_role in Liwet prepends <bos> manually)
    sample_pm = meta["prompts"][0]
    from tokenizers import Tokenizer

    orig_tok = Tokenizer.from_file(str(exp_dir / "tokenizer.json"))
    expected_no_bos = list(orig_tok.encode(sample_pm["formatted"], add_special_tokens=False).ids)
    tok_info = _check_tokenizer(exp_dir, sample_pm["formatted"], expected_no_bos)

    is_moe = cfg_dict.get("model_type") == "mixtral"
    if is_moe:
        hf_model = _load_mixtral_with_liwet_router(exp_dir)
    else:
        hf_model = LlamaForCausalLM.from_pretrained(exp_dir, torch_dtype=torch.float32)
        hf_model.eval()
        assert torch.equal(hf_model.lm_head.weight, hf_model.model.embed_tokens.weight)

    ref_ids_order = ref_fp32["prompt_ids"].tolist()
    ref_logits = ref_fp32["logits"]
    ref_experts = ref_fp32["moe_experts"] if "moe_experts" in ref_fp32 else np.empty((0,), dtype=np.int32)

    cos_list: List[float] = []
    max_abs_list: List[float] = []
    top1_ok = 0
    moe_layers_ok = 0
    moe_layers_tot = 0

    with torch.inference_mode():
        for idx, pid in enumerate(ref_ids_order):
            pm = prompt_by_id[pid]
            inp = torch.tensor([pm["ids"]], dtype=torch.long)
            out = hf_model(inp, use_cache=False)
            cand = out.logits[0, -1, :].float().numpy()
            ref = ref_logits[idx].astype(np.float64)
            cand64 = cand.astype(np.float64)
            cos_list.append(_cosine(ref, cand64))
            max_abs_list.append(float(np.max(np.abs(ref - cand64))))
            if int(np.argmax(ref)) == int(np.argmax(cand64)):
                top1_ok += 1
            if is_moe and ref_experts.size > 0:
                ref_exp = ref_experts[idx]
                for l_idx, layer in enumerate(hf_model.model.layers):
                    cand_set = set(layer.mlp.gate._last_indices[-1].tolist())
                    ref_set = set(ref_exp[l_idx].tolist())
                    moe_layers_tot += 1
                    if cand_set == ref_set:
                        moe_layers_ok += 1

    # P2 greedy comparison against reference Liwet model on CPU float32
    lm, ref_bundle = _load_ref_model_cpu(short_name)
    short_prompts = [p for p in meta["prompts"] if "long" not in p["id"]][:greedy_limit]
    greedy_matches = 0
    greedy_first_divs: List[int] = []
    with torch.inference_mode():
        for pm in short_prompts:
            ids = list(pm["ids"])
            inp = torch.tensor([ids], dtype=torch.long)
            # Ref CPU fp32 greedy
            ref_toks: List[int] = []
            r_out = ref_bundle.model.forward(inp, past_key_values=None, use_cache=True)
            r_pkv = r_out.past_key_values
            r_logits = r_out.logits[:, -1, :].float()
            for s in range(greedy_steps):
                t = int(torch.argmax(r_logits, dim=-1).item())
                ref_toks.append(t)
                if s + 1 < greedy_steps:
                    r_out = ref_bundle.model.forward(torch.tensor([[t]], dtype=torch.long), past_key_values=r_pkv, use_cache=True)
                    r_pkv = r_out.past_key_values
                    r_logits = r_out.logits[:, -1, :].float()

            # HF Llama/Mixtral CPU fp32 greedy
            hf_toks: List[int] = []
            h_out = hf_model(inp, use_cache=True)
            h_pkv = h_out.past_key_values
            h_logits = h_out.logits[:, -1, :].float()
            for s in range(greedy_steps):
                t = int(torch.argmax(h_logits, dim=-1).item())
                hf_toks.append(t)
                if s + 1 < greedy_steps:
                    h_out = hf_model(torch.tensor([[t]], dtype=torch.long), past_key_values=h_pkv, use_cache=True)
                    h_pkv = h_out.past_key_values
                    h_logits = h_out.logits[:, -1, :].float()

            div = greedy_steps
            for s in range(greedy_steps):
                if ref_toks[s] != hf_toks[s]:
                    div = s
                    break
            greedy_first_divs.append(div)
            if div == greedy_steps:
                greedy_matches += 1

    lm.clear_model_cache()
    del ref_bundle
    del hf_model
    gc.collect()

    min_cos = min(cos_list)
    mean_cos = statistics.mean(cos_list)
    max_abs = max(max_abs_list)
    top1_pct = 100.0 * top1_ok / max(1, len(ref_ids_order))
    moe_pct = (100.0 * moe_layers_ok / moe_layers_tot) if moe_layers_tot > 0 else None

    passed = (
        min_cos >= 0.99999
        and top1_pct == 100.0
        and greedy_matches == len(short_prompts)
        and (moe_pct is None or moe_pct == 100.0)
        and tok_info["exact_encode_match"]
    )
    return {
        "model": short_name,
        "hf_arch": cfg_dict["architectures"][0],
        "rope_theta": 10000.0,
        "tie_word_embeddings": True,
        "rms_norm_eps": float(cfg_dict["rms_norm_eps"]),
        "tokenizer": tok_info,
        "P1_cpu_fp32": {
            "n": len(ref_ids_order),
            "min_cosine": round(min_cos, 8),
            "mean_cosine": round(mean_cos, 8),
            "max_abs_diff": round(max_abs, 7),
            "top1_match_pct": round(top1_pct, 2),
        },
        "P2_cpu_fp32_greedy": {
            "prompts": len(short_prompts),
            "steps": greedy_steps,
            "exact_matches": greedy_matches,
            "first_divs": greedy_first_divs,
        },
        "P3_cpu_fp32_router_pct": round(moe_pct, 3) if moe_pct is not None else None,
        "PASSED": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT_ROOT)
    parser.add_argument("--ref-root", type=Path, default=DEFAULT_REF_ROOT)
    parser.add_argument("--greedy-steps", type=int, default=32)
    parser.add_argument("--greedy-limit", type=int, default=2)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    results = []
    all_passed = True
    for m in args.models:
        short = m.split("/")[-1]
        print(f"=== Verifying HF export parity for {short} ===", flush=True)
        res = verify_model(
            short,
            args.export_root,
            args.ref_root,
            greedy_steps=args.greedy_steps,
            greedy_limit=args.greedy_limit,
        )
        results.append(res)
        all_passed = all_passed and res["PASSED"]
        print(json.dumps(res, indent=2, ensure_ascii=False), flush=True)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if not all_passed:
        sys.exit(2)


if __name__ == "__main__":
    main()
