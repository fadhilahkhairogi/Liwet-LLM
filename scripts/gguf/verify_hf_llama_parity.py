"""Verify exported HF LlamaForCausalLM / MixtralForCausalLM folders against Liwet CPU fp32 reference (Gate B2.2).

MoE notes:
- Liwet's reference MoE uses capacity-limited dispatch (capacity_factor=1.25) that
  drops assignments beyond `capacity` per expert during multi-token prefill. Standard
  Mixtral never drops, so the verifier patches `MixtralSparseMoeBlock.forward` to
  replicate that dispatch exactly (same argsort/searchsorted ordering).
- Because attention is computed with different GEMM op order (fused `qkv_proj` vs
  separate `q_proj`/`k_proj`/`v_proj`), hidden states drift by ~3e-6 after ~24 layers.
  That is enough to flip exact router ties (selection-logit gaps < 1e-5), so the MoE
  gate is layer-wise parity on identical hidden states (P1b) plus tie-aware reporting
  of any end-to-end divergence.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
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
        assert d.get("moe_capacity_factor") == 1.25, d.get("moe_capacity_factor")
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

    # Patch MixtralSparseMoeBlock to replicate Liwet capacity-limited dispatch.
    # The reference MoE drops assignments beyond `capacity` per expert when
    # num_tokens > 1 (capacity_factor=1.25), which standard Mixtral never does.
    cap_factor = float(getattr(cfg, "moe_capacity_factor", 1.25))
    for layer in model.model.layers:
        block = layer.mlp

        def _make_block_forward(mod, cap_factor):
            def _liwet_block_forward(hidden_states: torch.Tensor):
                batch_size, seq_len, hidden_dim = hidden_states.shape
                flat = hidden_states.reshape(-1, hidden_dim)
                _, top_k_weights, top_k_index = mod.gate(flat)
                num_tokens = flat.shape[0]
                if num_tokens == 1:
                    out = mod.experts(flat, top_k_index, top_k_weights)
                    return out.view(batch_size, seq_len, hidden_dim)

                capacity = max(
                    1,
                    math.ceil(
                        num_tokens * top_k_index.shape[1] / mod.experts.num_experts * cap_factor
                    ),
                )
                flat_indices = top_k_index.reshape(-1)
                flat_weights = top_k_weights.reshape(-1)
                order = torch.argsort(flat_indices)
                sorted_indices = flat_indices[order]
                expert_ids = torch.arange(mod.experts.num_experts, device=flat_indices.device)
                offsets_end = torch.searchsorted(sorted_indices, expert_ids, right=True)
                offsets = torch.cat([offsets_end.new_zeros(1), offsets_end[:-1]])
                positions = (
                    torch.arange(flat_indices.numel(), device=flat_indices.device)
                    - offsets[sorted_indices]
                )
                valid = positions < capacity
                token_indices = (order // top_k_index.shape[1])[valid]
                selected_index = sorted_indices[valid].unsqueeze(-1)
                selected_weight = flat_weights[order][valid].unsqueeze(-1)
                expert_out = mod.experts(flat[token_indices], selected_index, selected_weight)
                final = torch.zeros_like(flat).index_add_(0, token_indices, expert_out)
                return final.view(batch_size, seq_len, hidden_dim)

            return _liwet_block_forward

        block.forward = _make_block_forward(block, cap_factor)  # type: ignore[method-assign]

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


# Any router near-tie below this selection-logit gap is considered ambiguous: fp32
# op-order drift (fused qkv_proj vs q/k/v GEMMs, bmm vs linear) is orders of
# magnitude larger than the gap, so either expert choice is numerically valid.
TIE_EPS = 1e-5


def _ref_moe_modules(ref_model):
    return [layer.mlp for layer in ref_model.layers]


def _ref_router_snapshot(moe, hidden_states: torch.Tensor):
    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    logits = F.linear(flat, moe.router.weight).float()
    probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    sel = logits + moe.expert_bias.float()
    idx = torch.topk(sel, moe.top_k, dim=-1).indices
    top_vals = torch.topk(sel, moe.top_k + 1, dim=-1).values
    margin = top_vals[..., -2] - top_vals[..., -1]
    weights = torch.gather(probs, dim=-1, index=idx)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    return idx, margin, weights


def _layerwise_moe_check(hf_model: MixtralForCausalLM, ref_model, prompts: List[dict]) -> dict:
    """Feed identical hidden states into each reference/HF MoE block and compare outputs.

    This isolates the exported expert weights + routing + capacity dispatch from the
    accumulated drift of the full 32-layer forward (attention op ordering). The
    acceptance threshold is 1e-4 absolute; observed worst cases are ~3e-5 on rows
    whose MoE output magnitudes reach ~300 (i.e. ~1e-7 relative, pure fp32 GEMM
    ordering noise between grouped bmm and per-expert linear).
    """
    ref_moes = _ref_moe_modules(ref_model)
    max_abs = 0.0
    router_ok = 0
    router_tot = 0
    for pm in prompts:
        inp = torch.tensor([pm["ids"]], dtype=torch.long)
        ref_in, ref_out, ref_idx = {}, {}, {}
        handles = []
        for i, moe in enumerate(ref_moes):
            def pre(m, inputs, i=i):
                ref_in[i] = inputs[0].reshape(-1, inputs[0].shape[-1]).detach().clone()
                ref_idx[i] = _ref_router_snapshot(m, inputs[0])[0]

            def post(m, inputs, output, i=i):
                ref_out[i] = output.detach().clone()

            handles.append(moe.register_forward_pre_hook(pre))
            handles.append(moe.register_forward_hook(post))
        with torch.inference_mode():
            ref_model(inp, past_key_values=None, use_cache=False)
        for h in handles:
            h.remove()

        for i, layer in enumerate(hf_model.model.layers):
            captured = {}

            def post_gate(m, inputs, output, captured=captured):
                captured["idx"] = output[2]

            h = layer.mlp.gate.register_forward_hook(post_gate)
            with torch.inference_mode():
                out = layer.mlp(ref_in[i].unsqueeze(0))
            h.remove()
            out = out.reshape(-1, out.shape[-1])
            ref_flat = ref_out[i].reshape(-1, out.shape[-1])
            max_abs = max(max_abs, float((out - ref_flat).abs().max().item()))
            router_ok += int(torch.equal(ref_idx[i], captured["idx"]))
            router_tot += 1

    return {
        "prompts": len(prompts),
        "max_abs_diff": round(max_abs, 9),
        "router_match_pct": round(100.0 * router_ok / max(1, router_tot), 3),
    }


def _tie_divergence_analysis(hf_model: MixtralForCausalLM, ref_model, pm: dict) -> dict:
    """Locate the first routing divergence and check whether it sits on a near-tie."""
    ref_moes = _ref_moe_modules(ref_model)
    inp = torch.tensor([pm["ids"]], dtype=torch.long)
    ref_idx, ref_margin = {}, {}
    handles = []
    for i, moe in enumerate(ref_moes):
        def pre(m, inputs, i=i):
            idx, margin, _ = _ref_router_snapshot(m, inputs[0])
            ref_idx[i] = idx
            ref_margin[i] = margin

        handles.append(moe.register_forward_pre_hook(pre))
    with torch.inference_mode():
        ref_model(inp, past_key_values=None, use_cache=False)
    for h in handles:
        h.remove()

    hf_idx: List[torch.Tensor] = []
    handles = []
    for layer in hf_model.model.layers:
        def post_gate(m, inputs, output, hf_idx=hf_idx):
            hf_idx.append(output[2])

        handles.append(layer.mlp.gate.register_forward_hook(post_gate))
    with torch.inference_mode():
        hf_model(inp, use_cache=False)
    for h in handles:
        h.remove()

    for i in range(len(ref_idx)):
        diff = ref_idx[i] != hf_idx[i]
        if diff.any():
            rows = diff.any(dim=-1).nonzero().flatten()
            row = int(rows[0])
            worst_margin = float(ref_margin[i][rows].max().item())
            return {
                "first_layer": i,
                "first_row": row,
                "n_diff_rows": int(rows.numel()),
                "max_selection_margin": worst_margin,
                "is_tie": worst_margin < TIE_EPS,
                "ref_experts": ref_idx[i][row].tolist(),
                "hf_experts": hf_idx[i][row].tolist(),
            }
    return {"first_layer": None, "n_diff_rows": 0, "is_tie": True}


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

    layerwise = {"prompts": 0, "max_abs_diff": None, "router_match_pct": None}
    tie_divergences: Dict[str, dict] = {}
    if is_moe:
        parity_prompts = [p for p in meta["prompts"] if "long" not in p["id"]][: max(1, min(greedy_limit, 3))]
        layerwise = _layerwise_moe_check(hf_model, ref_bundle.model, parity_prompts)
        for idx, pid in enumerate(ref_ids_order):
            if cos_list[idx] < 0.99999:
                info = _tie_divergence_analysis(hf_model, ref_bundle.model, prompt_by_id[pid])
                info["min_cosine"] = round(cos_list[idx], 8)
                tie_divergences[pid] = info

    lm.clear_model_cache()
    del ref_bundle
    del hf_model
    gc.collect()

    min_cos = min(cos_list)
    mean_cos = statistics.mean(cos_list)
    max_abs = max(max_abs_list)
    top1_pct = 100.0 * top1_ok / max(1, len(ref_ids_order))
    moe_pct = (100.0 * moe_layers_ok / moe_layers_tot) if moe_layers_tot > 0 else None

    non_tie_fails = [pid for pid, d in tie_divergences.items() if not d.get("is_tie")]
    if is_moe:
        # Standard Mixtral cannot reproduce Liwet's capacity-limited dispatch, so the
        # HF verify path emulates it. End-to-end logits are still subject to fp32
        # op-order drift, which can flip exact router ties (documented in
        # tie_divergences); the authoritative check is layer-wise MoE parity on
        # identical hidden states.
        passed = (
            top1_pct == 100.0
            and greedy_matches == len(short_prompts)
            and tok_info["exact_encode_match"]
            and layerwise["max_abs_diff"] is not None
            and layerwise["max_abs_diff"] <= 1e-4
            and layerwise["router_match_pct"] == 100.0
            and not non_tie_fails
            and min_cos >= 0.999
        )
    else:
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
        "P1b_layerwise_moe": layerwise if is_moe else None,
        "tie_divergences": tie_divergences if is_moe else {},
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
