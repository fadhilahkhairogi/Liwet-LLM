"""Compare candidate backend against parity reference dumps (Gates P1, P2, P3, P4)."""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCAL_MODELS = REPO_ROOT.parent / "TUBES_1_SAFETENSOR"
DEFAULT_REF_ROOT = Path(r"E:\liwet_gguf_build\parity_ref")


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _tvd(samples_a: List[int], samples_b: List[int]) -> float:
    ca = Counter(samples_a)
    cb = Counter(samples_b)
    na = max(1, len(samples_a))
    nb = max(1, len(samples_b))
    keys = set(ca.keys()) | set(cb.keys())
    return 0.5 * sum(abs(ca[k] / na - cb[k] / nb) for k in keys)


def _eval_p1(ref_logits: np.ndarray, cand_logits: np.ndarray, regime: str) -> dict:
    cos_list = []
    max_abs_list = []
    top1_matches = 0
    n = min(len(ref_logits), len(cand_logits))
    for i in range(n):
        r = ref_logits[i].astype(np.float64)
        c = cand_logits[i].astype(np.float64)
        cos_list.append(_cosine(r, c))
        max_abs_list.append(float(np.max(np.abs(r - c))))
        if int(np.argmax(r)) == int(np.argmax(c)):
            top1_matches += 1
    min_cos = min(cos_list) if cos_list else 0.0
    mean_cos = statistics.mean(cos_list) if cos_list else 0.0
    max_abs = max(max_abs_list) if max_abs_list else 0.0
    top1_pct = 100.0 * top1_matches / max(1, n)

    if regime == "same_dtype":
        passed = (min_cos >= 0.9999) and (top1_pct == 100.0)
    elif regime in {"gguf_bf16", "gguf_f16"}:
        passed = (min_cos >= 0.999) and (top1_pct >= 99.0)
    else:  # gguf_q8_0
        passed = (min_cos >= 0.995) and (top1_pct >= 97.0)
    return {
        "n": n,
        "min_cosine": round(min_cos, 6),
        "mean_cosine": round(mean_cos, 6),
        "max_abs_diff": round(max_abs, 5),
        "top1_match_pct": round(top1_pct, 2),
        "passed": passed,
    }


def _eval_p2(ref_rows: List[dict], cand_tokens_list: List[List[int]], regime: str) -> dict:
    first_divs = []
    exact_128 = 0
    bad_div_non_tie = []
    n = min(len(ref_rows), len(cand_tokens_list))
    for i in range(n):
        ref_toks = ref_rows[i]["tokens"]
        gaps = ref_rows[i]["logit_gaps"]
        cand_toks = cand_tokens_list[i]
        m = min(len(ref_toks), len(cand_toks))
        div = m
        for j in range(m):
            if ref_toks[j] != cand_toks[j]:
                div = j
                if gaps[j] >= 0.05:
                    bad_div_non_tie.append({"id": ref_rows[i]["id"], "step": j, "gap": gaps[j]})
                break
        if div == m:
            exact_128 += 1
        first_divs.append(div)

    exact_pct = 100.0 * exact_128 / max(1, n)
    med_div = statistics.median(first_divs) if first_divs else 0.0
    if regime == "same_dtype":
        passed = (exact_pct >= 95.0) and (len(bad_div_non_tie) == 0)
    elif regime in {"gguf_bf16", "gguf_f16"}:
        passed = med_div >= 96.0
    else:
        passed = med_div >= 48.0  # reported for Q8_0
    return {
        "n": n,
        "exact_128_pct": round(exact_pct, 2),
        "median_first_div": med_div,
        "min_first_div": min(first_divs) if first_divs else 0,
        "non_tie_divergences": bad_div_non_tie[:5],
        "passed": passed,
    }


def _eval_p3(ref_experts: np.ndarray, cand_experts: np.ndarray, regime: str) -> Optional[dict]:
    if ref_experts.size == 0 or cand_experts.size == 0:
        return None
    # shape [N, L, K] or flattened [M, K]
    r_flat = ref_experts.reshape(-1, ref_experts.shape[-1])
    c_flat = cand_experts.reshape(-1, cand_experts.shape[-1])
    total = min(len(r_flat), len(c_flat))
    matches = sum(1 for i in range(total) if set(r_flat[i].tolist()) == set(c_flat[i].tolist()))
    pct = 100.0 * matches / max(1, total)
    thresh = 97.0 if regime == "gguf_q8_0" else 99.5
    return {
        "total_token_layers": total,
        "exact_set_match_pct": round(pct, 3),
        "passed": pct >= thresh,
    }


def _run_torch_fast_candidate(
    repo_id: str,
    ref_dir: Path,
    device_name: str,
    dtype_name: str,
    greedy_ids_order: List[str],
    prefill_ids_order: Optional[List[str]] = None,
    p4_ids_order: Optional[List[str]] = None,
) -> dict:
    sys.path.insert(0, str(REPO_ROOT / "liwet-space"))
    import liwet_models as lm  # type: ignore

    orig_dl = lm._download_model_files
    lm._download_model_files = lambda r, token=None: lm._find_local_model_files(r) or orig_dl(r, token=token)

    meta = json.loads((ref_dir / "meta.json").read_text(encoding="utf-8"))
    prompts = meta["prompts"]
    by_id = {p["id"]: p for p in prompts}
    prompts_prefill = [by_id[pid] for pid in prefill_ids_order] if prefill_ids_order is not None else prompts
    prompts_greedy = [by_id[pid] for pid in greedy_ids_order]
    p4_target_ids = list(p4_ids_order) if p4_ids_order is not None else [p["id"] for p in prompts[:5]]
    p4_set = set(p4_target_ids)
    bundle = lm.load_liwet_model(repo_id, device_name=device_name, dtype_name=dtype_name)
    is_moe = bundle.spec.arch == "moe_400m"

    # Attach MoE capture only during prefill (must be removed before CUDA Graph capture in greedy)
    captured: List[np.ndarray] = []
    handles = []
    if is_moe:
        def make_hook():
            def pre_hook(mod, inputs):
                h = inputs[0]
                b, s, d = h.shape
                flat = h.reshape(b * s, d)
                rl = F.linear(flat, mod.router.weight).float()
                sel = rl + mod.expert_bias.float()
                idx = torch.topk(sel, mod.top_k, dim=-1).indices
                captured.append(idx.detach().cpu().numpy().astype(np.int32))
            return pre_hook
        for mod in bundle.model.modules():
            if hasattr(mod, "router") and hasattr(mod, "expert_bias") and hasattr(mod, "top_k"):
                handles.append(mod.register_forward_pre_hook(make_hook()))

    # 1. Prefill logits + P4 sampling
    prefill_logits = []
    prefill_experts = []
    logits_by_id: Dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for pm in prompts_prefill:
            captured.clear()
            inp = torch.tensor([pm["ids"]], dtype=torch.long, device=bundle.device)
            if hasattr(lm, "prefill_once"):
                logits = lm.prefill_once(bundle, inp)
            else:
                out = bundle.model.forward(inp, past_key_values=None, use_cache=False)
                logits = out.logits[:, -1, :].float()
            prefill_logits.append(logits[0].detach().cpu().numpy())
            if pm["id"] in p4_set:
                logits_by_id[pm["id"]] = logits.detach().clone()
            if is_moe and captured:
                prefill_experts.append(np.stack([c[-1] for c in captured], axis=0))

        # Remove hooks cleanly BEFORE any CUDA Graph capture in greedy!
        for h in handles:
            h.remove()
        handles.clear()

        p4_samples = []
        for pid in p4_target_ids:
            if pid not in logits_by_id:
                pm = by_id[pid]
                inp = torch.tensor([pm["ids"]], dtype=torch.long, device=bundle.device)
                if hasattr(lm, "prefill_once"):
                    lg_p4 = lm.prefill_once(bundle, inp)
                else:
                    lg_p4 = bundle.model.forward(inp, past_key_values=None, use_cache=False).logits[:, -1, :].float()
                logits_by_id[pid] = lg_p4.detach()
            logits_ref_p4 = logits_by_id[pid]
            s500 = []
            for seed in range(500):
                torch.manual_seed(seed)
                if bundle.device.type == "cuda":
                    torch.cuda.manual_seed_all(seed)
                temp = 0.65
                lg = logits_ref_p4 / temp
                kth = torch.topk(lg, 40, dim=-1).values[:, -1:]
                lg = lg.masked_fill(lg < kth, float("-inf"))
                sl, si = torch.sort(lg, descending=True, dim=-1)
                sp = torch.softmax(sl, dim=-1)
                rem = sp.cumsum(dim=-1) - sp > 0.9
                sl = sl.masked_fill(rem, float("-inf"))
                lg = torch.full_like(lg, float("-inf")).scatter(-1, si, sl)
                s500.append(int(torch.multinomial(torch.softmax(lg, dim=-1), 1).item()))
            p4_samples.append(s500)

    # 2. Greedy 128 steps via bundle's decode path
    def run_greedy(rep_penalty: float) -> List[List[int]]:
        all_toks = []
        orig_stop = lm._stop_token_ids
        lm._stop_token_ids = lambda tok: set()
        try:
            for pm in prompts_greedy:
                if hasattr(lm, "greedy_token_ids"):
                    toks = lm.greedy_token_ids(bundle, pm["ids"], steps=128, repetition_penalty=rep_penalty)
                else:
                    toks = _torch_greedy_fallback(bundle, pm["ids"], steps=128, rep_penalty=rep_penalty)
                all_toks.append(toks)
        finally:
            lm._stop_token_ids = orig_stop
        return all_toks

    greedy_norep = run_greedy(1.0)
    def_rep = float(meta["recommended_params"].get("repetition_penalty", 1.1))
    greedy_defrep = run_greedy(def_rep)

    lm.clear_model_cache()
    del bundle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "prefill_logits": np.stack(prefill_logits, axis=0),
        "prefill_experts": np.stack(prefill_experts, axis=0) if prefill_experts else np.empty((0,), dtype=np.int32),
        "greedy_norep": greedy_norep,
        "greedy_defrep": greedy_defrep,
        "p4_samples": p4_samples,
    }


def _torch_greedy_fallback(bundle, prompt_ids: List[int], steps: int, rep_penalty: float) -> List[int]:
    toks = []
    with torch.inference_mode():
        inp = torch.tensor([prompt_ids], dtype=torch.long, device=bundle.device)
        out = bundle.model.forward(inp, past_key_values=None, use_cache=True)
        pkv = out.past_key_values
        logits = out.logits[:, -1, :].float()
        seen = inp.view(-1).clone()
        for s in range(steps):
            adj = logits.clone()
            if rep_penalty != 1.0 and seen.numel() > 0:
                u = torch.unique(seen)
                v = adj[0, u]
                adj[0, u] = torch.where(v > 0, v / rep_penalty, v * rep_penalty)
            tok = int(torch.argmax(adj, dim=-1).item())
            toks.append(tok)
            if s + 1 < steps:
                nxt = torch.tensor([[tok]], dtype=torch.long, device=bundle.device)
                seen = torch.cat([seen, nxt.view(-1)])
                out = bundle.model.forward(nxt, past_key_values=pkv, use_cache=True)
                pkv = out.past_key_values
                logits = out.logits[:, -1, :].float()
    return toks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", default="torch_fast", choices=["torch_fast"])
    parser.add_argument("--regime", default="same_dtype", choices=["same_dtype", "gguf_bf16", "gguf_f16", "gguf_q8_0"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--ref-root", default=str(DEFAULT_REF_ROOT))
    parser.add_argument("--local-dir", default=str(DEFAULT_LOCAL_MODELS))
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    os.environ["LIWET_LOCAL_MODELS_DIR"] = args.local_dir
    short = args.model.split("/")[-1]
    ref_dir = Path(args.ref_root) / short

    ref_bf16 = np.load(ref_dir / "prefill_bf16_cuda.npz")
    ref_fp32 = np.load(ref_dir / "prefill_fp32_cpu.npz")
    ref_greedy_norep = json.loads((ref_dir / "greedy_no_rep.json").read_text(encoding="utf-8"))
    ref_greedy_defrep = json.loads((ref_dir / "greedy_default_rep.json").read_text(encoding="utf-8"))
    ref_p4 = np.load(ref_dir / "sampling_p4.npz")

    greedy_ids_order = [r["id"] for r in ref_greedy_norep]
    prefill_ids_order = ref_fp32["prompt_ids"].tolist() if (args.device == "cpu" and "prompt_ids" in ref_fp32) else None
    p4_ids_order = ref_p4["prompt_ids"].tolist() if "prompt_ids" in ref_p4 else None

    if args.backend == "torch_fast":
        cand = _run_torch_fast_candidate(
            args.model,
            ref_dir,
            args.device,
            args.dtype,
            greedy_ids_order=greedy_ids_order,
            prefill_ids_order=prefill_ids_order,
            p4_ids_order=p4_ids_order,
        )
    else:
        raise NotImplementedError(args.backend)

    ref_logits_target = ref_fp32["logits"] if args.device == "cpu" else ref_bf16["logits"]
    p1 = _eval_p1(ref_logits_target, cand["prefill_logits"], args.regime)
    p2_norep = _eval_p2(ref_greedy_norep, cand["greedy_norep"], args.regime)
    p2_defrep = _eval_p2(ref_greedy_defrep, cand["greedy_defrep"], args.regime)
    ref_experts_target = ref_fp32["moe_experts"] if (args.device == "cpu" and "moe_experts" in ref_fp32) else ref_bf16["moe_experts"]
    p3 = _eval_p3(ref_experts_target, cand["prefill_experts"], args.regime) if ref_experts_target.size > 0 else None

    tvds = [_tvd(ref_p4["samples"][i].tolist(), cand["p4_samples"][i]) for i in range(len(cand["p4_samples"]))]
    p4 = {
        "max_tvd": round(max(tvds), 4) if tvds else 0.0,
        "mean_tvd": round(statistics.mean(tvds), 4) if tvds else 0.0,
        "passed": (max(tvds) <= 0.05) if tvds else False,
    }

    report = {
        "model": args.model,
        "backend": args.backend,
        "regime": args.regime,
        "P1_logit": p1,
        "P2_greedy_norep": p2_norep,
        "P2_greedy_defrep": p2_defrep,
        "P3_router": p3,
        "P4_sampling": p4,
        "ALL_PASSED": p1["passed"] and p2_norep["passed"] and p2_defrep["passed"] and (p3["passed"] if p3 else True) and p4["passed"],
    }
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["ALL_PASSED"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
