"""Dump untouched PyTorch reference logits, 128-step greedy tokens, MoE routing, and P4 sampling distributions."""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCAL_MODELS = REPO_ROOT.parent / "TUBES_1_SAFETENSOR"
DEFAULT_OUT_ROOT = Path(r"E:\liwet_gguf_build\parity_ref")
PROMPTS_FILE = Path(__file__).resolve().parent / "prompts.jsonl"


def _load_ref_lm():
    ref_dir = Path(__file__).resolve().parent / "reference"
    sys.path.insert(0, str(ref_dir))
    import liwet_models_ref as lm  # type: ignore

    orig_dl = lm._download_model_files
    lm._download_model_files = lambda repo_id, token=None: lm._find_local_model_files(repo_id) or orig_dl(repo_id, token=token)
    return lm


def _attach_moe_hooks(model):
    """Capture top_indices [num_tokens, top_k] for every MoE layer using exact Liwet routing formula."""
    captured: List[np.ndarray] = []
    handles = []

    def make_hook():
        def pre_hook(mod, inputs):
            hidden_states = inputs[0]
            b, s, h = hidden_states.shape
            flat = hidden_states.reshape(b * s, h)
            router_logits = F.linear(flat, mod.router.weight).float()
            selection_logits = router_logits + mod.expert_bias.float()
            top_idx = torch.topk(selection_logits, mod.top_k, dim=-1).indices
            captured.append(top_idx.detach().cpu().numpy().astype(np.int32))
        return pre_hook

    for mod in model.modules():
        if hasattr(mod, "router") and hasattr(mod, "expert_bias") and hasattr(mod, "top_k"):
            handles.append(mod.register_forward_pre_hook(make_hook()))
    return captured, handles


def _apply_rep_penalty_ref(logits: torch.Tensor, seen_ids: torch.Tensor, penalty: float) -> torch.Tensor:
    if penalty == 1.0 or seen_ids.numel() == 0:
        return logits
    out = logits.clone()
    uniq = torch.unique(seen_ids)
    vals = out[0, uniq]
    out[0, uniq] = torch.where(vals > 0, vals / penalty, vals * penalty)
    return out


def _sample_next_token_ref(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
) -> int:
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
    return int(torch.multinomial(probs, num_samples=1).item())


def _stratify_prompts(prompt_meta: List[dict], limit: int) -> List[dict]:
    if limit <= 0 or limit >= len(prompt_meta):
        return prompt_meta
    picked_ids = set()
    picked: List[dict] = []
    # Guarantee at least one from each kind
    kinds_seen = set()
    for pm in prompt_meta:
        k = pm.get("kind", "standard")
        if k not in kinds_seen:
            kinds_seen.add(k)
            picked.append(pm)
            picked_ids.add(pm["id"])
            if len(picked) >= limit:
                return picked
    for pm in prompt_meta:
        if pm["id"] not in picked_ids:
            picked.append(pm)
            picked_ids.add(pm["id"])
            if len(picked) >= limit:
                break
    return picked


def dump_model(
    lm,
    repo_id: str,
    out_root: Path,
    greedy_steps: int = 128,
    greedy_limit: int = 6,
    cpu_prefill_limit: int = 10,
) -> None:
    short = repo_id.split("/")[-1]
    out_dir = out_root / short
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = lm.get_model_spec(repo_id)

    all_prompts = [json.loads(line) for line in PROMPTS_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    mode_prompts = [p for p in all_prompts if p["prompt_mode"] == spec.prompt_mode]

    # 1. Load on CUDA BF16 first (also get tokenizer + encoded IDs)
    bundle_cuda = lm.load_liwet_model(repo_id, device_name="cuda", dtype_name="auto")
    stop_ids = lm._stop_token_ids(bundle_cuda.tokenizer)

    prompt_meta = []
    for p in mode_prompts:
        formatted = lm.build_prompt_from_messages(
            spec,
            p["messages"],
            p["system_prompt"],
            use_chat_template=(spec.prompt_mode not in {"raw", "raw_bos"}),
        )
        ids, truncated = lm._encode_prompt(bundle_cuda, formatted, max_new_tokens=greedy_steps)
        prompt_meta.append(
            {
                "id": p["id"],
                "kind": p["kind"],
                "formatted": formatted,
                "ids": ids,
                "truncated": truncated,
            }
        )

    # Save meta
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "repo_id": repo_id,
                "prompt_mode": spec.prompt_mode,
                "context_len": bundle_cuda.context_len,
                "recommended_params": spec.recommended_params,
                "stop_ids": sorted(stop_ids),
                "prompts": prompt_meta,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    is_moe = spec.arch == "moe_400m"
    captured, handles = _attach_moe_hooks(bundle_cuda.model) if is_moe else ([], [])

    # 2. Prefill BF16 CUDA + P4 500-seed sampling on first 5 prompts
    bf16_logits = []
    bf16_experts = []
    p4_tokens = []
    with torch.inference_mode():
        for idx, pm in enumerate(prompt_meta):
            captured.clear()
            inp = torch.tensor([pm["ids"]], dtype=torch.long, device=bundle_cuda.device)
            out = bundle_cuda.model.forward(inp, past_key_values=None, use_cache=False)
            last_logits = out.logits[:, -1, :].float()
            bf16_logits.append(last_logits[0].cpu().numpy())
            if is_moe:
                # captured has 1 entry [seq_len, top_k] per layer -> take last token [-1]
                bf16_experts.append(np.stack([c[-1] for c in captured], axis=0))
            if idx < 5:
                samples_500 = []
                for seed in range(500):
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    samples_500.append(_sample_next_token_ref(last_logits, 0.65, 40, 0.9))
                p4_tokens.append(samples_500)

    np.savez_compressed(
        out_dir / "prefill_bf16_cuda.npz",
        logits=np.stack(bf16_logits, axis=0),
        moe_experts=np.stack(bf16_experts, axis=0) if is_moe else np.empty((0,), dtype=np.int32),
    )
    np.savez_compressed(
        out_dir / "sampling_p4.npz",
        prompt_ids=np.array([pm["id"] for pm in prompt_meta[:5]]),
        samples=np.array(p4_tokens, dtype=np.int32),
    )

    # 3. Greedy 128 tokens (no rep penalty, and default rep penalty) on BF16 CUDA
    greedy_subset = _stratify_prompts(prompt_meta, greedy_limit)

    def run_greedy(rep_penalty: float) -> List[dict]:
        rows = []
        with torch.inference_mode():
            for idx_p, pm in enumerate(greedy_subset):
                print(f"    greedy [{idx_p+1}/{len(greedy_subset)}] {pm['id']}...", flush=True)
                ids = list(pm["ids"])
                inp = torch.tensor([ids], dtype=torch.long, device=bundle_cuda.device)
                captured.clear()
                out = bundle_cuda.model.forward(inp, past_key_values=None, use_cache=True)
                pkv = out.past_key_values
                logits = out.logits[:, -1, :].float()
                seen = inp.view(-1).clone()

                gen_tokens = []
                logit_gaps = []
                experts_trace = []
                first_eos_step = None

                for step in range(greedy_steps):
                    if is_moe and step > 0:
                        pass
                    adj = _apply_rep_penalty_ref(logits, seen, rep_penalty)
                    # Token selection must mirror the reference model path exactly:
                    # torch.argmax. torch.topk tie-breaks exact bf16 ties differently
                    # (it may return the later index), which silently diverged from the
                    # eager reference on near-tie logits. Keep topk only for the gap.
                    tok = int(torch.argmax(adj[0]).item())
                    top2 = torch.topk(adj[0], 2)
                    gap = float((top2.values[0] - top2.values[1]).item())
                    gen_tokens.append(tok)
                    logit_gaps.append(round(gap, 5))
                    if is_moe:
                        # captured currently holds last step's [seq_or_1, top_k] per layer
                        experts_trace.append([c[-1].tolist() for c in captured])
                    if first_eos_step is None and tok in stop_ids:
                        first_eos_step = step

                    if step + 1 < greedy_steps:
                        nxt = torch.tensor([[tok]], dtype=torch.long, device=bundle_cuda.device)
                        seen = torch.cat([seen, nxt.view(-1)])
                        captured.clear()
                        out = bundle_cuda.model.forward(nxt, past_key_values=pkv, use_cache=True)
                        pkv = out.past_key_values
                        logits = out.logits[:, -1, :].float()

                rows.append(
                    {
                        "id": pm["id"],
                        "tokens": gen_tokens,
                        "logit_gaps": logit_gaps,
                        "first_eos_step": first_eos_step,
                        "experts_trace": experts_trace if is_moe else None,
                    }
                )
        return rows

    print(f"  [{short}] running greedy no-rep ({len(greedy_subset)} prompts × {greedy_steps} steps)...", flush=True)
    greedy_norep = run_greedy(1.0)
    (out_dir / "greedy_no_rep.json").write_text(
        json.dumps(greedy_norep, ensure_ascii=False), encoding="utf-8"
    )

    def_rep = float(spec.recommended_params.get("repetition_penalty", 1.1))
    print(f"  [{short}] running greedy default-rep={def_rep} ({len(greedy_subset)} prompts)...", flush=True)
    greedy_defrep = run_greedy(def_rep)
    (out_dir / "greedy_default_rep.json").write_text(
        json.dumps(greedy_defrep, ensure_ascii=False), encoding="utf-8"
    )

    for h in handles:
        h.remove()
    lm.clear_model_cache()
    del bundle_cuda
    gc.collect()
    torch.cuda.empty_cache()

    # 4. Prefill FP32 CPU (stratified subset)
    cpu_subset = _stratify_prompts(prompt_meta, cpu_prefill_limit)
    bundle_cpu = lm.load_liwet_model(repo_id, device_name="cpu", dtype_name="float32")
    captured_cpu, handles_cpu = _attach_moe_hooks(bundle_cpu.model) if is_moe else ([], [])
    cpu_logits = []
    cpu_experts = []
    print(f"  [{short}] running CPU fp32 prefill ({len(cpu_subset)} prompts)...", flush=True)
    with torch.inference_mode():
        for pm in cpu_subset:
            captured_cpu.clear()
            inp = torch.tensor([pm["ids"]], dtype=torch.long, device=bundle_cpu.device)
            out = bundle_cpu.model.forward(inp, past_key_values=None, use_cache=False)
            cpu_logits.append(out.logits[:, -1, :].numpy().astype(np.float32))
            if is_moe:
                cpu_experts.append(np.stack([c[-1] for c in captured_cpu], axis=0))
    for h in handles_cpu:
        h.remove()
    np.savez_compressed(
        out_dir / "prefill_fp32_cpu.npz",
        prompt_ids=np.array([pm["id"] for pm in cpu_subset]),
        logits=np.concatenate(cpu_logits, axis=0),
        moe_experts=np.stack(cpu_experts, axis=0) if is_moe else np.empty((0,), dtype=np.int32),
    )
    lm.clear_model_cache()
    del bundle_cpu
    gc.collect()
    print(f"  [{short}] reference dump complete in {out_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "fadhilahkhairogi/Liwet-400M-A188M-id-chat",
            "fadhilahkhairogi/Liwet-163M-id-chat",
            "fadhilahkhairogi/Liwet-66M-id-chat",
        ],
    )
    parser.add_argument("--local-dir", default=str(DEFAULT_LOCAL_MODELS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--greedy-steps", type=int, default=128)
    parser.add_argument("--greedy-limit", type=int, default=6)
    parser.add_argument("--cpu-prefill-limit", type=int, default=10)
    args = parser.parse_args()

    os.environ["LIWET_LOCAL_MODELS_DIR"] = args.local_dir
    lm = _load_ref_lm()
    out_root = Path(args.out_dir)
    for m in args.models:
        print(f"=== Dumping parity reference for {m} ===", flush=True)
        dump_model(
            lm,
            m,
            out_root,
            greedy_steps=args.greedy_steps,
            greedy_limit=args.greedy_limit,
            cpu_prefill_limit=args.cpu_prefill_limit,
        )


if __name__ == "__main__":
    main()
