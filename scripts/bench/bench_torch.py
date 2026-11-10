"""Headless PyTorch inference benchmark for Liwet-LLM (prefill, decode, loop overhead)."""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCAL_MODELS = REPO_ROOT.parent / "TUBES_1_SAFETENSOR"


def _import_liwet(use_ref: bool):
    if use_ref:
        ref_dir = REPO_ROOT / "scripts" / "parity" / "reference"
        sys.path.insert(0, str(ref_dir))
        import liwet_models_ref as lm  # type: ignore
    else:
        space_dir = REPO_ROOT / "liwet-space"
        sys.path.insert(0, str(space_dir))
        import liwet_models as lm  # type: ignore
    return lm


class GpuSampler:
    def __init__(self, interval_s: float = 0.25) -> None:
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.samples: List[dict] = []

    def _poll(self) -> None:
        cmd = [
            "nvidia-smi",
            "--query-gpu=clocks.sm,power.draw,clocks_event_reasons.sw_power_cap,clocks_event_reasons.sw_thermal_slowdown,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(cmd, text=True, timeout=2.0).strip()
                if out:
                    parts = [p.strip() for p in out.split(",")]
                    if len(parts) >= 5:
                        self.samples.append(
                            {
                                "sm_mhz": float(parts[0]) if parts[0] not in {"N/A", "[N/A]"} else None,
                                "power_w": float(parts[1]) if parts[1] not in {"N/A", "[N/A]"} else None,
                                "sw_power_cap": parts[2],
                                "sw_thermal": parts[3],
                                "temp_c": float(parts[4]) if parts[4] not in {"N/A", "[N/A]"} else None,
                            }
                        )
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self._stop.clear()
        self.samples.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        clocks = [s["sm_mhz"] for s in self.samples if s["sm_mhz"] is not None]
        powers = [s["power_w"] for s in self.samples if s["power_w"] is not None]
        sw_cap = any("Active" in str(s["sw_power_cap"]) for s in self.samples)
        sw_therm = any("Active" in str(s["sw_thermal"]) for s in self.samples)
        return {
            "n_samples": len(self.samples),
            "avg_sm_mhz": round(statistics.mean(clocks), 1) if clocks else None,
            "max_sm_mhz": max(clocks) if clocks else None,
            "avg_power_w": round(statistics.mean(powers), 2) if powers else None,
            "sw_power_cap_active": sw_cap,
            "sw_thermal_active": sw_therm,
        }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round((pct / 100.0) * (len(s) - 1)))))
    return s[idx]


def bench_model(
    lm,
    repo_id: str,
    device_choice: str,
    dtype_choice: str,
    decode_tokens: int = 256,
    prefill_lengths: Optional[List[int]] = None,
) -> dict:
    if torch.cuda.is_available() and device_choice in {"auto", "cuda"}:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    bundle = lm.load_liwet_model(repo_id, device_name=device_choice, dtype_name=dtype_choice)
    _sync(bundle.device)
    load_s = time.perf_counter() - t0

    spec = bundle.spec
    ctx = bundle.context_len
    if prefill_lengths is None:
        prefill_lengths = [32, 512, min(1500, ctx - decode_tokens - 16)]
    else:
        prefill_lengths = [min(L, ctx - 64) for L in prefill_lengths]

    # Build a synthetic base text of repeated Indonesian tokens >= ctx
    sample_seed = (
        "Pengembangan model bahasa Indonesia skala ringkas membutuhkan evaluasi kecepatan inferensi "
        "pada perangkat keras lokal dan server secara objektif. "
    )
    all_ids = bundle.tokenizer.encode(sample_seed * 60, add_special_tokens=False).ids

    use_fast = (
        hasattr(lm, "_use_fastpath")
        and lm._use_fastpath()
        and hasattr(lm, "_get_static_decode_artifacts")
        and hasattr(bundle.model, "forward_prefill_static")
    )

    # Warmup 1 short prefill + 5 decode steps (and capture CUDA Graph if fastpath)
    warm_ids = torch.tensor([all_ids[:16]], dtype=torch.long, device=bundle.device)
    with torch.inference_mode():
        if use_fast:
            cache, runner = lm._get_static_decode_artifacts(bundle, want_cuda_graph=True)
            logits = lm.prefill_once(bundle, warm_ids)
            nxt = logits.argmax(dim=-1, keepdim=True)
            for s in range(5):
                if runner is not None:
                    logits = runner.step(nxt, 16 + s)
                else:
                    pos_buf = cache.open_position(16 + s)
                    logits = bundle.model.forward_decode_static(nxt, pos_buf, cache)
                nxt = logits.argmax(dim=-1, keepdim=True)
        else:
            out = bundle.model.forward(warm_ids, past_key_values=None, use_cache=True)
            pkv = out.past_key_values
            nxt = out.logits[:, -1:, :].argmax(dim=-1)
            for _ in range(5):
                out = bundle.model.forward(nxt, past_key_values=pkv, use_cache=True)
                pkv = out.past_key_values
                nxt = out.logits[:, -1:, :].argmax(dim=-1)
    _sync(bundle.device)

    # 1. Prefill benchmarks
    prefill_results = {}
    for L in prefill_lengths:
        inp = torch.tensor([all_ids[:L]], dtype=torch.long, device=bundle.device)
        times_ms = []
        with torch.inference_mode():
            for _ in range(3):
                _sync(bundle.device)
                ts = time.perf_counter()
                if use_fast:
                    _ = lm.prefill_once(bundle, inp)
                else:
                    _ = bundle.model.forward(inp, past_key_values=None, use_cache=True)
                _sync(bundle.device)
                times_ms.append((time.perf_counter() - ts) * 1000.0)
        med_ms = statistics.median(times_ms)
        prefill_results[str(L)] = {
            "median_ms": round(med_ms, 2),
            "tok_per_s": round(L / (med_ms / 1000.0), 1),
        }

    # 2. Pure forward + argmax decode (256 tokens from 64-tok prompt, no EOS stop)
    prompt_64 = torch.tensor([all_ids[:64]], dtype=torch.long, device=bundle.device)
    sampler = GpuSampler() if bundle.device.type == "cuda" else None
    if sampler:
        sampler.start()

    step_ms: List[float] = []
    with torch.inference_mode():
        _sync(bundle.device)
        if use_fast:
            cache, runner = lm._get_static_decode_artifacts(bundle, want_cuda_graph=True)
            logits = lm.prefill_once(bundle, prompt_64)
            nxt = logits.argmax(dim=-1, keepdim=True)
            _sync(bundle.device)
            use_cuda_events = bundle.device.type == "cuda"
            events = (
                [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(decode_tokens)]
                if use_cuda_events
                else []
            )
            t_dec0 = time.perf_counter()
            for idx_s in range(decode_tokens):
                if use_cuda_events:
                    events[idx_s][0].record()
                else:
                    t_step0 = time.perf_counter()
                if runner is not None:
                    logits = runner.step(nxt, 64 + idx_s)
                else:
                    pos_buf = cache.open_position(64 + idx_s)
                    logits = bundle.model.forward_decode_static(nxt, pos_buf, cache)
                nxt = logits.argmax(dim=-1, keepdim=True)
                if use_cuda_events:
                    events[idx_s][1].record()
                else:
                    step_ms.append((time.perf_counter() - t_step0) * 1000.0)
            _sync(bundle.device)
            t_dec_total = time.perf_counter() - t_dec0
            if use_cuda_events:
                step_ms = [e0.elapsed_time(e1) for e0, e1 in events]
        else:
            out = bundle.model.forward(prompt_64, past_key_values=None, use_cache=True)
            pkv = out.past_key_values
            nxt = out.logits[:, -1:, :].argmax(dim=-1)
            _sync(bundle.device)
            t_dec0 = time.perf_counter()
            for _ in range(decode_tokens):
                t_step0 = time.perf_counter()
                out = bundle.model.forward(nxt, past_key_values=pkv, use_cache=True)
                pkv = out.past_key_values
                nxt = out.logits[:, -1:, :].argmax(dim=-1)
                _sync(bundle.device)
                step_ms.append((time.perf_counter() - t_step0) * 1000.0)
            t_dec_total = time.perf_counter() - t_dec0

    gpu_info_pure = sampler.stop() if sampler else {}

    # 3. Full stream_generate loop (a: fixed N tokens by temporarily disabling stop_ids)
    orig_stop_fn = lm._stop_token_ids
    lm._stop_token_ids = lambda tok: set()
    prompt_text_64 = bundle.tokenizer.decode(all_ids[:64], skip_special_tokens=True)
    # Build clean role prompt if needed
    prompt_formatted = lm.build_prompt_from_messages(
        spec,
        [{"role": "user", "content": "Jelaskan secara rinci sejarah dan perkembangan komputasi GPU untuk AI."}],
        lm.DEFAULT_SYSTEM_PROMPT,
    )
    yield_times: List[float] = []
    stats_fixed: dict = {}
    if sampler:
        sampler.start()
    try:
        _sync(bundle.device)
        t_sg0 = time.perf_counter()
        t_prev = t_sg0
        for _partial in lm.stream_generate(
            bundle,
            prompt_formatted,
            max_new_tokens=decode_tokens,
            temperature=0.0,
            repetition_penalty=float(spec.recommended_params.get("repetition_penalty", 1.1)),
            stats=stats_fixed,
        ):
            now = time.perf_counter()
            yield_times.append((now - t_prev) * 1000.0)
            t_prev = now
        _sync(bundle.device)
        sg_wall_s = time.perf_counter() - t_sg0
    finally:
        lm._stop_token_ids = orig_stop_fn
    gpu_info_sg = sampler.stop() if sampler else {}

    # 4. Natural stream_generate (with stop tokens intact, realistic chat response)
    stats_nat: dict = {}
    _sync(bundle.device)
    t_nat0 = time.perf_counter()
    final_nat = ""
    for partial in lm.stream_generate(
        bundle,
        prompt_formatted,
        max_new_tokens=decode_tokens,
        temperature=0.0,
        repetition_penalty=float(spec.recommended_params.get("repetition_penalty", 1.1)),
        stats=stats_nat,
    ):
        final_nat = partial
    _sync(bundle.device)
    nat_wall_s = time.perf_counter() - t_nat0

    peak_vram_mb = (
        round(torch.cuda.max_memory_allocated(bundle.device) / (1024 * 1024), 1)
        if bundle.device.type == "cuda"
        else 0.0
    )

    res = {
        "repo_id": repo_id,
        "device": str(bundle.device),
        "dtype": str(bundle.dtype),
        "load_time_s": round(load_s, 3),
        "peak_vram_mb": peak_vram_mb,
        "prefill": prefill_results,
        "pure_forward_decode": {
            "tokens": decode_tokens,
            "total_s": round(t_dec_total, 4),
            "tok_per_s": round(decode_tokens / t_dec_total, 2),
            "ms_per_tok_p50": round(_percentile(step_ms, 50), 2),
            "ms_per_tok_p95": round(_percentile(step_ms, 95), 2),
            "gpu": gpu_info_pure,
        },
        "stream_generate_fixed": {
            "tokens": stats_fixed.get("generated_tokens", 0),
            "yield_count": len(yield_times),
            "model_seconds": round(stats_fixed.get("seconds", 0.0), 4),
            "wall_seconds": round(sg_wall_s, 4),
            "tok_per_s": round(stats_fixed.get("tokens_per_second", 0.0), 2),
            "yield_ms_p50": round(_percentile(yield_times, 50), 2),
            "yield_ms_p95": round(_percentile(yield_times, 95), 2),
            "gpu": gpu_info_sg,
        },
        "stream_generate_natural": {
            "tokens": stats_nat.get("generated_tokens", 0),
            "model_seconds": round(stats_nat.get("seconds", 0.0), 4),
            "wall_seconds": round(nat_wall_s, 4),
            "tok_per_s": round(stats_nat.get("tokens_per_second", 0.0), 2),
            "preview": final_nat[:120],
        },
    }

    lm.clear_model_cache()
    del bundle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res


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
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--decode-tokens", type=int, default=256)
    parser.add_argument("--ref", action="store_true", help="Use scripts/parity/reference/liwet_models_ref.py")
    parser.add_argument("--local-dir", default=str(DEFAULT_LOCAL_MODELS))
    parser.add_argument("--out", default=str(REPO_ROOT / "docs" / "bench" / "baseline_torch.json"))
    args = parser.parse_args()

    os.environ["LIWET_LOCAL_MODELS_DIR"] = args.local_dir
    lm = _import_liwet(args.ref)
    # Force pure local offline load (avoid HF network calls)
    orig_dl = lm._download_model_files
    lm._download_model_files = lambda repo_id, token=None: lm._find_local_model_files(repo_id) or orig_dl(repo_id, token=token)

    results = []
    for m in args.models:
        print(f"=== Benchmarking {m} on {args.device} ({args.dtype}) ===", flush=True)
        r = bench_model(lm, m, args.device, args.dtype, decode_tokens=args.decode_tokens)
        results.append(r)
        print(
            f"  pure={r['pure_forward_decode']['tok_per_s']} tok/s | "
            f"stream_fixed={r['stream_generate_fixed']['tok_per_s']} tok/s | "
            f"vram={r['peak_vram_mb']} MB | "
            f"sm_clk={r['pure_forward_decode']['gpu'].get('avg_sm_mhz')} MHz",
            flush=True,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
