"""Profile 20 decode tokens with torch.profiler and summarize kernel count, GPU idle %, top ops."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCAL_MODELS = REPO_ROOT.parent / "TUBES_1_SAFETENSOR"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="fadhilahkhairogi/Liwet-400M-A188M-id-chat")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--local-dir", default=str(DEFAULT_LOCAL_MODELS))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "docs" / "bench"))
    args = parser.parse_args()

    os.environ["LIWET_LOCAL_MODELS_DIR"] = args.local_dir
    sys.path.insert(0, str(REPO_ROOT / "liwet-space"))
    import liwet_models as lm  # type: ignore

    orig_dl = lm._download_model_files
    lm._download_model_files = lambda repo_id, token=None: lm._find_local_model_files(repo_id) or orig_dl(repo_id, token=token)

    bundle = lm.load_liwet_model(args.model, device_name="cuda", dtype_name="auto")
    short = args.model.split("/")[-1]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / f"trace_{short}.json"
    summary_path = out_dir / f"profile_{short}.json"

    ids = bundle.tokenizer.encode("Teknologi kecerdasan buatan di Indonesia berkembang pesat. " * 20, add_special_tokens=False).ids[: args.prompt_len]
    inp = torch.tensor([ids], dtype=torch.long, device=bundle.device)

    with torch.inference_mode():
        out = bundle.model.forward(inp, past_key_values=None, use_cache=True)
        pkv = out.past_key_values
        nxt = out.logits[:, -1:, :].argmax(dim=-1)
        # Warmup 3 steps
        for _ in range(3):
            out = bundle.model.forward(nxt, past_key_values=pkv, use_cache=True)
            pkv = out.past_key_values
            nxt = out.logits[:, -1:, :].argmax(dim=-1)
        torch.cuda.synchronize()

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            with_stack=False,
        ) as prof:
            for _ in range(args.steps):
                out = bundle.model.forward(nxt, past_key_values=pkv, use_cache=True)
                pkv = out.past_key_values
                nxt = out.logits[:, -1:, :].argmax(dim=-1)
            torch.cuda.synchronize()

    prof.export_chrome_trace(str(trace_path))

    # Inspect Chrome trace events for exact kernel count and GPU busy time
    trace_data = json.loads(trace_path.read_text(encoding="utf-8"))
    events = trace_data.get("traceEvents", [])
    kernel_events = [
        e for e in events
        if e.get("ph") == "X" and e.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"}
    ]
    cuda_kernels = [e for e in events if e.get("ph") == "X" and e.get("cat") == "kernel"]
    if kernel_events:
        t_min = min(e["ts"] for e in kernel_events)
        t_max = max(e["ts"] + e["dur"] for e in kernel_events)
        span_us = max(1.0, t_max - t_min)
        # Merge intervals to avoid double-counting concurrent streams
        intervals = sorted((e["ts"], e["ts"] + e["dur"]) for e in kernel_events)
        merged = [list(intervals[0])]
        for s, end in intervals[1:]:
            if s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([s, end])
        gpu_busy_us = sum(e - s for s, e in merged)
        gpu_idle_pct = round(100.0 * (1.0 - gpu_busy_us / span_us), 2)
    else:
        span_us = 0.0
        gpu_busy_us = 0.0
        gpu_idle_pct = 100.0

    avgs = prof.key_averages()
    top_cpu = sorted(avgs, key=lambda x: x.self_cpu_time_total, reverse=True)[:15]
    top_cuda = sorted(avgs, key=lambda x: x.self_device_time_total, reverse=True)[:15]

    summary = {
        "model": args.model,
        "steps": args.steps,
        "total_cuda_kernels": len(cuda_kernels),
        "kernels_per_token": round(len(cuda_kernels) / args.steps, 1),
        "wall_span_ms": round(span_us / 1000.0, 2),
        "ms_per_token": round(span_us / 1000.0 / args.steps, 2),
        "tok_per_s": round(args.steps / (span_us / 1e6), 2) if span_us > 0 else 0.0,
        "gpu_busy_ms": round(gpu_busy_us / 1000.0, 2),
        "gpu_idle_pct": gpu_idle_pct,
        "trace_file": str(trace_path),
        "trace_mb": round(trace_path.stat().st_size / (1024 * 1024), 2),
        "top_cpu_ops": [
            {
                "name": item.key,
                "calls": item.count,
                "self_cpu_ms": round(item.self_cpu_time_total / 1000.0, 3),
                "cuda_ms": round(item.self_device_time_total / 1000.0, 3),
            }
            for item in top_cpu
        ],
        "top_cuda_ops": [
            {
                "name": item.key,
                "calls": item.count,
                "self_cuda_ms": round(item.self_device_time_total / 1000.0, 3),
                "self_cpu_ms": round(item.self_cpu_time_total / 1000.0, 3),
            }
            for item in top_cuda
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"top_cpu_ops", "top_cuda_ops"}}, indent=2))
    print("Top CPU ops:")
    for op in summary["top_cpu_ops"][:10]:
        print(f"  {op['self_cpu_ms']:7.2f} ms | calls={op['calls']:5d} | {op['name']}")


if __name__ == "__main__":
    main()
