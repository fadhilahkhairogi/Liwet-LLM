import argparse
import time
from pathlib import Path

import torch

from train_utils import (
    PROJECT_ROOT,
    autocast_context,
    get_optimizer,
    load_model,
    make_grad_scaler,
    maybe_compile,
    resolve_device,
    resolve_dtype,
    seed_everything,
)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "model")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--accum", type=int, default=10)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-checkpointing", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--full-bf16", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    model = load_model(args.model_dir, device)
    if args.full_bf16:
        model.to(torch.bfloat16)
    model.train()
    if args.no_checkpointing:
        model.gradient_checkpointing_disable()
    else:
        model.gradient_checkpointing_enable()
    if args.compile:
        model = maybe_compile(model)
    save_target = model._orig_mod if hasattr(model, "_orig_mod") else model

    optimizer = get_optimizer(model, args.lr, 0.1, (0.9, 0.95))
    scaler = make_grad_scaler(device, dtype)
    input_ids = torch.randint(
        4, save_target.config.vocab_size, (args.batch_size, args.seq_len), device=device
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    total_micro = (args.warmup_steps + args.steps) * args.accum
    measured_tokens = 0
    started = None
    optimizer.zero_grad(set_to_none=True)

    for micro_step in range(total_micro):
        if started is None and micro_step == args.warmup_steps * args.accum:
            synchronize(device)
            started = time.perf_counter()
            measured_tokens = 0

        with autocast_context(device, dtype):
            output = model(input_ids, labels=input_ids)
            loss = output.loss
        scaled_loss = loss / args.accum
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        measured_tokens += input_ids.numel()

        if (micro_step + 1) % args.accum == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    synchronize(device)
    elapsed = max(1e-6, time.perf_counter() - started)
    tokens_per_second = measured_tokens / elapsed
    peak = torch.cuda.max_memory_allocated() / 1e9
    reserved = torch.cuda.max_memory_reserved() / 1e9
    print(
        {
            "device": str(device),
            "dtype": dtype,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "accum": args.accum,
            "checkpointing": not args.no_checkpointing,
            "compile": args.compile,
            "full_bf16": args.full_bf16,
            "measured_steps": args.steps,
            "tokens": measured_tokens,
            "seconds": round(elapsed, 3),
            "tokens_per_second": round(tokens_per_second, 1),
            "peak_memory_gb": round(peak, 2),
            "reserved_memory_gb": round(reserved, 2),
        }
    )


if __name__ == "__main__":
    main()
