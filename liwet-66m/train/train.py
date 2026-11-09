"""Train MiniTransformer from pretraining through post-training on the prepared bins.

The script consumes the token files produced by ``prepare_data.py`` and runs two
stages with one model:

1. pretrain  : next-token prediction on ``data_id/bins/pretrain_*.bin``
2. posttrain : continued next-token prediction on ``data_id/bins/posttrain_*.bin``

Examples
--------
python train.py
python train.py --stage pretrain --pretrain-steps 2000 --batch-size 4 --grad-accum 8
python train.py --stage both --resume auto
python train.py --init-from checkpoints/pretrain.pt --stage posttrain
"""

from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import nullcontext
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from config import GPTConfig
from model import build_model
from prepare_data import TokenDataset, get_batch


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MiniTransformer on pretrain + posttrain token bins."
    )

    data = parser.add_argument_group("data")
    data.add_argument("--data-dir", default="data_id", help="Directory made by prepare_data.py.")
    data.add_argument("--out-dir", default="checkpoints", help="Checkpoint/log output directory.")
    data.add_argument(
        "--stage",
        choices=["both", "pretrain", "posttrain"],
        default="both",
        help="Which training stage(s) to run.",
    )
    data.add_argument(
        "--resume",
        default=None,
        help="Checkpoint to resume from, or 'auto' for <out-dir>/latest.pt.",
    )
    data.add_argument(
        "--init-from",
        default=None,
        help="Checkpoint whose model weights are loaded without resuming optimizer/step.",
    )

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--device", default="auto", help="auto, cuda, cuda:0, cpu, ...")
    runtime.add_argument("--seed", type=int, default=1234)
    runtime.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use mixed precision on CUDA.",
    )
    runtime.add_argument(
        "--amp-dtype",
        choices=["auto", "bfloat16", "float16"],
        default="auto",
        help="Mixed precision dtype (auto picks bfloat16 when supported).",
    )

    model = parser.add_argument_group("model")
    model.add_argument("--block-size", type=int, default=None)
    model.add_argument("--n-layer", type=int, default=None)
    model.add_argument("--n-head", type=int, default=None)
    model.add_argument("--n-embd", type=int, default=None)
    model.add_argument("--mlp-hidden", type=int, default=None)
    model.add_argument("--dropout", type=float, default=None)
    model.add_argument("--no-tie-weights", action="store_true")

    training = parser.add_argument_group("training")
    training.add_argument("--batch-size", type=int, default=4)
    training.add_argument("--grad-accum", type=int, default=8)
    training.add_argument("--pretrain-epochs", type=float, default=1.0)
    training.add_argument("--posttrain-epochs", type=float, default=1.0)
    training.add_argument("--pretrain-steps", type=int, default=None)
    training.add_argument("--posttrain-steps", type=int, default=None)
    training.add_argument("--pretrain-lr", type=float, default=3e-4)
    training.add_argument("--posttrain-lr", type=float, default=3e-5)
    training.add_argument("--min-lr-ratio", type=float, default=0.1)
    training.add_argument("--warmup-ratio", type=float, default=0.03)
    training.add_argument("--weight-decay", type=float, default=0.1)
    training.add_argument("--beta1", type=float, default=0.9)
    training.add_argument("--beta2", type=float, default=0.95)
    training.add_argument("--grad-clip", type=float, default=1.0)
    training.add_argument(
        "--mask-assistant",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For chat posttrain data, compute loss only on assistant responses.",
    )

    logging = parser.add_argument_group("logging")
    logging.add_argument("--eval-interval", type=int, default=250)
    logging.add_argument("--eval-iters", type=int, default=50)
    logging.add_argument("--log-interval", type=int, default=10)
    logging.add_argument("--save-interval", type=int, default=1000)

    args = parser.parse_args()
    validate_args(parser, args)
    return args


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.resume and args.init_from:
        parser.error("--resume and --init-from cannot be used together.")
    if args.batch_size < 1 or args.grad_accum < 1:
        parser.error("--batch-size and --grad-accum must be >= 1.")
    if args.pretrain_epochs <= 0 or args.posttrain_epochs <= 0:
        parser.error("epoch counts must be > 0.")
    if args.pretrain_steps is not None and args.pretrain_steps < 1:
        parser.error("--pretrain-steps must be >= 1.")
    if args.posttrain_steps is not None and args.posttrain_steps < 1:
        parser.error("--posttrain-steps must be >= 1.")
    if args.eval_interval < 1 or args.save_interval < 1 or args.log_interval < 1:
        parser.error("intervals must be >= 1.")
    if not (0.0 <= args.min_lr_ratio <= 1.0):
        parser.error("--min-lr-ratio must be in [0, 1].")
    if not (0.0 <= args.warmup_ratio < 1.0):
        parser.error("--warmup-ratio must be in [0, 1).")


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")
    return device


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def load_model_weights(model: torch.nn.Module, checkpoint: dict) -> bool:
    state = dict(checkpoint["model"])
    old_vocab = int(checkpoint.get("config", {}).get("vocab_size", model.config.vocab_size))
    new_vocab = model.config.vocab_size
    if old_vocab == new_vocab:
        model.load_state_dict(state)
        return False
    if old_vocab > new_vocab:
        raise SystemExit(
            f"Cannot shrink checkpoint vocab {old_vocab:,} to data vocab {new_vocab:,}."
        )

    old_tok = state["tok_emb.weight"]
    new_tok = old_tok.new_empty((new_vocab, old_tok.size(1)))
    new_tok[:old_vocab].copy_(old_tok)
    torch.nn.init.normal_(new_tok[old_vocab:], mean=0.0, std=0.02)
    state["tok_emb.weight"] = new_tok

    if model.config.tie_weights:
        state["lm_head.weight"] = new_tok
    else:
        old_head = state["lm_head.weight"]
        new_head = old_head.new_empty((new_vocab, old_head.size(1)))
        new_head[:old_vocab].copy_(old_head)
        torch.nn.init.normal_(new_head[old_vocab:], mean=0.0, std=0.02)
        state["lm_head.weight"] = new_head

    model.load_state_dict(state)
    log(f"loaded weights with expanded vocab: {old_vocab:,} -> {new_vocab:,}")
    return True


def build_config(args: argparse.Namespace, vocab_size: int, checkpoint: dict | None) -> GPTConfig:
    if checkpoint is not None:
        known = {field.name for field in fields(GPTConfig)}
        values = {k: v for k, v in checkpoint["config"].items() if k in known}
        config = GPTConfig(**values)
    else:
        config = GPTConfig(
            vocab_size=vocab_size,
            block_size=args.block_size if args.block_size is not None else GPTConfig().block_size,
            n_layer=args.n_layer if args.n_layer is not None else GPTConfig().n_layer,
            n_head=args.n_head if args.n_head is not None else GPTConfig().n_head,
            n_embd=args.n_embd if args.n_embd is not None else GPTConfig().n_embd,
            mlp_hidden=args.mlp_hidden if args.mlp_hidden is not None else GPTConfig().mlp_hidden,
            dropout=args.dropout if args.dropout is not None else GPTConfig().dropout,
            tie_weights=not args.no_tie_weights,
        )

    if config.vocab_size > vocab_size:
        raise SystemExit(
            f"Checkpoint vocab_size={config.vocab_size} is larger than data vocab_size={vocab_size}."
        )
    if config.vocab_size < vocab_size:
        log(
            f"expanding vocab {config.vocab_size:,} -> {vocab_size:,} "
            f"(new tokenizer special tokens)"
        )
        config.vocab_size = vocab_size
    if config.n_embd % config.n_head != 0:
        raise SystemExit(
            f"n_embd={config.n_embd} must be divisible by n_head={config.n_head}."
        )
    return config


def autocast_context(enabled: bool, device: torch.device, dtype: torch.dtype):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def plan_steps(args: argparse.Namespace, stage: str, num_tokens: int, tokens_per_step: int) -> int:
    explicit = args.pretrain_steps if stage == "pretrain" else args.posttrain_steps
    if explicit is not None:
        return explicit
    epochs = args.pretrain_epochs if stage == "pretrain" else args.posttrain_epochs
    return max(1, math.ceil(num_tokens * epochs / tokens_per_step))


def lr_at(step: int, total_steps: int, base_lr: float, min_lr_ratio: float, warmup_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    min_lr = base_lr * min_lr_ratio
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (base_lr - min_lr)


def build_assistant_mask(targets: torch.Tensor, assistant_id: int, end_id: int) -> torch.Tensor:
    starts = (targets == assistant_id).to(torch.int64)
    ends = (targets == end_id).to(torch.int64)
    cumulative = torch.cumsum(starts - ends, dim=1)
    mask = torch.zeros_like(targets, dtype=torch.bool)
    mask[:, 1:] = cumulative[:, :-1] > 0
    return mask


def compute_loss(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    if mask is None or not mask.any():
        _, loss = model(x, y)
        return loss
    logits, _ = model(x)
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_targets = y.reshape(-1)
    flat_mask = mask.reshape(-1)
    selected_targets = flat_targets[flat_mask]
    if selected_targets.numel() == 0:
        _, loss = model(x, y)
        return loss
    loss_sum = F.cross_entropy(flat_logits[flat_mask], selected_targets, reduction="sum")
    return loss_sum / selected_targets.numel()


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: TokenDataset,
    config: GPTConfig,
    args: argparse.Namespace,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    loss_ids: tuple[int, int] | None = None,
) -> float:
    model.eval()
    rng = np.random.default_rng(args.seed)
    total = 0.0
    count = 0
    for _ in range(args.eval_iters):
        x, y = get_batch(dataset.data, args.batch_size, config.block_size, rng)
        x = torch.from_numpy(x).to(device)
        y = torch.from_numpy(y).to(device)
        mask = build_assistant_mask(y, loss_ids[0], loss_ids[1]) if loss_ids else None
        with autocast_context(amp_enabled, device, amp_dtype):
            loss = compute_loss(model, x, y, mask)
        total += loss.item()
        count += 1
    model.train()
    return total / max(count, 1)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    config: GPTConfig,
    stage: str,
    step: int,
    steps: int,
    best_val: float,
    tokens_seen: int,
    args: argparse.Namespace,
    device: torch.device,
    rng: np.random.Generator,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "config": config.to_dict(),
        "stage": stage,
        "step": step,
        "steps": steps,
        "best_val": best_val,
        "tokens_seen": tokens_seen,
        "args": vars(args),
        "rng_np": rng.bit_generator.state,
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
    }
    torch.save(state, path)


def train_stage(
    stage: str,
    model: torch.nn.Module,
    config: GPTConfig,
    args: argparse.Namespace,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    scaler: torch.amp.GradScaler | None,
    resume: dict | None,
    loss_ids: tuple[int, int] | None = None,
) -> tuple[float, int]:
    train_ds = TokenDataset.from_out_dir(args.data_dir, stage, "train", block_size=config.block_size)
    val_ds = TokenDataset.from_out_dir(args.data_dir, stage, "val", block_size=config.block_size)

    tokens_per_step = args.batch_size * args.grad_accum * config.block_size
    total_steps = plan_steps(args, stage, len(train_ds), tokens_per_step)
    base_lr = args.pretrain_lr if stage == "pretrain" else args.posttrain_lr
    warmup_steps = min(total_steps, max(0, int(total_steps * args.warmup_ratio)))

    out_dir = Path(args.out_dir).expanduser().resolve()
    latest_path = out_dir / "latest.pt"
    stage_path = out_dir / f"{stage}.pt"
    best_path = out_dir / f"{stage}_best.pt"

    rng = np.random.default_rng(args.seed)
    optimizer = model.configure_optimizers(
        args.weight_decay, base_lr, (args.beta1, args.beta2), device.type
    )

    start_step = 0
    best_val = float("inf")
    tokens_seen = 0
    if resume is not None:
        start_step = int(resume.get("step", 0))
        best_val = float(resume.get("best_val", float("inf")))
        tokens_seen = int(resume.get("tokens_seen", 0))
        if resume.get("optimizer") is not None:
            optimizer.load_state_dict(resume["optimizer"])
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(device)
        if resume.get("rng_np") is not None:
            rng.bit_generator.state = resume["rng_np"]
        if resume.get("rng_torch") is not None:
            torch.set_rng_state(resume["rng_torch"])
        if device.type == "cuda" and resume.get("rng_cuda"):
            torch.cuda.set_rng_state_all(resume["rng_cuda"])

    log(
        f"[{stage}] train_tokens={len(train_ds):,} val_tokens={len(val_ds):,} "
        f"steps={total_steps:,} batch={args.batch_size} accum={args.grad_accum} "
        f"tokens/step={tokens_per_step:,} peak_lr={base_lr:g}"
    )
    if start_step >= total_steps:
        log(f"[{stage}] already completed at step {start_step:,}/{total_steps:,}; skipping.")
        return best_val, total_steps

    val_loss = evaluate(model, val_ds, config, args, device, amp_enabled, amp_dtype, loss_ids)
    if val_loss < best_val:
        best_val = val_loss
        save_checkpoint(
            best_path, model, None, config, stage, start_step, total_steps,
            best_val, tokens_seen, args, device, rng,
        )
    log(f"[{stage}] step {start_step:,}/{total_steps:,} | val {val_loss:.4f} | tokens {tokens_seen:,}")

    model.train()
    for step in range(start_step, total_steps):
        lr = lr_at(step, total_steps, base_lr, args.min_lr_ratio, warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        step_start = time.perf_counter()
        loss_sum = torch.zeros((), device=device)
        for _ in range(args.grad_accum):
            x, y = get_batch(train_ds.data, args.batch_size, config.block_size, rng)
            x = torch.from_numpy(x).to(device)
            y = torch.from_numpy(y).to(device)
            mask = build_assistant_mask(y, loss_ids[0], loss_ids[1]) if loss_ids else None
            with autocast_context(amp_enabled, device, amp_dtype):
                loss = compute_loss(model, x, y, mask)
            loss = loss / args.grad_accum
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            loss_sum += loss.detach()

        if scaler is not None:
            scaler.unscale_(optimizer)
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        tokens_seen += tokens_per_step
        elapsed = max(time.perf_counter() - step_start, 1e-9)
        step += 1

        if step % args.log_interval == 0 or step == total_steps:
            log(
                f"[{stage}] step {step:,}/{total_steps:,} | loss {loss_sum.item():.4f} "
                f"| lr {lr:.3e} | tokens {tokens_seen:,} "
                f"| {tokens_per_step / elapsed:,.0f} tok/s"
            )

        if step % args.eval_interval == 0 or step == total_steps:
            val_loss = evaluate(
                model, val_ds, config, args, device, amp_enabled, amp_dtype, loss_ids
            )
            improved = val_loss < best_val
            if improved:
                best_val = val_loss
                save_checkpoint(
                    best_path, model, None, config, stage, step, total_steps,
                    best_val, tokens_seen, args, device, rng,
                )
            log(
                f"[{stage}] eval step {step:,}/{total_steps:,} | val {val_loss:.4f} "
                f"| best {best_val:.4f}"
            )

        if step % args.save_interval == 0:
            save_checkpoint(
                latest_path, model, optimizer, config, stage, step, total_steps,
                best_val, tokens_seen, args, device, rng,
            )

    step = total_steps
    save_checkpoint(
        latest_path, model, optimizer, config, stage, step, total_steps,
        best_val, tokens_seen, args, device, rng,
    )
    save_checkpoint(
        stage_path, model, optimizer, config, stage, step, total_steps,
        best_val, tokens_seen, args, device, rng,
    )
    log(f"[{stage}] done | best_val {best_val:.4f} | checkpoint {stage_path}")
    return best_val, total_steps


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    bins_meta = load_json(Path(args.data_dir) / "bins" / "meta.json")
    vocab_size = int(bins_meta["vocab_size"])

    loss_ids = None
    if args.mask_assistant:
        tokenizer_meta_path = Path(args.data_dir) / "tokenizer" / "meta.json"
        if tokenizer_meta_path.exists():
            special = load_json(tokenizer_meta_path).get("special_tokens", {})
            assistant_id = special.get("<|assistant|>")
            end_id = special.get("<|end|>")
            if assistant_id is not None and end_id is not None:
                loss_ids = (int(assistant_id), int(end_id))
                log(
                    f"assistant loss mask: assistant_id={loss_ids[0]} end_id={loss_ids[1]}"
                )
            else:
                log("assistant loss mask: disabled (no chat special tokens)")
        else:
            log("assistant loss mask: disabled (tokenizer/meta.json missing)")

    checkpoint = None
    init_only = False
    if args.resume:
        resume_path = (
            Path(args.out_dir).expanduser().resolve() / "latest.pt"
            if args.resume == "auto"
            else Path(args.resume)
        )
        checkpoint = load_checkpoint(resume_path)
    elif args.init_from:
        checkpoint = load_checkpoint(Path(args.init_from))
        init_only = True

    config = build_config(args, vocab_size, checkpoint)
    model = build_model(config)
    if checkpoint is not None:
        expanded = load_model_weights(model, checkpoint)
        if expanded and args.resume:
            log("vocab expanded; optimizer state will be reset for this run")
            checkpoint["optimizer"] = None
    model.to(device)

    amp_enabled = bool(args.amp and device.type == "cuda")
    if amp_enabled:
        if args.amp_dtype == "auto":
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif args.amp_dtype == "bfloat16":
            amp_dtype = torch.bfloat16
        else:
            amp_dtype = torch.float16
        scaler = (
            torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
            if amp_dtype == torch.float16
            else None
        )
    else:
        amp_dtype = torch.float32
        scaler = None

    log(f"device        : {device}")
    log(f"amp           : {amp_enabled} ({amp_dtype})")
    log(f"config        : {config.to_dict()}")
    log(f"parameters    : {model.num_parameters():,} (~{model.num_parameters() / 1e6:.2f}M)")

    requested = ["pretrain", "posttrain"] if args.stage == "both" else [args.stage]
    stages = requested
    if args.resume and checkpoint is not None:
        checkpoint_stage = checkpoint.get("stage")
        if checkpoint_stage == "posttrain":
            if "posttrain" not in requested:
                raise SystemExit(
                    "Checkpoint is already in posttrain; use --stage posttrain (or both)."
                )
            stages = ["posttrain"]
        elif checkpoint_stage != "pretrain":
            raise SystemExit(f"Unknown checkpoint stage: {checkpoint_stage!r}")

    for stage in stages:
        resume_this = None
        if (
            not init_only
            and checkpoint is not None
            and checkpoint.get("stage") == stage
        ):
            resume_this = checkpoint
        train_stage(
            stage,
            model,
            config,
            args,
            device,
            amp_enabled,
            amp_dtype,
            scaler,
            resume_this,
            loss_ids if stage == "posttrain" else None,
        )
        if resume_this is not None:
            checkpoint = None

    log("all requested stages finished.")


if __name__ == "__main__":
    main()
