import argparse
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from train_utils import (
    PROJECT_ROOT,
    JsonlLogger,
    args_to_dict,
    autocast_context,
    copy_tokenizer,
    default_num_workers,
    get_optimizer,
    load_model,
    load_training_state,
    make_grad_scaler,
    make_scheduler,
    maybe_compile,
    model_summary,
    prevent_system_sleep,
    prune_checkpoints,
    resolve_data_path,
    resolve_device,
    resolve_dtype,
    save_checkpoint,
    save_json,
    seed_everything,
    setup_logging,
)


class TokenShards:
    def __init__(self, index_path: Path, split: str):
        self.index_path = Path(index_path)
        index = json.loads(self.index_path.read_text(encoding="utf-8"))
        self.base_dir = self.index_path.parent
        entries = index[split]["files"]
        self.arrays = []
        self.files = []
        for entry in entries:
            path = self.base_dir / entry["path"]
            array = np.memmap(path, dtype=np.uint16, mode="r")
            if len(array) == 0:
                continue
            self.arrays.append(array)
            self.files.append(path)
        if not self.arrays:
            raise ValueError(f"tidak ada file token untuk split {split}")
        self.lengths = np.array([len(array) for array in self.arrays], dtype=np.int64)
        self.total_tokens = int(self.lengths.sum())
        self.probabilities = self.lengths / self.total_tokens

    def sample(self, rng: np.random.Generator, seq_len: int):
        while True:
            file_index = int(rng.choice(len(self.arrays), p=self.probabilities))
            array = self.arrays[file_index]
            if len(array) <= seq_len:
                continue
            start = int(rng.integers(0, len(array) - seq_len + 1))
            sequence = np.asarray(array[start : start + seq_len], dtype=np.int64)
            return torch.from_numpy(sequence)


class RandomPackedDataset(IterableDataset):
    def __init__(self, corpus: TokenShards, seq_len: int, seed: int):
        self.corpus = corpus
        self.seq_len = seq_len
        self.seed = seed

    def __iter__(self):
        rng = np.random.default_rng(self.seed + torch.initial_seed() % (2**32))
        while True:
            yield {"input_ids": self.corpus.sample(rng, self.seq_len)}


class FixedPackedDataset(Dataset):
    def __init__(self, corpus: TokenShards, seq_len: int, max_sequences: int):
        self.corpus = corpus
        self.seq_len = seq_len
        self.positions = []
        for file_index, length in enumerate(corpus.lengths):
            count = int(length) // seq_len
            for offset in range(count):
                self.positions.append((file_index, offset * seq_len))
                if len(self.positions) >= max_sequences:
                    break
            if len(self.positions) >= max_sequences:
                break
        if not self.positions:
            raise ValueError("tidak ada sequence validasi")

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        file_index, start = self.positions[index]
        array = self.corpus.arrays[file_index]
        sequence = np.asarray(array[start : start + self.seq_len], dtype=np.int64)
        return {"input_ids": torch.from_numpy(sequence)}


def evaluate(model, loader, device, dtype, max_batches: int) -> float:
    model.eval()
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        with autocast_context(device, dtype):
            for batch in loader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                output = model(input_ids, labels=input_ids)
                total_loss += float(output.loss.detach())
                batches += 1
                if batches >= max_batches:
                    break
    model.train()
    return total_loss / max(1, batches)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "model")
    parser.add_argument("--data-index", type=Path, default=PROJECT_ROOT / "data" / "tokenized" / "index.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "checkpoints" / "pretrain")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-hours", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--keep-checkpoints", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--full-bf16", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=default_num_workers())
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--compile-mode",
        choices=[
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ],
        default="max-autotune-no-cudagraphs",
    )
    args = parser.parse_args()

    prevent_system_sleep()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    if args.full_bf16:
        dtype = "bf16"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.output_dir / "train.log")
    metrics = JsonlLogger(args.output_dir / "metrics.jsonl")
    logging.info("device=%s dtype=%s", device, dtype)
    args.data_index = resolve_data_path(args.data_index)

    train_corpus = TokenShards(args.data_index, "train")
    val_corpus = TokenShards(args.data_index, "val")
    tokens_per_step = args.batch_size * args.grad_accum * args.seq_len
    if args.max_steps > 0:
        total_steps = args.max_steps
    else:
        total_steps = int(train_corpus.total_tokens * args.epochs // tokens_per_step)
    total_steps = max(1, total_steps)
    logging.info(
        "train_tokens=%d val_tokens=%d tokens_per_step=%d total_steps=%d",
        train_corpus.total_tokens,
        val_corpus.total_tokens,
        tokens_per_step,
        total_steps,
    )

    if args.resume:
        model = load_model(args.resume, device)
    else:
        model = load_model(args.model_dir, device)
    if args.full_bf16:
        model.to(torch.bfloat16)
        logging.info("model weights: bf16")
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_disable()
    if args.compile:
        model = maybe_compile(
            model, mode=args.compile_mode, dynamic=False
        )
    save_target = model._orig_mod if hasattr(model, "_orig_mod") else model
    if args.seq_len > save_target.config.max_position_embeddings:
        raise SystemExit("seq_len melebihi max_position_embeddings model")
    logging.info("model: %s", model_summary(save_target))

    train_dataset = RandomPackedDataset(train_corpus, args.seq_len, args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    val_dataset = FixedPackedDataset(
        val_corpus, args.seq_len, max(args.eval_batches * args.batch_size, args.batch_size)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    optimizer = get_optimizer(model, args.lr, args.weight_decay, tuple(args.betas))
    scheduler = make_scheduler(optimizer, args.warmup_steps, total_steps, args.min_lr_ratio)
    scaler = make_grad_scaler(device, dtype)

    global_step = 0
    best_loss = float("inf")
    if args.resume and (args.resume / "training_state.pt").exists():
        global_step, best_loss = load_training_state(
            args.resume, optimizer, scheduler, scaler, device
        )
        logging.info("resume dari step %d", global_step)

    tokenizer_dir = PROJECT_ROOT / "data" / "tokenizer"
    save_json(args.output_dir / "trainer_args.json", args_to_dict(args))

    model.train()
    train_iterator = iter(train_loader)
    micro_step = 0
    tokens_processed = 0
    started_at = time.time()
    deadline = started_at + args.max_hours * 3600 if args.max_hours > 0 else None
    last_log_time = started_at
    last_log_tokens = 0
    optimizer.zero_grad(set_to_none=True)

    while global_step < total_steps:
        batch = next(train_iterator)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        with autocast_context(device, dtype):
            output = model(input_ids, labels=input_ids)
            loss = output.loss
        scaled_loss = loss / args.grad_accum
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        micro_step += 1
        tokens_processed += input_ids.numel()

        if micro_step % args.grad_accum != 0:
            continue

        if scaler is not None:
            scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

        if global_step % args.log_interval == 0 or global_step == 1:
            if device.type == "cuda":
                torch.cuda.synchronize()
            now = time.time()
            elapsed = max(1e-6, now - started_at)
            interval_tokens = tokens_processed - last_log_tokens
            interval_seconds = max(1e-6, now - last_log_time)
            lr = scheduler.get_last_lr()[0]
            tokens_per_second = interval_tokens / interval_seconds
            average_tokens_per_second = tokens_processed / elapsed
            peak_memory = (
                torch.cuda.max_memory_allocated() / 1e9
                if device.type == "cuda"
                else 0.0
            )
            aux_loss = getattr(output, "aux_loss", None)
            aux_loss_value = (
                float(aux_loss.detach()) if aux_loss is not None else 0.0
            )
            logging.info(
                "step=%d/%d loss=%.4f ppl=%.2f aux=%.4f lr=%.2e grad=%.3f tok/s=%.0f avg_tok/s=%.0f mem=%.1fGB",
                global_step,
                total_steps,
                float(loss.detach()),
                math.exp(min(20.0, float(loss.detach()))),
                aux_loss_value,
                lr,
                float(grad_norm),
                tokens_per_second,
                average_tokens_per_second,
                peak_memory,
            )
            metrics.log(
                {
                    "step": global_step,
                    "train_loss": float(loss.detach()),
                    "aux_loss": aux_loss_value,
                    "lr": lr,
                    "grad_norm": float(grad_norm),
                    "tokens": tokens_processed,
                    "tokens_per_second": tokens_per_second,
                    "average_tokens_per_second": average_tokens_per_second,
                    "peak_memory_gb": peak_memory,
                }
            )
            last_log_time = now
            last_log_tokens = tokens_processed

        if args.eval_interval > 0 and global_step % args.eval_interval == 0:
            val_loss = evaluate(model, val_loader, device, dtype, args.eval_batches)
            logging.info("eval step=%d val_loss=%.4f val_ppl=%.2f", global_step, val_loss, math.exp(min(20.0, val_loss)))
            metrics.log({"step": global_step, "val_loss": val_loss, "val_ppl": math.exp(min(20.0, val_loss))})
            if val_loss < best_loss:
                best_loss = val_loss
            model.train()
            last_log_time = time.time()
            last_log_tokens = tokens_processed

        if args.save_interval > 0 and global_step % args.save_interval == 0:
            save_checkpoint(
                args.output_dir / f"step_{global_step:07d}",
                save_target,
                optimizer,
                scheduler,
                scaler,
                global_step,
                best_loss,
                args_to_dict(args),
                tokenizer_dir,
            )
            prune_checkpoints(args.output_dir, args.keep_checkpoints)
            logging.info("checkpoint disimpan step=%d", global_step)
            last_log_time = time.time()
            last_log_tokens = tokens_processed

        if deadline is not None and time.time() >= deadline:
            logging.info("batas waktu %.1f jam tercapai pada step %d", args.max_hours, global_step)
            break

    val_loss = evaluate(model, val_loader, device, dtype, args.eval_batches)
    logging.info("eval final val_loss=%.4f val_ppl=%.2f", val_loss, math.exp(min(20.0, val_loss)))
    metrics.log({"step": global_step, "val_loss": val_loss, "val_ppl": math.exp(min(20.0, val_loss))})

    final_dir = args.output_dir / "final"
    save_checkpoint(
        final_dir,
        save_target,
        optimizer,
        scheduler,
        scaler,
        global_step,
        min(best_loss, val_loss),
        args_to_dict(args),
        tokenizer_dir,
    )
    copy_tokenizer(tokenizer_dir, final_dir)
    logging.info("selesai | step=%d | model=%s", global_step, final_dir)
    metrics.close()


if __name__ == "__main__":
    main()
