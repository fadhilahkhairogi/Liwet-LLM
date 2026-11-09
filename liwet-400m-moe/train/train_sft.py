import argparse
import json
import logging
import math
import time
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer

from chat_format import encode_sft, render_messages
from train_utils import (
    PROJECT_ROOT,
    BucketBatchSampler,
    JsonlLogger,
    args_to_dict,
    autocast_context,
    chunked_cross_entropy,
    copy_tokenizer,
    default_num_workers,
    get_optimizer,
    load_model,
    load_training_state,
    make_grad_scaler,
    make_scheduler,
    maybe_compile,
    model_summary,
    prune_checkpoints,
    resolve_data_path,
    resolve_device,
    resolve_dtype,
    save_checkpoint,
    save_json,
    seed_everything,
    setup_logging,
)


class SFTDataset(Dataset):
    def __init__(self, examples: list):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def read_jsonl(path: Path):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def prepare_examples(path: Path, tokenizer: Tokenizer, max_len: int, max_samples: int, label: str) -> list:
    examples = []
    skipped = 0
    seen = 0
    for record in read_jsonl(path):
        seen += 1
        encoded = encode_sft(tokenizer, record["messages"], max_len)
        if encoded is None:
            skipped += 1
            continue
        input_ids, labels = encoded
        if sum(1 for label in labels if label != -100) < 8:
            skipped += 1
            continue
        examples.append(
            {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            }
        )
        if seen % 20000 == 0:
            logging.info("%s | diproses=%d valid=%d", label, seen, len(examples))
        if max_samples > 0 and len(examples) >= max_samples:
            break
    logging.info("%s | total=%d valid=%d dilewati=%d", label, seen, len(examples), skipped)
    return examples


def collate_sft(batch: list, pad_token_id: int):
    max_len = max(item["input_ids"].numel() for item in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for row, item in enumerate(batch):
        length = item["input_ids"].numel()
        input_ids[row, :length] = item["input_ids"]
        labels[row, :length] = item["labels"]
        attention_mask[row, :length] = 1
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def evaluate_sft(model, loader, device, dtype, max_batches: int):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    batches = 0
    with torch.no_grad():
        with autocast_context(device, dtype):
            for batch in loader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                output = model(input_ids, attention_mask=attention_mask)
                loss, label_tokens = chunked_cross_entropy(output.logits, labels)
                total_loss += float(loss.detach()) * int(label_tokens)
                total_tokens += int(label_tokens)
                batches += 1
                if batches >= max_batches:
                    break
    model.train()
    return total_loss / max(1, total_tokens)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "pretrain" / "final")
    parser.add_argument("--train-file", type=Path, default=PROJECT_ROOT / "data" / "sft" / "train.jsonl")
    parser.add_argument("--val-file", type=Path, default=PROJECT_ROOT / "data" / "sft" / "val.jsonl")
    parser.add_argument("--tokenizer-dir", type=Path, default=PROJECT_ROOT / "data" / "tokenizer")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "checkpoints" / "sft")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--max-hours", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--keep-checkpoints", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--length-bucket",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--bucket-multiplier", type=int, default=8)
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
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
        default=False,
    )
    parser.add_argument(
        "--compile-mode",
        choices=[
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ],
        default="default",
    )
    parser.add_argument("--preview-samples", type=int, default=0)
    parser.add_argument("--no-final-save", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.output_dir / "train.log")
    metrics = JsonlLogger(args.output_dir / "metrics.jsonl")
    logging.info("device=%s dtype=%s", device, dtype)

    tokenizer = Tokenizer.from_file(str(args.tokenizer_dir / "tokenizer.json"))
    pad_token_id = tokenizer.token_to_id("<pad>")
    args.train_file = resolve_data_path(args.train_file)
    args.val_file = resolve_data_path(args.val_file)
    train_examples = prepare_examples(
        args.train_file, tokenizer, args.max_seq_len, args.max_samples, "train"
    )
    val_examples = prepare_examples(
        args.val_file, tokenizer, args.max_seq_len, min(args.max_samples, args.eval_batches * args.batch_size)
        if args.max_samples > 0
        else 0,
        "val",
    )
    if not train_examples or not val_examples:
        raise SystemExit("dataset SFT kosong")

    if args.length_bucket:
        batch_sampler = BucketBatchSampler(
            [item["input_ids"].numel() for item in train_examples],
            args.batch_size,
            shuffle=True,
            seed=args.seed,
            bucket_multiplier=args.bucket_multiplier,
        )
        train_loader = DataLoader(
            SFTDataset(train_examples),
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
            prefetch_factor=4 if args.num_workers > 0 else None,
            collate_fn=partial(collate_sft, pad_token_id=pad_token_id),
        )
        logging.info(
            "length-bucket aktif | batch_per_epoch=%d bucket=%d",
            len(batch_sampler),
            batch_sampler.bucket_size,
        )
    else:
        train_loader = DataLoader(
            SFTDataset(train_examples),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=True,
            persistent_workers=args.num_workers > 0,
            prefetch_factor=4 if args.num_workers > 0 else None,
            collate_fn=partial(collate_sft, pad_token_id=pad_token_id),
        )
    val_loader = DataLoader(
        SFTDataset(val_examples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=partial(collate_sft, pad_token_id=pad_token_id),
    )
    if len(train_loader) == 0:
        raise SystemExit("train_loader kosong, kurangi batch size")

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * args.epochs
    total_steps = max(1, total_steps)
    logging.info(
        "train=%d val=%d steps_per_epoch=%d total_steps=%d",
        len(train_examples),
        len(val_examples),
        steps_per_epoch,
        total_steps,
    )

    if args.resume:
        model = load_model(args.resume, device)
    else:
        model = load_model(args.base_checkpoint, device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_disable()
    if args.compile:
        model = maybe_compile(
            model, mode=args.compile_mode, dynamic=True
        )
    save_target = model._orig_mod if hasattr(model, "_orig_mod") else model
    if args.max_seq_len > save_target.config.max_position_embeddings:
        raise SystemExit("max_seq_len melebihi max_position_embeddings model")
    logging.info("model: %s", model_summary(save_target))

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

    save_json(args.output_dir / "trainer_args.json", args_to_dict(args))
    model.train()
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    tokens_processed = 0
    started_at = time.time()
    deadline = (
        started_at + args.max_hours * 3600 if args.max_hours > 0 else None
    )
    last_log_time = started_at
    last_log_tokens = 0
    stop_training = False

    def optimizer_step() -> None:
        nonlocal global_step, best_loss, last_log_time, last_log_tokens
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
            logging.info(
                "step=%d/%d loss=%.4f ppl=%.2f lr=%.2e grad=%.3f tok/s=%.0f avg_tok/s=%.0f mem=%.1fGB",
                global_step,
                total_steps,
                float(last_loss.detach()),
                math.exp(min(20.0, float(last_loss.detach()))),
                lr,
                float(grad_norm),
                tokens_per_second,
                average_tokens_per_second,
                peak_memory,
            )
            metrics.log(
                {
                    "step": global_step,
                    "train_loss": float(last_loss.detach()),
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
            val_loss = evaluate_sft(model, val_loader, device, dtype, args.eval_batches)
            logging.info("eval step=%d val_loss=%.4f val_ppl=%.2f", global_step, val_loss, math.exp(min(20.0, val_loss)))
            metrics.log({"step": global_step, "val_loss": val_loss})
            best_loss = min(best_loss, val_loss)
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
                args.tokenizer_dir,
            )
            prune_checkpoints(args.output_dir, args.keep_checkpoints)
            logging.info("checkpoint disimpan step=%d", global_step)
            last_log_time = time.time()
            last_log_tokens = tokens_processed

    last_loss = torch.tensor(0.0)
    for epoch in range(args.epochs):
        if args.length_bucket:
            train_loader.batch_sampler.set_epoch(epoch)
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with autocast_context(device, dtype):
                output = model(input_ids, attention_mask=attention_mask)
                loss, _ = chunked_cross_entropy(output.logits, labels)
                aux_loss = getattr(output, "aux_loss", None)
                if aux_loss is not None:
                    loss = loss + aux_loss
            last_loss = loss.detach()
            scaled_loss = loss / args.grad_accum
            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            micro_step += 1
            tokens_processed += int(attention_mask.sum())

            if micro_step % args.grad_accum == 0:
                optimizer_step()
                if args.max_steps > 0 and global_step >= args.max_steps:
                    stop_training = True
                    break
            if deadline is not None and time.time() >= deadline:
                logging.info(
                    "batas waktu %.1f jam tercapai pada step %d",
                    args.max_hours,
                    global_step,
                )
                stop_training = True
                break
        if stop_training:
            break
        if micro_step % args.grad_accum != 0:
            optimizer_step()
            if args.max_steps > 0 and global_step >= args.max_steps:
                break
        val_loss = evaluate_sft(model, val_loader, device, dtype, args.eval_batches)
        logging.info(
            "epoch=%d selesai val_loss=%.4f val_ppl=%.2f",
            epoch + 1,
            val_loss,
            math.exp(min(20.0, val_loss)),
        )
        metrics.log({"step": global_step, "epoch": epoch + 1, "val_loss": val_loss})
        best_loss = min(best_loss, val_loss)

    val_loss = evaluate_sft(model, val_loader, device, dtype, args.eval_batches)
    logging.info("eval final val_loss=%.4f val_ppl=%.2f", val_loss, math.exp(min(20.0, val_loss)))
    metrics.log({"step": global_step, "val_loss": val_loss})
    best_loss = min(best_loss, val_loss)

    final_dir = args.output_dir / "final"
    if not args.no_final_save:
        save_checkpoint(
            final_dir,
            save_target,
            optimizer,
            scheduler,
            scaler,
            global_step,
            best_loss,
            args_to_dict(args),
            args.tokenizer_dir,
        )
        copy_tokenizer(args.tokenizer_dir, final_dir)

    if args.preview_samples > 0:
        preview_prompt = "User: Jelaskan apa itu Bahasa Indonesia dan mengapa penting bagi masyarakat Indonesia.\n\nAssistant:"
        input_ids = [tokenizer.token_to_id("<bos>")] + tokenizer.encode(
            preview_prompt, add_special_tokens=False
        ).ids
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_tensor)
        save_target.eval()
        with torch.no_grad(), autocast_context(device, dtype):
            output = save_target.generate(
                input_tensor,
                attention_mask=attention_mask,
                max_new_tokens=80,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.token_to_id("<eos>"),
                pad_token_id=pad_token_id,
            )
        generated = tokenizer.decode(output[0, len(input_ids) :].tolist())
        logging.info("preview: %s", generated)

    if args.no_final_save:
        logging.info("selesai (dry-run) | step=%d", global_step)
    else:
        logging.info("selesai | step=%d | model=%s", global_step, final_dir)
    metrics.close()


if __name__ == "__main__":
    main()
