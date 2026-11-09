import argparse
import gc
import json
import logging
import math
import time
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer

from chat_format import encode_preference
from train_utils import (
    PROJECT_ROOT,
    JsonlLogger,
    args_to_dict,
    autocast_context,
    copy_tokenizer,
    get_optimizer,
    load_model,
    load_training_state,
    make_grad_scaler,
    make_scheduler,
    maybe_compile,
    model_summary,
    prune_checkpoints,
    resolve_device,
    resolve_dtype,
    save_checkpoint,
    save_json,
    seed_everything,
    setup_logging,
)


class PreferenceDataset(Dataset):
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
        chosen = encode_preference(tokenizer, record["prompt"], record["chosen"], max_len)
        rejected = encode_preference(tokenizer, record["prompt"], record["rejected"], max_len)
        if chosen is None or rejected is None:
            skipped += 1
            continue
        if (
            sum(1 for label in chosen[1] if label != -100) < 6
            or sum(1 for label in rejected[1] if label != -100) < 6
        ):
            skipped += 1
            continue
        examples.append(
            {
                "id": record["id"],
                "source": record["source"],
                "chosen_input_ids": torch.tensor(chosen[0], dtype=torch.long),
                "chosen_labels": torch.tensor(chosen[1], dtype=torch.long),
                "rejected_input_ids": torch.tensor(rejected[0], dtype=torch.long),
                "rejected_labels": torch.tensor(rejected[1], dtype=torch.long),
            }
        )
        if seen % 1000 == 0:
            logging.info("%s | diproses=%d valid=%d", label, seen, len(examples))
        if max_samples > 0 and len(examples) >= max_samples:
            break
    logging.info("%s | total=%d valid=%d dilewati=%d", label, seen, len(examples), skipped)
    return examples


def collate_preference(batch: list, pad_token_id: int):
    max_len = max(
        max(item["chosen_input_ids"].numel(), item["rejected_input_ids"].numel())
        for item in batch
    )
    chosen_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    chosen_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    chosen_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    rejected_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    rejected_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    rejected_mask = torch.zeros((len(batch), max_len), dtype=torch.long)

    for row, item in enumerate(batch):
        length = item["chosen_input_ids"].numel()
        chosen_ids[row, :length] = item["chosen_input_ids"]
        chosen_labels[row, :length] = item["chosen_labels"]
        chosen_mask[row, :length] = 1
        length = item["rejected_input_ids"].numel()
        rejected_ids[row, :length] = item["rejected_input_ids"]
        rejected_labels[row, :length] = item["rejected_labels"]
        rejected_mask[row, :length] = 1

    return {
        "ids": [item["id"] for item in batch],
        "chosen_input_ids": chosen_ids,
        "chosen_labels": chosen_labels,
        "chosen_attention_mask": chosen_mask,
        "rejected_input_ids": rejected_ids,
        "rejected_labels": rejected_labels,
        "rejected_attention_mask": rejected_mask,
    }


def sequence_logps(logits: torch.Tensor, labels: torch.Tensor, chunk_size: int = 256):
    batch_size, sequence_length, _ = logits.shape
    sums = torch.zeros(batch_size, dtype=torch.float32, device=logits.device)
    counts = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
    for start in range(0, sequence_length - 1, chunk_size):
        end = min(start + chunk_size, sequence_length - 1)
        chunk_logits = logits[:, start:end, :].float()
        chunk_labels = labels[:, start + 1 : end + 1]
        token_loss = F.cross_entropy(
            chunk_logits.transpose(1, 2),
            chunk_labels,
            ignore_index=-100,
            reduction="none",
        )
        sums = sums - token_loss.sum(dim=-1)
        counts = counts + (chunk_labels != -100).sum(dim=-1)
    averages = sums / counts.clamp(min=1)
    return sums, averages, counts


def model_logps(model, batch, device, dtype):
    input_ids = torch.cat(
        [batch["chosen_input_ids"], batch["rejected_input_ids"]], dim=0
    ).to(device, non_blocking=True)
    attention_mask = torch.cat(
        [batch["chosen_attention_mask"], batch["rejected_attention_mask"]], dim=0
    ).to(device, non_blocking=True)
    labels = torch.cat([batch["chosen_labels"], batch["rejected_labels"]], dim=0).to(
        device, non_blocking=True
    )
    with autocast_context(device, dtype):
        logits = model(input_ids, attention_mask=attention_mask).logits
    sums, averages, counts = sequence_logps(logits, labels)
    half = sums.numel() // 2
    return (
        sums[:half],
        averages[:half],
        counts[:half],
        sums[half:],
        averages[half:],
        counts[half:],
    )


def log_odds(average_logp: torch.Tensor) -> torch.Tensor:
    probability = torch.exp(average_logp.clamp(max=-1e-6))
    return average_logp - torch.log1p(-probability)


def preference_loss(
    algo: str,
    chosen_sums,
    chosen_avgs,
    rejected_sums,
    rejected_avgs,
    reference: list,
    beta: float,
    orpo_lambda: float,
    simpo_gamma: float,
):
    if algo == "dpo":
        ref_chosen, ref_rejected = reference
        margin = (chosen_sums - rejected_sums) - (ref_chosen - ref_rejected)
        loss = -F.logsigmoid(beta * margin).mean()
        accuracy = (chosen_sums > rejected_sums).float().mean()
        return loss, accuracy
    if algo == "simpo":
        margin = beta * (chosen_avgs - rejected_avgs) - simpo_gamma
        loss = -F.logsigmoid(margin).mean()
        accuracy = (chosen_avgs > rejected_avgs).float().mean()
        return loss, accuracy
    nll = -chosen_avgs.mean()
    or_loss = -F.logsigmoid(log_odds(chosen_avgs) - log_odds(rejected_avgs)).mean()
    loss = nll + orpo_lambda * or_loss
    accuracy = (chosen_avgs > rejected_avgs).float().mean()
    return loss, accuracy


def evaluate_preference(
    model,
    loader,
    device,
    dtype,
    algo: str,
    reference_cache: dict,
    beta: float,
    orpo_lambda: float,
    simpo_gamma: float,
    max_batches: int,
):
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_pairs = 0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            chosen_sums, chosen_avgs, _, rejected_sums, rejected_avgs, _ = model_logps(
                model, batch, device, dtype
            )
            if algo == "dpo":
                ref_chosen = torch.tensor(
                    [reference_cache[item_id][0] for item_id in batch["ids"]],
                    dtype=torch.float32,
                    device=device,
                )
                ref_rejected = torch.tensor(
                    [reference_cache[item_id][1] for item_id in batch["ids"]],
                    dtype=torch.float32,
                    device=device,
                )
                reference = [ref_chosen, ref_rejected]
            else:
                reference = None
            loss, accuracy = preference_loss(
                algo,
                chosen_sums,
                chosen_avgs,
                rejected_sums,
                rejected_avgs,
                reference,
                beta,
                orpo_lambda,
                simpo_gamma,
            )
            pairs = chosen_sums.numel()
            total_loss += float(loss.detach()) * pairs
            total_correct += float(accuracy) * pairs
            total_pairs += pairs
            batches += 1
            if batches >= max_batches:
                break
    model.train()
    return total_loss / max(1, total_pairs), total_correct / max(1, total_pairs)


def precompute_reference_logps(
    model,
    loaders: dict,
    device,
    dtype,
    cache_path: Path,
) -> dict:
    model.eval()
    cache = {}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as handle:
        for split, loader in loaders.items():
            logging.info("precompute reference logps split=%s", split)
            with torch.no_grad():
                for batch in loader:
                    with torch.no_grad():
                        chosen_sums, _, _, rejected_sums, _, _ = model_logps(
                            model, batch, device, dtype
                        )
                    for item_id, chosen_logp, rejected_logp in zip(
                        batch["ids"],
                        chosen_sums.tolist(),
                        rejected_sums.tolist(),
                    ):
                        cache[item_id] = (chosen_logp, rejected_logp)
                        handle.write(
                            json.dumps(
                                {
                                    "id": item_id,
                                    "split": split,
                                    "chosen_logp": chosen_logp,
                                    "rejected_logp": rejected_logp,
                                }
                            )
                            + "\n"
                        )
    return cache


def load_reference_cache(path: Path) -> dict:
    cache = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            cache[record["id"]] = (record["chosen_logp"], record["rejected_logp"])
    return cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["simpo", "orpo", "dpo"], default="simpo")
    parser.add_argument("--base-checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "sft" / "final")
    parser.add_argument("--reference-checkpoint", type=Path, default=None)
    parser.add_argument("--train-file", type=Path, default=PROJECT_ROOT / "data" / "preference" / "train.jsonl")
    parser.add_argument("--val-file", type=Path, default=PROJECT_ROOT / "data" / "preference" / "val.jsonl")
    parser.add_argument("--test-file", type=Path, default=PROJECT_ROOT / "data" / "preference" / "test.jsonl")
    parser.add_argument("--tokenizer-dir", type=Path, default=PROJECT_ROOT / "data" / "tokenizer")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--reference-cache", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--dpo-beta", type=float, default=0.1)
    parser.add_argument("--simpo-beta", type=float, default=2.0)
    parser.add_argument("--simpo-gamma", type=float, default=0.5)
    parser.add_argument("--orpo-lambda", type=float, default=0.1)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--keep-checkpoints", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--ref-batch-size", type=int, default=2)
    parser.add_argument("--overwrite-reference", action="store_true")
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = PROJECT_ROOT / "checkpoints" / args.algo
    if args.reference_cache is None:
        args.reference_cache = args.output_dir / "reference_logprobs.jsonl"
    if args.reference_checkpoint is None:
        args.reference_checkpoint = args.base_checkpoint
    if args.lr is None:
        args.lr = 1e-6 if args.algo == "simpo" else 5e-6
    loss_beta = args.simpo_beta if args.algo == "simpo" else args.dpo_beta

    seed_everything(args.seed)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.output_dir / "train.log")
    metrics = JsonlLogger(args.output_dir / "metrics.jsonl")
    logging.info("algo=%s device=%s dtype=%s", args.algo, device, dtype)

    tokenizer = Tokenizer.from_file(str(args.tokenizer_dir / "tokenizer.json"))
    pad_token_id = tokenizer.token_to_id("<pad>")

    train_examples = prepare_examples(args.train_file, tokenizer, args.max_seq_len, args.max_samples, "train")
    val_examples = prepare_examples(args.val_file, tokenizer, args.max_seq_len, args.max_samples, "val")
    test_examples = prepare_examples(args.test_file, tokenizer, args.max_seq_len, args.max_samples, "test")
    if not train_examples or not val_examples:
        raise SystemExit("dataset preference kosong")

    train_loader = DataLoader(
        PreferenceDataset(train_examples),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        collate_fn=partial(collate_preference, pad_token_id=pad_token_id),
    )
    val_loader = DataLoader(
        PreferenceDataset(val_examples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=partial(collate_preference, pad_token_id=pad_token_id),
    )
    test_loader = DataLoader(
        PreferenceDataset(test_examples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=partial(collate_preference, pad_token_id=pad_token_id),
    )
    if len(train_loader) == 0:
        raise SystemExit("train_loader kosong, kurangi batch size")

    reference_cache = {}
    if args.algo == "dpo":
        cache_exists = args.reference_cache.exists() and not args.overwrite_reference
        if cache_exists:
            meta_path = args.reference_cache.with_suffix(".meta.json")
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                expected_checkpoint = str(Path(args.reference_checkpoint).resolve())
                if meta.get("reference_checkpoint") != expected_checkpoint:
                    raise SystemExit(
                        "reference cache dibuat dengan checkpoint berbeda, gunakan --overwrite-reference"
                    )
                if meta.get("max_seq_len") != args.max_seq_len:
                    raise SystemExit(
                        "reference cache dibuat dengan max_seq_len berbeda, gunakan --overwrite-reference"
                    )
            reference_cache = load_reference_cache(args.reference_cache)
            required = {item["id"] for item in train_examples + val_examples + test_examples}
            missing = required - set(reference_cache)
            if missing:
                raise SystemExit(
                    f"reference cache tidak lengkap ({len(missing)} id hilang), gunakan --overwrite-reference"
                )
            logging.info("reference cache dimuat: %d pair", len(reference_cache))
        else:
            logging.info("menghitung reference logprobs dari %s", args.reference_checkpoint)
            reference_model = load_model(args.reference_checkpoint, device)
            reference_model.eval()
            reference_model.gradient_checkpointing_disable()
            ref_loader = DataLoader(
                PreferenceDataset(train_examples + val_examples + test_examples),
                batch_size=args.ref_batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=device.type == "cuda",
                drop_last=False,
                collate_fn=partial(collate_preference, pad_token_id=pad_token_id),
            )
            reference_cache = precompute_reference_logps(
                reference_model,
                {"all": ref_loader},
                device,
                dtype,
                args.reference_cache,
            )
            del reference_model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            save_json(
                args.reference_cache.with_suffix(".meta.json"),
                {
                    "reference_checkpoint": str(Path(args.reference_checkpoint).resolve()),
                    "max_seq_len": args.max_seq_len,
                    "count": len(reference_cache),
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
            )
            logging.info("reference cache selesai: %d pair", len(reference_cache))

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * args.epochs
    total_steps = max(1, total_steps)
    logging.info(
        "train=%d val=%d test=%d steps_per_epoch=%d total_steps=%d",
        len(train_examples),
        len(val_examples),
        len(test_examples),
        steps_per_epoch,
        total_steps,
    )

    if args.resume:
        model = load_model(args.resume, device)
    else:
        model = load_model(args.base_checkpoint, device)
    if args.no_gradient_checkpointing:
        model.gradient_checkpointing_disable()
    else:
        model.gradient_checkpointing_enable()
    if args.compile:
        model = maybe_compile(model)
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
    pairs_processed = 0
    started_at = time.time()
    stop_training = False
    last_loss = torch.tensor(0.0)
    last_accuracy = 0.0

    def optimizer_step() -> None:
        nonlocal global_step, best_loss
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
            elapsed = max(1e-6, time.time() - started_at)
            lr = scheduler.get_last_lr()[0]
            logging.info(
                "step=%d/%d loss=%.4f acc=%.3f lr=%.2e grad=%.3f pair/s=%.1f",
                global_step,
                total_steps,
                float(last_loss),
                last_accuracy,
                lr,
                float(grad_norm),
                pairs_processed / elapsed,
            )
            metrics.log(
                {
                    "step": global_step,
                    "train_loss": float(last_loss),
                    "train_accuracy": last_accuracy,
                    "lr": lr,
                    "grad_norm": float(grad_norm),
                    "pairs": pairs_processed,
                }
            )

        if args.eval_interval > 0 and global_step % args.eval_interval == 0:
            val_loss, val_accuracy = evaluate_preference(
                model,
                val_loader,
                device,
                dtype,
                args.algo,
                reference_cache,
                loss_beta,
                args.orpo_lambda,
                args.simpo_gamma,
                args.eval_batches,
            )
            logging.info(
                "eval step=%d val_loss=%.4f val_acc=%.4f",
                global_step,
                val_loss,
                val_accuracy,
            )
            metrics.log(
                {
                    "step": global_step,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                }
            )
            best_loss = min(best_loss, val_loss)
            model.train()

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

    for epoch in range(args.epochs):
        for batch in train_loader:
            chosen_sums, chosen_avgs, _, rejected_sums, rejected_avgs, _ = model_logps(
                model, batch, device, dtype
            )
            if args.algo == "dpo":
                ref_chosen = torch.tensor(
                    [reference_cache[item_id][0] for item_id in batch["ids"]],
                    dtype=torch.float32,
                    device=device,
                )
                ref_rejected = torch.tensor(
                    [reference_cache[item_id][1] for item_id in batch["ids"]],
                    dtype=torch.float32,
                    device=device,
                )
                reference = [ref_chosen, ref_rejected]
            else:
                reference = None
            loss, accuracy = preference_loss(
                args.algo,
                chosen_sums,
                chosen_avgs,
                rejected_sums,
                rejected_avgs,
                reference,
                loss_beta,
                args.orpo_lambda,
                args.simpo_gamma,
            )
            last_loss = loss.detach()
            last_accuracy = float(accuracy)
            scaled_loss = loss / args.grad_accum
            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            micro_step += 1
            pairs_processed += chosen_sums.numel()

            if micro_step % args.grad_accum == 0:
                optimizer_step()
                if args.max_steps > 0 and global_step >= args.max_steps:
                    stop_training = True
                    break
        if stop_training:
            break
        if micro_step % args.grad_accum != 0:
            optimizer_step()
            if args.max_steps > 0 and global_step >= args.max_steps:
                break
        val_loss, val_accuracy = evaluate_preference(
            model,
            val_loader,
            device,
            dtype,
            args.algo,
            reference_cache,
            loss_beta,
            args.orpo_lambda,
            args.simpo_gamma,
            args.eval_batches,
        )
        logging.info(
            "epoch=%d selesai val_loss=%.4f val_acc=%.4f",
            epoch + 1,
            val_loss,
            val_accuracy,
        )
        metrics.log(
            {
                "step": global_step,
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )
        best_loss = min(best_loss, val_loss)

    val_loss, val_accuracy = evaluate_preference(
        model,
        val_loader,
        device,
        dtype,
        args.algo,
        reference_cache,
        loss_beta,
        args.orpo_lambda,
        args.simpo_gamma,
        args.eval_batches,
    )
    test_loss, test_accuracy = evaluate_preference(
        model,
        test_loader,
        device,
        dtype,
        args.algo,
        reference_cache,
        loss_beta,
        args.orpo_lambda,
        args.simpo_gamma,
        args.eval_batches,
    )
    logging.info(
        "eval final val_loss=%.4f val_acc=%.4f test_loss=%.4f test_acc=%.4f",
        val_loss,
        val_accuracy,
        test_loss,
        test_accuracy,
    )
    metrics.log(
        {
            "step": global_step,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
        }
    )
    best_loss = min(best_loss, val_loss)

    final_dir = args.output_dir / "final"
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
    logging.info("selesai | step=%d | model=%s", global_step, final_dir)
    metrics.close()


if __name__ == "__main__":
    main()
