"""Compare the new chat checkpoint with the previous SFT checkpoint.

The new model was trained on chat-formatted data with assistant loss masking,
so its checkpoint validation loss is not directly comparable to the previous
full-text SFT validation loss. This script therefore also scores response
tokens on held-out chat examples under each model's own prompt format.

Examples
--------
python compare_chat.py
python compare_chat.py --num-samples 50 --max-new-tokens 96
python compare_chat.py --json-out checkpoints/chat_comparison.json
python compare_chat.py --markdown-out checkpoints/chat_comparison.md
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from eval_sentences import EOS_TOKEN, load_model, perplexity, resolve_device, special_id


BASE_DIR = Path(__file__).resolve().parent
TOKENIZER_PATH = BASE_DIR / "data_id/tokenizer/tokenizer.json"
CHAT_VAL_BIN = BASE_DIR.parent / "train/data_id/bins/posttrain_val.bin"
CHAT_DATA_META = BASE_DIR.parent / "train/data_id/bins/meta.json"

MODELS = {
    "Chat baru": BASE_DIR / "checkpoints/posttrain_chat_best.pt",
    "SFT sebelumnya": BASE_DIR / "checkpoints/posttrain_2epoch_best.pt",
}

PROMPTS = [
    "Halo, apa kabar?",
    "Apa ibu kota Indonesia?",
    "Jelaskan dengan singkat mengapa langit berwarna biru pada siang hari.",
    "Resep nasi goreng sederhana:",
    "Pada tahun 1945, Indonesia",
    "Pada suatu pagi di sebuah desa kecil,",
    "Ide proyek akhir semester tentang kecerdasan buatan adalah",
    "def hitung_faktorial(n):",
]


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare the new chat model with the previous SFT model."
    )
    parser.add_argument("--tokenizer", default=str(TOKENIZER_PATH))
    parser.add_argument("--val-bin", default=str(CHAT_VAL_BIN))
    parser.add_argument("--val-meta", default=str(CHAT_DATA_META))
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--text", action="append", default=None, help="Custom prompt.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--markdown-out", default=None)
    args = parser.parse_args()
    if args.num_samples < 1:
        parser.error("--num-samples must be >= 1.")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be >= 1.")
    if args.temperature <= 0:
        parser.error("--temperature must be > 0.")
    if not (0.0 < args.top_p < 1.0):
        parser.error("--top-p must be in (0, 1).")
    if args.top_k is not None and args.top_k < 1:
        parser.error("--top-k must be >= 1.")
    if args.repetition_penalty <= 0:
        parser.error("--repetition-penalty must be > 0.")
    return args


def load_pairs(path: Path, meta_path: Path, tokenizer: Tokenizer, limit: int):
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    dtype = np.dtype(meta.get("dtype", "uint16"))
    data = np.memmap(path, dtype=dtype, mode="r")
    ids = np.asarray(data)
    user_id = tokenizer.token_to_id("<|user|>")
    assistant_id = tokenizer.token_to_id("<|assistant|>")
    end_id = tokenizer.token_to_id("<|end|>")
    if None in (user_id, assistant_id, end_id):
        raise SystemExit("Tokenizer tidak memiliki token <|user|>, <|assistant|>, atau <|end|>.")

    pairs = []
    index = 0
    while index < len(ids) - 2 and len(pairs) < limit:
        user_index = index
        while user_index < len(ids) and ids[user_index] != user_id:
            user_index += 1
        if user_index >= len(ids):
            break
        assistant_index = user_index + 1
        while assistant_index < len(ids) and ids[assistant_index] != assistant_id:
            assistant_index += 1
        if assistant_index >= len(ids):
            break
        turn_end = assistant_index + 1
        while turn_end < len(ids) and ids[turn_end] not in (user_id, end_id):
            turn_end += 1
        if turn_end >= len(ids):
            break
        user_text = tokenizer.decode(
            ids[user_index + 1 : assistant_index].tolist(), skip_special_tokens=True
        ).strip()
        assistant_text = tokenizer.decode(
            ids[assistant_index + 1 : turn_end].tolist(), skip_special_tokens=True
        ).strip()
        if user_text and assistant_text:
            pairs.append((user_text, assistant_text))

        document_end = turn_end
        while document_end < len(ids) and ids[document_end] != end_id:
            document_end += 1
        index = document_end + 1 if document_end < len(ids) else turn_end + 1
    return pairs


def score_target(
    model,
    context_ids: Sequence[int],
    target_ids: Sequence[int],
    block_size: int,
    device: torch.device,
) -> tuple[float, int, int, bool] | None:
    target = list(target_ids)
    if not target:
        return None
    truncated = len(target) > block_size
    target = target[:block_size]
    max_context = max(1, block_size + 1 - len(target))
    context = list(context_ids)
    if len(context) > max_context:
        context = context[-max_context:]
        truncated = True
    sequence = context + target
    x = torch.tensor([sequence[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([sequence[1:]], dtype=torch.long, device=device)
    target_start = max(0, len(context) - 1)
    y[:, :target_start] = -1

    with torch.inference_mode():
        logits, _ = model(x)
    logits = logits.float()
    losses = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1),
        reduction="none",
        ignore_index=-1,
    )
    valid = y.reshape(-1) != -1
    loss_sum = float(losses[valid].sum().item())
    correct = int((logits.argmax(dim=-1).reshape(-1)[valid] == y.reshape(-1)[valid]).sum().item())
    return loss_sum, correct, int(valid.sum().item()), truncated


def heldout_metrics(
    model,
    tokenizer: Tokenizer,
    pairs: Sequence[tuple[str, str]],
    block_size: int,
    device: torch.device,
    chat_format: bool,
) -> dict:
    eos_id = special_id(tokenizer, EOS_TOKEN, 2)
    end_id = special_id(tokenizer, "<|end|>", 32002)
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    examples = 0
    truncated_examples = 0
    for user_text, assistant_text in pairs:
        if chat_format:
            context_text = f"<|user|>\n{user_text}\n<|assistant|>\n"
            target_text = f"{assistant_text}\n<|end|>\n"
            context_ids = tokenizer.encode(context_text, add_special_tokens=False).ids
            target_ids = tokenizer.encode(target_text, add_special_tokens=False).ids
        else:
            context_text = f"{user_text}\n\n"
            target_text = assistant_text
            context_ids = tokenizer.encode(context_text, add_special_tokens=False).ids
            target_ids = tokenizer.encode(target_text, add_special_tokens=False).ids + [eos_id]
        result = score_target(model, context_ids, target_ids, block_size, device)
        if result is None:
            continue
        loss_sum, correct, token_count, truncated = result
        total_loss += loss_sum
        total_correct += correct
        total_tokens += token_count
        examples += 1
        if truncated:
            truncated_examples += 1

    loss = total_loss / total_tokens if total_tokens else float("nan")
    return {
        "loss": loss,
        "perplexity": perplexity(loss),
        "top1_accuracy": total_correct / total_tokens if total_tokens else float("nan"),
        "tokens": total_tokens,
        "examples": examples,
        "truncated_examples": truncated_examples,
        "format": "chat" if chat_format else "plain",
    }


def generate(
    model,
    tokenizer: Tokenizer,
    prompt: str,
    chat_format: bool,
    eos_id: int,
    end_id: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[str, int, float]:
    if chat_format:
        prompt_text = f"<|user|>\n{prompt}\n<|assistant|>\n"
    else:
        prompt_text = prompt
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False).ids
    torch.manual_seed(args.seed)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            idx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
    elapsed = time.perf_counter() - started
    new_ids = output[0, len(prompt_ids) :].tolist()
    stops = [new_ids.index(token) for token in (end_id, eos_id) if token in new_ids]
    if stops:
        new_ids = new_ids[: min(stops)]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip(), len(new_ids), elapsed


def evaluate_model(
    label: str,
    checkpoint: Path,
    tokenizer: Tokenizer,
    pairs: Sequence[tuple[str, str]],
    prompts: Sequence[str],
    args: argparse.Namespace,
    device: torch.device,
    chat_format: bool,
) -> dict:
    model, config, payload = load_model(checkpoint, device)
    eos_id = special_id(tokenizer, EOS_TOKEN, 2)
    end_id = special_id(tokenizer, "<|end|>", 32002)
    metrics = heldout_metrics(model, tokenizer, pairs, config.block_size, device, chat_format)
    log(
        f"   held-out respons | loss {metrics['loss']:.4f} "
        f"| ppl {metrics['perplexity']:.2f} | top1 {metrics['top1_accuracy']:.1%} "
        f"| tokens {metrics['tokens']:,} | contoh {metrics['examples']}"
    )

    generations = []
    for prompt in prompts:
        text, new_tokens, seconds = generate(
            model, tokenizer, prompt, chat_format, eos_id, end_id, args, device
        )
        generations.append(
            {
                "text": prompt,
                "generation": text,
                "tokens": new_tokens,
                "seconds": seconds,
            }
        )
        log(f"   {prompt!r} -> {text!r}")

    train_args = payload.get("args", {})
    meta = {
        "checkpoint": str(checkpoint),
        "stage": payload.get("stage"),
        "best_val": payload.get("best_val"),
        "step": payload.get("step"),
        "steps": payload.get("steps"),
        "tokens_seen": payload.get("tokens_seen"),
        "vocab_size": config.vocab_size,
        "pretrain_epochs": train_args.get("pretrain_epochs"),
        "posttrain_epochs": train_args.get("posttrain_epochs"),
        "mask_assistant": train_args.get("mask_assistant"),
        "format": "chat" if chat_format else "plain",
    }
    del model
    del payload
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"meta": meta, "heldout": metrics, "prompts": generations}


def markdown_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        "# Perbandingan Model Chat Baru vs Model Sebelumnya",
        "",
        (
            f"Pengaturan generasi: temperature={report['settings']['temperature']}, "
            f"top-p={report['settings']['top_p']}, "
            f"max token={report['settings']['max_new_tokens']}, "
            f"seed={report['settings']['seed']}."
        ),
        "",
        "## Checkpoint",
        "",
        "| Model | Format | Best val | Step | Vocab | Catatan |",
        "|---|---|---:|---:|---:|---|",
    ]
    for label, variant in report["models"].items():
        meta = variant["meta"]
        note = "assistant-masked val" if meta.get("mask_assistant") else "full-text val"
        lines.append(
            f"| {label} | {meta['format']} | {meta['best_val']:.4f} "
            f"| {meta['step']:,}/{meta['steps']:,} | {meta['vocab_size']:,} | {note} |"
        )
    lines.extend(
        [
            "",
            "Val loss checkpoint chat memakai assistant masking dan data chat, "
            "sedangkan baseline memakai full-text; angka tersebut tidak "
            "dibandingkan langsung. Metrik held-out di bawah dihitung pada contoh "
            "chat yang sama, dengan format prompt masing-masing model, dan hanya "
            "pada token respons.",
            "",
            "## Held-out respons",
            "",
            "| Model | Response loss | Perplexity | Top-1 | Token | Contoh |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for label, variant in report["models"].items():
        metrics = variant["heldout"]
        lines.append(
            f"| {label} | {metrics['loss']:.4f} | {metrics['perplexity']:.2f} "
            f"| {metrics['top1_accuracy']:.1%} | {metrics['tokens']:,} "
            f"| {metrics['examples']} |"
        )
    lines.extend(["", "## Hasil generasi", ""])
    for index, prompt in enumerate(report["prompts"], 1):
        lines.append(f"### {index}. {prompt}")
        lines.append("")
        for label, variant in report["models"].items():
            entry = variant["prompts"][index - 1]
            lines.append(f"**{label}** ({entry['tokens']} token)")
            lines.append("")
            lines.append("`" + markdown_cell(entry["generation"]) + "`")
            lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    tokenizer_path = Path(args.tokenizer).expanduser()
    val_bin = Path(args.val_bin).expanduser()
    val_meta = Path(args.val_meta).expanduser()
    for path in (tokenizer_path, val_bin, val_meta):
        if not path.exists():
            raise SystemExit(f"File tidak ditemukan: {path}")
    for label, checkpoint in MODELS.items():
        if not checkpoint.exists():
            raise SystemExit(f"Checkpoint {label} tidak ditemukan: {checkpoint}")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    pairs = load_pairs(val_bin, val_meta, tokenizer, args.num_samples)
    if not pairs:
        raise SystemExit("Tidak ada pasangan user/assistant di val bin.")
    prompts = [text.strip() for text in (args.text or PROMPTS) if text.strip()]

    log(f"device         : {device}")
    log(f"held-out pairs : {len(pairs)}")
    log(f"prompts        : {len(prompts)}")

    report = {
        "settings": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "seed": args.seed,
            "device": str(device),
            "num_samples": len(pairs),
        },
        "prompts": prompts,
        "models": {},
    }

    for label, checkpoint in MODELS.items():
        chat_format = label == "Chat baru"
        log("")
        log(f"== {label} | {checkpoint.name} | format={'chat' if chat_format else 'plain'}")
        report["models"][label] = evaluate_model(
            label, checkpoint, tokenizer, pairs, prompts, args, device, chat_format
        )

    log("")
    log(
        "catatan: val loss checkpoint chat memakai assistant-masked loss pada data chat; "
        "baseline memakai full-text loss. Keduanya tidak dibandingkan langsung."
    )
    log("== Ringkasan held-out respons (contoh chat baru, hanya token respons)")
    for label, variant in report["models"].items():
        metrics = variant["heldout"]
        log(
            f"{label:<16} loss {metrics['loss']:.4f} | ppl {metrics['perplexity']:.2f} "
            f"| top1 {metrics['top1_accuracy']:.1%} | tokens {metrics['tokens']:,}"
        )
    labels = list(report["models"])
    first = report["models"][labels[0]]["heldout"]
    second = report["models"][labels[1]]["heldout"]
    log(
        f"selisih ({labels[1]} - {labels[0]}): loss {second['loss'] - first['loss']:+.4f} "
        f"| ppl {second['perplexity'] - first['perplexity']:+.2f}"
    )

    if args.json_out:
        path = Path(args.json_out).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"[done] wrote {path}")
    if args.markdown_out:
        path = Path(args.markdown_out).expanduser()
        write_markdown(path, report)
        log(f"[done] wrote {path}")


if __name__ == "__main__":
    main()
