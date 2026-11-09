"""Compare the 1-epoch and 2-epoch checkpoints on the same prompts.

For every prompt the script reports the prompt loss, perplexity, top-1 accuracy,
and a sampled continuation for both checkpoints. Results can be saved as JSON
and Markdown for the report.

Examples
--------
python compare_epochs.py
python compare_epochs.py --stage posttrain --max-new-tokens 96
python compare_epochs.py --json-out checkpoints/epoch_comparison.json
python compare_epochs.py --markdown-out checkpoints/epoch_comparison.md
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Sequence

import torch
from tokenizers import Tokenizer

from eval_sentences import (
    EOS_TOKEN,
    PAD_TOKEN,
    load_model,
    perplexity,
    resolve_device,
    score_batch,
    special_id,
)


BASE_DIR = Path(__file__).resolve().parent
TOKENIZER_PATH = BASE_DIR / "data_id/tokenizer/tokenizer.json"

GROUPS = {
    "pretrain": {
        "title": "Pretraining",
        "variants": {
            "1 epoch": BASE_DIR / "checkpoints/pretrain_1epoch_best.pt",
            "2 epoch": BASE_DIR / "checkpoints/pretrain_2epoch_best.pt",
        },
    },
    "posttrain": {
        "title": "Post-training (SFT)",
        "variants": {
            "Pretrain 1 epoch": BASE_DIR / "checkpoints/posttrain_1epoch_best.pt",
            "Pretrain 2 epoch": BASE_DIR / "checkpoints/posttrain_2epoch_best.pt",
        },
    },
}

PROMPTS = [
    "Halo, apa kabar?",
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
        description="Compare 1-epoch and 2-epoch checkpoints on the same prompts."
    )
    parser.add_argument(
        "--stage",
        choices=["both", "pretrain", "posttrain"],
        default="both",
        help="Which checkpoint pair to compare.",
    )
    parser.add_argument("--tokenizer", default=str(TOKENIZER_PATH))
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
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be >= 1.")
    if not (0.0 < args.top_p < 1.0):
        parser.error("--top-p must be in (0, 1).")
    if args.temperature <= 0:
        parser.error("--temperature must be > 0.")
    if args.top_k is not None and args.top_k < 1:
        parser.error("--top-k must be >= 1.")
    if args.repetition_penalty <= 0:
        parser.error("--repetition-penalty must be > 0.")
    return args


def generate(
    model,
    tokenizer: Tokenizer,
    prompt: str,
    eos_id: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[str, int, float]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
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
    if eos_id in new_ids:
        new_ids = new_ids[: new_ids.index(eos_id)]
    return tokenizer.decode(new_ids, skip_special_tokens=True), len(new_ids), elapsed


def compare_variant(
    label: str,
    checkpoint: Path,
    tokenizer: Tokenizer,
    prompts: Sequence[str],
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    model, config, payload = load_model(checkpoint, device)
    eos_id = special_id(tokenizer, EOS_TOKEN, 2)
    pad_id = special_id(tokenizer, PAD_TOKEN, 1)
    prompt_ids = [tokenizer.encode(text, add_special_tokens=False).ids for text in prompts]
    scored = score_batch(
        model,
        [ids + [eos_id] for ids in prompt_ids],
        pad_id,
        config.block_size,
        device,
        False,
        torch.float32,
    )

    entries = []
    for text, ids, (losses, correct, truncated) in zip(prompts, prompt_ids, scored):
        token_count = len(losses)
        loss = sum(losses) / token_count if token_count else float("nan")
        top1 = correct / token_count if token_count else float("nan")
        generation, new_tokens, seconds = generate(
            model, tokenizer, text, eos_id, args, device
        )
        entries.append(
            {
                "text": text,
                "prompt_tokens": len(ids),
                "scored_tokens": token_count,
                "loss": loss,
                "perplexity": perplexity(loss),
                "top1_accuracy": top1,
                "truncated": truncated,
                "generation": generation,
                "generated_tokens": new_tokens,
                "seconds": seconds,
            }
        )
        log(
            f"   {label:<8} | loss {loss:.4f} | ppl {perplexity(loss):.2f} "
            f"| {new_tokens} tok | {new_tokens / max(seconds, 1e-9):.1f} tok/s"
        )

    train_args = payload.get("args", {})
    meta = {
        "checkpoint": str(checkpoint),
        "stage": payload.get("stage"),
        "best_val": payload.get("best_val"),
        "step": payload.get("step"),
        "steps": payload.get("steps"),
        "tokens_seen": payload.get("tokens_seen"),
        "pretrain_epochs": train_args.get("pretrain_epochs"),
        "posttrain_epochs": train_args.get("posttrain_epochs"),
    }
    del model
    del payload
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"meta": meta, "prompts": entries}


def epoch_note(meta: dict) -> str:
    if meta.get("pretrain_epochs") is None:
        return "-"
    note = f"pretrain {meta['pretrain_epochs']:g}"
    if meta.get("stage") == "posttrain" and meta.get("posttrain_epochs") is not None:
        note += f" + SFT {meta['posttrain_epochs']:g}"
    return note


def print_group(report: dict) -> None:
    for group_key, group in report["groups"].items():
        log("")
        log(f"== {group['title']}")
        for label, variant in group["variants"].items():
            meta = variant["meta"]
            log(
                f"[{label}] val {meta['best_val']:.4f} | step {meta['step']:,}"
                f"/{meta['steps']:,} | tokens {meta['tokens_seen']:,} "
                f"| {epoch_note(meta)}"
            )
        for index, prompt in enumerate(report["prompts"], 1):
            log("")
            log(f"#{index} {prompt}")
            for label, variant in group["variants"].items():
                entry = variant["prompts"][index - 1]
                log(
                    f"   {label:<16} | loss {entry['loss']:.4f} "
                    f"| ppl {entry['perplexity']:.2f}"
                )
                log(f"     gen: {entry['generation']}")

    log("")
    log("== Ringkasan validasi")
    for group in report["groups"].values():
        entries = list(group["variants"].items())
        first_label, first_variant = entries[0]
        second_label, second_variant = entries[1]
        first = first_variant["meta"]["best_val"]
        second = second_variant["meta"]["best_val"]
        log(
            f"{group['title']}: {first_label} {first:.4f} | "
            f"{second_label} {second:.4f} | delta {second - first:+.4f}"
        )


def markdown_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()


def write_markdown(path: Path, report: dict) -> None:
    settings = report["settings"]
    lines = [
        "# Perbandingan Model 1 Epoch vs 2 Epoch",
        "",
        (
            f"Pengaturan generasi: temperature={settings['temperature']}, "
            f"top-p={settings['top_p']}, max token={settings['max_new_tokens']}, "
            f"seed={settings['seed']}."
        ),
        "",
    ]
    for group in report["groups"].values():
        lines.append(f"## {group['title']}")
        lines.append("")
        if any(
            variant["meta"].get("stage") == "posttrain"
            for variant in group["variants"].values()
        ):
            lines.append(
                "Kedua checkpoint memakai SFT 1 epoch; yang berbeda adalah "
                "jumlah epoch pretraining."
            )
            lines.append("")
        lines.append("| Model | Best val loss | Step | Epoch |")
        lines.append("|---|---:|---:|---|")
        for label, variant in group["variants"].items():
            meta = variant["meta"]
            lines.append(
                f"| {label} | {meta['best_val']:.4f} "
                f"| {meta['step']:,}/{meta['steps']:,} | {epoch_note(meta)} |"
            )
        lines.append("")
        for index, prompt in enumerate(report["prompts"], 1):
            lines.append(f"### {index}. {prompt}")
            lines.append("")
            for label, variant in group["variants"].items():
                entry = variant["prompts"][index - 1]
                lines.append(
                    f"**{label}** — loss {entry['loss']:.4f}, "
                    f"ppl {entry['perplexity']:.2f}, "
                    f"top-1 {entry['top1_accuracy']:.1%}"
                )
                lines.append("")
                lines.append("`" + markdown_cell(entry["generation"]) + "`")
                lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    tokenizer_path = Path(args.tokenizer).expanduser()
    if not tokenizer_path.exists():
        raise SystemExit(f"Tokenizer not found: {tokenizer_path}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    prompts = [text.strip() for text in (args.text or PROMPTS) if text.strip()]

    selected = ["pretrain", "posttrain"] if args.stage == "both" else [args.stage]
    report = {
        "settings": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "seed": args.seed,
            "device": str(device),
        },
        "prompts": prompts,
        "groups": {},
    }

    log(f"device  : {device}")
    log(f"prompts : {len(prompts)}")
    for group_key in selected:
        group = GROUPS[group_key]
        missing = [str(path) for path in group["variants"].values() if not path.exists()]
        if missing:
            raise SystemExit("Checkpoint tidak ditemukan: " + ", ".join(missing))
        log("")
        log(f"== {group['title']}")
        variants = {}
        for label, checkpoint in group["variants"].items():
            log(f"[{label}] {checkpoint.name}")
            variants[label] = compare_variant(
                label, checkpoint, tokenizer, prompts, args, device
            )
        report["groups"][group_key] = {"title": group["title"], "variants": variants}

    print_group(report)

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
