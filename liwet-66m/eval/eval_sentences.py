"""Score Indonesian sentences and optionally generate continuations.

The script loads a checkpoint produced by ``train.py`` (plus the tokenizer from
``data_id/tokenizer/tokenizer.json``), tokenizes every input sentence, and reports
the per-sentence next-token loss, perplexity, and top-1 accuracy. If
``--max-new-tokens`` is set, the model also continues each sentence.

Examples
--------
python eval_sentences.py
python eval_sentences.py --text "Halo, apa kabar?" --text "Ibu kota Indonesia adalah"
python eval_sentences.py --input prompts.txt --max-new-tokens 60 --temperature 0.8
python eval_sentences.py --checkpoint checkpoints/pretrain_best.pt --show-tokens
python eval_sentences.py --input prompts.txt --json-out scores.json --csv-out scores.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from config import GPTConfig
from model import build_model


EOS_TOKEN = "<|endoftext|>"
PAD_TOKEN = "<|pad|>"

DEFAULT_SENTENCES = [
    "Halo, nama saya Budi. Saya tinggal di Jakarta.",
    "Ibu kota Indonesia adalah Jakarta, sedangkan ibu kota Jepang adalah Tokyo.",
    "Tolong jelaskan mengapa langit berwarna biru pada siang hari.",
    "Pada tahun 1945, Indonesia memproklamasikan kemerdekaannya.",
    "Resep nasi goreng sederhana: tumis bawang, masukkan nasi, lalu tambahkan kecap.",
]


@dataclass
class SentenceResult:
    index: int
    text: str
    tokens: int
    loss: float
    perplexity: float
    top1_accuracy: float
    truncated: bool
    token_losses: list[tuple[str, float]]
    generation: str | None = None


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate sentences with a trained MiniTransformer checkpoint."
    )

    model = parser.add_argument_group("model")
    model.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path (default: best available under checkpoints/).",
    )
    model.add_argument(
        "--tokenizer",
        default="data_id/tokenizer/tokenizer.json",
        help="Path to the byte-level BPE tokenizer.json.",
    )
    model.add_argument("--device", default="auto", help="auto, cuda, cuda:0, cpu, ...")
    model.add_argument(
        "--batch-size", type=int, default=8, help="Sentences scored per forward pass."
    )
    model.add_argument(
        "--amp",
        action="store_true",
        help="Use bfloat16/float16 autocast on CUDA (losses are still computed in fp32).",
    )

    source = parser.add_argument_group("input")
    source.add_argument(
        "--text",
        action="append",
        default=None,
        help="Sentence to evaluate; repeat the flag to add more sentences.",
    )
    source.add_argument(
        "--input",
        action="append",
        default=None,
        help="UTF-8 text file with one sentence per line ('-' reads stdin).",
    )
    source.add_argument(
        "--show-tokens",
        action="store_true",
        help="Print the per-token negative log-likelihood for every sentence.",
    )

    generation = parser.add_argument_group("generation")
    generation.add_argument(
        "--max-new-tokens",
        type=int,
        default=0,
        help="Generate this many continuation tokens per sentence (0 disables).",
    )
    generation.add_argument("--temperature", type=float, default=0.8)
    generation.add_argument("--top-k", type=int, default=None)
    generation.add_argument("--top-p", type=float, default=None)
    generation.add_argument("--repetition-penalty", type=float, default=1.0)
    generation.add_argument(
        "--stop-at-eos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Truncate a continuation at the first end-of-text token.",
    )

    output = parser.add_argument_group("output")
    output.add_argument("--json-out", default=None, help="Write detailed results as JSON.")
    output.add_argument("--csv-out", default=None, help="Write one CSV row per sentence.")

    args = parser.parse_args()
    validate_args(parser, args)
    return args


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1.")
    if args.max_new_tokens < 0:
        parser.error("--max-new-tokens must be >= 0.")
    if args.max_new_tokens > 0:
        if args.temperature <= 0:
            parser.error("--temperature must be > 0.")
        if args.top_k is not None and args.top_k < 1:
            parser.error("--top-k must be >= 1.")
        if args.top_p is not None and not (0.0 < args.top_p < 1.0):
            parser.error("--top-p must be in (0, 1).")
        if args.repetition_penalty <= 0:
            parser.error("--repetition-penalty must be > 0.")


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available.")
    return device


def default_checkpoint() -> Path:
    candidates = [
        "posttrain_2epoch_best.pt",
        "posttrain_best.pt",
        "posttrain.pt",
        "pretrain_2epoch_best.pt",
        "pretrain_best.pt",
        "pretrain.pt",
        "latest.pt",
    ]
    root = Path("checkpoints")
    for name in candidates:
        path = root / name
        if path.exists():
            return path
    return root / candidates[0]


def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def build_config(payload: dict) -> GPTConfig:
    known = {field.name for field in fields(GPTConfig)}
    values = {k: v for k, v in payload.get("config", {}).items() if k in known}
    if not values:
        raise SystemExit("Checkpoint does not contain a GPTConfig.")
    return GPTConfig(**values)


def load_model(path: Path, device: torch.device):
    payload = load_checkpoint(path)
    config = build_config(payload)
    model = build_model(config)
    state = payload.get("model", payload)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, config, payload


def special_id(tokenizer: Tokenizer, token: str, fallback: int) -> int:
    token_id = tokenizer.token_to_id(token)
    return fallback if token_id is None else token_id


def read_sentences(args: argparse.Namespace) -> list[str]:
    sentences = [text.strip() for text in (args.text or []) if text.strip()]
    for source in args.input or []:
        if source == "-":
            content = sys.stdin.read()
        else:
            path = Path(source)
            if not path.exists():
                raise SystemExit(f"Input file not found: {path}")
            content = path.read_text(encoding="utf-8-sig")
        sentences.extend(line.strip() for line in content.splitlines() if line.strip())
    return sentences or list(DEFAULT_SENTENCES)


def autocast_context(enabled: bool, device: torch.device, dtype: torch.dtype):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


@torch.inference_mode()
def score_batch(
    model: torch.nn.Module,
    sequences: Sequence[Sequence[int]],
    pad_id: int,
    block_size: int,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> list[tuple[list[float], int, bool]]:
    prepared: list[list[int]] = []
    truncated_flags: list[bool] = []
    for ids in sequences:
        ids = list(ids)
        truncated = len(ids) > block_size + 1
        if truncated:
            ids = ids[:block_size] + [ids[-1]]
        prepared.append(ids)
        truncated_flags.append(truncated)

    max_len = max(len(ids) - 1 for ids in prepared)
    x = torch.full((len(prepared), max_len), pad_id, dtype=torch.long, device=device)
    y = torch.full((len(prepared), max_len), -1, dtype=torch.long, device=device)
    for row, ids in enumerate(prepared):
        length = len(ids) - 1
        x[row, :length] = torch.tensor(ids[:-1], dtype=torch.long, device=device)
        y[row, :length] = torch.tensor(ids[1:], dtype=torch.long, device=device)

    with autocast_context(amp_enabled, device, amp_dtype):
        logits, _ = model(x)
    logits = logits.float()
    vocab_size = logits.size(-1)
    nll = F.cross_entropy(
        logits.reshape(-1, vocab_size), y.reshape(-1), reduction="none", ignore_index=-1
    ).view_as(y)
    predictions = logits.argmax(dim=-1)

    results: list[tuple[list[float], int, bool]] = []
    for row in range(len(prepared)):
        valid = y[row] != -1
        token_count = int(valid.sum().item())
        if token_count == 0:
            results.append(([], 0, truncated_flags[row]))
            continue
        losses = nll[row][valid].tolist()
        correct = int(((predictions[row] == y[row]) & valid).sum().item())
        results.append((losses, correct, truncated_flags[row]))
    return results


def token_label(tokenizer: Tokenizer, token_id: int) -> str:
    token = tokenizer.id_to_token(token_id)
    if token is None:
        return str(token_id)
    if token.startswith("<|") and token.endswith("|>"):
        return token
    return tokenizer.decode([token_id], skip_special_tokens=False)


def generate_continuation(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    prompt_ids: Sequence[int],
    eos_id: int,
    args: argparse.Namespace,
    device: torch.device,
) -> str:
    idx = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    output = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    new_ids = output[0, len(prompt_ids) :].tolist()
    if args.stop_at_eos and eos_id in new_ids:
        new_ids = new_ids[: new_ids.index(eos_id)]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def perplexity(loss: float) -> float:
    if math.isnan(loss):
        return float("nan")
    if loss > 700:
        return float("inf")
    return math.exp(loss)


def evaluate(args: argparse.Namespace) -> tuple[list[SentenceResult], dict]:
    device = resolve_device(args.device)
    checkpoint_path = (
        Path(args.checkpoint).expanduser() if args.checkpoint else default_checkpoint()
    )
    tokenizer_path = Path(args.tokenizer).expanduser()
    if not tokenizer_path.exists():
        raise SystemExit(f"Tokenizer not found: {tokenizer_path}")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab_size = tokenizer.get_vocab_size()
    model, config, payload = load_model(checkpoint_path, device)
    if config.vocab_size > vocab_size:
        raise SystemExit(
            f"Checkpoint vocab_size={config.vocab_size} is larger than tokenizer "
            f"vocab_size={vocab_size}."
        )

    eos_id = special_id(tokenizer, EOS_TOKEN, 2)
    pad_id = special_id(tokenizer, PAD_TOKEN, 1)
    amp_enabled = bool(args.amp and device.type == "cuda")
    if amp_enabled:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        amp_dtype = torch.float32

    sentences = read_sentences(args)
    prompt_ids = [tokenizer.encode(text, add_special_tokens=False).ids for text in sentences]
    sequences = [ids + [eos_id] for ids in prompt_ids]
    scored = [(index, ids) for index, ids in enumerate(sequences) if len(ids) >= 2]
    scored_by_index: dict[int, tuple[list[float], int, bool]] = {}
    for start in range(0, len(scored), args.batch_size):
        chunk = scored[start : start + args.batch_size]
        batch_results = score_batch(
            model,
            [ids for _, ids in chunk],
            pad_id,
            config.block_size,
            device,
            amp_enabled,
            amp_dtype,
        )
        for (index, _), result in zip(chunk, batch_results):
            scored_by_index[index] = result

    results: list[SentenceResult] = []
    for index, (text, ids) in enumerate(zip(sentences, sequences)):
        losses, correct, truncated = scored_by_index.get(index, ([], 0, False))
        token_count = len(losses)
        if token_count:
            mean_loss = sum(losses) / token_count
            top1 = correct / token_count
        else:
            mean_loss = float("nan")
            top1 = float("nan")
        token_losses = [
            (token_label(tokenizer, token_id), loss)
            for token_id, loss in zip(ids[1:], losses)
        ]
        generation = None
        if args.max_new_tokens > 0 and prompt_ids[index]:
            generation = generate_continuation(
                model, tokenizer, prompt_ids[index], eos_id, args, device
            )
        results.append(
            SentenceResult(
                index=index + 1,
                text=text,
                tokens=token_count,
                loss=mean_loss,
                perplexity=perplexity(mean_loss),
                top1_accuracy=top1,
                truncated=truncated,
                token_losses=token_losses,
                generation=generation,
            )
        )

    total_tokens = sum(result.tokens for result in results)
    total_loss = sum(result.loss * result.tokens for result in results if result.tokens)
    total_correct = sum(
        result.top1_accuracy * result.tokens for result in results if result.tokens
    )
    aggregate = {
        "sentences": len(results),
        "tokens": total_tokens,
        "loss": total_loss / total_tokens if total_tokens else float("nan"),
        "top1_accuracy": total_correct / total_tokens if total_tokens else float("nan"),
    }
    aggregate["perplexity"] = perplexity(aggregate["loss"])

    meta = {
        "checkpoint": str(checkpoint_path),
        "stage": payload.get("stage"),
        "best_val": payload.get("best_val"),
        "tokenizer": str(tokenizer_path),
        "device": str(device),
        "block_size": config.block_size,
        "max_new_tokens": args.max_new_tokens,
    }
    return results, {"meta": meta, "aggregate": aggregate}


def print_results(results: Iterable[SentenceResult], aggregate: dict, show_tokens: bool) -> None:
    for result in results:
        ppl = result.perplexity
        ppl_text = "inf" if math.isinf(ppl) else f"{ppl:.2f}"
        top1 = "n/a" if math.isnan(result.top1_accuracy) else f"{result.top1_accuracy:.1%}"
        truncated = " (truncated)" if result.truncated else ""
        log(
            f"#{result.index} | tokens {result.tokens} | loss {result.loss:.4f} "
            f"| ppl {ppl_text} | top1 {top1}{truncated}"
        )
        log(f"   text : {result.text}")
        if show_tokens:
            for token, token_loss in result.token_losses:
                log(f"     {token!r:<24} {token_loss:.4f}")
        if result.generation is not None:
            log(f"   gen  : {result.generation}")

    ppl = aggregate["perplexity"]
    ppl_text = "inf" if math.isinf(ppl) else f"{ppl:.2f}"
    log("")
    log(
        f"sentences {aggregate['sentences']} | tokens {aggregate['tokens']} "
        f"| loss {aggregate['loss']:.4f} | ppl {ppl_text} "
        f"| top1 {aggregate['top1_accuracy']:.1%}"
    )


def write_json(path: Path, results: Sequence[SentenceResult], report: dict) -> None:
    payload = dict(report)
    payload["sentences"] = [
        {
            **{k: v for k, v in asdict(result).items() if k != "token_losses"},
            "token_losses": [
                {"token": token, "loss": loss} for token, loss in result.token_losses
            ],
        }
        for result in results
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, results: Sequence[SentenceResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "index",
                "text",
                "tokens",
                "loss",
                "perplexity",
                "top1_accuracy",
                "truncated",
                "generation",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.index,
                    result.text,
                    result.tokens,
                    f"{result.loss:.6f}" if not math.isnan(result.loss) else "",
                    f"{result.perplexity:.6f}" if not math.isnan(result.perplexity) else "",
                    (
                        f"{result.top1_accuracy:.6f}"
                        if not math.isnan(result.top1_accuracy)
                        else ""
                    ),
                    result.truncated,
                    result.generation or "",
                ]
            )


def main() -> None:
    args = parse_args()
    results, report = evaluate(args)
    meta = report["meta"]
    log(f"checkpoint : {meta['checkpoint']} (stage={meta['stage']})")
    log(f"tokenizer  : {meta['tokenizer']}")
    log(f"device     : {meta['device']} | block_size {meta['block_size']}")
    log("")
    print_results(results, report["aggregate"], args.show_tokens)

    if args.json_out:
        path = Path(args.json_out).expanduser()
        write_json(path, results, report)
        log(f"[done] wrote {path}")
    if args.csv_out:
        path = Path(args.csv_out).expanduser()
        write_csv(path, results)
        log(f"[done] wrote {path}")


if __name__ == "__main__":
    main()
