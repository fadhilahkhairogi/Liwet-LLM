"""Indonesian pretraining + SFT data pipeline: download -> BPE -> .bin -> batches.

Stages
------
1. download : stream Indonesian datasets from the Hugging Face Hub into JSONL.
2. bpe      : train a byte-level BPE tokenizer on a sample of the local corpus.
3. tokenize : encode every document, append EOS, and pack train/val .bin files.
4. batches  : inspect ready-to-train batches from the .bin files.

Examples
--------
python prepare_data.py all --out-dir data_id --vocab-size 32000
python prepare_data.py download --stage pretrain --max-docs 100000
python prepare_data.py bpe --out-dir data_id --vocab-size 32000
python prepare_data.py tokenize --out-dir data_id
python prepare_data.py batches --out-dir data_id --stage posttrain --split train

The training loop can import the batch helpers directly:

    from prepare_data import TokenDataset
    train = TokenDataset.from_out_dir("data_id", "pretrain", "train", block_size=1024)
    x, y = train.get_batch(batch_size=8, device="cuda")

Layout produced under --out-dir:
    raw/pretrain/*.jsonl
    raw/posttrain/*.jsonl
    tokenizer/tokenizer.json
    tokenizer/meta.json
    bins/pretrain_train.bin
    bins/pretrain_val.bin
    bins/posttrain_train.bin
    bins/posttrain_val.bin
    bins/meta.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
from tqdm import tqdm

UNK_TOKEN = "<|unk|>"
PAD_TOKEN = "<|pad|>"
EOS_TOKEN = "<|endoftext|>"

SPECIAL_TOKENS = [UNK_TOKEN, PAD_TOKEN, EOS_TOKEN]

PRETRAIN_DATASETS = [
    {
        "id": "wikimedia/wikipedia",
        "config": "20231101.id",
        "split": "train",
        "field": "text",
    },
    {
        "id": "allenai/c4",
        "config": "id",
        "split": "train",
        "field": "text",
    },
    {
        "id": "HuggingFaceFW/fineweb-2",
        "config": "ind_Latn",
        "split": "train",
        "field": "text",
    },
]

POSTTRAIN_DATASETS = [
    {
        "id": "OpenPathAI/alpaca-gpt4-indonesian-cleaned",
        "config": None,
        "split": "train",
        "format": "prompt_response",
    },
    {
        "id": "FreedomIntelligence/evol-instruct-indonesian",
        "config": None,
        "split": "train",
        "format": "conversations",
    },
    {
        "id": "LorthGyu/indonesian-instructions",
        "config": None,
        "split": "train",
        "format": "alpaca",
    },
    {
        "id": "hanifabdlh/LaMini-Instruction-Indonesian-Google-Translated",
        "config": None,
        "split": "train",
        "format": "context_instruction_response",
    },
]

DEFAULT_PRETRAIN_DOCS_PER_DATASET = 200_000
DEFAULT_POSTTRAIN_DOCS_PER_DATASET = 100_000


def log(message: str) -> None:
    print(message, flush=True)


def _slug(spec: dict) -> str:
    raw = f"{spec['id']}_{spec.get('config') or 'default'}"
    return re.sub(r"[^A-Za-z0-9._-]+", "__", raw).strip("_")


def _json_line(text: str) -> str:
    return json.dumps({"text": text}, ensure_ascii=False) + "\n"


def _clean_text(text, min_chars: int, max_chars: int) -> str | None:
    if not isinstance(text, str):
        return None
    text = text.strip()
    if len(text) < min_chars:
        return None
    if max_chars > 0 and len(text) > max_chars:
        return None
    return text or None


def _batched(iterable: Iterable, size: int) -> Iterator[list]:
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _iter_jsonl_text(paths: Sequence[Path]) -> Iterator[str]:
    handles = [open(path, "r", encoding="utf-8") for path in paths]
    try:
        active = list(range(len(handles)))
        while active:
            for index in list(active):
                line = handles[index].readline()
                if not line:
                    active.remove(index)
                    continue
                try:
                    text = json.loads(line).get("text", "")
                except json.JSONDecodeError:
                    continue
                if text:
                    yield text
    finally:
        for handle in handles:
            handle.close()


def _plain_join(*parts: str) -> str:
    cleaned = [part.strip() for part in parts if part and part.strip()]
    return "\n\n".join(cleaned)


def _format_conversations(example: dict) -> str:
    values = []
    for turn in example.get("conversations") or []:
        value = turn.get("value") or ""
        if value.strip():
            values.append(value)
    return _plain_join(*values)


def _format_prompt_response(example: dict) -> str:
    return _plain_join(
        example.get("prompt") or "",
        example.get("response") or "",
    )


def _format_alpaca(example: dict) -> str:
    return _plain_join(
        example.get("instruction") or "",
        example.get("input") or "",
        example.get("output") or "",
    )


def _format_context_instruction_response(example: dict) -> str:
    return _plain_join(
        example.get("context") or "",
        example.get("instruction") or "",
        example.get("response") or "",
    )


POSTTRAIN_FORMATTERS = {
    "conversations": _format_conversations,
    "prompt_response": _format_prompt_response,
    "alpaca": _format_alpaca,
    "context_instruction_response": _format_context_instruction_response,
}


def _load_streaming_dataset(spec: dict):
    from datasets import load_dataset

    kwargs = {"split": spec.get("split", "train"), "streaming": True}
    if spec.get("revision"):
        kwargs["revision"] = spec["revision"]
    return load_dataset(spec["id"], spec.get("config"), **kwargs)


def _iter_stage_texts(stage: str, spec: dict, dataset) -> Iterator[str]:
    if stage == "pretrain":
        field = spec.get("field", "text")
        for example in dataset:
            text = example.get(field)
            if text:
                yield text
    else:
        formatter = POSTTRAIN_FORMATTERS[spec["format"]]
        for example in dataset:
            text = formatter(example)
            if text:
                yield text


def _download_one(
    stage: str,
    spec: dict,
    path: Path,
    limit: int,
    force: bool,
    shuffle_buffer: int,
    seed: int,
    min_chars: int,
    max_chars: int,
) -> None:
    if path.exists() and not force:
        log(f"[skip] {stage}/{path.name} already exists (use --force to redo)")
        return

    dataset = _load_streaming_dataset(spec)
    texts = _iter_stage_texts(stage, spec, dataset)
    rng = random.Random(seed)
    tmp_path = path.with_name(path.name + ".tmp")
    buffer: list[str] = []
    written = 0
    progress = tqdm(
        texts,
        total=limit if limit > 0 else None,
        desc=f"{stage}/{_slug(spec)}",
        unit="doc",
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            for text in progress:
                if limit > 0 and written >= limit:
                    break
                text = _clean_text(text, min_chars, max_chars)
                if text is None:
                    continue

                if shuffle_buffer > 1:
                    buffer.append(text)
                    if len(buffer) >= shuffle_buffer:
                        handle.write(_json_line(buffer.pop(rng.randrange(len(buffer)))))
                else:
                    handle.write(_json_line(text))
                written += 1
                progress.set_postfix(written=written)

            while buffer:
                handle.write(_json_line(buffer.pop(rng.randrange(len(buffer)))))
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    log(f"[done] {stage}/{path.name}: {written:,} documents")


def cmd_download(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).expanduser().resolve()
    stages = ["pretrain", "posttrain"] if args.stage == "both" else [args.stage]

    failures = 0
    for stage in stages:
        specs = PRETRAIN_DATASETS if stage == "pretrain" else POSTTRAIN_DATASETS
        stage_dir = out_dir / "raw" / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        default_limit = (
            DEFAULT_PRETRAIN_DOCS_PER_DATASET
            if stage == "pretrain"
            else DEFAULT_POSTTRAIN_DOCS_PER_DATASET
        )
        limit = default_limit if args.max_docs is None else args.max_docs
        limit_text = "all" if limit <= 0 else f"{limit:,}"

        log(f"== {stage}: {len(specs)} dataset(s), limit={limit_text} docs/dataset")
        for index, spec in enumerate(specs):
            path = stage_dir / f"{_slug(spec)}.jsonl"
            try:
                _download_one(
                    stage=stage,
                    spec=spec,
                    path=path,
                    limit=limit,
                    force=args.force,
                    shuffle_buffer=args.shuffle_buffer,
                    seed=args.seed + index,
                    min_chars=args.min_chars,
                    max_chars=args.max_chars,
                )
            except Exception as error:
                failures += 1
                log(f"[error] {spec['id']}: {type(error).__name__}: {error}")
                if args.fail_fast:
                    raise

    if failures:
        log(f"[warn] {failures} dataset(s) failed; rerun the command to retry.")


def _sample_texts(
    pretrain_files: Sequence[Path],
    posttrain_files: Sequence[Path],
    total_bytes: int,
) -> Iterator[str]:
    used = 0
    pretrain_budget = int(total_bytes * 0.9) if posttrain_files else total_bytes
    if pretrain_files:
        for text in _iter_jsonl_text(pretrain_files):
            if used >= pretrain_budget:
                break
            used += len(text.encode("utf-8"))
            yield text
    if posttrain_files:
        for text in _iter_jsonl_text(posttrain_files):
            if used >= total_bytes:
                break
            used += len(text.encode("utf-8"))
            yield text


def train_bpe(
    out_dir: Path,
    vocab_size: int,
    sample_mb: int,
    min_frequency: int,
    seed: int,
) -> None:
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

    raw_dir = out_dir / "raw"
    pretrain_files = sorted((raw_dir / "pretrain").glob("*.jsonl"))
    posttrain_files = sorted((raw_dir / "posttrain").glob("*.jsonl"))
    if not pretrain_files and not posttrain_files:
        raise SystemExit(f"No JSONL files under {raw_dir}. Run the download stage first.")

    if vocab_size <= len(SPECIAL_TOKENS) + 256:
        raise SystemExit("--vocab-size must be greater than the byte alphabet plus special tokens.")

    total_bytes = sample_mb * 1024 * 1024
    log(
        f"== bpe: sampling up to {sample_mb:,} MB from "
        f"{len(pretrain_files)} pretrain / {len(posttrain_files)} posttrain file(s)"
    )
    texts = _sample_texts(pretrain_files, posttrain_files, total_bytes)
    tokenizer = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)

    actual_vocab = tokenizer.get_vocab_size()
    tokenizer_dir = out_dir / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(tokenizer_dir / "tokenizer.json"))

    special_ids = {name: tokenizer.token_to_id(name) for name in SPECIAL_TOKENS}
    meta = {
        "vocab_size": actual_vocab,
        "requested_vocab_size": vocab_size,
        "special_tokens": special_ids,
        "unk_id": special_ids[UNK_TOKEN],
        "pad_id": special_ids[PAD_TOKEN],
        "eos_id": special_ids[EOS_TOKEN],
        "model": "byte-level BPE",
        "sample_mb": sample_mb,
        "min_frequency": min_frequency,
        "seed": seed,
    }
    (tokenizer_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if actual_vocab < vocab_size:
        log(f"[warn] requested vocab {vocab_size:,} but corpus only produced {actual_vocab:,}")
    log(f"[done] tokenizer/tokenizer.json: vocab_size={actual_vocab:,}")
    log(f"       special ids: {special_ids}")

    sample = "Halo, dunia! Ini adalah uji coba tokenizer bahasa Indonesia."
    ids = tokenizer.encode(sample, add_special_tokens=False).ids
    log(f"       roundtrip: {tokenizer.decode(ids, skip_special_tokens=False)!r}")


def cmd_bpe(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).expanduser().resolve()
    train_bpe(
        out_dir=out_dir,
        vocab_size=args.vocab_size,
        sample_mb=args.bpe_sample_mb,
        min_frequency=args.min_frequency,
        seed=args.seed,
    )


def _stage_files(out_dir: Path, stage: str) -> list[Path]:
    return sorted((out_dir / "raw" / stage).glob("*.jsonl"))


def _ensure_val_split(
    train_path: Path,
    val_path: Path,
    dtype: np.dtype,
    min_tokens: int = 4096,
) -> int:
    dtype = np.dtype(dtype)
    val_tokens = val_path.stat().st_size // dtype.itemsize if val_path.exists() else 0
    train_tokens = train_path.stat().st_size // dtype.itemsize if train_path.exists() else 0
    if train_tokens == 0 or val_tokens > 0:
        return val_tokens

    take = min(train_tokens, max(min_tokens, min(100_000, train_tokens // 10)))
    data = np.memmap(train_path, dtype=dtype, mode="r")
    tail = np.asarray(data[-take:])
    del data
    with open(val_path, "wb") as handle:
        tail.tofile(handle)
    return len(tail)


def _tokenize_stage(
    stage: str,
    files: Sequence[Path],
    out_dir: Path,
    tokenizer,
    dtype: np.dtype,
    val_frac: float,
    encode_batch_size: int,
    seed: int,
) -> dict:
    bins_dir = out_dir / "bins"
    train_path = bins_dir / f"{stage}_train.bin"
    val_path = bins_dir / f"{stage}_val.bin"
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    rng = random.Random(seed)

    train_docs = 0
    val_docs = 0
    train_tokens = 0
    val_tokens = 0
    texts = tqdm(_iter_jsonl_text(files), desc=f"tokenize/{stage}", unit="doc")

    with open(train_path, "wb") as train_handle, open(val_path, "wb") as val_handle:
        for chunk in _batched(texts, encode_batch_size):
            encodings = tokenizer.encode_batch(chunk, add_special_tokens=False)
            train_chunk: list[int] = []
            val_chunk: list[int] = []

            for encoding in encodings:
                ids = list(encoding.ids)
                ids.append(eos_id)
                if rng.random() < val_frac:
                    val_chunk.extend(ids)
                    val_docs += 1
                else:
                    train_chunk.extend(ids)
                    train_docs += 1

            if train_chunk:
                np.asarray(train_chunk, dtype=dtype).tofile(train_handle)
                train_tokens += len(train_chunk)
            if val_chunk:
                np.asarray(val_chunk, dtype=dtype).tofile(val_handle)
                val_tokens += len(val_chunk)

    val_tokens = _ensure_val_split(train_path, val_path, dtype)
    if val_docs == 0 and val_tokens > 0:
        val_docs = 1
    info = {
        "train_docs": train_docs,
        "val_docs": val_docs,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "val_frac": val_frac,
        "train_file": str(train_path.relative_to(out_dir)).replace("\\", "/"),
        "val_file": str(val_path.relative_to(out_dir)).replace("\\", "/"),
    }
    log(
        f"[done] {stage}: train={train_tokens:,} tokens / {train_docs:,} docs, "
        f"val={val_tokens:,} tokens / {val_docs:,} docs"
    )
    return info


def cmd_tokenize(args: argparse.Namespace) -> None:
    from tokenizers import Tokenizer

    out_dir = Path(args.out_dir).expanduser().resolve()
    tokenizer_path = out_dir / "tokenizer" / "tokenizer.json"
    if not tokenizer_path.exists():
        raise SystemExit(f"Missing {tokenizer_path}. Run the bpe stage first.")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab_size = tokenizer.get_vocab_size()
    dtype = np.uint16 if vocab_size <= 65536 else np.uint32
    stages = ["pretrain", "posttrain"] if args.stage == "both" else [args.stage]
    bins_dir = out_dir / "bins"
    bins_dir.mkdir(parents=True, exist_ok=True)

    bins_meta = {
        "vocab_size": vocab_size,
        "dtype": np.dtype(dtype).name,
        "eos_id": tokenizer.token_to_id(EOS_TOKEN),
        "stages": {},
    }

    for index, stage in enumerate(stages):
        files = _stage_files(out_dir, stage)
        if not files:
            log(f"[warn] no raw/{stage}/*.jsonl files; skipping")
            continue
        default_val_frac = 0.001 if stage == "pretrain" else 0.02
        val_frac = default_val_frac if args.val_frac is None else args.val_frac
        bins_meta["stages"][stage] = _tokenize_stage(
            stage=stage,
            files=files,
            out_dir=out_dir,
            tokenizer=tokenizer,
            dtype=dtype,
            val_frac=val_frac,
            encode_batch_size=args.encode_batch_size,
            seed=args.seed + index,
        )

    if not bins_meta["stages"]:
        raise SystemExit("Nothing tokenized. Run the download stage first.")

    (bins_dir / "meta.json").write_text(
        json.dumps(bins_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"[done] wrote {bins_dir / 'meta.json'}")
    log(f"[next] set GPTConfig.vocab_size = {vocab_size}")


def get_batch(
    data: np.ndarray,
    batch_size: int,
    block_size: int,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample random contiguous windows; returns int64 arrays x and y (shifted by 1)."""
    rng = rng or np.random.default_rng()
    max_start = len(data) - block_size - 1
    if max_start <= 0:
        raise ValueError(
            f"Not enough tokens ({len(data):,}) for block_size={block_size}. "
            "Reduce --block-size."
        )
    starts = rng.integers(0, max_start, size=batch_size)
    x = np.stack([np.asarray(data[i : i + block_size], dtype=np.int64) for i in starts])
    y = np.stack(
        [np.asarray(data[i + 1 : i + block_size + 1], dtype=np.int64) for i in starts]
    )
    return x, y


def iter_batches(
    data: np.ndarray,
    batch_size: int,
    block_size: int,
    shuffle: bool = True,
    drop_last: bool = True,
    seed: int = 0,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield non-overlapping blocks in shuffled order (one pass over the data)."""
    max_start = len(data) - block_size - 1
    if max_start <= 0:
        return
    starts = np.arange(0, max_start, block_size, dtype=np.int64)
    if drop_last:
        starts = starts[: (len(starts) // batch_size) * batch_size]
    if shuffle:
        np.random.default_rng(seed).shuffle(starts)

    for offset in range(0, len(starts), batch_size):
        batch_starts = starts[offset : offset + batch_size]
        if len(batch_starts) < batch_size and drop_last:
            break
        x = np.stack(
            [np.asarray(data[i : i + block_size], dtype=np.int64) for i in batch_starts]
        )
        y = np.stack(
            [np.asarray(data[i + 1 : i + block_size + 1], dtype=np.int64) for i in batch_starts]
        )
        yield x, y


def to_torch(x: np.ndarray, y: np.ndarray, device: str | None = None):
    import torch

    x_t = torch.from_numpy(x)
    y_t = torch.from_numpy(y)
    if device is not None:
        x_t = x_t.to(device)
        y_t = y_t.to(device)
    return x_t, y_t


class TokenDataset:
    """Memory-mapped token stream with random-window and sequential batch helpers."""

    def __init__(self, path: str | Path, dtype: str = "uint16", block_size: int | None = None):
        self.path = Path(path)
        self.dtype = np.dtype(dtype)
        self.data = np.memmap(self.path, dtype=self.dtype, mode="r")
        self.block_size = block_size

    @classmethod
    def from_out_dir(
        cls,
        out_dir: str | Path,
        stage: str,
        split: str,
        block_size: int | None = None,
    ) -> "TokenDataset":
        bins_dir = Path(out_dir) / "bins"
        meta_path = bins_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing {meta_path}. Run the tokenize stage first.")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        path = bins_dir / f"{stage}_{split}.bin"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run: python prepare_data.py tokenize "
                f"--out-dir {out_dir} --stage {stage}"
            )
        return cls(path, dtype=meta.get("dtype", "uint16"), block_size=block_size)

    def __len__(self) -> int:
        return len(self.data)

    def get_batch(
        self,
        batch_size: int,
        block_size: int | None = None,
        device: str | None = None,
        seed: int | None = None,
    ):
        rng = np.random.default_rng(seed)
        x, y = get_batch(
            self.data,
            batch_size=batch_size,
            block_size=block_size or self.block_size,
            rng=rng,
        )
        if device is not None:
            return to_torch(x, y, device)
        return x, y

    def iter_batches(
        self,
        batch_size: int,
        block_size: int | None = None,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0,
        device: str | None = None,
    ):
        for x, y in iter_batches(
            self.data,
            batch_size=batch_size,
            block_size=block_size or self.block_size,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=seed,
        ):
            if device is not None:
                yield to_torch(x, y, device)
            else:
                yield x, y


def cmd_batches(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).expanduser().resolve()
    try:
        dataset = TokenDataset.from_out_dir(
            out_dir, args.stage, args.split, block_size=args.block_size
        )
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from error
    log(
        f"== {dataset.path}: {len(dataset):,} tokens, "
        f"dtype={dataset.dtype}, block_size={args.block_size}"
    )

    tokenizer = None
    tokenizer_path = out_dir / "tokenizer" / "tokenizer.json"
    if tokenizer_path.exists():
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    for step in range(args.num_batches):
        x, y = dataset.get_batch(
            batch_size=args.batch_size,
            block_size=args.block_size,
            seed=args.seed + step,
        )
        if args.torch:
            x, y = to_torch(x, y, args.device)
            log(
                f"step {step}: x={tuple(x.shape)} y={tuple(y.shape)} "
                f"dtype={x.dtype} device={x.device}"
            )
        else:
            log(f"step {step}: x={x.shape} y={y.shape} dtype={x.dtype}")

        if tokenizer is not None:
            preview = tokenizer.decode(x[0][:64].tolist(), skip_special_tokens=False)
            log(f"  preview: {preview!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Indonesian pretraining/SFT data pipeline (download, BPE, bins, batches)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--out-dir",
            default="data_id",
            help="Output directory (default: ./data_id).",
        )
        sub.add_argument("--seed", type=int, default=1234, help="Random seed.")

    def add_download_args(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--stage",
            choices=["pretrain", "posttrain", "both"],
            default="both",
            help="Which corpus to download.",
        )
        sub.add_argument(
            "--max-docs",
            type=int,
            default=None,
            help=(
                "Max documents per dataset (0 = no limit). Default: "
                f"{DEFAULT_PRETRAIN_DOCS_PER_DATASET:,} for pretrain, "
                f"{DEFAULT_POSTTRAIN_DOCS_PER_DATASET:,} for posttrain."
            ),
        )
        sub.add_argument("--force", action="store_true", help="Redownload existing files.")
        sub.add_argument(
            "--fail-fast",
            action="store_true",
            help="Stop on the first dataset download error.",
        )
        sub.add_argument(
            "--shuffle-buffer",
            type=int,
            default=10_000,
            help="In-memory shuffle buffer while writing JSONL (0/1 disables).",
        )
        sub.add_argument(
            "--min-chars", type=int, default=32, help="Skip documents shorter than this."
        )
        sub.add_argument(
            "--max-chars",
            type=int,
            default=200_000,
            help="Skip documents longer than this (0 = no limit).",
        )

    def add_bpe_args(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--vocab-size", type=int, default=32_000, help="BPE vocabulary size.")
        sub.add_argument(
            "--bpe-sample-mb",
            type=int,
            default=128,
            help="Corpus sample size in MB used to train BPE.",
        )
        sub.add_argument(
            "--min-frequency", type=int, default=2, help="Minimum pair frequency for BPE."
        )

    def add_tokenize_args(sub: argparse.ArgumentParser, with_stage: bool = True) -> None:
        if with_stage:
            sub.add_argument(
                "--stage",
                choices=["pretrain", "posttrain", "both"],
                default="both",
                help="Which corpus to tokenize.",
            )
        sub.add_argument(
            "--val-frac",
            type=float,
            default=None,
            help="Validation fraction (default: 0.001 pretrain, 0.02 posttrain).",
        )
        sub.add_argument(
            "--encode-batch-size",
            type=int,
            default=256,
            help="Documents encoded per batch.",
        )

    download = subparsers.add_parser("download", help="Download Indonesian datasets to JSONL.")
    add_common(download)
    add_download_args(download)
    download.set_defaults(func=cmd_download)

    bpe = subparsers.add_parser("bpe", help="Train the byte-level BPE tokenizer.")
    add_common(bpe)
    add_bpe_args(bpe)
    bpe.set_defaults(func=cmd_bpe)

    tokenize = subparsers.add_parser("tokenize", help="Encode JSONL into train/val .bin files.")
    add_common(tokenize)
    add_tokenize_args(tokenize)
    tokenize.set_defaults(func=cmd_tokenize)

    batches = subparsers.add_parser("batches", help="Inspect batches from the .bin files.")
    add_common(batches)
    batches.add_argument(
        "--stage", choices=["pretrain", "posttrain"], default="pretrain"
    )
    batches.add_argument("--split", choices=["train", "val"], default="train")
    batches.add_argument("--batch-size", type=int, default=4)
    batches.add_argument("--block-size", type=int, default=1024)
    batches.add_argument("--num-batches", type=int, default=3)
    batches.add_argument("--torch", action="store_true", help="Return torch tensors.")
    batches.add_argument("--device", default=None, help="Torch device, e.g. cuda or cpu.")
    batches.set_defaults(func=cmd_batches)

    all_cmd = subparsers.add_parser("all", help="Run download, bpe, and tokenize.")
    add_common(all_cmd)
    add_download_args(all_cmd)
    add_bpe_args(all_cmd)
    add_tokenize_args(all_cmd, with_stage=False)
    all_cmd.set_defaults(func=None)

    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "all":
        cmd_download(args)
        cmd_bpe(args)
        cmd_tokenize(args)
        log(f"[next] python prepare_data.py batches --out-dir {args.out_dir}")
    else:
        args.func(args)


if __name__ == "__main__":
    main()
