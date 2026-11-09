import contextlib
import json
import logging
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

if os.name != "nt":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Sampler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "model"))

from model import (
    IndonesianChatLM,
    ModelConfig,
    count_active_parameters,
    count_parameters,
    has_model_weights,
)

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer_info.json",
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def default_num_workers(max_workers: int = 4) -> int:
    if os.name == "nt":
        return 0
    return max(1, min(max_workers, os.cpu_count() or 1))


class BucketBatchSampler(Sampler):
    def __init__(
        self,
        lengths,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
        bucket_multiplier: int = 8,
    ) -> None:
        lengths = [int(length) for length in lengths]
        if not lengths:
            raise ValueError("panjang sequence tidak boleh kosong")
        if batch_size < 1:
            raise ValueError("batch_size harus >= 1")
        self.lengths = lengths
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.seed = int(seed)
        self.bucket_size = max(
            self.batch_size, self.batch_size * max(1, int(bucket_multiplier))
        )
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        total = 0
        for start in range(0, len(self.lengths), self.bucket_size):
            size = min(start + self.bucket_size, len(self.lengths)) - start
            total += size // self.batch_size
        return total

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(indices)
        indices.sort(key=self.lengths.__getitem__)
        buckets = [
            indices[start : start + self.bucket_size]
            for start in range(0, len(indices), self.bucket_size)
        ]
        if self.shuffle:
            rng.shuffle(buckets)
        batches = []
        for bucket in buckets:
            if self.shuffle:
                rng.shuffle(bucket)
            limit = len(bucket) - self.batch_size + 1
            for start in range(0, max(0, limit), self.batch_size):
                batches.append(bucket[start : start + self.batch_size])
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches


def resolve_data_path(path: Path) -> Path:
    path = Path(path)
    if path.exists():
        return path
    try:
        relative = path.relative_to(PROJECT_ROOT)
    except ValueError:
        return path
    fallback = PROJECT_ROOT.parent / "TUBES_1_160M" / relative
    if fallback.exists():
        logging.warning("data %s tidak ada, memakai %s", path, fallback)
        return fallback
    return path


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_dtype(dtype: str, device: torch.device) -> str:
    if dtype != "auto":
        return dtype
    if device.type == "cuda":
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    return "fp32"


def autocast_context(device: torch.device, dtype: str):
    if device.type == "cuda" and dtype == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if device.type == "cuda" and dtype == "fp16":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def make_grad_scaler(device: torch.device, dtype: str):
    if device.type == "cuda" and dtype == "fp16":
        return torch.amp.GradScaler("cuda", enabled=True)
    return None


def load_model(model_path, device: torch.device) -> IndonesianChatLM:
    model_path = Path(model_path)
    if model_path.is_file():
        config_path = model_path.parent / "config.json"
        if not config_path.exists():
            config_path = PROJECT_ROOT / "model" / "config.json"
        model = IndonesianChatLM.from_pretrained(
            model_path, map_location="cpu", config_path=config_path
        )
    elif has_model_weights(model_path) and (
        model_path / "config.json"
    ).exists():
        model = IndonesianChatLM.from_pretrained(model_path, map_location="cpu")
    elif (model_path / "config.json").exists():
        config = ModelConfig.from_json(model_path / "config.json")
        model = IndonesianChatLM(config)
        logging.warning(
            "checkpoint %s tidak memiliki bobot, memakai inisialisasi acak",
            model_path,
        )
    else:
        raise FileNotFoundError(f"checkpoint model tidak ditemukan: {model_path}")
    model.to(device)
    return model


def get_optimizer(model: torch.nn.Module, lr: float, weight_decay: float, betas: tuple) -> AdamW:
    decay = []
    no_decay = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    fused = any(parameter.is_cuda for parameter in model.parameters())
    if fused:
        try:
            return AdamW(
                groups, lr=lr, betas=betas, eps=1e-8, fused=True
            )
        except (RuntimeError, TypeError) as error:
            logging.warning("AdamW fused tidak tersedia: %s", error)
    return AdamW(groups, lr=lr, betas=betas, eps=1e-8)


def make_scheduler(optimizer: AdamW, warmup_steps: int, total_steps: int, min_lr_ratio: float) -> LambdaLR:
    total_steps = max(1, total_steps)
    warmup_steps = max(0, min(warmup_steps, total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    save_dir: Path,
    model: IndonesianChatLM,
    optimizer: AdamW,
    scheduler: LambdaLR,
    scaler,
    global_step: int,
    best_loss: float,
    args: dict,
    tokenizer_dir: Path,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    copy_tokenizer(tokenizer_dir, save_dir)
    state = {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "global_step": global_step,
        "best_loss": best_loss,
        "args": args,
    }
    torch.save(state, save_dir / "training_state.pt")


def load_training_state(checkpoint_dir: Path, optimizer, scheduler, scaler, device: torch.device):
    path = checkpoint_dir / "training_state.pt"
    if not path.exists():
        return 0, float("inf")
    state = torch.load(path, map_location=device, weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    return state.get("global_step", 0), state.get("best_loss", float("inf"))


def copy_tokenizer(tokenizer_dir: Path, save_dir: Path) -> None:
    tokenizer_dir = Path(tokenizer_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for filename in TOKENIZER_FILES:
        source = tokenizer_dir / filename
        if source.exists():
            shutil.copy2(source, save_dir / filename)


class JsonlLogger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(self.path, "a", encoding="utf-8")

    def log(self, record: dict) -> None:
        record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **record}
        self.handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def setup_logging(log_file: Path) -> None:
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


def model_summary(model: torch.nn.Module) -> str:
    total = count_parameters(model)
    active = count_active_parameters(model)
    if active != total:
        return (
            f"{total:,} parameter ({total / 1e6:.1f}M total, "
            f"{active / 1e6:.1f}M aktif)"
        )
    return f"{total:,} parameter ({total / 1e6:.1f}M)"


def save_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
def args_to_dict(args) -> dict:
    result = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value)
        elif isinstance(value, tuple):
            result[key] = list(value)
        else:
            result[key] = value
    return result


def chunked_cross_entropy(logits: torch.Tensor, labels: torch.Tensor, chunk_size: int = 256):
    total_loss = logits.new_zeros((), dtype=torch.float32)
    total_tokens = torch.zeros((), dtype=torch.long, device=logits.device)
    sequence_length = logits.shape[1]
    for start in range(0, sequence_length - 1, chunk_size):
        end = min(start + chunk_size, sequence_length - 1)
        chunk_logits = logits[:, start:end, :].float()
        chunk_labels = labels[:, start + 1 : end + 1]
        loss = F.cross_entropy(
            chunk_logits.transpose(1, 2),
            chunk_labels,
            ignore_index=-100,
            reduction="sum",
        )
        total_loss = total_loss + loss
        total_tokens = total_tokens + (chunk_labels != -100).sum()
    return total_loss / total_tokens.clamp(min=1), total_tokens

def parse_step_number(path: Path) -> int:
    try:
        return int(path.name.split("_")[1])
    except Exception:
        return -1

def prune_checkpoints(output_dir: Path, keep_last: int, protected_steps: tuple = (8000, 16000, 2500, 2600)) -> None:
    if keep_last < 0:
        return
    output_dir = Path(output_dir)
    checkpoints = sorted(path for path in output_dir.glob("step_*") if path.is_dir())
    candidates = [p for p in checkpoints if parse_step_number(p) not in protected_steps]
    if keep_last == 0:
        for path in candidates:
            shutil.rmtree(path, ignore_errors=True)
        return
    for path in candidates[:-keep_last]:
        shutil.rmtree(path, ignore_errors=True)

def maybe_compile(model, mode: str = "default", dynamic: bool = False):
    try:
        import triton  # noqa: F401
    except Exception as error:
        logging.warning("torch.compile dilewati, Triton tidak tersedia: %s", error)
        return model
    try:
        import torch._dynamo

        torch._dynamo.config.suppress_errors = True
        torch._dynamo.config.cache_size_limit = max(
            int(getattr(torch._dynamo.config, "cache_size_limit", 8)), 32
        )
    except Exception:
        pass
    try:
        compiled = torch.compile(model, mode=mode, dynamic=dynamic)
        logging.info("torch.compile aktif | mode=%s dynamic=%s", mode, dynamic)
        return compiled
    except Exception as error:
        logging.warning("torch.compile mode=%s gagal: %s", mode, error)
        if mode != "default":
            try:
                compiled = torch.compile(model, mode="default", dynamic=dynamic)
                logging.info(
                    "torch.compile aktif | mode=default dynamic=%s", dynamic
                )
                return compiled
            except Exception as fallback_error:
                logging.warning("torch.compile gagal: %s", fallback_error)
        return model


def prevent_system_sleep() -> None:
    import ctypes
    import os

    if os.name != "nt":
        return
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception:
        pass

