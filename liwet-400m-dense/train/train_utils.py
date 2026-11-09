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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "model"))

from model import IndonesianChatLM, ModelConfig, count_parameters

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
        config = ModelConfig.from_json(config_path)
        model = IndonesianChatLM(config)
        state = torch.load(model_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state)
    elif (model_path / "pytorch_model.bin").exists():
        model = IndonesianChatLM.from_pretrained(model_path, map_location="cpu")
    elif (model_path / "config.json").exists():
        config = ModelConfig.from_json(model_path / "config.json")
        model = IndonesianChatLM(config)
        logging.warning("checkpoint %s tidak memiliki pytorch_model.bin, memakai inisialisasi acak", model_path)
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

def prune_checkpoints(output_dir: Path, keep_last: int) -> None:
    if keep_last < 0:
        return
    output_dir = Path(output_dir)
    checkpoints = sorted(path for path in output_dir.glob("step_*") if path.is_dir())
    if keep_last == 0:
        for path in checkpoints:
            shutil.rmtree(path, ignore_errors=True)
        return
    for path in checkpoints[:-keep_last]:
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

