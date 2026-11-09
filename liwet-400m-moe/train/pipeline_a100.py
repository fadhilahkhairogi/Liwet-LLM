import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_ROOT / "train"
MODEL_DIR = PROJECT_ROOT / "model"
DEFAULT_AUTOTUNE_JSON = PROJECT_ROOT / "autotune_a100_results.json"
STAGES = ("pretrain", "sft", "simpo")

sys.path.insert(0, str(TRAIN_DIR))
from train_utils import (
    default_num_workers,
    has_model_weights,
    resolve_data_path,
)


def build_env() -> dict:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    if os.name != "nt":
        env.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
        )
    env.setdefault(
        "TORCHINDUCTOR_CACHE_DIR", str(PROJECT_ROOT / ".cache" / "inductor")
    )
    env.setdefault("TRITON_CACHE_DIR", str(PROJECT_ROOT / ".cache" / "triton"))
    return env


ENV = build_env()


def log(message: str) -> None:
    print(f"[pipeline] {message}", flush=True)


def gpu_info() -> dict:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"name": "cpu", "memory_gb": 0.0}
        properties = torch.cuda.get_device_properties(0)
        return {
            "name": properties.name,
            "memory_gb": round(properties.total_memory / 1e9, 1),
            "capability": list(torch.cuda.get_device_capability(0)),
            "bf16": torch.cuda.is_bf16_supported(),
            "torch": torch.__version__,
        }
    except Exception as error:
        return {"error": str(error)}


def run_command(name: str, command: list, log_path: Path) -> tuple:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"[{name}] " + " ".join(str(part) for part in command))
    started = time.time()
    with open(log_path, "w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            [str(part) for part in command],
            cwd=str(PROJECT_ROOT),
            env=ENV,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            if process.stdout is not None:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    handle.write(line)
            returncode = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
            raise
    return returncode, time.time() - started


def is_complete(stage_dir: Path) -> bool:
    final = stage_dir / "final"
    return (final / "config.json").exists() and has_model_weights(final)


def is_valid_step(path: Path) -> bool:
    pt = path / "pytorch_model.bin"
    st = path / "training_state.pt"
    if not pt.exists() or not st.exists():
        return False
    if pt.stat().st_size < 1.5e9 or st.stat().st_size < 3.0e9:
        return False
    try:
        with open(pt, "rb") as f:
            f.seek(max(0, pt.stat().st_size - 65536))
            if b"PK\x05\x06" not in f.read():
                return False
        with open(st, "rb") as f:
            f.seek(max(0, st.stat().st_size - 65536))
            if b"PK\x05\x06" not in f.read():
                return False
    except Exception:
        return False
    return True


def latest_step(stage_dir: Path):
    steps = sorted(
        [path for path in stage_dir.glob("step_*") if is_valid_step(path)],
        key=lambda p: int(p.name.split("_")[1])
    )
    return steps[-1] if steps else None


def read_metrics(stage_dir: Path) -> dict:
    path = stage_dir / "metrics.jsonl"
    best_tokens = None
    last_train_loss = None
    last_val_loss = None
    last_val_accuracy = None
    records = 0
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            records += 1
            tokens = record.get("tokens_per_second")
            if tokens is not None:
                best_tokens = max(best_tokens or 0.0, float(tokens))
            if record.get("train_loss") is not None:
                last_train_loss = float(record["train_loss"])
            if record.get("val_loss") is not None:
                last_val_loss = float(record["val_loss"])
            if record.get("val_accuracy") is not None:
                last_val_accuracy = float(record["val_accuracy"])
    return {
        "records": records,
        "best_tokens_per_second": (
            round(best_tokens, 1) if best_tokens is not None else None
        ),
        "last_train_loss": last_train_loss,
        "last_val_loss": last_val_loss,
        "last_val_accuracy": last_val_accuracy,
    }


def tuned_value(results: dict, stage: str, key: str, default):
    entry = results.get(stage) or {}
    if entry.get("skipped"):
        return default
    value = entry.get(key)
    return default if value is None else value


def tuned_compile_mode(results: dict, stage: str, default):
    entry = results.get(stage) or {}
    if entry.get("skipped"):
        return default
    if "compile" in entry and not entry["compile"]:
        return None
    return entry.get("compile_mode", default)


def compile_args(mode) -> list:
    if mode is None or str(mode).lower() in {"none", "no", "off", "false"}:
        return ["--no-compile"]
    return ["--compile", "--compile-mode", str(mode)]


def maybe_autotune(args) -> tuple:
    path = Path(args.autotune_json)
    if not args.run_autotune:
        if path.exists():
            log(f"memakai hasil autotune: {path}")
            return json.loads(path.read_text(encoding="utf-8")), path
        log("autotune dimatikan dan hasil belum ada, memakai default A100")
        return {}, None
    if path.exists() and not args.rerun_autotune:
        log(f"memakai hasil autotune: {path}")
        return json.loads(path.read_text(encoding="utf-8")), path
    command = [
        sys.executable,
        "-u",
        str(TRAIN_DIR / "autotune_a100.py"),
        "--output-json",
        str(path),
        "--max-minutes",
        str(args.autotune_minutes),
        "--steps",
        str(args.autotune_steps),
        "--stage-steps",
        str(args.autotune_stage_steps),
        "--num-workers",
        str(args.num_workers),
        "--stages",
        "pretrain,sft,simpo",
    ]
    if args.autotune_quick:
        command.append("--quick")
    if args.autotune_keep_temp:
        command.append("--keep-temp")
    returncode, seconds = run_command(
        "autotune", command, Path(args.logs_dir) / "autotune.log"
    )
    if returncode != 0 or not path.exists():
        message = f"autotune gagal (returncode={returncode})"
        if args.allow_autotune_failure:
            log(message + ", memakai default")
            return {}, path if path.exists() else None
        raise SystemExit(message)
    log(f"autotune selesai dalam {seconds / 60.0:.1f} menit")
    return json.loads(path.read_text(encoding="utf-8")), path


def resolve(path) -> Path:
    return Path(resolve_data_path(Path(path)))


def preflight(args, stages: list) -> None:
    missing = []
    model_config = Path(args.model_dir) / "config.json"
    if not model_config.exists():
        missing.append(str(model_config))
    else:
        try:
            model_values = json.loads(
                model_config.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError:
            model_values = {}
        if not model_values.get("use_moe", False):
            message = f"{model_config} tidak mengaktifkan use_moe"
            if args.dry_run:
                log("PERINGATAN: " + message)
            else:
                raise SystemExit(message)
    tokenizer = Path(args.tokenizer_dir) / "tokenizer.json"
    if not tokenizer.exists():
        missing.append(str(tokenizer))
    if "pretrain" in stages:
        path = resolve(args.data_index)
        if not path.exists():
            missing.append(f"{path} (index data pretrain)")
    if "sft" in stages:
        for item in (args.sft_train_file, args.sft_val_file):
            path = resolve(item)
            if not path.exists():
                missing.append(str(path))
    if "simpo" in stages:
        for item in (
            args.pref_train_file,
            args.pref_val_file,
            args.pref_test_file,
        ):
            path = resolve(item)
            if not path.exists():
                missing.append(str(path))
    if missing:
        message = "file berikut tidak ditemukan:\n  - " + "\n  - ".join(missing)
        if args.dry_run:
            log("PERINGATAN: " + message)
            return
        raise SystemExit(message)


def resolve_base(args, stage: str, stage_dir: Path) -> Path:
    override = getattr(args, f"{stage}_base")
    if override is not None:
        return Path(override)
    if stage == "sft":
        return stage_dir["pretrain"] / "final"
    return stage_dir["sft"] / "final"


def pretrain_command(args, results, output_dir: Path, resume, data_index: Path) -> list:
    seq_len = args.seq_len_pretrain or int(
        tuned_value(results, "pretrain", "seq_len", 1024)
    )
    batch_size = args.batch_size_pretrain or int(
        tuned_value(results, "pretrain", "batch_size", 16)
    )
    grad_accum = args.grad_accum_pretrain or int(
        tuned_value(results, "pretrain", "grad_accum", 4)
    )
    compile_mode = args.compile_mode_pretrain
    if compile_mode is None:
        compile_mode = tuned_compile_mode(
            results, "pretrain", "max-autotune-no-cudagraphs"
        )
    command = [
        sys.executable,
        "-u",
        str(TRAIN_DIR / "train_pretrain.py"),
        "--model-dir",
        str(args.model_dir),
        "--data-index",
        str(data_index),
        "--output-dir",
        str(output_dir),
        "--seq-len",
        str(seq_len),
        "--batch-size",
        str(batch_size),
        "--grad-accum",
        str(grad_accum),
        "--dtype",
        args.dtype,
        "--num-workers",
        str(args.num_workers),
        "--log-interval",
        str(args.log_interval),
        "--eval-interval",
        str(args.eval_interval_pretrain),
        "--save-interval",
        str(args.save_interval_pretrain),
        "--eval-batches",
        str(args.eval_batches),
        "--keep-checkpoints",
        str(args.keep_checkpoints),
        "--epochs",
        str(args.epochs_pretrain),
    ]
    command += compile_args(compile_mode)
    if args.lr_pretrain is not None:
        command += ["--lr", str(args.lr_pretrain)]
    if args.max_steps_pretrain > 0:
        command += ["--max-steps", str(args.max_steps_pretrain)]
    if args.max_hours_pretrain > 0:
        command += ["--max-hours", str(args.max_hours_pretrain)]
    if args.pretrain_gradient_checkpointing:
        command.append("--gradient-checkpointing")
    if args.full_bf16:
        command.append("--full-bf16")
    if resume is not None:
        command += ["--resume", str(resume)]
    return command


def sft_command(args, results, output_dir: Path, base: Path, resume) -> list:
    seq_len = args.seq_len_sft or int(
        tuned_value(results, "sft", "seq_len", 2048)
    )
    batch_size = args.batch_size_sft or int(
        tuned_value(results, "sft", "batch_size", 4)
    )
    grad_accum = args.grad_accum_sft or int(
        tuned_value(results, "sft", "grad_accum", 8)
    )
    compile_mode = args.compile_mode_sft
    if compile_mode is None:
        compile_mode = tuned_compile_mode(results, "sft", "default")
    command = [
        sys.executable,
        "-u",
        str(TRAIN_DIR / "train_sft.py"),
        "--base-checkpoint",
        str(base),
        "--train-file",
        str(resolve(args.sft_train_file)),
        "--val-file",
        str(resolve(args.sft_val_file)),
        "--tokenizer-dir",
        str(args.tokenizer_dir),
        "--output-dir",
        str(output_dir),
        "--max-seq-len",
        str(seq_len),
        "--batch-size",
        str(batch_size),
        "--grad-accum",
        str(grad_accum),
        "--epochs",
        str(args.epochs_sft),
        "--dtype",
        args.dtype,
        "--num-workers",
        str(args.num_workers),
        "--log-interval",
        str(args.log_interval),
        "--eval-interval",
        str(args.eval_interval_sft),
        "--save-interval",
        str(args.save_interval_sft),
        "--eval-batches",
        str(args.eval_batches),
        "--keep-checkpoints",
        str(args.keep_checkpoints),
    ]
    if args.length_bucket:
        command += [
            "--bucket-multiplier",
            str(args.bucket_multiplier),
            "--length-bucket",
        ]
    command += compile_args(compile_mode)
    if args.lr_sft is not None:
        command += ["--lr", str(args.lr_sft)]
    if args.max_steps_sft > 0:
        command += ["--max-steps", str(args.max_steps_sft)]
    if args.max_hours_sft > 0:
        command += ["--max-hours", str(args.max_hours_sft)]
    if args.max_samples_sft > 0:
        command += ["--max-samples", str(args.max_samples_sft)]
    if args.sft_gradient_checkpointing:
        command.append("--gradient-checkpointing")
    if resume is not None:
        command += ["--resume", str(resume)]
    return command


def simpo_command(args, results, output_dir: Path, base: Path, resume) -> list:
    seq_len = args.seq_len_simpo or int(
        tuned_value(results, "simpo", "seq_len", 1024)
    )
    batch_size = args.batch_size_simpo or int(
        tuned_value(results, "simpo", "batch_size", 4)
    )
    grad_accum = args.grad_accum_simpo or int(
        tuned_value(results, "simpo", "grad_accum", 16)
    )
    compile_mode = args.compile_mode_simpo
    if compile_mode is None:
        compile_mode = tuned_compile_mode(results, "simpo", "max-autotune-no-cudagraphs")
    command = [
        sys.executable,
        "-u",
        str(TRAIN_DIR / "train_preference.py"),
        "--algo",
        "simpo",
        "--base-checkpoint",
        str(base),
        "--train-file",
        str(resolve(args.pref_train_file)),
        "--val-file",
        str(resolve(args.pref_val_file)),
        "--test-file",
        str(resolve(args.pref_test_file)),
        "--tokenizer-dir",
        str(args.tokenizer_dir),
        "--output-dir",
        str(output_dir),
        "--max-seq-len",
        str(seq_len),
        "--batch-size",
        str(batch_size),
        "--grad-accum",
        str(grad_accum),
        "--epochs",
        str(args.epochs_simpo),
        "--lr",
        str(args.lr_simpo),
        "--simpo-beta",
        str(args.simpo_beta),
        "--simpo-gamma",
        str(args.simpo_gamma),
        "--dtype",
        args.dtype,
        "--num-workers",
        str(args.num_workers),
        "--log-interval",
        str(args.log_interval),
        "--eval-interval",
        str(args.eval_interval_simpo),
        "--save-interval",
        str(args.save_interval_simpo),
        "--eval-batches",
        str(args.eval_batches),
        "--keep-checkpoints",
        str(args.keep_checkpoints),
    ]
    if args.length_bucket:
        command += [
            "--bucket-multiplier",
            str(args.bucket_multiplier),
            "--length-bucket",
        ]
    command += compile_args(compile_mode)
    if args.max_steps_simpo > 0:
        command += ["--max-steps", str(args.max_steps_simpo)]
    if args.max_hours_simpo > 0:
        command += ["--max-hours", str(args.max_hours_simpo)]
    if args.max_samples_simpo > 0:
        command += ["--max-samples", str(args.max_samples_simpo)]
    if args.simpo_gradient_checkpointing:
        command.append("--gradient-checkpointing")
    if resume is not None:
        command += ["--resume", str(resume)]
    return command


def write_summary(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline MoE 400M: autotune -> pretrain -> SFT -> SimPO"
    )
    parser.add_argument("--stages", default="pretrain,sft,simpo")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument(
        "--output-root", type=Path, default=PROJECT_ROOT / "checkpoints"
    )
    parser.add_argument("--logs-dir", type=Path, default=None)
    parser.add_argument(
        "--data-index",
        type=Path,
        default=PROJECT_ROOT / "data" / "tokenized" / "index.json",
    )
    parser.add_argument(
        "--sft-train-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "sft" / "train.jsonl",
    )
    parser.add_argument(
        "--sft-val-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "sft" / "val.jsonl",
    )
    parser.add_argument(
        "--pref-train-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "preference" / "train.jsonl",
    )
    parser.add_argument(
        "--pref-val-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "preference" / "val.jsonl",
    )
    parser.add_argument(
        "--pref-test-file",
        type=Path,
        default=PROJECT_ROOT / "data" / "preference" / "test.jsonl",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "tokenizer",
    )
    parser.add_argument(
        "--sft-base", type=Path, default=None, help="override checkpoint SFT"
    )
    parser.add_argument(
        "--simpo-base", type=Path, default=None, help="override checkpoint SimPO"
    )
    parser.add_argument("--pretrain-base", type=Path, default=None)

    parser.add_argument(
        "--autotune-json", type=Path, default=DEFAULT_AUTOTUNE_JSON
    )
    parser.add_argument(
        "--run-autotune",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--rerun-autotune", action="store_true")
    parser.add_argument("--autotune-quick", action="store_true")
    parser.add_argument("--autotune-minutes", type=float, default=45.0)
    parser.add_argument("--autotune-steps", type=int, default=3)
    parser.add_argument("--autotune-stage-steps", type=int, default=6)
    parser.add_argument("--autotune-keep-temp", action="store_true")
    parser.add_argument("--allow-autotune-failure", action="store_true")

    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--keep-checkpoints", type=int, default=6)
    parser.add_argument(
        "--length-bucket",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--bucket-multiplier", type=int, default=8)
    parser.add_argument("--full-bf16", action="store_true")

    parser.add_argument("--epochs-pretrain", type=float, default=1.0)
    parser.add_argument("--seq-len-pretrain", type=int, default=None)
    parser.add_argument("--batch-size-pretrain", type=int, default=None)
    parser.add_argument("--grad-accum-pretrain", type=int, default=None)
    parser.add_argument("--compile-mode-pretrain", default=None)
    parser.add_argument("--lr-pretrain", type=float, default=None)
    parser.add_argument("--max-steps-pretrain", type=int, default=0)
    parser.add_argument("--max-hours-pretrain", type=float, default=0.0)
    parser.add_argument(
        "--pretrain-gradient-checkpointing", action="store_true"
    )
    parser.add_argument("--eval-interval-pretrain", type=int, default=1000)
    parser.add_argument("--save-interval-pretrain", type=int, default=1000)

    parser.add_argument("--epochs-sft", type=int, default=1)
    parser.add_argument("--seq-len-sft", type=int, default=None)
    parser.add_argument("--batch-size-sft", type=int, default=None)
    parser.add_argument("--grad-accum-sft", type=int, default=None)
    parser.add_argument("--compile-mode-sft", default=None)
    parser.add_argument("--lr-sft", type=float, default=None)
    parser.add_argument("--max-steps-sft", type=int, default=0)
    parser.add_argument("--max-hours-sft", type=float, default=0.0)
    parser.add_argument("--max-samples-sft", type=int, default=0)
    parser.add_argument("--sft-gradient-checkpointing", action="store_true")
    parser.add_argument("--eval-interval-sft", type=int, default=500)
    parser.add_argument("--save-interval-sft", type=int, default=500)

    parser.add_argument("--epochs-simpo", type=int, default=1)
    parser.add_argument("--seq-len-simpo", type=int, default=None)
    parser.add_argument("--batch-size-simpo", type=int, default=None)
    parser.add_argument("--grad-accum-simpo", type=int, default=None)
    parser.add_argument("--compile-mode-simpo", default=None)
    parser.add_argument("--lr-simpo", type=float, default=1e-6)
    parser.add_argument("--simpo-beta", type=float, default=2.0)
    parser.add_argument("--simpo-gamma", type=float, default=0.5)
    parser.add_argument("--max-steps-simpo", type=int, default=0)
    parser.add_argument("--max-hours-simpo", type=float, default=0.0)
    parser.add_argument("--max-samples-simpo", type=int, default=0)
    parser.add_argument("--simpo-gradient-checkpointing", action="store_true")
    parser.add_argument("--eval-interval-simpo", type=int, default=30)
    parser.add_argument("--save-interval-simpo", type=int, default=30)

    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.logs_dir is None:
        args.logs_dir = args.output_root / "logs"
    if args.num_workers is None:
        args.num_workers = default_num_workers(8)
    return args


def main() -> None:
    args = parse_args()
    selected = [item.strip() for item in args.stages.split(",") if item.strip()]
    unknown = [item for item in selected if item not in STAGES]
    if unknown:
        raise SystemExit(f"stage tidak dikenal: {unknown}")

    stage_dir = {name: args.output_root / name for name in STAGES}
    pending = []
    skipped = {}
    for name in selected:
        if is_complete(stage_dir[name]) and not args.force:
            skipped[name] = "final checkpoint sudah ada"
        else:
            pending.append(name)

    if args.pretrain_base is not None:
        args.model_dir = args.pretrain_base

    info = gpu_info()
    log(f"GPU={info.get('name')} VRAM={info.get('memory_gb')} GB")
    if info.get("name") and "A100" not in info["name"]:
        log("PERINGATAN: GPU bukan A100, hasil tuning bisa berbeda")

    preflight(args, pending)
    if args.dry_run:
        for name, reason in skipped.items():
            log(f"{name}: SKIP ({reason})")
        results = {}
        path = Path(args.autotune_json)
        if path.exists():
            results = json.loads(path.read_text(encoding="utf-8"))
            log(f"dry-run memakai autotune: {path}")
        else:
            log("dry-run: autotune belum ada, memakai default A100")
    else:
        results, _ = maybe_autotune(args)

    summary = {
        "gpu": info,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "autotune_json": str(args.autotune_json),
        "stages": {},
    }
    summary_path = args.output_root / "pipeline_summary.json"
    started = time.time()

    for name, reason in skipped.items():
        log(f"{name}: SKIP ({reason})")
        summary["stages"][name] = {
            "skipped": True,
            "reason": reason,
            "output_dir": str(stage_dir[name]),
            "metrics": read_metrics(stage_dir[name]),
        }
        if not args.dry_run:
            write_summary(summary_path, summary)

    for name in pending:
        if name == "pretrain":
            resume = (
                latest_step(stage_dir[name]) if args.resume else None
            )
            command = pretrain_command(
                args,
                results,
                stage_dir[name],
                resume,
                resolve(args.data_index),
            )
        elif name == "sft":
            base = resolve_base(args, "sft", stage_dir)
            if not has_model_weights(base):
                if args.dry_run:
                    log(f"PERINGATAN: checkpoint dasar SFT belum ada: {base}")
                else:
                    raise SystemExit(
                        f"checkpoint dasar SFT tidak ditemukan: {base}"
                    )
            resume = (
                latest_step(stage_dir[name]) if args.resume else None
            )
            command = sft_command(args, results, stage_dir[name], base, resume)
        else:
            base = resolve_base(args, "simpo", stage_dir)
            if not has_model_weights(base):
                if args.dry_run:
                    log(f"PERINGATAN: checkpoint dasar SimPO belum ada: {base}")
                else:
                    raise SystemExit(
                        f"checkpoint dasar SimPO tidak ditemukan: {base}"
                    )
            resume = (
                latest_step(stage_dir[name]) if args.resume else None
            )
            command = simpo_command(args, results, stage_dir[name], base, resume)

        if args.dry_run:
            log(f"[dry-run][{name}] " + " ".join(str(part) for part in command))
            summary["stages"][name] = {
                "dry_run": True,
                "command": [str(part) for part in command],
                "output_dir": str(stage_dir[name]),
            }
            continue

        log(f"=== {name.upper()} ===")
        returncode, seconds = run_command(
            name, command, Path(args.logs_dir) / f"{name}.log"
        )
        record = {
            "skipped": False,
            "returncode": returncode,
            "seconds": round(seconds, 1),
            "minutes": round(seconds / 60.0, 2),
            "command": [str(part) for part in command],
            "output_dir": str(stage_dir[name]),
            "metrics": read_metrics(stage_dir[name]),
        }
        summary["stages"][name] = record
        write_summary(summary_path, summary)
        if returncode != 0:
            log(f"{name} gagal dengan returncode={returncode}")
            if not args.keep_going:
                raise SystemExit(returncode)

    summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    summary["elapsed_minutes"] = round((time.time() - started) / 60.0, 2)
    if not args.dry_run:
        write_summary(summary_path, summary)
        log(f"ringkasan pipeline: {summary_path}")
    log("=== HASIL ===")
    for name in selected:
        record = summary["stages"].get(name, {})
        if record.get("skipped"):
            log(f"{name}: SKIP ({record.get('reason')})")
            continue
        metrics = record.get("metrics") or {}
        tokens = metrics.get("best_tokens_per_second")
        tokens_text = f"{tokens:,.0f} tok/s" if tokens else "-"
        log(
            f"{name}: {record.get('minutes', 0):.1f} menit | "
            f"peak {tokens_text} | "
            f"val_loss={metrics.get('last_val_loss')}"
        )


if __name__ == "__main__":
    main()
