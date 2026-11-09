import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model"
TRAIN_DIR = PROJECT_ROOT / "train"
TOKENIZER_DIR = PROJECT_ROOT / "data" / "tokenizer"
EFFECTIVE_TOKENS = 65536
COMPILE_MODES = ["default", "max-autotune-no-cudagraphs"]

sys.path.insert(0, str(TRAIN_DIR))
from train_utils import has_model_weights, resolve_data_path


def log(message):
    print(message, flush=True)


def build_env():
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["TORCHINDUCTOR_CACHE_DIR"] = str(
        PROJECT_ROOT / ".cache" / "inductor"
    )
    env["TRITON_CACHE_DIR"] = str(PROJECT_ROOT / ".cache" / "triton")
    return env


ENV = build_env()


def triton_available():
    try:
        import triton  # noqa: F401

        return True
    except Exception:
        return False


def run_command(command, timeout):
    started = time.time()
    try:
        process = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=ENV,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
        return process.returncode, process.stdout, time.time() - started
    except subprocess.TimeoutExpired as error:
        output = error.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        return 124, output + "\nTIMEOUT", time.time() - started


def failure_kind(output):
    lowered = output.lower()
    if "out of memory" in lowered or "cuda error" in lowered:
        return "oom"
    if "no module named 'triton'" in lowered:
        return "no_triton"
    return "error"


def last_float(output, pattern):
    matches = re.findall(pattern, output)
    return float(matches[-1]) if matches else None


def print_tail(output, lines=25):
    tail = output.strip().splitlines()[-lines:]
    for line in tail:
        log("    | " + line)


def pretrain_command(args, seq_len, batch_size, full_bf16, compile_mode):
    json_out = args.tmp_dir / (
        f"pretrain_s{seq_len}_b{batch_size}_"
        f"{compile_mode or 'nocompile'}_{'full' if full_bf16 else 'mixed'}.json"
    )
    command = [
        sys.executable,
        "-u",
        str(TRAIN_DIR / "benchmark_pretrain.py"),
        "--model-dir",
        str(args.model_dir),
        "--seq-len",
        str(seq_len),
        "--batch-size",
        str(batch_size),
        "--accum",
        "1",
        "--steps",
        str(args.steps),
        "--warmup-steps",
        str(args.warmup_steps),
        "--json-out",
        str(json_out),
        "--no-gradient-checkpointing",
    ]
    if compile_mode:
        command += ["--compile", "--compile-mode", compile_mode]
    else:
        command += ["--no-compile"]
    if full_bf16:
        command.append("--full-bf16")
    return command, json_out


def run_pretrain_bench(args, seq_len, batch_size, full_bf16, compile_mode):
    command, json_out = pretrain_command(
        args, seq_len, batch_size, full_bf16, compile_mode
    )
    timeout = args.compile_timeout if compile_mode else args.timeout
    code, output, seconds = run_command(command, timeout)
    if code != 0 or not json_out.exists():
        return None, output, failure_kind(output)
    result = json.loads(json_out.read_text(encoding="utf-8"))
    result["wall_seconds"] = round(seconds, 2)
    return result, output, None


def choose_fastest(results, safe_limit_gb=None):
    if not results:
        return None
    if safe_limit_gb is not None:
        safe = [
            result
            for result in results
            if result.get("peak_memory_gb") is not None
            and result["peak_memory_gb"] <= safe_limit_gb
        ]
        if safe:
            return max(safe, key=lambda item: item["tokens_per_second"])
    return max(results, key=lambda item: item["tokens_per_second"])


def find_max_pretrain_batch(
    args, seq_len, candidates, compile_mode, full_bf16, deadline
):
    results = []
    for batch_size in candidates:
        if time.time() > deadline:
            log("  [pretrain] batas waktu tuning tercapai")
            break
        log(
            f"  [pretrain] seq={seq_len} batch={batch_size} "
            f"compile={compile_mode or 'no'} "
            f"bf16={'full' if full_bf16 else 'mixed'}"
        )
        result, output, error = run_pretrain_bench(
            args, seq_len, batch_size, full_bf16, compile_mode
        )
        if result is None:
            log(f"    gagal ({error})")
            if error == "oom":
                break
            print_tail(output)
            break
        log(
            f"    {result['tokens_per_second']:,.0f} tok/s | "
            f"peak {result['peak_memory_gb']:.2f} GB"
        )
        results.append(result)
    return choose_fastest(results, args.safe_memory_gb)


def tune_pretrain(args, deadline):
    if time.time() > deadline:
        return None
    seq_len = 1024
    candidates = [1, 2, 4] if args.quick else [4, 8, 12, 16, 20, 24, 28, 32]
    baseline = find_max_pretrain_batch(
        args, seq_len, candidates, None, False, deadline
    )
    if baseline is None:
        return None
    batch_size = baseline["batch_size"]
    modes = [None]
    if triton_available():
        modes += COMPILE_MODES
    runs = []
    for mode in modes:
        if time.time() > deadline:
            break
        if mode is None:
            runs.append(baseline)
            continue
        log(
            f"  [pretrain] uji compile={mode} pada batch={batch_size}"
        )
        result, output, error = run_pretrain_bench(
            args, seq_len, batch_size, False, mode
        )
        if result is None:
            log(f"    compile gagal ({error})")
            if error != "oom":
                print_tail(output, 15)
            continue
        log(
            f"    {result['tokens_per_second']:,.0f} tok/s | "
            f"peak {result['peak_memory_gb']:.2f} GB"
        )
        runs.append(result)
    best = choose_fastest(runs, args.safe_memory_gb)
    full_bf16 = None
    if not args.quick and time.time() < deadline:
        log("  [pretrain] cek full-bf16 pada konfigurasi terbaik")
        full_bf16, output, error = run_pretrain_bench(
            args, seq_len, batch_size, True, best["compile_mode"]
        )
        if full_bf16 is not None:
            log(
                f"    {full_bf16['tokens_per_second']:,.0f} tok/s | "
                f"peak {full_bf16['peak_memory_gb']:.2f} GB"
            )
        else:
            log(f"    full-bf16 gagal ({error})")
    seq_2048 = None
    if args.include_2048 and not args.quick and time.time() < deadline:
        log("  [pretrain] uji seq-len 2048")
        candidates_2048 = [2, 4, 6, 8, 10, 12, 16]
        seq_2048 = find_max_pretrain_batch(
            args,
            2048,
            candidates_2048,
            best["compile_mode"],
            False,
            deadline,
        )
    grad_accum = max(
        1, int(round(EFFECTIVE_TOKENS / (batch_size * seq_len)))
    )
    return {
        "seq_len": seq_len,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "tokens_per_step": batch_size * grad_accum * seq_len,
        "compile": best["compile"],
        "compile_mode": best["compile_mode"],
        "dtype": "bf16_autocast",
        "tokens_per_second": best["tokens_per_second"],
        "peak_memory_gb": best["peak_memory_gb"],
        "full_bf16": full_bf16,
        "seq_len_2048": seq_2048,
        "all_runs": runs,
    }


def stage_data_ready(args, stage):
    if stage == "sft":
        return args.sft_train_file.exists() and args.sft_val_file.exists()
    return (
        args.pref_train_file.exists()
        and args.pref_val_file.exists()
        and args.pref_test_file.exists()
    )


def stage_command(args, stage, base_checkpoint, seq_len, batch_size, compile_mode):
    output_dir = args.tmp_dir / (
        f"{stage}_s{seq_len}_b{batch_size}_{compile_mode or 'nocompile'}"
    )
    if stage == "sft":
        command = [
            sys.executable,
            "-u",
            str(TRAIN_DIR / "train_sft.py"),
            "--base-checkpoint",
            str(base_checkpoint),
            "--train-file",
            str(args.sft_train_file),
            "--val-file",
            str(args.sft_val_file),
            "--tokenizer-dir",
            str(args.tokenizer_dir),
            "--output-dir",
            str(output_dir),
            "--max-seq-len",
            str(seq_len),
            "--batch-size",
            str(batch_size),
            "--grad-accum",
            "1",
            "--max-steps",
            str(args.stage_steps),
            "--warmup-steps",
            "10",
            "--log-interval",
            "2",
            "--eval-interval",
            "0",
            "--eval-batches",
            "2",
            "--save-interval",
            "0",
            "--keep-checkpoints",
            "0",
            "--max-samples",
            str(args.max_samples),
            "--num-workers",
            str(args.num_workers),
            "--dtype",
            "bf16",
            "--no-gradient-checkpointing",
            "--no-final-save",
        ]
    else:
        command = [
            sys.executable,
            "-u",
            str(TRAIN_DIR / "train_preference.py"),
            "--algo",
            "simpo",
            "--base-checkpoint",
            str(base_checkpoint),
            "--train-file",
            str(args.pref_train_file),
            "--val-file",
            str(args.pref_val_file),
            "--test-file",
            str(args.pref_test_file),
            "--tokenizer-dir",
            str(args.tokenizer_dir),
            "--output-dir",
            str(output_dir),
            "--max-seq-len",
            str(seq_len),
            "--batch-size",
            str(batch_size),
            "--grad-accum",
            "1",
            "--max-steps",
            str(args.stage_steps),
            "--warmup-steps",
            "10",
            "--log-interval",
            "2",
            "--eval-interval",
            "0",
            "--eval-batches",
            "2",
            "--save-interval",
            "0",
            "--keep-checkpoints",
            "0",
            "--max-samples",
            str(args.max_samples),
            "--num-workers",
            str(args.num_workers),
            "--dtype",
            "bf16",
            "--lr",
            "1e-6",
            "--simpo-beta",
            "2.0",
            "--simpo-gamma",
            "0.5",
            "--no-gradient-checkpointing",
            "--no-final-save",
        ]
    if compile_mode:
        command += ["--compile", "--compile-mode", compile_mode]
    else:
        command += ["--no-compile"]
    return command


def run_stage_bench(args, stage, base_checkpoint, seq_len, batch_size, compile_mode):
    command = stage_command(
        args, stage, base_checkpoint, seq_len, batch_size, compile_mode
    )
    timeout = args.compile_timeout if compile_mode else args.timeout
    code, output, seconds = run_command(command, timeout)
    tokens_per_second = last_float(output, r"tok/s=([0-9.]+)")
    pairs_per_second = last_float(output, r"pair/s=([0-9.]+)")
    peak_memory = last_float(output, r"mem=([0-9.]+)GB")
    if code != 0 or tokens_per_second is None:
        return None, output, failure_kind(output)
    return {
        "tokens_per_second": round(tokens_per_second, 1),
        "pairs_per_second": (
            round(pairs_per_second, 1)
            if pairs_per_second is not None
            else None
        ),
        "peak_memory_gb": peak_memory,
        "wall_seconds": round(seconds, 2),
        "compile": compile_mode is not None,
        "compile_mode": compile_mode,
    }, output, None


def find_max_stage_batch(
    args, stage, base_checkpoint, seq_len, candidates, compile_mode, deadline
):
    results = []
    for batch_size in candidates:
        if time.time() > deadline:
            log(f"  [{stage}] batas waktu tuning tercapai")
            break
        log(
            f"  [{stage}] seq={seq_len} batch={batch_size} "
            f"compile={compile_mode or 'no'}"
        )
        result, output, error = run_stage_bench(
            args, stage, base_checkpoint, seq_len, batch_size, compile_mode
        )
        if result is None:
            log(f"    gagal ({error})")
            if error == "oom":
                break
            print_tail(output)
            break
        log(
            f"    {result['tokens_per_second']:,.0f} tok/s | "
            f"peak {result['peak_memory_gb'] or 0:.2f} GB"
        )
        result["batch_size"] = batch_size
        result["seq_len"] = seq_len
        results.append(result)
    return choose_fastest(results, args.safe_memory_gb)


def tune_stage(args, stage, base_checkpoint, seq_len, candidates, deadline):
    if base_checkpoint is None or not stage_data_ready(args, stage):
        return {"skipped": True, "reason": "checkpoint atau data tidak tersedia"}
    baseline = find_max_stage_batch(
        args, stage, base_checkpoint, seq_len, candidates, None, deadline
    )
    if baseline is None:
        return {"skipped": True, "reason": "semua konfigurasi gagal"}
    batch_size = baseline["batch_size"]
    best = baseline
    modes = COMPILE_MODES if triton_available() else []
    for compile_mode in modes:
        if time.time() > deadline:
            break
        log(
            f"  [{stage}] uji compile={compile_mode} pada batch={batch_size}"
        )
        compiled, output, error = run_stage_bench(
            args, stage, base_checkpoint, seq_len, batch_size, compile_mode
        )
        if compiled is not None:
            compiled["batch_size"] = batch_size
            compiled["seq_len"] = seq_len
            log(
                f"    {compiled['tokens_per_second']:,.0f} tok/s | "
                f"peak {compiled['peak_memory_gb'] or 0:.2f} GB"
            )
            if compiled["tokens_per_second"] > best["tokens_per_second"]:
                best = compiled
        else:
            log(f"    compile gagal ({error})")
            if error != "oom":
                print_tail(output, 15)
    grad_accum = max(
        1, int(round(EFFECTIVE_TOKENS / (batch_size * seq_len)))
    )
    best = dict(best)
    best["grad_accum"] = grad_accum
    best["tokens_per_step"] = batch_size * grad_accum * seq_len
    best["dtype"] = "bf16_autocast"
    return best


def create_random_checkpoint(args):
    cached = getattr(args, "_random_checkpoint", None)
    if cached is not None and has_model_weights(cached):
        return cached
    import torch

    sys.path.insert(0, str(args.model_dir.resolve()))
    from model import IndonesianChatLM, ModelConfig

    config = ModelConfig.from_json(args.model_dir / "config.json")
    model = IndonesianChatLM(config)
    target = args.tmp_dir / "random_base"
    model.save_pretrained(target)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    args._random_checkpoint = target
    log(f"  checkpoint acak dibuat untuk benchmark: {target}")
    return target


def resolve_base_checkpoint(args):
    if args.base_checkpoint is not None:
        path = Path(args.base_checkpoint)
        if has_model_weights(path):
            return path
    final = PROJECT_ROOT / "checkpoints" / "pretrain" / "final"
    if has_model_weights(final):
        return final
    return create_random_checkpoint(args)


def print_summary(summary):
    log("\n=== RINGKASAN AUTOTUNE A100 ===")
    gpu = summary.get("gpu", {})
    log(f"GPU: {gpu.get('name')} | VRAM {gpu.get('memory_gb')} GB")
    for name in ("pretrain", "sft", "simpo"):
        result = summary.get(name)
        if not result:
            log(f"{name}: tidak dijalankan")
            continue
        if result.get("skipped"):
            log(f"{name}: SKIP ({result.get('reason')})")
            continue
        log(
            f"{name}: seq={result['seq_len']} batch={result['batch_size']} "
            f"accum={result['grad_accum']} "
            f"compile={result['compile_mode'] or 'no'} "
            f"tok/s={result['tokens_per_second']:,.0f} "
            f"peak={result['peak_memory_gb'] or 0:.1f}GB"
        )
        if result.get("seq_len_2048"):
            alt = result["seq_len_2048"]
            log(
                f"  alternatif seq=2048: batch={alt['batch_size']} "
                f"tok/s={alt['tokens_per_second']:,.0f}"
            )
        if result.get("full_bf16"):
            alt = result["full_bf16"]
            log(
                f"  full-bf16: tok/s={alt['tokens_per_second']:,.0f} "
                f"(tidak direkomendasikan untuk stabilitas Adam)"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument(
        "--base-checkpoint", type=Path, default=None
    )
    parser.add_argument(
        "--sft-train-file",
        type=Path,
        default=resolve_data_path(PROJECT_ROOT / "data" / "sft" / "train.jsonl"),
    )
    parser.add_argument(
        "--sft-val-file",
        type=Path,
        default=resolve_data_path(PROJECT_ROOT / "data" / "sft" / "val.jsonl"),
    )
    parser.add_argument(
        "--pref-train-file",
        type=Path,
        default=resolve_data_path(
            PROJECT_ROOT / "data" / "preference" / "train.jsonl"
        ),
    )
    parser.add_argument(
        "--pref-val-file",
        type=Path,
        default=resolve_data_path(
            PROJECT_ROOT / "data" / "preference" / "val.jsonl"
        ),
    )
    parser.add_argument(
        "--pref-test-file",
        type=Path,
        default=resolve_data_path(
            PROJECT_ROOT / "data" / "preference" / "test.jsonl"
        ),
    )
    parser.add_argument(
        "--tokenizer-dir", type=Path, default=TOKENIZER_DIR
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "autotune_a100_results.json",
    )
    parser.add_argument(
        "--stages", default="pretrain,sft,simpo"
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--include-2048", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-minutes", type=float, default=45.0)
    parser.add_argument("--compile-timeout", type=float, default=1800.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--stage-steps", type=int, default=6)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.max_minutes = min(args.max_minutes, 12.0)
        args.max_samples = min(args.max_samples, 64)
        args.stage_steps = min(args.stage_steps, 4)
        args.include_2048 = False

    started = time.time()
    deadline = started + args.max_minutes * 60.0
    args.tmp_dir = Path(
        tempfile.mkdtemp(prefix="autotune_a100_", dir=str(PROJECT_ROOT))
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    stages = [item.strip() for item in args.stages.split(",") if item.strip()]

    import torch

    gpu = {
        "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "memory_gb": round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 1
        )
        if torch.cuda.is_available()
        else 0.0,
        "torch": torch.__version__,
        "triton": triton_available(),
    }
    log(f"[autotune] GPU={gpu['name']} | VRAM={gpu['memory_gb']} GB")
    args.safe_memory_gb = (
        gpu["memory_gb"] * 0.9 if gpu["memory_gb"] > 0 else None
    )
    if "A100" not in gpu["name"]:
        log("[autotune] PERINGATAN: GPU bukan A100; angka bukan baseline A100.")

    summary = {
        "gpu": gpu,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "expected_effective_tokens_per_step": EFFECTIVE_TOKENS,
    }

    if "pretrain" in stages:
        log("\n[autotune] === PRETRAIN ===")
        summary["pretrain"] = tune_pretrain(args, deadline)
    if "sft" in stages:
        log("\n[autotune] === SFT ===")
        base = resolve_base_checkpoint(args)
        summary["sft"] = tune_stage(
            args,
            "sft",
            base,
            2048,
            [1, 2] if args.quick else [1, 2, 4, 6, 8],
            deadline,
        )
    if "simpo" in stages:
        log("\n[autotune] === SIMPO ===")
        base_checkpoint = resolve_base_checkpoint(args)
        summary["simpo"] = tune_stage(
            args,
            "simpo",
            base_checkpoint,
            1024,
            [1] if args.quick else [1, 2, 3, 4],
            deadline,
        )

    summary["elapsed_minutes"] = round((time.time() - started) / 60.0, 2)
    args.output_json.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print_summary(summary)
    log(f"\n[autotune] hasil disimpan: {args.output_json}")
    if not args.keep_temp:
        shutil.rmtree(args.tmp_dir, ignore_errors=True)
    else:
        log(f"[autotune] temp dipertahankan: {args.tmp_dir}")


if __name__ == "__main__":
    main()
