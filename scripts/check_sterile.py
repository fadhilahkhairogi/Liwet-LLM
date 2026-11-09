#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
import sys
from pathlib import Path

MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_JSON_BYTES = 100 * 1024
IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    ".venv",
    "venv",
    "env",
    "node_modules",
}
FORBIDDEN_DIRS = {"checkpoints", "logs", "data", "data_id", "__pycache__"}
FORBIDDEN_SUFFIXES = {
    ".pt",
    ".pth",
    ".bin",
    ".safetensors",
    ".ckpt",
    ".onnx",
    ".gguf",
    ".ggml",
    ".tar",
    ".tgz",
    ".zip",
    ".7z",
    ".rar",
    ".pkl",
    ".pickle",
    ".npy",
    ".npz",
    ".parquet",
    ".arrow",
    ".jsonl",
    ".gz",
    ".bz2",
    ".xz",
    ".h5",
    ".hdf5",
    ".sqlite",
    ".db",
    ".log",
    ".part",
    ".tmp",
    ".pid",
    ".out",
    ".err",
    ".pyc",
    ".pyo",
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
}
FORBIDDEN_NAMES = {".env"}
LOCAL_PATH_PATTERNS = [
    ("windows_drive", re.compile(r"(?i)(?<![A-Za-z0-9_])[A-Za-z]:[\\/]")),
    ("unc_path", re.compile(r"\\\\[A-Za-z0-9._-]+\\[A-Za-z0-9._$-]+")),
    ("unix_home", re.compile(r"/(?:content|kaggle|home|mnt|Users|root|Volumes)/")),
    ("file_url", re.compile("file" + "://")),
]
TOKEN_PATTERNS = [
    ("hf_token", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("bearer_token", re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._-]{20,}\b")),
    ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    (
        "env_token",
        re.compile(
            r"(?i)\b(?:HF_TOKEN|HUGGING_FACE_HUB_TOKEN|OPENAI_API_KEY|"
            r"ANTHROPIC_API_KEY|GITHUB_TOKEN)\s*[:=]\s*[\"']?[A-Za-z0-9_\-]{10,}"
        ),
    ),
]
IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def iter_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        for name in list(dirnames):
            if name in IGNORED_DIRS:
                dirnames.remove(name)
            elif name in FORBIDDEN_DIRS:
                yield current / name, "forbidden_dir"
                dirnames.remove(name)
        for name in filenames:
            yield current / name, "file"


def is_text_file(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            chunk = handle.read(8192)
    except OSError:
        return False
    return b"\x00" not in chunk


def iter_json_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from iter_json_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from iter_json_strings(item)


def line_number(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def scan_text(path: Path, text: str, violations: list[str]) -> None:
    seen = set()
    for label, pattern in LOCAL_PATH_PATTERNS + TOKEN_PATTERNS:
        for match in pattern.finditer(text):
            key = (label, match.start())
            if key in seen:
                continue
            seen.add(key)
            violations.append(
                f"{path}:{line_number(text, match.start())}: {label}: "
                f"{match.group(0)[:80]!r}"
            )
    for match in IPV4_RE.finditer(text):
        candidate = match.group(0)
        try:
            address = ipaddress.ip_address(candidate)
        except ValueError:
            continue
        if address.is_private or address.is_link_local or address.is_loopback:
            violations.append(
                f"{path}:{line_number(text, match.start())}: private_ip: {candidate}"
            )


def audit(root: Path) -> list[str]:
    violations: list[str] = []
    for path, kind in iter_files(root):
        if kind == "forbidden_dir":
            violations.append(f"{path}: forbidden directory")
            continue
        try:
            size = path.stat().st_size
        except OSError as exc:
            violations.append(f"{path}: cannot stat: {exc}")
            continue
        rel = path.relative_to(root)
        if size > MAX_FILE_BYTES:
            violations.append(
                f"{rel}: file too large ({size} bytes > {MAX_FILE_BYTES})"
            )
        name = path.name.lower()
        suffix = path.suffix.lower()
        if name in FORBIDDEN_NAMES or name.startswith(".env."):
            violations.append(f"{rel}: forbidden env file")
        if suffix in FORBIDDEN_SUFFIXES:
            violations.append(f"{rel}: forbidden file type ({suffix})")
        if suffix == ".json" and size > MAX_JSON_BYTES:
            violations.append(
                f"{rel}: JSON config too large ({size} bytes > {MAX_JSON_BYTES})"
            )
        if is_text_file(path):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                violations.append(f"{rel}: cannot read: {exc}")
                continue
            if suffix == ".ipynb":
                try:
                    notebook = json.loads(text)
                    text = "\n".join(iter_json_strings(notebook))
                except (json.JSONDecodeError, TypeError):
                    pass
            scan_text(rel, text, violations)
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit Liwet-LLM for sterile public release."
    )
    parser.add_argument(
        "--root", default=str(Path(__file__).resolve().parent.parent)
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"error: root not found: {root}", file=sys.stderr)
        return 2
    violations = audit(root)
    if violations:
        print(f"STERILE CHECK FAILED: {len(violations)} issue(s)")
        for violation in violations:
            print(f"  - {violation}")
        return 1
    print("STERILE CHECK PASSED")
    print(f"  root: {root}")
    print(f"  max file size: {MAX_FILE_BYTES // (1024 * 1024)} MiB")
    print(f"  max JSON config: {MAX_JSON_BYTES // 1024} KiB")
    print(
        "  no forbidden binaries, checkpoints, logs, local paths, tokens, "
        "or private IPs"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
