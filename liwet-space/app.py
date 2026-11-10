from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import gradio as gr

try:
    import spaces
except ImportError:
    spaces = None


def _app_fastpath() -> bool:
    return os.environ.get("LIWET_FASTPATH", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def gpu_decorator(duration: int = 60):
    def _wrap(fn):
        if spaces is not None:
            return spaces.GPU(duration=duration)(fn)
        return fn

    return _wrap

from liwet_models import (
    DEFAULT_SYSTEM_PROMPT,
    LIWET_CATALOG,
    ModelUnavailableError,
    build_prompt_from_messages,
    check_repo_availability,
    get_model_spec,
    load_liwet_model,
    stream_generate,
)

GRADIO_MAJOR = int(gr.__version__.split(".")[0])

PARAM_PRESETS: Dict[str, Dict[str, float]] = {
    "Chat Seimbang": {
        "max_new_tokens": 192,
        "temperature": 0.65,
        "top_k": 40,
        "top_p": 0.9,
        "repetition_penalty": 1.12,
    },
    "Kreatif": {
        "max_new_tokens": 224,
        "temperature": 0.95,
        "top_k": 80,
        "top_p": 0.95,
        "repetition_penalty": 1.05,
    },
    "Deterministik": {
        "max_new_tokens": 160,
        "temperature": 0.0,
        "top_k": 0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
    },
    "Cepat": {
        "max_new_tokens": 64,
        "temperature": 0.55,
        "top_k": 20,
        "top_p": 0.85,
        "repetition_penalty": 1.08,
    },
}

_AVAIL_LOCK = threading.Lock()
_AVAIL_CACHE: Dict[str, Tuple[float, bool]] = {}
_AVAIL_TTL_SECONDS = 300.0 if _app_fastpath() else 60.0


def _is_repo_available(repo_id: str, force_refresh: bool = False) -> bool:
    if not repo_id:
        return False
    now = time.monotonic()
    with _AVAIL_LOCK:
        cached = _AVAIL_CACHE.get(repo_id)
        if not force_refresh and cached is not None and (now - cached[0]) < _AVAIL_TTL_SECONDS:
            return cached[1]
    avail = check_repo_availability(repo_id, timeout_s=2.5)
    with _AVAIL_LOCK:
        _AVAIL_CACHE[repo_id] = (time.monotonic(), avail)
    return avail


def _refresh_all_availability(force_refresh: bool = False) -> Dict[str, bool]:
    repos = [spec.repo_id for spec in LIWET_CATALOG]
    with ThreadPoolExecutor(max_workers=min(len(repos), 11)) as pool:
        results = list(
            pool.map(lambda r: _is_repo_available(r, force_refresh=force_refresh), repos)
        )
    return dict(zip(repos, results))


def _build_model_choices(force_refresh: bool = False) -> List[Tuple[str, str]]:
    avail_map = _refresh_all_availability(force_refresh=force_refresh)
    choices: List[Tuple[str, str]] = []
    for spec in LIWET_CATALOG:
        tag = "Live" if avail_map.get(spec.repo_id, False) else "Coming Soon"
        choices.append((f"[{tag}] {spec.label} · {spec.scale}", spec.repo_id))
    return choices


def _availability_badge(force_refresh: bool = False) -> str:
    avail_map = _refresh_all_availability(force_refresh=force_refresh)
    live_count = sum(1 for v in avail_map.values() if v)
    soon_count = len(avail_map) - live_count
    if soon_count > 0:
        return f"{live_count} Live · {soon_count} Coming Soon"
    return f"{live_count} Model Live"


MODEL_CHOICES: List[Tuple[str, str]] = _build_model_choices(force_refresh=False)
DEFAULT_MODEL = "fadhilahkhairogi/Liwet-400M-A188M-id-chat"
ARENA_MODEL_B = "fadhilahkhairogi/Liwet-163M-id-chat"
ARENA_MODEL_C = "fadhilahkhairogi/Liwet-66M-id-chat"
MODEL_IDS = {spec.repo_id for spec in LIWET_CATALOG}
if DEFAULT_MODEL not in MODEL_IDS:
    DEFAULT_MODEL = LIWET_CATALOG[0].repo_id
if ARENA_MODEL_B not in MODEL_IDS:
    ARENA_MODEL_B = LIWET_CATALOG[1].repo_id
if ARENA_MODEL_C not in MODEL_IDS:
    ARENA_MODEL_C = ""

if spaces is not None and _app_fastpath():
    try:
        load_liwet_model(DEFAULT_MODEL, device_name="auto")
    except Exception:
        pass


def _render_hero(force_refresh: bool = False) -> str:
    badge = _availability_badge(force_refresh=force_refresh)
    return (
        '<div class="liwet-hero">'
        '<h1>Liwet-LLM</h1>'
        f'<span class="liwet-status-pill">{badge}</span>'
        '</div>'
    )


CUSTOM_CSS = """
.gradio-container {
    max-width: 1440px !important;
    margin: 0 auto !important;
}
.liwet-hero {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 14px 22px;
    border-radius: 14px;
    color: #f7fee7;
    background:
        radial-gradient(circle at 88% 18%, rgba(190, 242, 100, 0.28), transparent 34%),
        linear-gradient(135deg, #052e16 0%, #14532d 48%, #365314 100%);
    box-shadow: 0 12px 32px rgba(5, 46, 22, 0.18);
    margin-bottom: 10px;
}
.liwet-hero h1 {
    margin: 0;
    font-size: 1.65rem;
    line-height: 1.1;
    letter-spacing: -0.03em;
    color: #f7fee7;
}
.liwet-status-pill {
    border: 1px solid rgba(190, 242, 100, 0.4);
    background: rgba(5, 46, 22, 0.42);
    border-radius: 999px;
    padding: 5px 12px;
    font-size: 0.82rem;
    font-weight: 600;
    color: #ecfccb;
    white-space: nowrap;
}
#arena-models-row,
#arena-outputs-row,
.arena-row-3 {
    display: grid !important;
    grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
    gap: 14px !important;
    align-items: stretch !important;
    width: 100% !important;
}
#arena-models-row > *,
#arena-outputs-row > *,
.arena-row-3 > * {
    min-width: 0 !important;
    width: 100% !important;
    max-width: 100% !important;
    flex: unset !important;
}
@media (max-width: 768px) {
    #arena-models-row,
    #arena-outputs-row,
    .arena-row-3 {
        grid-template-columns: 1fr !important;
    }
}
"""

THEME = gr.themes.Soft(primary_hue="green", neutral_hue="slate")


def _normalize_messages(history: Optional[list]) -> List[dict]:
    messages: List[dict] = []
    for item in history or []:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant", "system"}:
                text = "" if content is None else str(content)
                if text.strip():
                    messages.append({"role": role, "content": text})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            user, assistant = item
            if str(user or "").strip():
                messages.append({"role": "user", "content": str(user)})
            if str(assistant or "").strip():
                messages.append({"role": "assistant", "content": str(assistant)})
    return messages


def _model_badge(repo_id: str, include_rec: bool = False) -> str:
    if not repo_id:
        return ""
    spec = get_model_spec(repo_id)
    tag = "Live" if _is_repo_available(repo_id) else "Coming Soon"
    base = (
        f"`{tag}` · {spec.scale} · {spec.stage} · "
        f"{spec.total_params_label}/{spec.active_params_label} · "
        f"ctx {spec.context_len:,} · "
        f"[HF](https://huggingface.co/{spec.repo_id})"
    )
    if not include_rec:
        return base
    rec = spec.recommended_params
    return (
        f"{base}  \n"
        f"`{spec.prompt_mode}` · `max={rec.get('max_new_tokens')}` · "
        f"`temp={rec.get('temperature')}` · `top_k={rec.get('top_k')}` · "
        f"`top_p={rec.get('top_p')}` · `rep={rec.get('repetition_penalty')}`"
    )


def _model_card(repo_id: str) -> str:
    return _model_badge(repo_id, include_rec=False)


def _lab_card(repo_id: str) -> str:
    return _model_badge(repo_id, include_rec=True)


def _bundle_status(bundle) -> str:
    dtype = str(bundle.dtype).replace("torch.", "")
    return f"{bundle.device.type.upper()} `{dtype}`"


def _error_text(err: Exception) -> str:
    if isinstance(err, (ModelUnavailableError, ValueError)):
        return str(err)
    return (
        f"Terjadi kesalahan saat memuat atau menjalankan model "
        f"(`{type(err).__name__}`). Coba lagi atau pilih model lain."
    )


def _resolve_seed(use_seed: bool, seed_value) -> Optional[int]:
    if not use_seed:
        return None
    try:
        return int(seed_value)
    except (TypeError, ValueError):
        return None


def _fit_prompt(
    bundle,
    messages: List[dict],
    system_prompt: str,
    max_new_tokens: int,
) -> Tuple[str, bool, int]:
    budget = max(32, bundle.context_len - int(max_new_tokens) - 8)
    working = [
        dict(message)
        for message in messages
        if str(message.get("content", "")).strip()
    ]
    trimmed = False
    while True:
        prompt = build_prompt_from_messages(bundle.spec, working, system_prompt)
        try:
            token_count = len(
                bundle.tokenizer.encode(prompt, add_special_tokens=False).ids
            )
        except Exception:
            token_count = len(prompt) // 4
        if token_count <= budget or len(working) <= 1:
            return prompt, trimmed, token_count
        first_user = next(
            (
                index
                for index, message in enumerate(working)
                if message["role"] == "user"
            ),
            None,
        )
        if first_user is None:
            return prompt, trimmed, token_count
        del working[first_user]
        if first_user < len(working) and working[first_user]["role"] == "assistant":
            del working[first_user]
        trimmed = True


def _stats_md(stats: dict, extra: str = "") -> str:
    if not stats:
        return ""
    parts = [
        f"**{stats.get('seconds', 0.0):.2f} dtk**",
        f"**{stats.get('tokens_per_second', 0.0):.1f} tok/s**",
        f"{stats.get('generated_tokens', 0)}/{stats.get('prompt_tokens', 0)} tok",
    ]
    if stats.get("truncated"):
        parts.append("konteks dipotong")
    if extra:
        parts.append(extra)
    return " · ".join(parts)


def _lab_stats_table(stats: dict, bundle, extra: str = "") -> str:
    if not stats:
        return ""
    dtype = str(bundle.dtype).replace("torch.", "")
    note = []
    if stats.get("truncated"):
        note.append("potong konteks")
    if extra:
        note.append(extra)
    catatan = ", ".join(note) if note else "—"
    return "\n".join(
        [
            "| Prompt tok | Output tok | Latensi | Kecepatan | Device | Catatan |",
            "|---:|---:|---:|---:|---|---|",
            f"| {stats.get('prompt_tokens', 0)} | {stats.get('generated_tokens', 0)} | "
            f"{stats.get('seconds', 0.0):.2f} dtk | {stats.get('tokens_per_second', 0.0):.1f} tok/s | "
            f"{bundle.device.type.upper()} `{dtype}` | {catatan} |",
        ]
    )


def _apply_preset(name: str):
    preset = PARAM_PRESETS.get(name, PARAM_PRESETS["Chat Seimbang"])
    return (
        preset["max_new_tokens"],
        preset["temperature"],
        preset["top_k"],
        preset["top_p"],
        preset["repetition_penalty"],
    )


def _reset_chat_params():
    return (
        DEFAULT_SYSTEM_PROMPT,
        192,
        0.65,
        40,
        0.9,
        1.12,
        False,
        42,
    )


def _reset_arena_params():
    return (
        DEFAULT_SYSTEM_PROMPT,
        192,
        0.65,
        40,
        0.9,
        1.12,
        True,
        42,
    )


def _reset_lab_params():
    return (
        True,
        "Chat Seimbang",
        DEFAULT_SYSTEM_PROMPT,
        192,
        0.65,
        40,
        0.9,
        1.12,
        False,
        42,
    )


@gpu_decorator(duration=20 if _app_fastpath() else 60)
def chat_send(
    message,
    history,
    repo_id,
    system_prompt,
    max_new_tokens,
    temperature,
    top_k,
    top_p,
    repetition_penalty,
    use_seed,
    seed_value,
):
    message = str(message or "").strip()
    history = _normalize_messages(history)
    if not message:
        yield history, "", "", "Tulis pesan."
        return
    if not repo_id:
        yield history, message, "", "Pilih model."
        return

    spec = get_model_spec(repo_id)
    fastpath = _app_fastpath()
    if spec.arch == "moe_400m" and not _is_repo_available(repo_id, force_refresh=not fastpath):
        display = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "Coming Soon — bobot 400M belum diunggah."},
        ]
        yield display, "", "", "Coming Soon."
        return
    messages = history + [{"role": "user", "content": message}]
    display = messages + [{"role": "assistant", "content": ""}]
    yield display, "", "", "Memuat model..."

    try:
        bundle = load_liwet_model(repo_id)
    except Exception as err:
        display[-1]["content"] = _error_text(err)
        yield display, "", "", f"Gagal ({type(err).__name__})."
        return

    prompt_text, trimmed, _ = _fit_prompt(
        bundle, messages, system_prompt, max_new_tokens
    )
    seed = _resolve_seed(use_seed, seed_value)
    stats: dict = {}
    final_text = ""

    try:
        ticks = 0
        last_yield_t = time.perf_counter()
        for partial in stream_generate(
            bundle,
            prompt_text,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            repetition_penalty=float(repetition_penalty),
            seed=seed,
            stats=stats,
        ):
            final_text = partial
            ticks += 1
            now = time.perf_counter()
            if not fastpath or ticks >= 8 or (now - last_yield_t) >= 0.05:
                display[-1]["content"] = partial
                yield display, "", "", "Menghasilkan..."
                ticks = 0
                last_yield_t = time.perf_counter()

        display[-1]["content"] = final_text if final_text.strip() else "(Kosong)"
        extra = "riwayat dipangkas" if trimmed else ""
        yield (
            display,
            "",
            _stats_md(stats, extra),
            _bundle_status(bundle),
        )
    except Exception as err:
        display[-1]["content"] = _error_text(err)
        yield display, "", "", f"Gagal ({type(err).__name__})."


def clear_chat():
    return [], "", "", ""


@gpu_decorator(duration=45 if _app_fastpath() else 90)
def arena_run(
    prompt,
    system_prompt,
    model_a,
    model_b,
    model_c,
    max_new_tokens,
    temperature,
    top_k,
    top_p,
    repetition_penalty,
    use_seed,
    seed_value,
):
    prompt = str(prompt or "").strip()
    if not prompt:
        yield "Tulis prompt perbandingan terlebih dahulu.", "", "", "", ""
        return

    slots = [model_a, model_b, model_c]
    selected = [repo for repo in slots if repo]
    if len(selected) < 2:
        yield "Pilih minimal dua model yang berbeda.", "", "", "", ""
        return
    if len(set(selected)) != len(selected):
        yield "Setiap model dalam arena harus berbeda.", "", "", "", ""
        return

    specs = {repo: get_model_spec(repo) for repo in selected}
    texts = {repo: "" for repo in selected}
    rows: Dict[str, dict] = {
        repo: {
            "status": "Menunggu",
            "params": (
                f"{specs[repo].total_params_label} / "
                f"{specs[repo].active_params_label}"
            ),
        }
        for repo in selected
    }
    seed = _resolve_seed(use_seed, seed_value)

    def snapshot(status: str):
        outputs = [texts.get(repo, "") if repo else "" for repo in slots]
        return status, outputs[0], outputs[1], outputs[2], _arena_metrics(slots, rows)

    yield snapshot("Menyiapkan arena perbandingan...")
    started_all = time.perf_counter()
    fastpath = _app_fastpath()

    for repo in selected:
        spec = specs[repo]
        if spec.arch == "moe_400m" and not _is_repo_available(repo, force_refresh=not fastpath):
            rows[repo]["status"] = "Coming Soon"
            texts[repo] = (
                f"Coming Soon — bobot SafeTensors `{spec.repo_id}` belum diunggah ke "
                "Hugging Face Hub."
            )
            yield snapshot(f"**{spec.label}** berstatus **Coming Soon**.")
            continue
        rows[repo]["status"] = "Memuat model..."
        yield snapshot(f"Mengunduh dan memuat **{spec.label}**...")

        try:
            bundle = load_liwet_model(repo)
        except Exception as err:
            rows[repo]["status"] = f"Belum tersedia ({type(err).__name__})"
            texts[repo] = f"Model tidak dapat dimuat: {_error_text(err)}"
            yield snapshot(f"**{spec.label}** belum dapat digunakan.")
            continue

        prompt_text, trimmed, _ = _fit_prompt(
            bundle, [{"role": "user", "content": prompt}], system_prompt, max_new_tokens
        )
        stats: dict = {}
        final_text = ""
        rows[repo]["status"] = "Menghasilkan..."
        yield snapshot(f"Menghasilkan jawaban dengan **{spec.label}**...")

        try:
            ticks = 0
            last_yield_t = time.perf_counter()
            for partial in stream_generate(
                bundle,
                prompt_text,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_k=int(top_k),
                top_p=float(top_p),
                repetition_penalty=float(repetition_penalty),
                seed=seed,
                stats=stats,
            ):
                final_text = partial
                ticks += 1
                now = time.perf_counter()
                if not fastpath or ticks >= 8 or (now - last_yield_t) >= 0.05:
                    texts[repo] = partial
                    yield snapshot(f"Menghasilkan jawaban dengan **{spec.label}**...")
                    ticks = 0
                    last_yield_t = time.perf_counter()

            texts[repo] = final_text if final_text.strip() else "(Model tidak menghasilkan token.)"
            rows[repo].update(
                {
                    "status": "Selesai"
                    + (" · prompt dipotong" if stats.get("truncated") else "")
                    + (" · riwayat dipangkas" if trimmed else ""),
                    "params": (
                        f"{bundle.total_params / 1_000_000:.1f}M / "
                        f"{bundle.active_params / 1_000_000:.1f}M"
                    ),
                    "seconds": stats.get("seconds", 0.0),
                    "tokens": stats.get("generated_tokens", 0),
                    "tps": stats.get("tokens_per_second", 0.0),
                }
            )
            yield snapshot(f"**{spec.label}** selesai.")
        except Exception as err:
            rows[repo]["status"] = f"Gagal ({type(err).__name__})"
            texts[repo] = f"Generasi gagal: {_error_text(err)}"
            yield snapshot(f"**{spec.label}** gagal menghasilkan teks.")

    total_seconds = time.perf_counter() - started_all
    yield snapshot(f"Selesai · {total_seconds:.1f} dtk")


def _arena_metrics(slots: List[str], rows: Dict[str, dict]) -> str:
    if not any(slots):
        return ""
    lines = [
        "### Ringkasan perbandingan",
        "",
        "| Model | Parameter total / aktif | Latensi | Token baru | tok/s | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for repo in slots:
        if not repo:
            continue
        spec = get_model_spec(repo)
        row = rows.get(repo, {})
        latency = (
            f"{row['seconds']:.2f} detik" if "seconds" in row else "—"
        )
        tokens = str(row.get("tokens", "—"))
        tps = f"{row['tps']:.1f}" if "tps" in row else "—"
        status = str(row.get("status", "—"))
        lines.append(
            f"| {spec.label} | {row.get('params', '—')} | {latency} | "
            f"{tokens} | {tps} | {status} |"
        )
    return "\n".join(lines)


def lab_preview(prompt, system_prompt, repo_id, use_format):
    prompt = str(prompt or "").strip()
    if not prompt:
        return "Prompt kosong."
    if not repo_id:
        return "Pilih model."
    spec = get_model_spec(repo_id)
    formatted = build_prompt_from_messages(
        spec,
        [{"role": "user", "content": prompt}],
        system_prompt,
        use_chat_template=bool(use_format),
    )
    mode = "raw" if (not use_format or spec.prompt_mode in {"raw", "raw_bos"}) else spec.prompt_mode
    return f"Format: `{mode}`\n\n{formatted}"


@gpu_decorator(duration=20 if _app_fastpath() else 60)
def lab_generate(
    prompt,
    system_prompt,
    repo_id,
    use_format,
    max_new_tokens,
    temperature,
    top_k,
    top_p,
    repetition_penalty,
    use_seed,
    seed_value,
):
    prompt = str(prompt or "").strip()
    if not prompt:
        yield "", "", "Prompt kosong.", ""
        return
    if not repo_id:
        yield "", "", "Pilih model.", ""
        return

    spec = get_model_spec(repo_id)
    preview = lab_preview(prompt, system_prompt, repo_id, use_format)
    fastpath = _app_fastpath()
    if spec.arch == "moe_400m" and not _is_repo_available(repo_id, force_refresh=not fastpath):
        yield "", "", "Coming Soon — bobot 400M belum diunggah.", preview
        return
    yield "", "", f"Memuat {spec.label}...", preview

    try:
        bundle = load_liwet_model(repo_id)
    except Exception as err:
        yield "", "", f"Gagal ({type(err).__name__}).", preview
        return

    prompt_text, trimmed, _ = _fit_prompt(
        bundle,
        [{"role": "user", "content": prompt}],
        system_prompt,
        max_new_tokens,
    )
    if not use_format or spec.prompt_mode in {"raw", "raw_bos"}:
        prompt_text = prompt

    seed = _resolve_seed(use_seed, seed_value)
    stats: dict = {}
    final_text = ""
    started = time.perf_counter()

    try:
        ticks = 0
        last_yield_t = time.perf_counter()
        for partial in stream_generate(
            bundle,
            prompt_text,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            repetition_penalty=float(repetition_penalty),
            seed=seed,
            stats=stats,
        ):
            final_text = partial
            ticks += 1
            now = time.perf_counter()
            if not fastpath or ticks >= 8 or (now - last_yield_t) >= 0.05:
                elapsed = now - started
                yield (
                    final_text,
                    f"Berjalan · {elapsed:.1f} dtk",
                    "Menghasilkan...",
                    preview,
                )
                ticks = 0
                last_yield_t = time.perf_counter()

        if not final_text.strip():
            final_text = "(Kosong)"
        extra = "riwayat dipangkas" if trimmed else ""
        yield (
            final_text,
            _lab_stats_table(stats, bundle, extra),
            f"Selesai · {_bundle_status(bundle)}",
            preview,
        )
    except Exception as err:
        yield final_text, "", f"Gagal ({type(err).__name__}).", preview


BLOCKS_KWARGS = {"title": "Liwet-LLM"}
LAUNCH_KWARGS = {}
if GRADIO_MAJOR >= 6:
    LAUNCH_KWARGS.update(theme=THEME, css=CUSTOM_CSS)
else:
    BLOCKS_KWARGS.update(theme=THEME, css=CUSTOM_CSS)

CHATBOT_KWARGS = {}
if GRADIO_MAJOR < 6:
    CHATBOT_KWARGS["type"] = "messages"

with gr.Blocks(**BLOCKS_KWARGS) as demo:
    hero_html = gr.HTML(_render_hero(force_refresh=False))

    with gr.Tab("Percakapan"):
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=540,
                    **CHATBOT_KWARGS,
                )
                with gr.Row():
                    chat_message = gr.Textbox(
                        show_label=False,
                        placeholder="Tulis pesan...",
                        lines=2,
                        max_lines=6,
                        scale=5,
                    )
                    chat_send_button = gr.Button(
                        "Kirim", variant="primary", scale=1
                    )
                with gr.Row():
                    chat_clear_button = gr.Button("Bersihkan")
                chat_stats = gr.Markdown()
                chat_status = gr.Markdown()
            with gr.Column(scale=2):
                chat_model = gr.Dropdown(
                    choices=MODEL_CHOICES,
                    value=DEFAULT_MODEL,
                    label="Model",
                )
                chat_card = gr.Markdown(_model_card(DEFAULT_MODEL))
                with gr.Accordion("Pengaturan", open=False):
                    chat_system = gr.Textbox(
                        label="System prompt",
                        value=DEFAULT_SYSTEM_PROMPT,
                        lines=3,
                    )
                    chat_max_new = gr.Slider(
                        minimum=8,
                        maximum=512,
                        step=8,
                        value=192,
                        label="Max new tokens",
                    )
                    chat_temperature = gr.Slider(
                        minimum=0.0,
                        maximum=1.5,
                        step=0.05,
                        value=0.65,
                        label="Temperature",
                    )
                    with gr.Row():
                        chat_top_k = gr.Slider(
                            minimum=0,
                            maximum=100,
                            step=1,
                            value=40,
                            label="Top-k",
                        )
                        chat_top_p = gr.Slider(
                            minimum=0.05,
                            maximum=1.0,
                            step=0.05,
                            value=0.9,
                            label="Top-p",
                        )
                    chat_rep_penalty = gr.Slider(
                        minimum=1.0,
                        maximum=1.5,
                        step=0.01,
                        value=1.12,
                        label="Repetition penalty",
                    )
                    with gr.Row():
                        chat_use_seed = gr.Checkbox(
                            label="Seed tetap", value=False, scale=1
                        )
                        chat_seed = gr.Number(
                            label="Nilai seed", value=42, precision=0, scale=1
                        )
                    chat_reset_button = gr.Button("Reset parameter")

        chat_model.change(_model_card, chat_model, chat_card)
        chat_reset_button.click(
            _reset_chat_params,
            outputs=[
                chat_system,
                chat_max_new,
                chat_temperature,
                chat_top_k,
                chat_top_p,
                chat_rep_penalty,
                chat_use_seed,
                chat_seed,
            ],
            queue=False,
        )
        chat_send_button.click(
            chat_send,
            inputs=[
                chat_message,
                chatbot,
                chat_model,
                chat_system,
                chat_max_new,
                chat_temperature,
                chat_top_k,
                chat_top_p,
                chat_rep_penalty,
                chat_use_seed,
                chat_seed,
            ],
            outputs=[chatbot, chat_message, chat_stats, chat_status],
        )
        chat_message.submit(
            chat_send,
            inputs=[
                chat_message,
                chatbot,
                chat_model,
                chat_system,
                chat_max_new,
                chat_temperature,
                chat_top_k,
                chat_top_p,
                chat_rep_penalty,
                chat_use_seed,
                chat_seed,
            ],
            outputs=[chatbot, chat_message, chat_stats, chat_status],
        )
        chat_clear_button.click(
            clear_chat,
            outputs=[chatbot, chat_message, chat_stats, chat_status],
        )

    with gr.Tab("Arena Perbandingan"):
        gr.Markdown("Bandingkan 2–3 model Liwet pada prompt dan parameter yang sama.")
        arena_prompt = gr.Textbox(
            label="Prompt",
            placeholder="Tulis prompt perbandingan...",
            lines=3,
            max_lines=10,
        )
        with gr.Row(elem_id="arena-models-row", elem_classes=["arena-row-3"]):
            with gr.Column(scale=1, min_width=0, elem_classes=["arena-col"]):
                arena_model_a = gr.Dropdown(
                    choices=MODEL_CHOICES,
                    value=DEFAULT_MODEL,
                    label="Model A",
                    min_width=0,
                )
            with gr.Column(scale=1, min_width=0, elem_classes=["arena-col"]):
                arena_model_b = gr.Dropdown(
                    choices=MODEL_CHOICES,
                    value=ARENA_MODEL_B,
                    label="Model B",
                    min_width=0,
                )
            with gr.Column(scale=1, min_width=0, elem_classes=["arena-col"]):
                arena_model_c = gr.Dropdown(
                    choices=[("Tidak dipakai", "")] + MODEL_CHOICES,
                    value=ARENA_MODEL_C,
                    label="Model C (opsional)",
                    min_width=0,
                )
        arena_run_button = gr.Button("Jalankan arena", variant="primary")
        arena_status = gr.Markdown()
        with gr.Row(
            elem_id="arena-outputs-row",
            elem_classes=["arena-row-3"],
            equal_height=True,
        ):
            with gr.Column(scale=1, min_width=0, elem_classes=["arena-col"]):
                gr.Markdown("### Model A")
                arena_output_a = gr.Textbox(
                    label="Keluaran A",
                    lines=12,
                    max_lines=24,
                    interactive=False,
                )
            with gr.Column(scale=1, min_width=0, elem_classes=["arena-col"]):
                gr.Markdown("### Model B")
                arena_output_b = gr.Textbox(
                    label="Keluaran B",
                    lines=12,
                    max_lines=24,
                    interactive=False,
                )
            with gr.Column(scale=1, min_width=0, elem_classes=["arena-col"]):
                gr.Markdown("### Model C")
                arena_output_c = gr.Textbox(
                    label="Keluaran C",
                    lines=12,
                    max_lines=24,
                    interactive=False,
                )
        with gr.Accordion("Pengaturan", open=False):
            arena_system = gr.Textbox(
                label="System prompt",
                value=DEFAULT_SYSTEM_PROMPT,
                lines=3,
            )
            arena_max_new = gr.Slider(
                minimum=8,
                maximum=512,
                step=8,
                value=192,
                label="Max new tokens",
            )
            arena_temperature = gr.Slider(
                minimum=0.0,
                maximum=1.5,
                step=0.05,
                value=0.65,
                label="Temperature",
            )
            with gr.Row():
                arena_top_k = gr.Slider(
                    minimum=0,
                    maximum=100,
                    step=1,
                    value=40,
                    label="Top-k",
                )
                arena_top_p = gr.Slider(
                    minimum=0.05,
                    maximum=1.0,
                    step=0.05,
                    value=0.9,
                    label="Top-p",
                )
            arena_rep_penalty = gr.Slider(
                minimum=1.0,
                maximum=1.5,
                step=0.01,
                value=1.12,
                label="Repetition penalty",
            )
            with gr.Row():
                arena_use_seed = gr.Checkbox(label="Seed tetap", value=True)
                arena_seed = gr.Number(
                    label="Nilai seed", value=42, precision=0
                )
            arena_reset_button = gr.Button("Reset parameter")
        arena_metrics_md = gr.Markdown()

        arena_reset_button.click(
            _reset_arena_params,
            outputs=[
                arena_system,
                arena_max_new,
                arena_temperature,
                arena_top_k,
                arena_top_p,
                arena_rep_penalty,
                arena_use_seed,
                arena_seed,
            ],
            queue=False,
        )

        arena_run_button.click(
            arena_run,
            inputs=[
                arena_prompt,
                arena_system,
                arena_model_a,
                arena_model_b,
                arena_model_c,
                arena_max_new,
                arena_temperature,
                arena_top_k,
                arena_top_p,
                arena_rep_penalty,
                arena_use_seed,
                arena_seed,
            ],
            outputs=[
                arena_status,
                arena_output_a,
                arena_output_b,
                arena_output_c,
                arena_metrics_md,
            ],
        )

    with gr.Tab("Uji Parameter"):
        gr.Markdown("Uji preset, parameter decoding, dan pratinjau format prompt.")
        with gr.Row():
            with gr.Column(scale=3):
                lab_prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Tulis prompt...",
                    lines=5,
                    max_lines=12,
                )
                with gr.Row():
                    lab_preview_button = gr.Button("Pratinjau prompt")
                    lab_run_button = gr.Button(
                        "Generate", variant="primary"
                    )
                lab_preview_box = gr.Textbox(
                    label="Prompt terformat",
                    lines=10,
                    max_lines=22,
                    interactive=False,
                )
                lab_output = gr.Textbox(
                    label="Keluaran",
                    lines=12,
                    max_lines=26,
                    interactive=False,
                )
                lab_stats = gr.Markdown()
                lab_status = gr.Markdown()
            with gr.Column(scale=2):
                lab_model = gr.Dropdown(
                    choices=MODEL_CHOICES,
                    value=DEFAULT_MODEL,
                    label="Model",
                )
                lab_card = gr.Markdown(_lab_card(DEFAULT_MODEL))
                lab_format = gr.Checkbox(
                    label="Terapkan format prompt",
                    value=True,
                )
                lab_preset = gr.Dropdown(
                    choices=list(PARAM_PRESETS.keys()),
                    value="Chat Seimbang",
                    label="Preset decoding",
                )
                with gr.Accordion("Parameter decoding", open=True):
                    lab_system = gr.Textbox(
                        label="System prompt",
                        value=DEFAULT_SYSTEM_PROMPT,
                        lines=3,
                    )
                    lab_max_new = gr.Slider(
                        minimum=8,
                        maximum=512,
                        step=8,
                        value=192,
                        label="Max new tokens",
                    )
                    lab_temperature = gr.Slider(
                        minimum=0.0,
                        maximum=1.5,
                        step=0.05,
                        value=0.65,
                        label="Temperature",
                    )
                    lab_top_k = gr.Slider(
                        minimum=0,
                        maximum=100,
                        step=1,
                        value=40,
                        label="Top-k",
                    )
                    lab_top_p = gr.Slider(
                        minimum=0.05,
                        maximum=1.0,
                        step=0.05,
                        value=0.9,
                        label="Top-p",
                    )
                    lab_rep_penalty = gr.Slider(
                        minimum=1.0,
                        maximum=1.5,
                        step=0.01,
                        value=1.12,
                        label="Repetition penalty",
                    )
                    with gr.Row():
                        lab_use_seed = gr.Checkbox(
                            label="Seed tetap", value=False
                        )
                        lab_seed = gr.Number(
                            label="Nilai seed", value=42, precision=0
                        )
                    lab_reset_button = gr.Button("Reset parameter")

        lab_model.change(_lab_card, lab_model, lab_card)
        lab_reset_button.click(
            _reset_lab_params,
            outputs=[
                lab_format,
                lab_preset,
                lab_system,
                lab_max_new,
                lab_temperature,
                lab_top_k,
                lab_top_p,
                lab_rep_penalty,
                lab_use_seed,
                lab_seed,
            ],
            queue=False,
        )
        lab_preset.change(
            _apply_preset,
            inputs=lab_preset,
            outputs=[
                lab_max_new,
                lab_temperature,
                lab_top_k,
                lab_top_p,
                lab_rep_penalty,
            ],
        )
        lab_preview_button.click(
            lab_preview,
            inputs=[lab_prompt, lab_system, lab_model, lab_format],
            outputs=lab_preview_box,
        )
        lab_format.change(
            lab_preview,
            inputs=[lab_prompt, lab_system, lab_model, lab_format],
            outputs=lab_preview_box,
        )
        lab_run_button.click(
            lab_generate,
            inputs=[
                lab_prompt,
                lab_system,
                lab_model,
                lab_format,
                lab_max_new,
                lab_temperature,
                lab_top_k,
                lab_top_p,
                lab_rep_penalty,
                lab_use_seed,
                lab_seed,
            ],
            outputs=[lab_output, lab_stats, lab_status, lab_preview_box],
        )


    def _refresh_catalog_ui():
        choices = _build_model_choices(force_refresh=False)
        return (
            _render_hero(force_refresh=False),
            gr.update(choices=choices),
            gr.update(choices=choices),
            gr.update(choices=choices),
            gr.update(choices=[("Tidak dipakai", "")] + choices),
            gr.update(choices=choices),
        )

    demo.load(
        _refresh_catalog_ui,
        outputs=[
            hero_html,
            chat_model,
            arena_model_a,
            arena_model_b,
            arena_model_c,
            lab_model,
        ],
        queue=False,
        show_progress="hidden",
    )

demo.queue(default_concurrency_limit=1)

if __name__ == "__main__":
    demo.launch(**LAUNCH_KWARGS)
