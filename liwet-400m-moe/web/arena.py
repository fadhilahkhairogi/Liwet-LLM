import re
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from liwet_models import (
    DEFAULT_SYSTEM_PROMPT,
    LIWET_CATALOG,
    build_prompt_from_messages,
    clear_model_cache,
    discover_local_checkpoints,
    generate_text,
    load_liwet_local,
    load_liwet_model,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FALLBACK_TOKENIZER = PROJECT_ROOT / "data" / "tokenizer" / "tokenizer.json"

SCALES = (
    {
        "key": "66m",
        "title": "66M",
        "arch": "mini_66m",
        "default_repo": "fadhilahkhairogi/Liwet-66M-id-chat",
    },
    {
        "key": "163m",
        "title": "163M",
        "arch": "dense_163m",
        "default_repo": "fadhilahkhairogi/Liwet-163M-id-chat",
    },
    {
        "key": "400m",
        "title": "400M-A188M",
        "arch": "moe_400m",
        "default_repo": "fadhilahkhairogi/Liwet-400M-A188M-id-chat",
    },
)

DEFAULT_ARENA_PROMPT = (
    "Jelaskan apa itu Mixture of Experts (MoE) dengan bahasa yang sederhana, "
    "lalu sebutkan satu kelebihan dan satu tantangannya."
)


def evaluate_quality(text: str) -> dict:
    cleaned = text.strip()
    words = re.findall(r"[^\W_]+", cleaned.lower(), flags=re.UNICODE)
    word_count = len(words)
    if word_count == 0:
        return {
            "score": 0.0,
            "word_count": 0,
            "unique_ratio": 0.0,
            "repetition_ratio": 0.0,
        }
    unique_ratio = len(set(words)) / word_count
    if word_count >= 3:
        trigrams = list(zip(words, words[1:], words[2:]))
        repetition_ratio = 1.0 - len(set(trigrams)) / len(trigrams)
    else:
        repetition_ratio = 0.0
    length_factor = min(word_count / 45.0, 1.0)
    if word_count > 220:
        length_factor *= 220.0 / word_count
    diversity = min(unique_ratio / 0.65, 1.0)
    repetition = max(0.0, 1.0 - repetition_ratio * 2.0)
    score = 100.0 * (
        0.45 * diversity + 0.35 * repetition + 0.20 * length_factor
    )
    return {
        "score": round(score, 1),
        "word_count": word_count,
        "unique_ratio": unique_ratio,
        "repetition_ratio": repetition_ratio,
    }


def load_arena_bundle(
    spec,
    device_name: str,
    dtype_name: str,
    hf_token,
    allow_local: bool,
):
    try:
        bundle = load_liwet_model(
            spec.repo_id,
            device_name=device_name,
            dtype_name=dtype_name,
            token=hf_token,
        )
        return bundle, None
    except Exception as error:
        if allow_local and spec.arch == "moe_400m":
            for candidate in discover_local_checkpoints(PROJECT_ROOT):
                try:
                    bundle = load_liwet_local(
                        candidate["path"],
                        device_name=device_name,
                        dtype_name=dtype_name,
                        fallback_tokenizer=FALLBACK_TOKENIZER,
                        prompt_mode=candidate["prompt_mode"],
                    )
                    note = (
                        f"HF Hub tidak tersedia ({error}). "
                        f"Memakai {candidate['label']}."
                    )
                    return bundle, note
                except Exception:
                    continue
        raise


def run_arena(
    selected: dict,
    prompt_text: str,
    system_prompt: str,
    params: dict,
    device_name: str,
    dtype_name: str,
    hf_token,
    allow_local: bool,
) -> dict:
    results = {}
    progress = st.progress(0.0, text="Menyiapkan arena...")
    total = len(SCALES)
    for index, scale in enumerate(SCALES, start=1):
        progress.progress(
            (index - 1) / total,
            text=f"Memuat model {scale['title']}...",
        )
        record = {"spec": selected[scale["key"]]}
        try:
            bundle, note = load_arena_bundle(
                selected[scale["key"]],
                device_name,
                dtype_name,
                hf_token,
                allow_local,
            )
            record["note"] = note
            prompt = build_prompt_from_messages(
                bundle.spec,
                [{"role": "user", "content": prompt_text}],
                system_prompt,
            )
            progress.progress(
                (index - 0.5) / total,
                text=f"Menjalankan {bundle.spec.label}...",
            )
            text, stats = generate_text(
                bundle,
                prompt,
                max_new_tokens=params["max_new_tokens"],
                temperature=params["temperature"],
                top_k=params["top_k"],
                top_p=params["top_p"],
                repetition_penalty=params["repetition_penalty"],
                do_sample=params["do_sample"],
                seed=params["seed"],
            )
            text = text.strip() or stats.get("text", "")
            record.update(
                {
                    "text": text,
                    "stats": stats,
                    "quality": evaluate_quality(text),
                    "total_params": bundle.total_params,
                    "active_params": bundle.active_params,
                    "device": str(bundle.device),
                    "dtype": str(bundle.dtype).replace("torch.", ""),
                    "spec": bundle.spec,
                }
            )
        except Exception as error:
            record["error"] = str(error)
        results[scale["key"]] = record
        progress.progress(
            index / total, text=f"{scale['title']} selesai"
        )
    progress.empty()
    return results


def render_result(column, scale: dict, result) -> None:
    with column:
        st.subheader(scale["title"])
        if result is None:
            st.caption("Belum ada hasil.")
            return
        spec = result["spec"]
        st.caption(f"{spec.label} · {spec.stage} · konteks {spec.context_len}")
        if result.get("error"):
            st.error(result["error"])
            return
        if result.get("note"):
            st.info(result["note"])
        st.markdown(result["text"] or "_(kosong)_")
        stats = result["stats"]
        quality = result["quality"]
        first_left, first_right = st.columns(2)
        first_left.metric("Latensi generasi", f"{stats['seconds']:.2f} s")
        first_right.metric(
            "Kecepatan", f"{stats['tokens_per_second']:.1f} tok/s"
        )
        second_left, second_right = st.columns(2)
        second_left.metric("Token output", f"{stats['generated_tokens']}")
        second_right.metric("Skor kualitas", f"{quality['score']:.0f}/100")
        st.progress(min(max(quality["score"] / 100.0, 0.0), 1.0))
        st.caption(
            f"{quality['word_count']} kata · "
            f"keragaman {quality['unique_ratio']:.0%} · "
            f"pengulangan 3-gram {quality['repetition_ratio']:.0%} · "
            f"{result['total_params'] / 1e6:.1f}M total / "
            f"{result['active_params'] / 1e6:.1f}M aktif · "
            f"{result['device'].upper()} · {result['dtype']}"
        )
        if stats.get("truncated"):
            st.warning("Prompt dipotong agar muat dalam konteks model.")


def render_summary(results: dict) -> None:
    rows = []
    for scale in SCALES:
        result = results.get(scale["key"])
        if result is None or result.get("error"):
            rows.append(
                {
                    "Skala": scale["title"],
                    "Model": result["spec"].label if result else "-",
                    "Param total": "-",
                    "Param aktif": "-",
                    "Kata": "-",
                    "Latensi (s)": "-",
                    "Kecepatan (tok/s)": "-",
                    "Kualitas (0-100)": "-",
                }
            )
            continue
        stats = result["stats"]
        quality = result["quality"]
        rows.append(
            {
                "Skala": scale["title"],
                "Model": result["spec"].label,
                "Param total": f"{result['total_params'] / 1e6:.1f}M",
                "Param aktif": f"{result['active_params'] / 1e6:.1f}M",
                "Kata": quality["word_count"],
                "Latensi (s)": round(stats["seconds"], 2),
                "Kecepatan (tok/s)": round(stats["tokens_per_second"], 1),
                "Kualitas (0-100)": quality["score"],
            }
        )
    st.dataframe(rows, hide_index=True, use_container_width=True)


with st.sidebar:
    st.divider()
    st.subheader("Arena")
    device_name = st.selectbox(
        "Device", ("auto", "cuda", "cpu"), key="arena_device"
    )
    dtype_name = st.selectbox(
        "Dtype",
        ("auto", "bfloat16", "float16", "float32"),
        key="arena_dtype",
    )
    hf_token_value = st.text_input(
        "HF token (opsional)", type="password", key="arena_hf_token"
    )
    allow_local = st.checkbox(
        "Fallback ke checkpoint 400M lokal",
        value=True,
        key="arena_local_fallback",
    )
    if st.button("Kosongkan cache model", use_container_width=True):
        clear_model_cache()
        st.session_state.pop("arena_results", None)
        st.success("Cache model dibersihkan.")
    if st.button("Bersihkan hasil arena", use_container_width=True):
        st.session_state.pop("arena_results", None)

st.title("Arena Perbandingan 3 Skala")
st.caption("Bandingkan 66M, 163M, dan 400M-A188M pada prompt dan parameter yang sama.")

st.subheader("Pilih model")
selection_columns = st.columns(3)
selected = {}
for column, scale in zip(selection_columns, SCALES):
    options = [spec for spec in LIWET_CATALOG if spec.arch == scale["arch"]]
    labels = [spec.label for spec in options]
    default_index = next(
        (
            index
            for index, spec in enumerate(options)
            if spec.repo_id == scale["default_repo"]
        ),
        0,
    )
    with column:
        st.markdown(f"**{scale['title']}**")
        choice = st.selectbox(
            "Model",
            labels,
            index=default_index,
            key=f"arena_model_{scale['key']}",
            label_visibility="collapsed",
        )
        selected[scale["key"]] = options[labels.index(choice)]
        spec = selected[scale["key"]]
        st.caption(f"{spec.scale} · konteks {spec.context_len} token")

if "arena_prompt" not in st.session_state:
    st.session_state.arena_prompt = DEFAULT_ARENA_PROMPT
if "arena_system_prompt" not in st.session_state:
    st.session_state.arena_system_prompt = DEFAULT_SYSTEM_PROMPT

with st.form("arena_form"):
    prompt_text = st.text_area(
        "Prompt pengguna", key="arena_prompt", height=110
    )
    system_prompt = st.text_area(
        "System prompt", key="arena_system_prompt", height=80
    )
    parameter_columns = st.columns(5)
    max_new_tokens = parameter_columns[0].slider(
        "Max token", 32, 512, 192, 16, key="arena_max_tokens"
    )
    temperature = parameter_columns[1].slider(
        "Temperature", 0.0, 1.5, 0.7, 0.05, key="arena_temperature"
    )
    top_k = parameter_columns[2].slider(
        "Top-k", 1, 100, 50, 1, key="arena_top_k"
    )
    top_p = parameter_columns[3].slider(
        "Top-p", 0.1, 1.0, 0.9, 0.05, key="arena_top_p"
    )
    repetition_penalty = parameter_columns[4].slider(
        "Repetition penalty",
        1.0,
        1.5,
        1.1,
        0.01,
        key="arena_repetition_penalty",
    )
    seed_columns = st.columns([1, 1])
    use_seed = seed_columns[0].checkbox(
        "Gunakan seed", value=True, key="arena_use_seed"
    )
    seed = seed_columns[1].number_input(
        "Seed",
        min_value=0,
        max_value=2**31 - 1,
        value=42,
        step=1,
        key="arena_seed",
    )
    do_sample = st.checkbox(
        "Aktifkan sampling", value=True, key="arena_do_sample"
    )
    submitted = st.form_submit_button(
        "Jalankan Arena", type="primary", use_container_width=True
    )

if submitted:
    if not prompt_text.strip():
        st.warning("Prompt tidak boleh kosong.")
    else:
        st.session_state.arena_results = run_arena(
            selected,
            prompt_text,
            system_prompt,
            {
                "max_new_tokens": int(max_new_tokens),
                "temperature": float(temperature),
                "top_k": int(top_k),
                "top_p": float(top_p),
                "repetition_penalty": float(repetition_penalty),
                "do_sample": bool(do_sample),
                "seed": int(seed) if use_seed else None,
            },
            device_name,
            dtype_name,
            hf_token_value.strip() or None,
            allow_local,
        )

results = st.session_state.get("arena_results")

st.subheader("Hasil berdampingan")
result_columns = st.columns(3)
for column, scale in zip(result_columns, SCALES):
    render_result(
        column,
        scale,
        results.get(scale["key"]) if results else None,
    )

if results:
    st.subheader("Ringkasan metrik")
    render_summary(results)
    st.caption("Skor: heuristik keragaman kata, pengulangan 3-gram, dan panjang.")
