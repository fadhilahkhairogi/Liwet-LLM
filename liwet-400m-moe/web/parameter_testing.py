import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    DEFAULT_SYSTEM_PROMPT,
    build_prompt,
    model_sidebar,
    stream_generate,
)

bundle = model_sidebar()

st.title("Testing Parameter")
st.caption("Uji pengaruh preset dan parameter decoding terhadap output model.")

if "test_result" not in st.session_state:
    st.session_state.test_result = None

with st.sidebar:
    st.divider()
    st.subheader("Testing Parameter")
    if st.button("Reset hasil", use_container_width=True):
        st.session_state.test_result = None
        st.rerun()

if bundle is None:
    st.stop()

PRESETS = {
    "Chat seimbang": {
        "max_new_tokens": 256,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "do_sample": True,
        "use_seed": True,
    },
    "Kreatif": {
        "max_new_tokens": 320,
        "temperature": 1.1,
        "top_k": 80,
        "top_p": 0.95,
        "repetition_penalty": 1.05,
        "do_sample": True,
        "use_seed": True,
    },
    "Deterministik": {
        "max_new_tokens": 256,
        "temperature": 0.0,
        "top_k": 0,
        "top_p": 1.0,
        "repetition_penalty": 1.1,
        "do_sample": False,
        "use_seed": True,
    },
    "Cepat": {
        "max_new_tokens": 96,
        "temperature": 0.6,
        "top_k": 40,
        "top_p": 0.85,
        "repetition_penalty": 1.1,
        "do_sample": True,
        "use_seed": True,
    },
}
DEFAULTS = {
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
    "prompt_text": (
        "Jelaskan apa itu Mixture of Experts (MoE) "
        "dengan bahasa yang sederhana."
    ),
    "use_chat_format": True,
    "do_sample": True,
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "use_seed": True,
    "seed": 42,
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


def render_metrics(record: dict) -> None:
    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "Token output", f"{record['generated_tokens']}"
    )
    metric_columns[1].metric(
        "Token prompt", f"{record['prompt_tokens']}"
    )
    metric_columns[2].metric("Latensi", f"{record['seconds']:.2f} s")
    metric_columns[3].metric(
        "Throughput", f"{record['tokens_per_second']:.1f} tok/s"
    )
    total_tokens = record["prompt_tokens"] + record["generated_tokens"]
    st.caption(
        f"Konteks terpakai: {total_tokens} / "
        f"{bundle.model.config.max_position_embeddings} token · "
        f"{bundle.device.type.upper()} · "
        f"{str(bundle.dtype).replace('torch.', '')}"
    )
    if record["truncated"]:
        st.warning("Prompt dipotong agar muat dalam konteks model.")
    if not record["params"]["do_sample"]:
        st.caption("Mode greedy: temperature/top-k/top-p diabaikan.")
    with st.expander("Prompt yang dikirim"):
        st.code(record["prompt"], language=None)
    with st.expander("Parameter dan hasil"):
        st.json(
            {
                "parameter": record["params"],
                "system_prompt": record["system_prompt"],
                "output_tokens": record["generated_tokens"],
                "latency_seconds": round(record["seconds"], 4),
                "tokens_per_second": round(
                    record["tokens_per_second"], 2
                ),
            }
        )
    st.download_button(
        "Unduh output (.txt)",
        data=record["text"],
        file_name="moe_output.txt",
        mime="text/plain",
        use_container_width=True,
    )


left, right = st.columns([1, 1], gap="large")

request = None
with left:
    st.subheader("Input & Parameter")
    st.markdown("**Preset terbaik:**")
    preset_names = list(PRESETS)
    for start in range(0, len(preset_names), 2):
        preset_columns = st.columns(2)
        for column, name in zip(
            preset_columns, preset_names[start : start + 2]
        ):
            if column.button(
                name,
                key=f"preset_{name}",
                type="primary" if name == "Chat seimbang" else "secondary",
                use_container_width=True,
            ):
                for key, value in PRESETS[name].items():
                    st.session_state[key] = value
                st.session_state["active_preset"] = name
                st.rerun()
    if st.session_state.get("active_preset"):
        st.caption(
            f"Preset terakhir: **{st.session_state.active_preset}**"
        )
    with st.form("parameter_form"):
        system_prompt = st.text_area(
            "System prompt",
            key="system_prompt",
            height=100,
        )
        prompt_text = st.text_area(
            "Prompt pengguna",
            key="prompt_text",
            height=140,
        )
        use_chat_format = st.checkbox(
            "Format chat (System/User/Assistant)",
            key="use_chat_format",
            help="Matikan untuk mengirim prompt mentah tanpa template.",
        )
        do_sample = st.checkbox(
            "Aktifkan sampling",
            key="do_sample",
            help=(
                "Jika dimatikan, model memakai greedy decoding dan "
                "temperature/top-k/top-p diabaikan."
            ),
        )
        max_new_tokens = st.slider(
            "max_new_tokens",
            min_value=8,
            max_value=512,
            step=8,
            key="max_new_tokens",
        )
        temperature = st.slider(
            "temperature",
            min_value=0.0,
            max_value=2.0,
            step=0.05,
            key="temperature",
            help="Nilai 0 berarti greedy atau deterministik.",
        )
        top_k = st.slider(
            "top_k",
            min_value=0,
            max_value=200,
            step=1,
            key="top_k",
            help="0 berarti top-k dinonaktifkan.",
        )
        top_p = st.slider(
            "top_p",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            key="top_p",
            help="1.0 berarti top-p dinonaktifkan.",
        )
        repetition_penalty = st.slider(
            "repetition_penalty",
            min_value=1.0,
            max_value=2.0,
            step=0.01,
            key="repetition_penalty",
        )
        use_seed = st.checkbox("Gunakan seed", key="use_seed")
        seed = st.number_input(
            "seed",
            min_value=0,
            max_value=2**31 - 1,
            step=1,
            key="seed",
        )
        submitted = st.form_submit_button(
            "Generate", type="primary", use_container_width=True
        )
    if submitted:
        if not prompt_text.strip():
            st.warning("Prompt tidak boleh kosong.")
        else:
            if use_chat_format:
                prompt = build_prompt(
                    [{"role": "user", "content": prompt_text}],
                    system_prompt,
                )
            else:
                prompt = prompt_text
            request = {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "params": {
                    "max_new_tokens": int(max_new_tokens),
                    "temperature": float(temperature),
                    "top_k": int(top_k),
                    "top_p": float(top_p),
                    "repetition_penalty": float(repetition_penalty),
                    "seed": int(seed) if use_seed else None,
                    "do_sample": bool(do_sample),
                    "use_chat_format": bool(use_chat_format),
                },
            }

with right:
    st.subheader("Output & Metrik")
    if request is not None:
        params = request["params"]
        stats = {}
        try:
            full_text = st.write_stream(
                stream_generate(
                    bundle,
                    request["prompt"],
                    max_new_tokens=params["max_new_tokens"],
                    temperature=params["temperature"],
                    top_k=params["top_k"],
                    top_p=params["top_p"],
                    repetition_penalty=params["repetition_penalty"],
                    seed=params["seed"],
                    do_sample=params["do_sample"],
                    stats=stats,
                )
            )
            text = str(full_text).strip() or stats.get("text", "")
            record = {
                "text": text,
                "prompt": request["prompt"],
                "system_prompt": request["system_prompt"],
                "params": params,
                "prompt_tokens": stats.get("prompt_tokens", 0),
                "generated_tokens": stats.get("generated_tokens", 0),
                "seconds": stats.get("seconds", 0.0),
                "tokens_per_second": stats.get(
                    "tokens_per_second", 0.0
                ),
                "truncated": stats.get("truncated", False),
            }
            st.session_state.test_result = record
            if not text:
                st.markdown("_(kosong)_")
            render_metrics(record)
        except Exception as error:
            st.error(f"Gagal menjalankan model: {error}")
    elif st.session_state.test_result is not None:
        record = st.session_state.test_result
        st.markdown(record["text"] or "_(kosong)_")
        st.divider()
        render_metrics(record)
    else:
        st.caption("Siap.")
