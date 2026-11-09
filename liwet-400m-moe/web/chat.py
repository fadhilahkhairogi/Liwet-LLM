import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    DEFAULT_SYSTEM_PROMPT,
    build_prompt,
    model_sidebar,
    stream_generate,
    trim_messages_for_context,
)

CHAT_MAX_NEW_TOKENS = 256
CHAT_TEMPERATURE = 0.7
CHAT_TOP_K = 50
CHAT_TOP_P = 0.9
CHAT_REPETITION_PENALTY = 1.1

bundle = model_sidebar()

st.title("Chat")

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

with st.sidebar:
    st.divider()
    if st.button("Bersihkan chat", use_container_width=True):
        st.session_state.chat_messages = []
        st.rerun()
    st.caption(
        f"temp {CHAT_TEMPERATURE} · top-k {CHAT_TOP_K} · top-p {CHAT_TOP_P}"
    )

if bundle is None:
    st.stop()

for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Tulis pesan untuk model...")
if prompt:
    st.session_state.chat_messages.append(
        {"role": "user", "content": prompt}
    )
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            history, removed = trim_messages_for_context(
                bundle,
                st.session_state.chat_messages,
                DEFAULT_SYSTEM_PROMPT,
                CHAT_MAX_NEW_TOKENS,
            )
            if removed:
                st.caption(
                    "Riwayat lama dipotong otomatis agar muat dalam konteks."
                )
            prompt_text = build_prompt(history, DEFAULT_SYSTEM_PROMPT)
            stats = {}
            full_text = st.write_stream(
                stream_generate(
                    bundle,
                    prompt_text,
                    max_new_tokens=CHAT_MAX_NEW_TOKENS,
                    temperature=CHAT_TEMPERATURE,
                    top_k=CHAT_TOP_K,
                    top_p=CHAT_TOP_P,
                    repetition_penalty=CHAT_REPETITION_PENALTY,
                    seed=None,
                    stats=stats,
                )
            )
            full_text = str(full_text).strip()
            if not full_text:
                st.markdown("_(kosong)_")
            if stats.get("truncated"):
                st.caption("Prompt dipotong agar muat dalam konteks model.")
            if stats:
                st.caption(
                    f"{stats['generated_tokens']} token · "
                    f"{stats['seconds']:.2f} s · "
                    f"{stats['tokens_per_second']:.1f} tok/s"
                )
            st.session_state.chat_messages.append(
                {"role": "assistant", "content": full_text}
            )
        except Exception as error:
            st.error(f"Gagal menghasilkan respons: {error}")
