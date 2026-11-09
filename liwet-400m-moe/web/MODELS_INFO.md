# Liwet Model Family — 3-Scale Arena & Web Integration Guide

The main Streamlit web application lives in `web/` (`web/app.py`, `web/chat.py`, `web/arena.py`, `web/parameter_testing.py`, `web/common.py`). All **Liwet (`Lightweight Transformer`)** models across three parameter scales are available via `web/liwet_models.py` (from Hugging Face Hub or local `checkpoints/`).

## Arena Page (`web/arena.py`)

- Compares the `66M`, `163M`, and `400M-A188M` chat models side-by-side with one identical prompt.
- Reports generation speed (`latency`, `tokens/second`, output tokens) and an automatic quality score (0-100) derived from word diversity, 3-gram repetition, and response length.
- Falls back to a local `checkpoints/` 400M model when the Hugging Face repo is unavailable.

## Available Models Across All 3 Scales

| Scale / Arch | Source (`repo_id`) | Stage & Method | Context | Prompt Format | Status |
|---|---|---|---:|---|---|
| **66.7M** (`MiniTransformer`) | `fadhilahkhairogi/Liwet-66M-id-base-v1` | Pretrain (1 Epoch, ~428M tok) | 1,024 | `raw` | Live on HF Hub |
| **66.7M** (`MiniTransformer`) | `fadhilahkhairogi/Liwet-66M-id-base-v2` | Pretrain (2 Epochs, ~852M tok) | 1,024 | `raw` | Live on HF Hub |
| **66.7M** (`MiniTransformer`) | `fadhilahkhairogi/Liwet-66M-id-instruct-v1` | SFT Instruct (1-Epoch Base) | 1,024 | `instruct_plain` (`prompt\n\n`) | Live on HF Hub |
| **66.7M** (`MiniTransformer`) | `fadhilahkhairogi/Liwet-66M-id-instruct-v2` | SFT Instruct (2-Epoch Base) | 1,024 | `instruct_plain` (`prompt\n\n`) | Live on HF Hub |
| **66.7M** (`MiniTransformer`) | `fadhilahkhairogi/Liwet-66M-id-chat` | Conversational SFT (`vocab=32,003`) | 1,024 | `chat_tags` (`<\|user\|>\n...\n<\|assistant\|>\n`) | Live on HF Hub |
| **163M** (`Dense GQA`) | `fadhilahkhairogi/Liwet-163M-id-base` | Pretrain (2.10B tok) | 2,048 | `raw_bos` | Live on HF Hub |
| **163M** (`Dense GQA`) | `fadhilahkhairogi/Liwet-163M-id-instruct` | SFT Instruct (165.6k conv) | 2,048 | `chat_role` (`System/User/Assistant`) | Live on HF Hub |
| **163M** (`Dense GQA`) | `fadhilahkhairogi/Liwet-163M-id-chat` | SimPO Aligned (`beta=2.0, gamma=0.5`) | 2,048 | `chat_role` (`System/User/Assistant`) | Live on HF Hub |
| **400M-A188M** (`MoE GQA`) | `fadhilahkhairogi/Liwet-400M-A188M-id-base` | MoE Pretrain (32L, 8E Top-2) | 2,048 | `raw_bos` | Hub / Local `checkpoints/pretrain` |
| **400M-A188M** (`MoE GQA`) | `fadhilahkhairogi/Liwet-400M-A188M-id-instruct` | MoE SFT Instruct | 2,048 | `chat_role` (`System/User/Assistant`) | Hub / Local `checkpoints/sft` |
| **400M-A188M** (`MoE GQA`) | `fadhilahkhairogi/Liwet-400M-A188M-id-chat` | MoE SimPO Aligned | 2,048 | `chat_role` (`System/User/Assistant`) | Hub / Local `checkpoints/simpo` |

## Using `web/liwet_models.py`

- `LIWET_CATALOG` / `LIST_REPOS` / `get_model_spec(repo_id)`: ordered metadata for all 11 models across `66M`, `163M`, and `400M-A188M`.
- `load_liwet_model(repo_id, device_name="auto", dtype_name="auto")` -> downloads & caches model bundles (`MAX_CACHED_BUNDLES=2` LRU cache to fit 6 GB VRAM), raising `ModelUnavailableError` cleanly if a 400M HF Hub repo is not yet uploaded.
- `discover_local_checkpoints(project_root)` -> finds any local `checkpoints/{simpo,sft,pretrain}/{final,step_*}` folders (for offline/local 400M checkpoints).
- `load_liwet_local(ckpt_dir, device_name="auto", dtype_name="auto", fallback_tokenizer=..., prompt_mode=...)` -> loads local checkpoints into the exact same `LiwetBundle` API.
- `build_prompt_from_messages(spec, messages, system_prompt="", use_chat_template=True)` -> formats the exact prompt template required by each scale (`66M`, `163M`, `400M`).
- `stream_generate(bundle, prompt_text, ...)` & `generate_text(bundle, prompt_text, ...)` -> KV-cached streaming/generation yielding clean incremental replies and populating `stats` (`prompt_tokens`, `generated_tokens`, `seconds`, `tokens_per_second`, `truncated`, `text`).

## Architecture & Cleanup Notes
- Keep all interface pages (`web/app.py`, `web/chat.py`, `web/arena.py`, `web/parameter_testing.py`, `web/common.py`) cleanly organized inside `web/`.
- Legacy root interface scripts (`app.py`, `Jalankan_Streamlit_HP.bat`) have been removed. Ensure no local absolute drive paths, personal usernames, personal IP addresses, or secrets remain anywhere in the project.
