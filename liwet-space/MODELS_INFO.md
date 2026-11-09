# Liwet Model Family — Repository Catalog & Integration Guide

**Liwet (Lightweight Transformer)** is an Indonesian decoder-only causal language model family trained from scratch in PyTorch by Group 11, Telkom University. The family contains **11 variants across three architectures**, stored on the Hugging Face Hub as custom PyTorch SafeTensors (`bfloat16`) with `config.json`, `model.safetensors`, and `tokenizer.json`.

Use `liwet_models.py` for loading and generation. Do not use `transformers.AutoModelForCausalLM`; these checkpoints use custom `MiniTransformer`, `DenseIndonesianChatLM`, and `MoEIndonesianChatLM` implementations.

## Architecture summary

| Family | Parameters | Layers | Attention | Hidden size | Context | Variants |
|---|---:|---:|---|---:|---:|---:|
| **Liwet-66M** (`MiniTransformer`) | 66.7M total / 66.7M active | 12 | 8 query heads, multi-head attention | 512 | 1,024 | 5 |
| **Liwet-163M** (`Dense GQA`) | 163.0M total / 163.0M active | 16 | 12Q / 4KV grouped-query attention | 768 | 2,048 | 3 |
| **Liwet-400M-A188M** (`Sparse MoE GQA`) | 400.1M total / 187.8M active | 32 | 16Q/4KV grouped-query attention | — | 2,048 | 3 |

The **Liwet-400M-A188M** architecture is a sparse mixture-of-experts decoder with **32 layers (32L)**, **8 routed experts using top-2 routing**, and **no shared expert**. Each token activates roughly 187.8M of the 400.1M total parameters. All three families use RoPE and SwiGLU-style feed-forward blocks.

## Repository catalog

| # | Hugging Face repository | Family | Stage / method | Context | Prompt mode | Release |
|---:|---|---|---|---:|---|---|
| 1 | `fadhilahkhairogi/Liwet-66M-id-base-v1` | Liwet-66M | Pretraining, 1 epoch (~428M tokens) | 1,024 | `raw` | Live |
| 2 | `fadhilahkhairogi/Liwet-66M-id-base-v2` | Liwet-66M | Pretraining, 2 epochs (~852M tokens) | 1,024 | `raw` | Live |
| 3 | `fadhilahkhairogi/Liwet-66M-id-instruct-v1` | Liwet-66M | SFT instruction tuning from base-v1 | 1,024 | `instruct_plain` | Live |
| 4 | `fadhilahkhairogi/Liwet-66M-id-instruct-v2` | Liwet-66M | SFT instruction tuning from base-v2 | 1,024 | `instruct_plain` | Live |
| 5 | `fadhilahkhairogi/Liwet-66M-id-chat` | Liwet-66M | Conversational SFT, assistant-token masking, vocab 32,003 | 1,024 | `chat_tags` | Live |
| 6 | `fadhilahkhairogi/Liwet-163M-id-base` | Liwet-163M | Pretraining on 2.10B Indonesian tokens | 2,048 | `raw_bos` | Live |
| 7 | `fadhilahkhairogi/Liwet-163M-id-instruct` | Liwet-163M | SFT instruction tuning on 165.6k conversations | 2,048 | `chat_role` | Live |
| 8 | `fadhilahkhairogi/Liwet-163M-id-chat` | Liwet-163M | SimPO alignment (`beta=2.0`, `gamma=0.5`) | 2,048 | `chat_role` | Live |
| 9 | `fadhilahkhairogi/Liwet-400M-A188M-id-base` | Liwet-400M-A188M | Sparse MoE pretraining | 2,048 | `raw_bos` | Live |
| 10 | `fadhilahkhairogi/Liwet-400M-A188M-id-instruct` | Liwet-400M-A188M | Sparse MoE SFT instruction tuning | 2,048 | `chat_role` | Live |
| 11 | `fadhilahkhairogi/Liwet-400M-A188M-id-chat` | Liwet-400M-A188M | Sparse MoE SimPO alignment | 2,048 | `chat_role` | Live |

All 11 repositories across the three families are published Live on the Hugging Face Hub, and the demo verifies each repository with `check_repo_availability()` at runtime.

## Prompt formats

| Prompt mode | Format | Used by |
|---|---|---|
| `raw` | Latest user text only, no chat formatting | 66M base-v1/v2 |
| `raw_bos` | `<bos>` followed by the raw user text | 163M base, 400M base |
| `instruct_plain` | Instruction text followed by two newlines | 66M instruct-v1/v2 |
| `chat_tags` | `<\|user\|>\n...\n<\|assistant\|>\n...\n<\|end\|>` | 66M chat |
| `chat_role` | `System: ...\n\nUser: ...\n\nAssistant:` | 163M instruct/chat, 400M instruct/chat |

`build_prompt_from_messages(spec, messages, system_prompt="")` applies the correct format for single-turn or multi-turn history.

## Loader and generation API

- `LIST_REPOS` / `LIWET_CATALOG`: ordered `ModelSpec` entries containing `repo_id`, `label`, `scale`, `arch`, `stage`, `prompt_mode`, `total_params_label`, `active_params_label`, `context_len`, `recommended_params`, and `description`.
- `get_model_spec(repo_id)` returns the matching `ModelSpec`.
- `check_repo_availability(repo_id, timeout_s=3.0)` checks whether `config.json` and a SafeTensors/bin weight file exist on the Hub.
- `load_liwet_model(repo_id, device_name="auto", dtype_name="auto", token=None)` downloads files on demand, instantiates the correct architecture, caches the bundle in a bounded LRU cache, and raises a friendly `ModelUnavailableError` when a repository is not published yet.
- `resolve_device("auto")` selects CUDA when available and otherwise uses CPU. CPU execution always uses `float32`; GPU execution selects `bfloat16` or `float16`.
- `build_prompt_from_messages(spec, messages, system_prompt="")` formats single-turn and multi-turn prompts for the model's exact training format.
- `stream_generate(bundle, prompt_text, max_new_tokens=..., temperature=..., top_k=..., top_p=..., repetition_penalty=..., do_sample=..., seed=..., stats=stats)` yields accumulated clean response text with KV-cached decoding and updates `prompt_tokens`, `generated_tokens`, `seconds`, `tokens_per_second`, `truncated`, and `text`.
- `generate_text(bundle, prompt_text, ...)` is the non-streaming helper returning `(text, stats)`.

The loader keeps at most two model bundles in memory so the free CPU tier can release memory between model switches.

## Demo contract (`app.py`)

1. **Percakapan (Chat)**: multi-turn conversation across all available repositories, system prompt control, history trimming, reset controls, and per-response latency plus throughput.
2. **Arena Perbandingan (Comparison Arena)**: fixed 1x3 comparison of two or three models on the same prompt with identical decoding settings. Each answer streams into its own output card; the `Pengaturan` accordion remains below the three cards. A summary table reports model parameters, latency, generated tokens, and `tok/s`.
3. **Uji Parameter (Parameter Lab)**: decoding playground with presets (`Chat Seimbang`, `Kreatif`, `Deterministik`, `Cepat`), raw versus formatted prompt toggle, decoding sliders, optional fixed seed, formatted prompt preview, and a token/speed telemetry table.

## Runtime constraints

- The demo is CPU-first and supports the free Hugging Face Spaces CPU tier with `float32`.
- GPU runtimes are handled automatically through `resolve_device("auto")`; no app-level device changes are required.
- Arena loads models sequentially to avoid simultaneous memory spikes on CPU.
- Only public Hugging Face Hub repositories are used. No private tokens, local filesystem paths, or secrets are required.
- All 11 model repositories are live on Hugging Face Hub; any unavailable repository falls back gracefully to `Coming Soon` without crashing the app.

## Quick start

```bash
pip install -r requirements.txt
python app.py
```

## Project team

**Group 11, Telkom University**

- Fadhilah Akbar Khairogi — `103012300394`
- Faishal Fakhrillah — `103012300474`
- Deril Diaz — `103012300074`
