---
title: Liwet-LLM Indonesian Suite
emoji: 🍚
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 5.23.3
app_file: app.py
pinned: false
short_description: 11-model interactive suite for Liwet Indonesian LLMs
---

# Liwet-LLM — Indonesian Decoder-Only Transformer Family

Public demo workspace for **Liwet (Lightweight Transformer)**, a family of Indonesian decoder-only causal language models trained from scratch in PyTorch by **Group 11, Telkom University**. The app downloads weights and tokenizers directly from the Hugging Face Hub and runs them through custom PyTorch + SafeTensors loaders with KV-cached streaming generation.

## English

### Overview

Liwet spans three model families and **11 published or announced variants**:

- **Liwet-66M (`MiniTransformer`)**: 66.7M parameters, 12 layers, 8 heads, 512 hidden dimensions, RoPE + SwiGLU, 1,024-token context. Variants: `base-v1`, `base-v2`, `instruct-v1`, `instruct-v2`, `chat`.
- **Liwet-163M (`Dense GQA`)**: 163.0M parameters, 16 layers, 12 query / 4 KV grouped-query attention, 768 hidden dimensions, RoPE + SwiGLU, 2,048-token context. Variants: `base`, `instruct` (SFT), `chat` (SimPO).
- **Liwet-400M-A188M (`Sparse MoE GQA`)**: 400.1M total / 187.8M active parameters, 32 layers (32L), 16Q/4KV grouped-query attention, 8 routed experts with top-2 routing, and **no shared expert**, 2,048-token context. Variants: `base`, `instruct`, `chat`.

### Model variants

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

Availability is checked against the Hugging Face Hub by the app. A repository is enabled automatically when its `config.json`, `model.safetensors`, and `tokenizer.json` are available.

### Prompt formats

- `raw`: the latest user text is sent without chat formatting.
- `raw_bos`: a beginning-of-sequence token is prepended before the raw prompt.
- `instruct_plain`: the instruction is followed by two newlines.
- `chat_tags`: turns use `<|user|>`, `<|assistant|>`, and `<|end|>` markers.
- `chat_role`: turns use `System:`, `User:`, and `Assistant:` role labels.

### Demo tabs

1. **Percakapan (Chat)**: multi-turn conversation across all repositories, system prompt control, history trimming, reset controls, and per-response latency plus `tok/s` telemetry.
2. **Arena Perbandingan (Comparison Arena)**: compare two or three models on the same prompt with identical decoding settings. The arena keeps a fixed 1x3 layout, streams each answer into its own card, and places the `Pengaturan` accordion below the three output cards. A summary table reports parameters, latency, generated tokens, and throughput.
3. **Uji Parameter (Parameter Lab)**: decoding playground with `Chat Seimbang`, `Kreatif`, `Deterministik`, and `Cepat` presets; raw versus formatted prompt toggle; `max_new_tokens`, `temperature`, `top_k`, `top_p`, and `repetition_penalty` sliders; optional fixed seed; formatted prompt preview; and a telemetry table.

### Runtime and compatibility

- The Space is CPU-first and runs on the free CPU tier without CUDA.
- `load_liwet_model()` calls `resolve_device("auto")`: it uses CUDA when a GPU runtime is available and otherwise falls back to CPU with `float32`.
- Arena runs models sequentially to keep memory bounded on CPU. The loader keeps a small LRU cache of loaded model bundles.
- Weights are downloaded from the public Hugging Face Hub on first use; no private access token is required.
- All 11 model variants across 66M, 163M, and 400M-A188M are live on the Hugging Face Hub, while missing/unpublished repos still fall back gracefully to `Coming Soon`.

### Quick start

```bash
pip install -r requirements.txt
python app.py
```

### Team

**Group 11, Telkom University**

- Fadhilah Akbar Khairogi — `103012300394`
- Faishal Fakhrillah — `103012300474`
- Deril Diaz — `103012300074`

### Links

- Hugging Face organization: [fadhilahkhairogi](https://huggingface.co/fadhilahkhairogi)
- Integration reference: [MODELS_INFO.md](MODELS_INFO.md)
- Entry point: [app.py](app.py)

## Bahasa Indonesia

### Ringkasan

Liwet mencakup tiga keluarga model dan **11 varian yang sudah tersedia atau diumumkan**:

- **Liwet-66M (`MiniTransformer`)**: 66,7M parameter, 12 layer, 8 head, 512 dimensi tersembunyi, RoPE + SwiGLU, konteks 1.024 token. Varian: `base-v1`, `base-v2`, `instruct-v1`, `instruct-v2`, `chat`.
- **Liwet-163M (`Dense GQA`)**: 163,0M parameter, 16 layer, 12 query / 4 KV grouped-query attention, 768 dimensi tersembunyi, RoPE + SwiGLU, konteks 2.048 token. Varian: `base`, `instruct` (SFT), `chat` (SimPO).
- **Liwet-400M-A188M (`Sparse MoE GQA`)**: 400,1M parameter total / 187,8M parameter aktif, 32 layer (32L), 16Q/4KV grouped-query attention, 8 routed experts dengan routing top-2, dan **tanpa shared expert**, konteks 2.048 token. Varian: `base`, `instruct`, `chat`.

### Varian model

| # | Repositori Hugging Face | Keluarga | Tahap / metode | Konteks | Mode prompt | Rilis |
|---:|---|---|---|---:|---|---|
| 1 | `fadhilahkhairogi/Liwet-66M-id-base-v1` | Liwet-66M | Pretraining, 1 epoch (~428M token) | 1.024 | `raw` | Live |
| 2 | `fadhilahkhairogi/Liwet-66M-id-base-v2` | Liwet-66M | Pretraining, 2 epoch (~852M token) | 1.024 | `raw` | Live |
| 3 | `fadhilahkhairogi/Liwet-66M-id-instruct-v1` | Liwet-66M | SFT instruction tuning dari base-v1 | 1.024 | `instruct_plain` | Live |
| 4 | `fadhilahkhairogi/Liwet-66M-id-instruct-v2` | Liwet-66M | SFT instruction tuning dari base-v2 | 1.024 | `instruct_plain` | Live |
| 5 | `fadhilahkhairogi/Liwet-66M-id-chat` | Liwet-66M | Conversational SFT, masking token assistant, vocab 32.003 | 1.024 | `chat_tags` | Live |
| 6 | `fadhilahkhairogi/Liwet-163M-id-base` | Liwet-163M | Pretraining pada 2,10B token Bahasa Indonesia | 2.048 | `raw_bos` | Live |
| 7 | `fadhilahkhairogi/Liwet-163M-id-instruct` | Liwet-163M | SFT instruction tuning pada 165,6k percakapan | 2.048 | `chat_role` | Live |
| 8 | `fadhilahkhairogi/Liwet-163M-id-chat` | Liwet-163M | Alignment SimPO (`beta=2.0`, `gamma=0.5`) | 2.048 | `chat_role` | Live |
| 9 | `fadhilahkhairogi/Liwet-400M-A188M-id-base` | Liwet-400M-A188M | Pretraining Sparse MoE | 2.048 | `raw_bos` | Live |
| 10 | `fadhilahkhairogi/Liwet-400M-A188M-id-instruct` | Liwet-400M-A188M | SFT instruction tuning Sparse MoE | 2.048 | `chat_role` | Live |
| 11 | `fadhilahkhairogi/Liwet-400M-A188M-id-chat` | Liwet-400M-A188M | Alignment SimPO Sparse MoE | 2.048 | `chat_role` | Live |

Ketersediaan diperiksa langsung ke Hugging Face Hub oleh aplikasi. Repositori diaktifkan otomatis setelah `config.json`, `model.safetensors`, dan `tokenizer.json` tersedia.

### Format prompt

- `raw`: teks pengguna terakhir dikirim tanpa format percakapan.
- `raw_bos`: token beginning-of-sequence ditambahkan sebelum prompt mentah.
- `instruct_plain`: instruksi diikuti dua baris baru.
- `chat_tags`: giliran percakapan memakai penanda `<|user|>`, `<|assistant|>`, dan `<|end|>`.
- `chat_role`: giliran percakapan memakai label peran `System:`, `User:`, dan `Assistant:`.

### Tab demo

1. **Percakapan (Chat)**: percakapan multi-turn untuk semua repositori, kontrol system prompt, pemangkasan riwayat, tombol reset, serta telemetri latensi dan `tok/s` untuk setiap respons.
2. **Arena Perbandingan (Comparison Arena)**: membandingkan dua atau tiga model pada prompt yang sama dengan pengaturan decoding identik. Arena mempertahankan layout 1x3 tetap, menampilkan jawaban secara streaming pada kartu masing-masing, dan meletakkan accordion `Pengaturan` di bawah tiga kartu output. Tabel ringkasan menampilkan parameter, latensi, token baru, dan throughput.
3. **Uji Parameter (Parameter Lab)**: playground decoding dengan preset `Chat Seimbang`, `Kreatif`, `Deterministik`, dan `Cepat`; toggle prompt mentah versus terformat; slider `max_new_tokens`, `temperature`, `top_k`, `top_p`, dan `repetition_penalty`; seed tetap opsional; pratinjau prompt terformat; serta tabel telemetri.

### Runtime dan kompatibilitas

- Space dioptimalkan untuk CPU dan berjalan pada tier CPU gratis tanpa CUDA.
- `load_liwet_model()` memanggil `resolve_device("auto")`: CUDA dipakai bila runtime GPU tersedia, sedangkan CPU memakai `float32`.
- Arena menjalankan model secara berurutan agar penggunaan memori tetap terbatas di CPU. Loader menyimpan cache LRU kecil untuk bundel model yang sudah dimuat.
- Bobot diunduh dari Hugging Face Hub publik saat pertama kali dipakai; tidak memerlukan token akses privat.
- Semua 11 varian model (66M, 163M, dan 400M-A188M) telah tersedia publik di Hugging Face Hub, sementara repositori yang belum rilis tetap ditangani aman sebagai `Coming Soon`.

### Mulai cepat

```bash
pip install -r requirements.txt
python app.py
```

### Tim

**Kelompok 11, Telkom University**

- Fadhilah Akbar Khairogi — `103012300394`
- Faishal Fakhrillah — `103012300474`
- Deril Diaz — `103012300074`

### Tautan

- Organisasi Hugging Face: [fadhilahkhairogi](https://huggingface.co/fadhilahkhairogi)
- Referensi integrasi: [MODELS_INFO.md](MODELS_INFO.md)
- Entry point: [app.py](app.py)
