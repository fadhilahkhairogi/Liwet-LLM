# Liwet-LLM

**English** | [Bahasa Indonesia](#bahasa-indonesia)

Liwet (Lightweight Transformer) is a family of Indonesian decoder-only language models trained from scratch in PyTorch. This public monorepo contains source code only. Model weights, tokenizers, datasets, checkpoints, logs, and binary artifacts are intentionally excluded and distributed through Hugging Face.

## Team — Group 11, Telkom University

| Name | Student ID |
| --- | --- |
| Fadhilah Akbar Khairogi | 103012300394 |
| Faishal Fakhrillah | 103012300474 |
| Deril Diaz | 103012300074 |

## Model family

All published repositories live under [`fadhilahkhairogi/Liwet-*`](https://huggingface.co/fadhilahkhairogi).

| Scale | Architecture | Parameters | Context | Hugging Face |
| --- | --- | ---: | ---: | --- |
| **66M** | 12 layers, 8 heads, 512 hidden, RoPE + SwiGLU, 32k BPE (chat: 32,003) | 66.7M | 1,024 | [base-v2](https://huggingface.co/fadhilahkhairogi/Liwet-66M-id-base-v2), [instruct-v2](https://huggingface.co/fadhilahkhairogi/Liwet-66M-id-instruct-v2), [chat](https://huggingface.co/fadhilahkhairogi/Liwet-66M-id-chat) |
| **163M** | 16 layers, 12Q/4KV GQA, 768 hidden, 3,072 FFN, RoPE + SwiGLU | 163.0M | 2,048 | [base](https://huggingface.co/fadhilahkhairogi/Liwet-163M-id-base), [instruct](https://huggingface.co/fadhilahkhairogi/Liwet-163M-id-instruct), [chat](https://huggingface.co/fadhilahkhairogi/Liwet-163M-id-chat) |
| **400M** | Dense: 32 layers, 16Q/4KV GQA, 1,024 hidden, 2,880 FFN. MoE: 32 layers, 16Q/4KV GQA, 8 routed experts, top-2 routing, no shared expert | 399.84M dense; 400.1M total / 187.8M active (MoE) | 2,048 | Dense: source-only. MoE: [base](https://huggingface.co/fadhilahkhairogi/Liwet-400M-A188M-id-base), [instruct](https://huggingface.co/fadhilahkhairogi/Liwet-400M-A188M-id-instruct), [chat](https://huggingface.co/fadhilahkhairogi/Liwet-400M-A188M-id-chat) |

## Repository layout

| Path | Contents |
| --- | --- |
| `liwet-66m/` | 66M model, training code, evaluation CLIs, and model README |
| `liwet-163m/` | 163M model and training code; `eval/` placeholder |
| `liwet-400m-dense/` | Dense 400M model, training code, and sanitized Colab pipeline |
| `liwet-400m-moe/` | 400M-A188M Sparse MoE model, training code, Streamlit suite, and sanitized Colab pipeline |
| `liwet-space/` | Hugging Face Space Gradio app (`app.py`, `liwet_models.py`) for chat, 3-scale arena, and parameter lab |
| `scripts/check_sterile.py` | Release audit for size, paths, tokens, and binary artifacts |

## CPU-only quickstart

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements.txt
python scripts/check_sterile.py
```

Inference and evaluation are CPU-only by design. The training scripts are device-agnostic and are included for reproducibility; CUDA is optional and is not required by this repository.

## Evaluation

```bash
cd liwet-66m/eval
python evaluate.py --checkpoint checkpoints/posttrain_chat_best.pt
python compare_epochs.py
python compare_chat.py
```

Checkpoints and tokenizers are not committed. The evaluation CLIs expect PyTorch training checkpoints (`.pt`) and the matching tokenizer. Published safetensors weights are available on Hugging Face for inference; the Space loader lives in `liwet-space/`.

## Sterile release audit

```bash
python scripts/check_sterile.py
```

The audit fails on any file larger than 5 MiB, JSON config larger than 100 KiB, forbidden binary or archive extension, `checkpoints/`, `logs/`, `data/`, `data_id/`, `__pycache__/`, `.env` files, local absolute paths, API keys or Hugging Face tokens, and private IP addresses.

## Status

- All three model families (11 repositories across 66M, 163M, and 400M-A188M Sparse MoE) are published publicly on Hugging Face.
- The 400M dense variant is included as source; its weights are not published in this repository.
- The 400M-A188M Sparse MoE variant (`liwet-400m-moe/`) contains full model, training, A100 Colab pipeline, and Streamlit suite source code (32 layers, 16Q/4KV GQA, 8 routed experts with top-2 routing, no shared expert, 2,048 context, ~400.1M total / ~187.8M active parameters).
- **Live Interactive Web Demo (ZeroGPU A10G, Gradio 5.23.3):** [`fadhilahkhairogi/mini-transformer-id`](https://huggingface.co/spaces/fadhilahkhairogi/mini-transformer-id) ([direct app](https://fadhilahkhairogi-mini-transformer-id.hf.space/)) runs all 11 models, 3-Scale Comparison Arena, and Parameter Lab interactively, while [`fadhilahkhairogi/Liwet-Space`](https://huggingface.co/spaces/fadhilahkhairogi/Liwet-Space) hosts the static ecosystem portal (source in `liwet-space/`).

---

## Bahasa Indonesia

[English](#liwet-llm) | **Bahasa Indonesia**

Liwet (Lightweight Transformer) adalah keluarga model bahasa Indonesia decoder-only yang dilatih dari nol menggunakan PyTorch. Monorepo publik ini hanya berisi kode sumber. Bobot model, tokenizer, dataset, checkpoint, log, dan artefak biner sengaja tidak disertakan dan didistribusikan melalui Hugging Face.

### Tim — Kelompok 11, Telkom University

| Nama | NIM |
| --- | --- |
| Fadhilah Akbar Khairogi | 103012300394 |
| Faishal Fakhrillah | 103012300474 |
| Deril Diaz | 103012300074 |

### Keluarga model

Semua repositori yang sudah terbit berada di bawah [`fadhilahkhairogi/Liwet-*`](https://huggingface.co/fadhilahkhairogi).

| Skala | Arsitektur | Parameter | Konteks | Hugging Face |
| --- | --- | ---: | ---: | --- |
| **66M** | 12 layer, 8 head, hidden 512, RoPE + SwiGLU, BPE 32k (chat: 32.003) | 66,7M | 1.024 | [base-v2](https://huggingface.co/fadhilahkhairogi/Liwet-66M-id-base-v2), [instruct-v2](https://huggingface.co/fadhilahkhairogi/Liwet-66M-id-instruct-v2), [chat](https://huggingface.co/fadhilahkhairogi/Liwet-66M-id-chat) |
| **163M** | 16 layer, 12Q/4KV GQA, hidden 768, FFN 3.072, RoPE + SwiGLU | 163,0M | 2.048 | [base](https://huggingface.co/fadhilahkhairogi/Liwet-163M-id-base), [instruct](https://huggingface.co/fadhilahkhairogi/Liwet-163M-id-instruct), [chat](https://huggingface.co/fadhilahkhairogi/Liwet-163M-id-chat) |
| **400M** | Dense: 32 layer, 16Q/4KV GQA, hidden 1.024, FFN 2.880. MoE: 32 layer, 16Q/4KV GQA, 8 routed expert, routing top-2, tanpa shared expert | 399,84M dense; 400,1M total / 187,8M aktif (MoE) | 2.048 | Dense: sumber saja. MoE: [base](https://huggingface.co/fadhilahkhairogi/Liwet-400M-A188M-id-base), [instruct](https://huggingface.co/fadhilahkhairogi/Liwet-400M-A188M-id-instruct), [chat](https://huggingface.co/fadhilahkhairogi/Liwet-400M-A188M-id-chat) |

### Struktur repositori

| Path | Isi |
| --- | --- |
| `liwet-66m/` | Model 66M, kode training, CLI evaluasi, dan README model |
| `liwet-163m/` | Model 163M dan kode training; `eval/` masih placeholder |
| `liwet-400m-dense/` | Model 400M dense, kode training, dan pipeline Colab yang sudah disanitasi |
| `liwet-400m-moe/` | Model Sparse MoE 400M-A188M, kode training, aplikasi Streamlit, dan pipeline Colab ter-sanitasi |
| `liwet-space/` | Aplikasi Gradio Hugging Face Space (`app.py`, `liwet_models.py`) untuk chat, arena 3 skala, dan lab parameter |
| `scripts/check_sterile.py` | Audit rilis untuk ukuran, path, token, dan artefak biner |

### Mulai cepat CPU-only

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements.txt
python scripts/check_sterile.py
```

Inferensi dan evaluasi dirancang CPU-only. Skrip training bersifat device-agnostic dan disertakan untuk reproduksibilitas; CUDA opsional dan tidak diwajibkan oleh repositori ini.

### Evaluasi

```bash
cd liwet-66m/eval
python evaluate.py --checkpoint checkpoints/posttrain_chat_best.pt
python compare_epochs.py
python compare_chat.py
```

Checkpoint dan tokenizer tidak di-commit. CLI evaluasi membutuhkan checkpoint training PyTorch (`.pt`) dan tokenizer yang cocok. Bobot safetensors yang sudah terbit tersedia di Hugging Face untuk inferensi; loader Space tersedia di `liwet-space/`.

### Audit rilis steril

```bash
python scripts/check_sterile.py
```

Audit akan gagal jika ada file lebih dari 5 MiB, konfigurasi JSON lebih dari 100 KiB, ekstensi biner atau arsip terlarang, folder `checkpoints/`, `logs/`, `data/`, `data_id/`, `__pycache__/`, file `.env`, path absolut lokal, API key atau token Hugging Face, serta alamat IP privat.

### Status

- Ketiga keluarga model (11 repositori di skala 66M, 163M, dan Sparse MoE 400M-A188M) sudah terbit publik di Hugging Face.
- Varian 400M dense disertakan sebagai kode sumber; bobotnya tidak diterbitkan di repositori ini.
- Varian Sparse MoE 400M-A188M (`liwet-400m-moe/`) memuat kode model, pelatihan, pipeline Colab A100, dan antarmuka Streamlit lengkap (32 layer, 16Q/4KV GQA, 8 routed expert dengan routing top-2, tanpa shared expert, konteks 2.048, ~400,1M parameter total / ~187,8M parameter aktif).
- **Demo Web Interaktif Live (ZeroGPU A10G, Gradio 5.23.3):** [`fadhilahkhairogi/mini-transformer-id`](https://huggingface.co/spaces/fadhilahkhairogi/mini-transformer-id) ([aplikasi langsung](https://fadhilahkhairogi-mini-transformer-id.hf.space/)) menjalankan seluruh 11 model, Arena Perbandingan 3 Skala, dan Uji Parameter secara daring, sedangkan [`fadhilahkhairogi/Liwet-Space`](https://huggingface.co/spaces/fadhilahkhairogi/Liwet-Space) menyajikan portal statis ekosistem 11 model (kode sumber di `liwet-space/`).
