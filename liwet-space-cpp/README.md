---
title: Liwet-LLM GGUF (llama.cpp)
emoji: 🍚
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 5.23.3
app_file: app.py
pinned: false
short_description: Liwet GGUF suite served through llama.cpp
---

# Liwet-LLM — GGUF/llama.cpp edition

A 1:1 clone of the original `liwet-space` UI that serves the **same 11 model
variants** from GGUF files through `llama-cpp-python` instead of PyTorch +
SafeTensors. `app.py` is byte-identical to the original except for one import
line (`liwet_models` → `liwet_backend`); prompt building, tokenizer helpers,
stop ids, and reply cleanup still come from `liwet-space/liwet_models.py`.

## Backends

Select with `LIWET_BACKEND`:

| Value | Module | Use |
|---|---|---|
| `llamacpp` (default) | `backends/llamacpp_inproc.py` | In-process GGUF, CPU or CUDA offload. Parity sampler replayed in NumPy. |
| `server` | `backends/llamacpp_server.py` | Talks to a running `llama-server` over HTTP (`LIWET_SERVER_URL`). Native sampler — fast mode, not bit-parity. |
| `torch` | `backends/torch_backend.py` | Original PyTorch path, unchanged. |

If `LIWET_SERVER_URL` is set and `LIWET_BACKEND` is unset, the server backend is
chosen automatically.

Relevant environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `LIWET_GGUF_DIR` | `E:\liwet_gguf_build\gguf` | Folder containing `<Model>-<QUANT>.gguf` |
| `LIWET_GGUF_QUANT` | `Q8_0` | `BF16`, `F16`, or `Q8_0` |
| `LIWET_GGUF_NGL` | auto (`-1` with GPU build) | Layers offloaded; set `0` on CPU-only hosts |
| `LIWET_CPP_SAMPLER` | `parity` | `native` is accepted but currently shares the NumPy sampler |
| `LIWET_CPU_THREADS` | physical cores | Threads for llama.cpp |
| `LIWET_SERVER_URL` | — | e.g. `http://127.0.0.1:8080` |

On a CPU-only Space set `LIWET_GGUF_NGL=0`; the default Q8_0 files are small
enough for the free CPU tier (66M ≈ 72 MB, 163M ≈ 174 MB, 400M ≈ 516 MB).

## Parity status (Gate B3, 2026-11-10)

* **66M dense** — `ALL_PASSED` on BF16/F16/Q8_0 (P1 cos ≥ 0.9994, top1 100 %,
  P4 TVD ≤ 0.036).
* **163M dense** — P1/P4 pass; P2-defrep misses the strict median because of
  bf16 near-ties (all divergences are within one bf16 ulp of the reference).
* **400M MoE** — **numerically approximate, not bit-parity**: BF16/F16 median
  cosine 0.997, top1 100 %/97.1 %, router 88.8 %/88.4 %; Q8_0 top1 94.3 %.
  Router flips start at layer ~9 from near-ties of ~1 bf16 ulp. Accepted under
  the owner-approved MoE criteria (median/mean cosine ≥ 0.99, top1 ≥ 95 % for
  BF16/F16). When `LIWET_BACKEND=torch`, the 400M path remains exact.

Full numbers: `docs/bench/RESULTS.md` in the source repository.

## Local run

```powershell
# In-process GGUF (GPU if the wheel supports CUDA)
$env:LIWET_GGUF_DIR = "E:\liwet_gguf_build\gguf"
$env:LIWET_GGUF_QUANT = "Q8_0"
python app.py
```

Or use `scripts/run_local_servers.ps1` from the repository root, which can also
start `llama-server.exe` and point the UI at it (`-Backend server`).

## Notes

* No UI elements were added or removed relative to `liwet-space/app.py`.
* The 400M GGUF cards must show the “numerically approximate” warning; this
  app inherits the original card text and the caveat is documented here.
