# Liwet-LLM — Benchmark & Parity Results

> Log hasil pengukuran bertahap sesuai [INFERENCE_OPTIMIZATION_PLAN.md](../INFERENCE_OPTIMIZATION_PLAN.md).

## Lingkungan Pengukuran
- **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (6141 MiB VRAM, driver 610.88, max SM clock 3105 MHz, power cap 55 W / default 35 W)
- **CPU:** Intel Core 5 210H (8 cores, 12 logical processors)
- **Python / Torch:** Python 3.12.14, PyTorch `2.13.0+cu130` (`bf16=True`), `gradio 6.31.0` — **khusus angka Fase 0 di bawah**. Lingkungan Jalur A berbeda (torch `2.9.1+cu128`, gradio `5.23.3`); lihat [Lingkungan Pengukuran Jalur A](#lingkungan-pengukuran-jalur-a-berbeda-dari-fase-0).
- **Status Intel DTT Baseline:** Tercatat di [gpu_state_baseline.txt](gpu_state_baseline.txt). Selama beban GPU baseline, `SW Power Cap` & `SW Thermal Slowdown` terpantau **Active** (`avg_sm_mhz = 288–713 MHz`, `avg_power_w = 2.9–5.3 W`).

---

## Fase 0 — Baseline PyTorch Eager & Profiling

### 1. Throughput & Latensi Baseline (Headless, tanpa Gradio)

| Model | Device / Dtype | Load (s) | Prefill 32 (tok/s) | Prefill 512 (tok/s) | Prefill 1500* (tok/s) | Pure Forward Decode (tok/s) | `stream_generate` Fixed (tok/s) | `stream_generate` Natural (tok/s) | Peak VRAM (MB) | Avg SM MHz |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Liwet-400M-A188M-id-chat` | GPU (`bfloat16`) | 18.34 | 41.4 | 538.6 | 1082.9 | **1.73** (p50 494ms, p95 891ms) | **1.73** | **1.05** | 1118.9 | 713.3 |
| `Liwet-163M-id-chat` | GPU (`bfloat16`) | 15.96 | 141.8 | 2241.6 | 8765.0 | **5.07** (p50 180ms, p95 328ms) | **4.20** | **4.17** | 464.4 | 694.4 |
| `Liwet-66M-id-chat` | GPU (`bfloat16`) | 8.95 | 96.8 | 1624.5 | 2596.3 | **3.10** (p50 320ms, p95 416ms) | **2.86** | **2.70** | 229.4 | 288.0 |
| `Liwet-400M-A188M-id-chat` | CPU (`float32`) | 10.96 | 2.2 | 11.3 | 27.7 | **0.25** (p50 3755ms) | **0.25** | **0.25** | — | — |
| `Liwet-163M-id-chat` | CPU (`float32`) | 6.04 | 23.9 | 36.3 | 56.8 | **0.77** (p50 1126ms) | **0.50** | **0.57** | — | — |
| `Liwet-66M-id-chat` | CPU (`float32`) | 4.43 | 17.9 | 48.8 | 81.3 | **1.01** (p50 877ms) | **2.14** | **0.78** | — | — |

*\*Untuk `66M` (konteks 1024), panjang prefill terbesar adalah `752` (GPU) / `976` (CPU).*

### 2. Hasil Profiling `torch.profiler` (`Liwet-400M-A188M-id-chat`, 20 langkah decode CUDA)
Sumber: [profile_Liwet-400M-A188M-id-chat.json](profile_Liwet-400M-A188M-id-chat.json), trace mentah disimpan di `E:\liwet_gguf_build\traces\trace_Liwet-400M-A188M-id-chat.json` (`79.61 MB`).

- **Total CUDA kernels (20 token):** `26,380` → **`1,319.0` kernel per token**
- **Wall span:** `19,739.22 ms` (`986.96 ms/token` / `1.01 tok/s` di bawah profiler)
- **GPU busy vs idle:** GPU hanya sibuk **`2,029.07 ms`** → **GPU menganggur (idle) `89.72%` dari total waktu**!
- **Top CPU overhead (20 token):**
  - `cudaLaunchKernel` (`25,080` panggilan): `5,126.93 ms` + `cuLaunchKernel` (`1,300` panggilan): `358.29 ms`
  - `aten::mul`: `853.70 ms`, `aten::mm`: `784.51 ms`, `aten::add`: `659.72 ms`, `aten::empty` (`7,760` alokasi): `611.07 ms`, `aten::bmm`: `601.95 ms`, `aten::cat` (`3,200` panggilan): `537.21 ms`, `aten::copy_`: `352.86 ms`, `aten::index`: `313.87 ms`
- **Top CUDA ops (20 token, total `2,029 ms` pada clock ter-throttle):**
  - `aten::mm` (`1,940` calls): `976.63 ms`
  - `aten::index` (`1,320` calls — `vectorized_gather_kernel` untuk `self.gate_up_proj[experts]` & `self.down_proj[experts]`): **`555.62 ms`** (~27% waktu GPU habis hanya untuk menyalin bobot expert tiap layer!)
  - `aten::bmm`: `104.13 ms`, `aten::add`: `58.10 ms`, `aten::cat`: `51.01 ms`, `aten::_efficient_attention_forward`: `50.85 ms`, `aten::_fused_rms_norm`: `40.04 ms`

### 3. Overhead Level App (`bench_gradio_overhead.py` pada `400M-chat`, 128 token, riwayat 6 turn)
Sumber: [gradio_overhead.json](gradio_overhead.json)
- `_is_repo_available(..., force_refresh=True)` per request: **`0.629 detik`** panggilan jaringan ke HF Hub.
- `chat_send` direct generator (126 yield, 128 token): `109.504 dtk` (**`1.2 tok/s`**) — memicu `yield` ulang seluruh list riwayat chat di setiap token.

### 4. Kesimpulan Diagnosis Terbukti (Gate Fase 0 Lulus)
1. **Akar penyebab utama #1 (B1):** Python eager mode me-launch **`1,319` kernel CUDA per token**, membuat GPU idle **`89.72%`** dan bertahan di P-state/clock rendah (`288–713 MHz`), sementara CPU habis waktu untuk `cudaLaunchKernel` + alokasi tensor kecil (`aten::empty`).
2. **Akar penyebab #2 (B3 & B2):** Di `MoE.forward`, `self.gate_up_proj[experts]` & `down_proj[experts]` (`aten::index`) menghabiskan **27% waktu GPU** untuk menggandakan bobot expert ke buffer sementara; ditambah `torch.cat` + `repeat_interleave` KV-cache sebanyak `3,200` kali per 20 token.
3. **Akar penyebab #3 (B5–B7):** Pada 163M/66M, loop `stream_generate` mulai memotong throughput (`5.07 → 4.20 tok/s`), plus `_is_repo_available(force_refresh=True)` menambah `~0.63 s` per pesan 400M, dan `yield` tiap 1 token akan mencekik UI begitu GPU menembus >100 tok/s.
4. **Faktor pengali (B8):** `SW Power Cap` & `SW Thermal Slowdown` aktif di baseline; angka puncak tanpa throttle akan diukur terpisah setelah pemilik menjalankan `.bat`.

---

## Jalur A — PyTorch Fastpath (Static KV Cache + CUDA Graph + Incremental Detokenization)

### Lingkungan Pengukuran Jalur A (berbeda dari Fase 0)
- **Torch:** `2.9.1+cu128` — instalasi lama ternyata **korup** (`ModuleNotFoundError: torch.nn`, paket namespace 7.8 MB tanpa `__init__.py`/`nn`/`lib`), sehingga di-*force reinstall* dari index cu128. Seluruh angka Jalur A di bawah diukur pada instalasi yang sudah diperbaiki.
- **Gradio:** `5.23.3` (sesuai `liwet-space/requirements.txt`; `gradio 6.31.0` di Fase 0 tidak lagi terpasang).
- **Driver/GPU:** driver 610.88, RTX 4050 Laptop GPU.
- **Intel DTT:** `SW Power Cap` & `SW Thermal Slowdown` **tetap Active di seluruh run** (belum ada intervensi `.bat`; hanya pemilik yang boleh menjalankannya sebagai Administrator). Karena itu angka di bawah adalah batas **dengan throttle aktif**, bukan performa puncak.
- **Artefak:** [track_a_cuda.json](track_a_cuda.json), [track_a_cpu.json](track_a_cpu.json), [gradio_overhead_track_a.json](gradio_overhead_track_a.json), [gpu_state_track_a_before.txt](gpu_state_track_a_before.txt) / [gpu_state_track_a_after.txt](gpu_state_track_a_after.txt). Skrip: `scripts/bench/bench_torch.py`, `scripts/bench/bench_gradio_overhead.py`, `scripts/parity/{reference_dump,compare}.py`.

### 1. Throughput Headless Jalur A — GPU (`bfloat16`)

| Model | Load (s) | Prefill 32 (tok/s) | Prefill 512 (tok/s) | Prefill 1500* (tok/s) | Pure Forward Decode (tok/s) | `stream_generate` Fixed (tok/s) | `stream_generate` Natural (tok/s) | Peak VRAM (MB) | Avg SM MHz | Avg Power (W) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Liwet-400M-A188M-id-chat` | 3.772 | 263.6 | 3975.6 | 12853.0 | **105.28** (p50 9.42 ms, p95 9.81 ms) | **90.18** (p50 10.02 ms, p95 10.88 ms) | **92.41** | 1026.8 | 2370.0 | 24.34 |
| `Liwet-163M-id-chat` | 2.949 | 1373.1 | 21690.7 | 38833.1 | **204.43** (p50 4.79 ms, p95 6.05 ms) | **183.68** (p50 5.17 ms, p95 5.50 ms) | **183.31** | 425.1 | 1312.5 | 10.48 |
| `Liwet-66M-id-chat` | 1.275 | 1328.5 | 19499.8 | 39703.7 | **384.21** (p50 2.24 ms, p95 3.78 ms) | **361.87** | **373.33** | 215.8 | 1275.0 | 14.14 |

*\*Untuk `66M` (konteks 1024), panjang prefill terbesar adalah `992`; untuk `163M`/`400M` konteks 2048 sehingga `1500` valid.*

**Delta Fase 0 → Jalur A (same-throttle):**

| Model | Pure decode Fase 0 (tok/s) | Pure decode Jalur A (tok/s) | Speedup |
|---|---:|---:|---:|
| `Liwet-400M-A188M-id-chat` | 1.73 | 105.28 | **×60.9** |
| `Liwet-163M-id-chat` | 5.07 | 204.43 | **×40.3** |
| `Liwet-66M-id-chat` | 3.10 | 384.21 | **×124.0** |

**Overhead jalur `stream_generate` vs pure forward** (wall time relatif): 400M **+16.7%**, 163M **+11.3%**, 66M **+6.2%** — semuanya di bawah batas 20%.

### 2. Throughput Headless Jalur A — CPU (`float32`, 16 token)

| Model | Load (s) | Prefill 32 (tok/s) | Prefill 512 (tok/s) | Prefill 1500/992 (tok/s) | Pure Forward Decode (tok/s) | `stream_generate` Fixed (tok/s) | `stream_generate` Natural (tok/s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Liwet-66M-id-chat` | 1.084 | 760.0 | 1206.3 | 1079.1 (992) | **48.46** (p50 21.3 ms) | **36.70** | **37.39** |
| `Liwet-163M-id-chat` | 2.269 | 345.6 | 403.9 | 348.6 | **7.22** (p50 119.15 ms) | **6.19** | **4.93** |
| `Liwet-400M-A188M-id-chat` | 3.223 | 46.0 | 142.6 | 167.2 | **3.42** (p50 266.48 ms) | **3.28** | **3.43** |

Catatan: dibanding instalasi torch lama, CPU ikut membaik (400M `1.16 → 3.42`, 163M `4.3 → 7.22` tok/s) — indikasi instalasi lama memang memakai kernel/build yang tidak sehat. Target CPU ≥25 tok/s tetap **belum tercapai** untuk 163M/400M; ini dijadwalkan lewat Jalur B (GGUF `Q8_0`), bukan fastpath PyTorch.

### 3. Paritas P1–P4 (vs referensi eager pristine)

Artefak: [parity_track_a_cuda_Liwet-66M-id-chat.json](parity_track_a_cuda_Liwet-66M-id-chat.json), [parity_track_a_cuda_Liwet-163M-id-chat.json](parity_track_a_cuda_Liwet-163M-id-chat.json), [parity_track_a_cuda_Liwet-400M-A188M-id-chat.json](parity_track_a_cuda_Liwet-400M-A188M-id-chat.json); *fastpath off*: `parity_fastpath_off_*.json` dengan `LIWET_FASTPATH=0`.

| Model | P1 min cosine / max abs / top1 | P2 norep (exact-128 / min first div / non-tie div) | P2 defrep | P3 router | P4 max TVD | `ALL_PASSED` |
|---|---|---|---|---|---:|---|
| `Liwet-66M-id-chat` | 0.999996 / 0.0625 / 100% | 100% / 128 / `[]` | 100% / 128 / `[]` | — (dense) | 0.0 | ✅ |
| `Liwet-163M-id-chat` | 1.0 / 0.0 / 100% | 100% / 128 / `[]` | 100% / 128 / `[]` | — (dense) | 0.0 | ✅ |
| `Liwet-400M-A188M-id-chat` | 1.0 / 0.03125 / 100% | 100% / 128 / `[]` | 100% / 128 / `[]` | 100% (`1120` token-layer) | 0.0 | ✅ |

**`LIWET_FASTPATH=0` exact-old-behavior:** seluruh model `ALL_PASSED` dengan **P1 `max_abs_diff = 0.0`** dan P2 100% — membuktikan jalur lama tetap bit-identik terhadap referensi.

⚠️ **Caveat penting (lingkungan):** artefak `parity_track_a_cuda_*` / `parity_fastpath_off_*` di folder ini dibuat **sebelum** torch di-reinstall. Setelah reinstall, *rounding bf16 antar-build torch berbeda*, sehingga dump referensi lama tidak lagi cocok secara numerik (divergensi pertama pada 66M di langkah ke-9 dari 36 run, **semua** berupa tie, `non_tie_divergences: []`). Setelah dump 66M diregenerasi di lingkungan saat ini, hasilnya **kembali `ALL_PASSED`** (P2 100%, P4 `0.0`), dan perilaku identik terbukti juga saat `LIWET_FASTPATH=0` — jadi perbedaannya adalah cross-build, bukan regresi fastpath. Regenerasi dump **163M & 400M** sedang menunggu drive `E:` (lokasi artefak besar) tersambung kembali; sampai itu selesai, baris 163M/400M di tabel paritas di atas adalah hasil lingkungan lama.

**Diagnosis lama yang dikoreksi:** penyebab divergensi paritas sebelumnya **bukan** `allow_tf32=True` / `set_float32_matmul_precision("high")` / fused QKV — ketiganya ada juga di referensi pristine. Penyebab sebenarnya adalah `SDPA enable_gqa=True` yang memilih kernel berbeda; fastpath sekarang memakai `repeat_interleave` secara default (`LIWET_SDPA_GQA=1` untuk opt-in).

### 4. Overhead UI Gradio (`Liwet-400M-A188M-id-chat`, 128 token, riwayat 6 turn)
Sumber: [gradio_overhead_track_a.json](gradio_overhead_track_a.json) (`bench_gradio_overhead.py` dengan warmup 1 generasi pendek agar CUDA-graph capture tidak terhitung).

- `_is_repo_available(..., force_refresh=True)`: **`0.362 s`** per request (dengan `LIWET_FASTPATH` aktif, TTL cache 300 s; sebelumnya `0.629 s`).
- **Direct `chat_send` generator:** 24 yield, wall **`1.375 s`**, stats server **`93.7 tok/s`**. Headless natural pada model yang sama **`92.41 tok/s`** → overhead UI **≈ +1.2% waktu** (batas 20% ✅).
- **HTTP/SSE penuh (`gradio_client`):** wall **`3.355 s`** untuk 98 token, 21 event, stats server **`80.8 tok/s`** (≈14.4% di bawah headless). Selisih wall vs server ≈ `2.0 s` hampir seluruhnya biaya pengiriman ~21 event SSE via `gradio_client` di Windows (~100 ms/event), bukan komputasi model.

### 5. Checklist Gate Jalur A

| Gate | Target | Hasil | Status |
|---|---|---|---|
| 400M GPU headless pure decode | ≥100 tok/s | **105.28 tok/s** | ✅ |
| `stream_generate` overhead | ≤20% | 400M +16.7%, 163M +11.3%, 66M +6.2% | ✅ |
| UI vs headless | ≤20% | Direct `chat_send` **+1.2%**; HTTP/SSE server-side −14.4% | ✅ |
| P1 logit | cosine ≥0.9999, top1 100% | min cosine ≥0.999996, top1 100% | ✅ |
| P2 greedy 128-step | exact ≥95%, tanpa non-tie divergence | 100%, `non_tie_divergences: []` | ✅ (66M terverifikasi ulang pasca-reinstall; 163M/400M menunggu dump baru) |
| P3 router MoE | exact-set ≥99.5% | 100% (`1120` token-layer) | ✅ |
| P4 sampling | max TVD ≤0.05 | 0.0 | ✅ |
| `LIWET_FASTPATH=0` | identik perilaku lama | P1 `max_abs 0.0`, P2 100% | ✅ |
| P5 UI checklist | tidak ada elemen UI ditambah/hilang | `git diff liwet-space/app.py` hanya perubahan handler/warmup/throttle; tidak ada tab/kontrol/preset baru | ✅ |
| CPU ≥25 tok/s | — | 66M 48.46 ✅; 163M 7.22 ❌; 400M 3.42 ❌ | ⏭️ target Jalur B (GGUF) |

### 6. Catatan Operasional
- **Drive `E:` (yang menyimpan `liwet_gguf_build/`, dump paritas, build `llama.cpp`, trace) hilang di tengah sesi**; hanya `C:` dan `D:` ter-mount. Semua langkah yang bergantung pada dump besar (regenerasi 400M, promosi `parity_ref_v2 → parity_ref`, parity 3-model ulang) **diblokir** sampai drive tersambung lagi. Tidak ada artefak besar yang ditulis ke `C:`.
- Instalasi torch yang korup juga menghapus `torchvision`; belum dibutuhkan oleh fastpath/app, tetapi perlu dipasang ulang bila profiling berbasis torchvision diperlukan.
