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

✅ **Status verifikasi (diperbarui 2026-11-10):** seluruh artefak paritas di atas sudah **diregenerasi ulang di lingkungan torch `2.9.1+cu128`** — dump referensi baru ditulis ke `E:\liwet_gguf_build\parity_ref_v2\` lalu dipromosikan ke `parity_ref` (backup lama: `parity_ref_backup_20261110_0520`). Hasil: ketiga model **`ALL_PASSED`** dengan fastpath menyala, dan dengan **`LIWET_FASTPATH=0` P1 `max_abs_diff = 0.0` untuk ketiganya**. P2 `exact_128 = 100%` dan P4 `max_tvd = 0.0` di semua kombinasi.

**Catatan historis (cross-build):** dump yang dibuat pada build torch lama tidak cocok secara numerik dengan build baru — pada 66M divergensi pertama muncul di langkah ke-9 dari 36 run dan **semuanya tie** (`non_tie_divergences: []`), lantas direproduksi identik saat `LIWET_FASTPATH=0`. Jadi itu perbedaan rounding bf16 antar-build, bukan regresi fastpath. Karena itu dump lama **tidak boleh** dipakai untuk menyimpulkan kegagalan paritas setelah upgrade/reinstall torch; regenerasi dump lebih dulu.

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
| P2 greedy 128-step | exact ≥95%, tanpa non-tie divergence | 100%, `non_tie_divergences: []` | ✅ (ketiga model terverifikasi ulang pasca-reinstall) |
| P3 router MoE | exact-set ≥99.5% | 100% (`1120` token-layer) | ✅ |
| P4 sampling | max TVD ≤0.05 | 0.0 | ✅ |
| `LIWET_FASTPATH=0` | identik perilaku lama | P1 `max_abs 0.0`, P2 100% | ✅ |
| P5 UI checklist | tidak ada elemen UI ditambah/hilang | `git diff liwet-space/app.py` hanya perubahan handler/warmup/throttle; tidak ada tab/kontrol/preset baru | ✅ |
| CPU ≥25 tok/s | — | 66M 48.46 ✅; 163M 7.22 ❌; 400M 3.42 ❌ | ⏭️ target Jalur B (GGUF) |

### 6. Catatan Operasional
- **Drive `E:` sempat hilang di tengah sesi** (hanya `C:` dan `D:` ter-mount), lalu tersambung kembali. Setelah itu dump 400M dilengkapi, `parity_ref_v2` dipromosikan ke `parity_ref`, dan paritas 3 model dijalankan ulang (lihat §3). Tidak ada artefak besar yang ditulis ke `C:`.
- Instalasi torch yang korup juga menghapus `torchvision`; belum dibutuhkan oleh fastpath/app, tetapi perlu dipasang ulang bila profiling berbasis torchvision diperlukan.

---

## Jalur B — Gate B2: Ekspor HF Llama/Mixtral & Verifikasi Paritas Perantara

**Lingkungan:** CPU `float32` (referensi Python murni `scripts/parity/reference/liwet_models_ref.py` vs `transformers 5.17.0`), CPU `torch 2.9.1+cu128`. Folder ekspor: `E:\liwet_gguf_build\hf_llama_tmp\`. Artefak lengkap: [hf_llama_parity.json](hf_llama_parity.json). Dump referensi: `E:\liwet_gguf_build\parity_ref\` (10 prompt prefill CPU).

| Model | Arsitektur HF | Tokenizer exact | P1 min cosine / max abs / top1 | P1b layer-wise MoE (max abs / router) | P3 router (end-to-end) | P2 greedy 32-step | `PASSED` |
|---|---|---|---:|---|---:|---|---|
| `Liwet-400M-A188M-id-chat` | `MixtralForCausalLM` | ✅ | 0.999061 / 0.7247 / 100% | **3.05e-05 / 100%** | 99.688% | 2/2 (`[32, 32]`) | ✅ |
| `Liwet-163M-id-chat` | `LlamaForCausalLM` | ✅ | 1.0 / 1.14e-05 / 100% | — (dense) | — | 2/2 (`[32, 32]`) | ✅ |
| `Liwet-66M-id-chat` | `LlamaForCausalLM` | ✅ | 1.0 / 6.2e-06 / 100% | — (dense) | — | 2/2 (`[32, 32]`) | ✅ |

**Temuan #1 — MoE Liwet memakai capacity-limited dispatch yang tidak dimiliki Mixtral standar.** Referensi `MoE.forward` membatasi tiap expert ke `capacity = ceil(num_tokens × top_k / num_experts × capacity_factor)` assignment ketika `num_tokens > 1` (`capacity_factor = 1.25` untuk 400M). Pada prefill 40–45 token, jumlah assignment per expert mencapai **24–26** sementara capacity hanya **13–15**, sehingga **127–182 dari 2560–2880 assignment (≈5–6%) dibuang**. Ekspor Mixtral standar tidak membuang apa pun → divergensi besar (cosine 0.966, P3 85.9%). Verifier sekarang mem-patch `MixtralSparseMoeBlock.forward` untuk meniru dispatch referensi **persis** (`argsort` → `searchsorted` → posisi < capacity), dan `export_hf_llama.py` menulis `moe_capacity_factor` ke `config.json`.

**Temuan #2 — satu prompt tersangkut router near-tie di layer 24.** Pada `chat_role_01` (42 token), token ke-12 layer 24 memiliki selection logit expert 3 dan 7 yang hanya berbeda **2.4e-7**; drift fp32 antar implementasi attention (fused `qkv_proj` vs tiga GEMM `q/k/v`) sebesar ~3e-6 membalik pilihan (referensi `[6, 7]`, HF `[6, 3]`). Perbedaan itu mengubah bookkeeping capacity expert 3 (12 vs 13 assignment) sehingga output prompt tersebut bergeser (cos 0.999061, max abs 0.7247) — **sembilan prompt lain cos = 1.0**. Karena celah logitnya jauh di bawah galat numerik, kedua pilihan valid secara matematis; ini bukan kesalahan bobot ekspor.

**Kriteria Gate B2 yang diterapkan:**
- Dense (`66M`, `163M`): `P1 min cosine ≥ 0.99999`, top1 100%, P2 exact 32/32, tokenizer identik — **lulus ketat**.
- MoE (`400M`): top1 100%, P2 exact 32/32, tokenizer identik, **P1b layer-wise MoE ≤ 1e-4 dengan router 100%** (diukur dengan memberi hidden state referensi yang identik ke tiap blok MoE HF), `P1 min cosine ≥ 0.999`, dan setiap divergensi end-to-end wajib terklasifikasi sebagai tie (celah selection logit < 1e-5). Hasil: worst layer-wise **3.05e-05** pada baris dengan magnitudo output MoE ~302 (≈1e-7 relatif, murni noise urutan GEMM fp32), router **100%**, satu-satunya divergensi end-to-end = tie 2.4e-7 yang terdokumentasi.

**Implikasi untuk Jalur B (`B4`/Gate B3):** fork `llama.cpp` **wajib** mengimplementasikan dispatch capacity-limited yang sama agar prefill MoE cocok dengan referensi. Capacity dihitung dari total token pada satu forward; karena `llama.cpp` memecah prefill menjadi ubatch (`-ub`, default 512), prompt lebih panjang dari `n_ubatch` akan memakai capacity berbeda dan boleh divergen — dokumentasikan dan/atau set `-ub` ≥ panjang prompt saat benchmark paritas. Model dense tidak terpengaruh.

---

## Jalur B — Gate B3: Fork `llama.cpp`, Paritas GGUF & Benchmark

**Build:** fork `E:\liwet_gguf_build\llama.cpp` branch `liwet-moe`, basis vendor `0c1e57098` + 3 commit lokal:
- `9196caf89` — dispatch MoE capacity-limited (`ggml-cpu/ops.cpp` + `llama-graph.cpp`), KV baru `moe_capacity_factor` di GGUF, workflow CPU wheel.
- `f003973d4` — `llama-eval-callback`: filter tensor via env (`LLAMA_DEBUG_CB_FILTER`) + opsi BOS untuk menangkap router MoE.
- `cae551692` — capacity dispatch memakai `std::stable_sort` agar urutan keep per expert identik dengan `torch.argsort` (stable radix) di CUDA: assignment token-major.

Binary CUDA `SM89` (`ARCHS=890`, CUDA 12.9 + MSVC 14.50, `-allow-unsupported-compiler`): `llama-cli`, `llama-server`, `llama-quantize`, `llama-tokenize`, `llama-bench`, `llama-eval-callback` (`build 11315, commit cae551692`). `llama-cpp-python 0.3.36` diarahkan ke fork lokal (`GGML_CUDA=on`, `CMAKE_CUDA_ARCHITECTURES=89`) → `E:\liwet_gguf_build\wheels\llama_cpp_python-0.3.36-py3-none-win_amd64.whl` (±87 MiB), terpasang dan GPU offload terverifikasi.

### 1. Artefak GGUF & Paritas Tokenizer

| Model | BF16 / F16 (byte) | Q8_0 (byte) | Tokenizer `llama-tokenize` |
|---|---:|---:|---|
| `Liwet-66M-id-chat` | 134,598,912 (128.4 MiB) | 72,051,552 (68.7 MiB) | 35/35 exact |
| `Liwet-163M-id-chat` | 327,202,400 (312.0 MiB) | 174,401,120 (166.3 MiB) | 35/35 exact |
| `Liwet-400M-A188M-id-chat` | 801,993,920 (764.8 MiB) | 515,683,520 (491.8 MiB) | 35/35 exact |

Total **105/105 prompt exact** (100%) terhadap id `tokenizers` di dump referensi. Prompt `chat_role` membutuhkan BOS eksplisit karena GGUF menulis `add_bos_token=false`. Detail & perintah: [gguf_tokenizer_parity.txt](gguf_tokenizer_parity.txt).

### 2. Metodologi Paritas (backend `llamacpp`)

`scripts/parity/compare.py --backend llamacpp --regime gguf_{bf16,f16,q8_0}`:
- **P1 (logit):** 35 prompt prefill, dibandingkan dengan logit referensi **bf16 CUDA** (`prefill_bf16_cuda.npz`) pada token terakhir → cosine, max abs, top1. `n_ctx` = `context_len` model, `n_batch = n_ubatch` = 1920 (163M/400M) atau 896 (66M) — yaitu panjang prompt terpanjang dibulatkan ke 32 — agar capacity MoE dihitung sekali per prompt.
- **P2 (greedy 128):** 6 prompt × 128 token, `llama-cpp-python` (`n_gpu_layers=-1`, `logits_all=True`, `seed=0`), repetition penalty maksimum yang sama dengan referensi, argmax murni. Divergensi diklasifikasikan: **tie** bila celah top1–top2 referensi `< 0.05`, selain itu **non-tie** (dilaporkan beserta langkah & celahnya).
- **P3 (router MoE):** `llama-eval-callback` menangkap tensor `ffn_moe_topk` per layer (env filter), dibandingkan sebagai **himpunan** expert referensi. Dua prompt `long_2048`/`long_1024` dikecualikan (tidak dapat dikirim sebagai teks karena melewati `n_ctx`) → 33 prompt × 32 layer = **1056 token-layer**.
- **P4 (sampling):** 6 prompt × 500 seed, `temp 0.65`, `top-k 40`, `top-p 0.9` → histogram token, TVD maksimum.
- **Koreksi penting P4:** dump referensi dibuat dengan `torch.multinomial` pada tensor **CUDA**. Sampling kandidat di CPU memakai RNG stream berbeda sehingga TVD naik sampai 0.112 hanya karena noise RNG. `_p4_sample_from_logits` sekarang memindahkan logits ke CUDA bila tersedia → kedua sisi memakai urutan angka acak yang sama, dan P4 mengukur fidelity logit (replay logits referensi sendiri memberi TVD 0.0).

### 3. Hasil Paritas Dense (66M & 163M)

| Model | Quant | P1 min cos / max abs / top1 | P2 norep exact / median / non-tie | P2 defrep exact / median / non-tie | P4 max TVD | `ALL_PASSED` |
|---|---|---|---:|---:|---:|---|
| `66M` | BF16 | 0.999863 / 0.0908 / 100% | 83.3% / 128 / — | 83.3% / 128 / — | 0.028 | ✅ |
| `66M` | F16 | 0.999875 / 0.0879 / 100% | 83.3% / 128 / — | 66.7% / 128 / — | 0.036 | ✅ |
| `66M` | Q8_0 | 0.999480 / 0.2102 / 100% | 83.3% / 128 / `chat_tags_mt_00@86` (0.0625) | 83.3% / 128 / `chat_tags_00@79` (0.0544) | 0.032 | ✅ |
| `163M` | BF16 | 0.999973 / 0.1725 / 100% | 66.7% / 128 / `long_2048@22` (0.125), `chat_role_01@12` (0.0625) | 16.7% / **33.5** / `chat_role_01@12` (0.0625) | 0.016 | ❌ (P2d) |
| `163M` | F16 | 0.999974 / 0.1602 / 100% | 66.7% / 128 / idem | 16.7% / **33.5** / idem | 0.036 | ❌ (P2d) |
| `163M` | Q8_0 | 0.999819 / 0.3181 / 100% | 50.0% / 87.5 / `chat_role_26@47`, `chat_role_01@12` (0.0625) | 0.0% / **26.5** / `mt_00@25` (0.0679), `chat_role_01@12` (0.0625) | 0.048 | ❌ (P2d) |

Artefak: [gguf_b3/](gguf_b3/) (`gateB3_Liwet-{66M,163M}-id-chat_{BF16,F16,Q8_0}.json`; salinan mentah di `E:\liwet_gguf_build\scratch\`).

**Interpretasi dense:** P1 lulus dengan margin besar (cos ≥0.9995, top1 100%), P4 ≤0.048 di semua kombinasi. P2 selalu gagal hanya pada prompt-prompt yang di lintasan referensinya sendiri mengandung **tie/near-tie bf16** — statistik `logit_gaps` referensi (768 langkah greedy):

| Model | Trace | celah = 0 | celah < 0.05 | celah ≤ 0.0625 (1 bf16 ulp) |
|---|---|---:|---:|---:|
| 66M | norep | 5 | 1 | 10 |
| 66M | defrep | 2 | 2 | 10 |
| 163M | norep | 5 | 0 | 18 |
| 163M | defrep | 9 | 9 | 26 |
| 400M | norep | 11 | 1 | 11 |
| 400M | defrep | 6 | 10 | 27 |

Kandidat f32 `llama.cpp` memiliki galat hingga ~0.17 pada logit, yaitu **lebih besar dari 1 bf16 ulp** (0.0625 pada magnitudo ~10). Karena itu tie bf16 referensi tidak dapat direproduksi oleh runtime independen mana pun: argmax bisa memilih token yang berbeda walau kedua logit secara matematis sama. Eksperimen membulatkan logits kandidat ke grid bf16 sebelum argmax sempat dicoba, tetapi justru memperburuk `163M norep` (dari 100% → 33.3% exact) karena galat f32 > 1 ulp — jadi dibatalkan dan kandidat tetap f32 jujur. **Median ≥96 pada 163M defrep tidak tercapai** (33.5), tetapi seluruh divergensinya berada pada celah referensi ≤0.0625; tidak ada bukti kesalahan implementasi. Hal serupa yang lebih ringan terlihat di Q8_0.

Catatan kebijakan: rencana menyatakan P2 untuk **Q8_0 dilaporkan, bukan blocker**, sedangkan `compare.py` saat ini tetap menegakkan ambang Q8_0 (set ≥97%, median ≥48) sehingga `ALL_PASSED=false`. Perbedaan definisi ini dicatat, bukan diubah diam-diam.

### 4. Hasil Paritas MoE (`Liwet-400M-A188M-id-chat`) — **TIDAK LULUS**

| Quant | P1 min cos / mean / max abs / top1 | P2 norep exact / median | P2 defrep exact / median | P3 router | P4 max TVD | `ALL_PASSED` |
|---|---|---|---:|---:|---:|---:|---|
| BF16 | **0.970377** / 0.995388 / 1.964 / 100% | 16.7% / **6** | 0.0% / **3.5** | **88.826%** | **0.094** | ❌ |
| F16 | **0.972936** / 0.995441 / 2.047 / 97.14% | 16.7% / **6** | 0.0% / **3.5** | **88.352%** | **0.082** | ❌ |
| Q8_0 | **0.968586** / 0.994333 / 2.833 / 94.29% | 16.7% / 28 | 0.0% / **9** | **87.879%** | **0.172** | ❌ |

Artefak: [gguf_b3/](gguf_b3/) (`gateB3_Liwet-400M-A188M-id-chat_{BF16,F16,Q8_0}.json`). Hasil **identik** sebelum dan sesudah binary dibangun ulang di `cae551692`, jadi bukan artefak binary lama.

**Diagnosis (bukan bug dispatch):** dispatch capacity sudah diverifikasi semantik-identik dengan referensi (`stable_sort` → keep token-major per expert; capacity `ceil(n_tokens·k/E·1.25)`; `n_ubatch` = panjang prompt sehingga capacity dihitung sekali). Penyebab divergensi adalah **kegagalan router pada near-tie bf16**:
- Layer awal cocok; flip mulai sekitar layer 9 dan menumpuk. Contoh string match per layer (`chat_role_00`): `11111111101001111111111111100111` (28/32 layer).
- Contoh konkret layer 9: kandidat memberi nilai selection `[-1.1426 (exp 4), -1.1959 (exp 4), -1.2048 (exp 7) …]`, referensi memilih `[0, 7]`. Expert 4 vs 7 hanya berbeda **~0.0089 ≈ 1 bf16 ulp** pada magnitudo 1.2.
- Membulatkan `ffn_moe_probs_biased` ke grid bf16 secara offline memberi jumlah layer cocok yang sama (28/32) → pembulatan tidak menyelamatkan; selisih f32 vs bf16 sudah mengubah urutan sebelum pembulatan.
- Mengganti referensi ke dump **fp32 CPU** juga tidak lulus: cos min **0.9411** (`long_2048`), mean 0.9875, top1 100% pada 10 prompt.
- KV cache f32 (`type_k=type_v=0`) hanya menaikkan P1 5 prompt dari 0.9927 → 0.9945; tetap jauh dari 0.999.

Rantai sebab: satu flip router mengubah campuran expert → hidden state bergeser → layer berikutnya makin banyak near-tie yang terbalik → drift kumulatif 32 layer. Inilah yang menurunkan P1/P3/P4, bukan kesalahan konversi atau dispatch. Karena celah selection-nya ~1 ulp bf16, hasil ini **inheren** untuk runtime f32 independen dan tidak bisa diperbaiki dengan tuning threshold numerik kecil.

**Implikasi:** gate paritas MoE perlu kriteria khusus MoE (mis. P1 cos ≥0.99 + top1 ≥95% + P3 ≥85% dilaporkan; P1 layer-wise `ffn_moe_probs_biased` ≤1e-3 dengan router ≥99%), atau jalur compute bf16 di llama.cpp. Sampai itu diputuskan, klaim "GGUF 400M setara referensi secara numerik" **tidak boleh** dibuat; secara kualitatif output tetap terbaca dan top1 masih 94–100%.

### 5. Benchmark `llama-bench` (`commit cae551692`, build 11315)

GPU `-ngl 99`, 5 rep (pp512 + tg128); angka tg 400M dari run stabilitas 8 rep karena run 5 rep mengalami outlier throttle (lihat catatan).

| Model | Quant | Prefill 512 (tok/s) | Decode 128 (tok/s) | Decode terbaik vs Jalur A |
|---|---|---:|---:|---|
| `66M` | BF16 | 41,987 ± 5,799 | 672 ± 64 | ×1.75 |
| `66M` | F16 | 58,330 ± 13,530 | 649 ± 25 | ×1.69 |
| `66M` | Q8_0 | 51,596 ± 8,529 | **898 ± 34** | ×2.34 |
| `163M` | BF16 | 30,600 ± 4,553 | 397 ± 1 | ×1.94 |
| `163M` | F16 | 32,577 ± 6,592 | 390 ± 7 | ×1.91 |
| `163M` | Q8_0 | 36,629 ± 5,629 | **559 ± 25** | ×2.74 |
| `400M` | BF16 | 7,157 ± 131 | 29.6 ± 1.3 | **×0.28** |
| `400M` | F16 | 5,652 ± 202 | 25.8 ± 0.8 | ×0.25 |
| `400M` | Q8_0 | 8,570 ± 641 | **33.6 ± 0.7** | **×0.32** |

CPU `-ngl 0 -t 8`, tg32, run stabilitas 8 rep (Q8_0):

| Model | Decode CPU Q8_0 (tok/s, median) | Target ≥25 | vs Jalur A CPU (`fp32`) |
|---|---:|---|---:|
| `66M` | **326.8** | ✅ | ×6.7 (48.46) |
| `163M` | **155.7** | ✅ | ×21.5 (7.22) |
| `400M` | **103.2** | ✅ | ×30.2 (3.42) |

CPU BF16 (3 rep, informatif): 66M 152.8 ± 31.1, 163M 75.2 ± 19.2, 400M 27.1 ± 11.4 tok/s.

**Catatan stabilitas:** run 5 rep pada 400M GPU sempat menghasilkan sampel liar (`28.7 → 1.8 tok/s`) tanpa proses lain; run 8 rep berikutnya stabil (sd ≤1.3). Rep pertama juga sering jauh lebih lambat (warm-up/page-in). Karena itu decode dilaporkan dari run stabilitas, dan angka absolut GPU adalah **batas bawah** selama Intel DTT aktif: pengamatan `nvidia-smi` selama decode menunjukkan clock SM median ~0.8 GHz (maks 1.6 GHz) dari batas 3.1 GHz, sementara Jalur A terukur ~2.37 GHz. Pemilik perlu menjalankan skrip DTT (Administrator) untuk angka puncak.

**Interpretasi produk:**
- **CPU:** GGUF `Q8_0` memenuhi target ≥25 tok/s dengan margin besar (103–327 tok/s), **15–30× lebih cepat** dari jalur PyTorch CPU. Ini kemenangan utama Jalur B.
- **GPU dense:** llama.cpp 1.7–2.7× lebih cepat dari Jalur A pada decode, dan 1.4–2.2× pada prefill → kandidat kuat untuk 66M/163M.
- **GPU 400M (MoE):** llama.cpp justru ~3× **lebih lambat** dari Jalur A CUDA-graph (30–34 vs 105 tok/s). Rekomendasi: tetap pakai Jalur A (PyTorch fastpath) untuk 400M di GPU; GGUF 400M dipakai untuk CPU/server tanpa PyTorch.

### 6. Verdict Gate B3

| Kriteria | Hasil | Status |
|---|---|---|
| Tokenizer 105/105 exact | 105/105 | ✅ |
| Fork build CUDA SM89 + wheel terpasang | build 11315 `cae551692`, wheel ±87 MiB | ✅ |
| Dense 66M P1/P2/P4 | cos ≥0.9994, top1 100%, P2 median 128, TVD ≤0.036 | ✅ |
| Dense 163M P1/P4 | cos ≥0.9999, top1 100%, TVD ≤0.036 | ✅ |
| Dense 163M P2-defrep median ≥96 | 33.5; semua divergensi pada celah referensi ≤0.0625 | ⚠️ (tie bf16, terdokumentasi) |
| MoE 400M P1 cos ≥0.999 | 0.9686–0.9729 | ❌ (near-tie router) |
| MoE 400M P3 ≥99.5% | 87.9–88.8% | ❌ (hulu dari P1) |
| MoE 400M P4 TVD ≤0.05 | 0.082–0.172 | ❌ (hulu dari P1) |
| CPU Q8_0 ≥25 tok/s | 103–327 tok/s | ✅ |
| GPU decode Jalur B | dense ×1.7–2.7 vs Jalur A; 400M ×0.3 | ✅ dense / ⚠️ MoE |

**Kesimpulan:** Gate B3 **lulus untuk dense 66M** dan **lulus P1/P4 untuk 163M**, tetapi **tidak lulus penuh**: (a) P2 163M defrep jatuh karena tie bf16 referensi; (b) seluruh metrik paritas 400M MoE gagal akibat near-tie router yang inheren. Keduanya dianalisis dan didokumentasikan, bukan disembunyikan. Artefak mentah ada di [gguf_b3/](gguf_b3/) (`gateB3_*.json`, `bench_final_*.json`, `bench_stab_*.json`; salinan asli di `E:\liwet_gguf_build\scratch\`).
