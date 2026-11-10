# Plan Progress So Far — Liwet-LLM Inference Optimization

> **Terakhir Diperbarui:** `2026-11-10` (sesudah reinstall torch `2.9.1+cu128`)  
> **Estimasi Progres Total:** **~45% Selesai**  
> **Spesifikasi Rencana:** [INFERENCE_OPTIMIZATION_PLAN.md](INFERENCE_OPTIMIZATION_PLAN.md)  
> **Konteks Detail & Diagnosis Sesi Sebelumnya:** [PREVIOUS_AGENT_SESSION.md](PREVIOUS_AGENT_SESSION.md)  
> **Data Benchmark & Profiler:** [bench/RESULTS.md](bench/RESULTS.md)

---

## Ringkasan Status Deliverable (`D1`–`D9`)

| Kode | Deliverable | Lokasi Target | Progres | Status Saat Ini |
|---|---|---|---:|---|
| **D1** | Skrip benchmark & profiler + baseline | `scripts/bench/`, `docs/bench/RESULTS.md` | **100%** | Selesai & lulus Gate Fase 0 |
| **D2** | Harness paritas + dump referensi | `scripts/parity/`, `E:\liwet_gguf_build\parity_ref\` | **90%** | Dump referensi 3 model chat selesai, tetapi dibuat pada instalasi torch lama; `66M` sudah diregenerasi & lulus di lingkungan baru, `163M`/`400M` menunggu drive `E:`. `compare.py` perlu mode `llamacpp` di Jalur B |
| **D3** | PyTorch fastpath (`LIWET_FASTPATH`) | `liwet-space/liwet_models.py`, `liwet-space/app.py` (`perf/torch-fastpath`) | **95%** | Gate A terpenuhi: `400M` **105.28 tok/s** headless (dari `1.73 tok/s`), `stream_generate` +16.7%, UI +1.2%, P1–P4 lulus, `LIWET_FASTPATH=0` identik. Sisa: regenerasi dump 163M/400M saat drive `E:` kembali |
| **D4** | Fork `llama.cpp` + patch MoE + CI CPU wheel | `E:\liwet_gguf_build\llama.cpp` (`liwet-moe`) | **45%** | Patch C++ & converter Python selesai ditulis di atas commit vendor `0c1e57098`; belum dikompilasi & belum ada workflow GitHub Actions |
| **D5** | Skrip ekspor HF-Llama/Mixtral & konversi GGUF | `scripts/gguf/` | **50%** | `export_hf_llama.py` & `verify_hf_llama_parity.py` selesai dibuat; skrip batch konversi/kuantisasi GGUF & eksekusinya menunggu Gate A selesai |
| **D6** | 11 repo `-GGUF` (staging lokal dulu) | `E:\liwet_gguf_build\hf_gguf_staging\` | **0%** | Belum dimulai (menunggu Gate B2/B3 lulus & izin upload pemilik) |
| **D7** | Web klon 1:1 `liwet-space-cpp/` | `liwet-space-cpp/`, `scripts/run_local_servers.ps1` | **0%** | Belum dimulai (menunggu Gate B3 lulus) |
| **D8** | Diagnosa Ollama & `Modelfile` dense (`66M`, `163M`) | `scripts/ollama/` | **0%** | Belum dimulai (bonus tahap akhir) |
| **D9** | Laporan akhir seluruh model × backend × dtype | `docs/bench/RESULTS.md` | **40%** | Bagian Fase 0 & **Jalur A** selesai; bagian Jalur B, 8 varian sisa, dan Ollama menyusul |

---

## Checklist Detail Per Fase & Gate

### Fase 0 — Benchmark, Profiling Baseline & Referensi Paritas (`§3`, `§4`) — **LULUS (100%)**
- [x] Salin referensi murni yang belum disentuh ke `scripts/parity/reference/liwet_models_ref.py`.
- [x] Buat dataset uji paritas `scripts/parity/prompts.jsonl` (≥ 30 prompt per `prompt_mode`: `raw`, `raw_bos`, `instruct_plain`, `chat_tags`, `chat_role`, termasuk multi-turn, non-ASCII/emoji, dan konteks panjang).
- [x] Buat & jalankan `scripts/bench/bench_torch.py` untuk baseline GPU (`bf16`) dan CPU (`fp32`) pada 3 model chat (`400M`, `163M`, `66M`).
- [x] Jalankan `torch.profiler` 20 langkah decode (`scripts/bench/profile_decode.py`) → ekspor Chrome trace ke `E:\liwet_gguf_build\traces\` dan ringkasan ke `docs/bench/profile_Liwet-400M-A188M-id-chat.json`.
- [x] Ukur overhead level aplikasi Gradio (`scripts/bench/bench_gradio_overhead.py` → `docs/bench/gradio_overhead.json`).
- [x] Catat seluruh angka baseline + bukti akar masalah ke `docs/bench/RESULTS.md` (**Gate Fase 0 Lulus**).
- [x] Jalankan `scripts/parity/reference_dump.py` hingga tuntas 100% untuk `Liwet-400M-A188M-id-chat`, `Liwet-163M-id-chat`, dan `Liwet-66M-id-chat` di `E:\liwet_gguf_build\parity_ref\`.

---

### Jalur A — Optimasi PyTorch Tanpa Mengubah Output (`§5`, branch `perf/torch-fastpath`) — **SEDANG BERJALAN (~70%)**
- [x] **A1 (Level App di `liwet-space/app.py`):**
  - [x] Hilangkan `force_refresh=True` per request pada `_is_repo_available` (gunakan cache TTL).
  - [x] Throttle `yield` streaming UI (maksimal tiap `50 ms` atau `8` token, plus update final wajib).
  - [x] Pisahkan waktu generasi model murni (`seconds`) dari total end-to-end (`wall_seconds`).
  - [x] Implementasi detokenisasi inkremental dengan fallback/final decode identik dan mode assert `LIWET_DEBUG_DECODE=1`.
- [x] **A2–A5 (`liwet-space/liwet_models.py` & `requirements.txt`):**
  - [x] `seen_mask` boolean di GPU untuk repetition penalty tanpa `torch.unique`.
  - [x] `StaticKVCache` untuk ketiga keluarga (`MiniTransformer`, `DenseIndonesianChatLM`, `MoEIndonesianChatLM`). SDPA memakai **`repeat_interleave`** (bukan `enable_gqa=True`) agar bit-identik dengan referensi; `enable_gqa` hanya opt-in via `LIWET_SDPA_GQA=1`.
  - [x] Manual `CudaGraphDecodeRunner` dengan fallback otomatis ke static eager bila capture/replay gagal.
  - [x] Konfigurasi thread fisik CPU (`psutil` dipasang di `tubes_venv`) dengan tetap mempertahankan `float32` di CPU.
  - [x] Penyesuaian durasi `@spaces.GPU` realistis (`20s` chat/lab, `45s` arena) & preload model default pada lingkungan ZeroGPU.
- [x] **Verifikasi & Pemenuhan Gate Jalur A (LULUS — dengan 1 caveat lingkungan):**
  - [x] Paritas `P1`/`P3` `400M`: **diagnosis lama salah** — penyebabnya bukan `allow_tf32`/`matmul_precision("high")`/fused QKV (semua ada di referensi pristine), melainkan `SDPA enable_gqa=True` (kernel berbeda). Diganti `repeat_interleave` → `P1` top1 100%, `P3` 100%.
  - [x] `P2` `163M`/`66M`: tie-break disamakan dengan referensi (`torch.argmax` murni untuk pemilihan token) → `exact_128 = 100%`, `non_tie_divergences = []`.
  - [x] `≥ 100 tok/s` pada `400M` GPU: `MoE.forward` single-token dioptimalkan (hindari salinan bobot expert / `aten::index`) + incremental detokenisasi → **105.28 tok/s** headless, `stream_generate` **90.18–92.41 tok/s** (gap +16.7%).
  - [x] `LIWET_FASTPATH=0` identik: `P1 max_abs_diff = 0.0`, `P2` 100%, `ALL_PASSED` untuk ketiga model.
  - [x] Benchmark CPU + overhead UI Gradio: UI direct `chat_send` **+1.2%** vs headless (≤ 20%); HTTP/SSE server-side −14.4% (biaya SSE ~100 ms/event).
  - [x] Checklist UI `P5`: diff `app.py` tidak menambah/menghapus elemen UI apa pun.
  - [x] Update `docs/bench/RESULTS.md` (bagian Jalur A) & commit branch `perf/torch-fastpath` (tanpa push).
  - [ ] **Caveat / sisa:** instalasi torch lama ternyata korup dan sudah di-reinstall ke `2.9.1+cu128`. Rounding bf16 antar-build berubah, sehingga dump referensi lama tidak berlaku lagi untuk verifikasi numerik. `66M` sudah diregenerasi & `ALL_PASSED`; regenerasi `163M`/`400M` + promosi `parity_ref_v2 → parity_ref` **diblokir karena drive `E:` tidak ter-mount**.

---

### Jalur B — Konversi GGUF & Fork `llama.cpp` (`§6`) — **PERSIAPAN SELESAI (~45%)**
- [x] **B0:** Pin `llama-cpp-python` di `v0.3.36` (`1652066`) dan siapkan `E:\liwet_gguf_build\llama.cpp` branch `liwet-moe` tepat di commit submodule vendor `0c1e57098bba43ac29e6e3b677cdceebdd22334f`.
- [x] **B1 & B2.1:** Buat skrip ekspor `scripts/gguf/export_hf_llama.py` (mendukung `66M` `rms_norm_eps=1e-6`, `163M` `rms_norm_eps=1e-5`, dan `400M` `MixtralForCausalLM` + `e_score_correction.bias` + validasi keamanan `F16`) serta verifier `scripts/gguf/verify_hf_llama_parity.py`.
- [x] **B3 (Patch Kode Fork):** Patch `gguf-py/gguf/constants.py`, `conversion/base.py` (termasuk hash `chkhsh` tokenizer Liwet → `"gpt-2"`), `src/llama-model.cpp`, dan `src/models/llama.cpp` untuk mendukung `blk.%d.exp_probs_b`, `SOFTMAX_WEIGHT`, dan `expert_weights_norm=false`. Editable `gguf-py` sudah terpasang di `tubes_venv`.
- [ ] **B2.2:** Jalankan `export_hf_llama.py` & `verify_hf_llama_parity.py` untuk 3 model chat prioritas (**Gate Perantara B2**).
- [ ] **B4 (Build Lokal Windows CUDA `SM89`):**
  - [ ] Kompilasi `llama.cpp` (`llama-server.exe`, `llama-cli.exe`, `llama-quantize.exe`, `llama-tokenize.exe`, `llama-bench.exe`, `llama-eval-callback.exe`).
  - [ ] Konversi 3 model chat ke `BF16`, `F16`, dan `Q8_0` di `E:\liwet_gguf_build\gguf\`, serta verifikasi `llama-tokenize.exe` identik dengan `tokenizers` pada `prompts.jsonl`.
  - [ ] Arahkan `llama-cpp-python` ke fork lokal, build & install wheel CUDA `SM89` ke `tubes_venv`.
- [ ] **B5:** Buat `.github/workflows/build-cpu-wheel.yml` di fork `llama.cpp` untuk wheel CPU Linux x86_64 AVX2, lalu commit perubahan fork di branch `liwet-moe`.
- [ ] **Gate B3:** Perluas `scripts/parity/compare.py` untuk backend `llamacpp` (`P1`–`P4` termasuk router `P3` via `llama-eval-callback`), jalankan benchmark `llama-bench.exe`, dan catat di `docs/bench/RESULTS.md`.

---

### Web Klon 1:1 `liwet-space-cpp/` (`§7`) — **BELUM DIMULAI (0%)**
- [ ] Buat struktur lengkap `liwet-space-cpp/` (`app.py`, `liwet_backend.py`, `backends/torch_backend.py`, `backends/llamacpp_inproc.py`, `backends/llamacpp_server.py`, `gguf_catalog.py`, `requirements.txt`, `README.md`) dan `scripts/run_local_servers.ps1`.
- [ ] Pastikan single-source import dari `liwet-space/liwet_models.py` untuk katalog, prompt builder, tokenisasi, stop-id, dan pembersihan reply.
- [ ] Verifikasi mode paritas & mode cepat sampler native, benchmark kecepatan GPU (`≥ 100 tok/s`) & CPU lokal (`≥ 25 tok/s`), serta lulus checklist `P5` dengan screenshot berdampingan di `docs/bench/ui/`.

---

### Tahap Akhir — 8 Varian Sisa, Staging `-GGUF`, Ollama & Laporan Final (`§6.B6`, `§8`, `§9`) — **BELUM DIMULAI (0%)**
- [ ] Ekspor, konversi (`BF16`, `F16`, `Q8_0`), uji paritas `P1`/`P2`, dan benchmark ke-8 varian model sisanya.
- [ ] Susun 11 folder staging repo `fadhilahkhairogi/<Model>-GGUF` secara lokal di `E:\liwet_gguf_build\hf_gguf_staging\` (jangan upload sebelum izin pemilik).
- [ ] Diagnosa masalah startup Ollama lokal (`%LOCALAPPDATA%\Ollama\server.log`) & buat `Modelfile` untuk seluruh varian dense (`66M`, `163M`) di `scripts/ollama/`.
- [ ] Finalisasi laporan komprehensif di `docs/bench/RESULTS.md` dan minta konfirmasi pemilik untuk langkah remote/Admin.
