# Plan Progress So Far — Liwet-LLM Inference Optimization

> **Terakhir Diperbarui:** `2026-11-10` (Gate B3 selesai diukur: dense lulus, MoE 400M gagal paritas karena near-tie router bf16)  
> **Estimasi Progres Total:** **~56% Selesai**  
> **Spesifikasi Rencana:** [INFERENCE_OPTIMIZATION_PLAN.md](INFERENCE_OPTIMIZATION_PLAN.md)  
> **Konteks Detail & Diagnosis Sesi Sebelumnya:** [PREVIOUS_AGENT_SESSION.md](PREVIOUS_AGENT_SESSION.md)  
> **Data Benchmark & Profiler:** [bench/RESULTS.md](bench/RESULTS.md)

---

## Ringkasan Status Deliverable (`D1`–`D9`)

| Kode | Deliverable | Lokasi Target | Progres | Status Saat Ini |
|---|---|---|---:|---|
| **D1** | Skrip benchmark & profiler + baseline | `scripts/bench/`, `docs/bench/RESULTS.md` | **100%** | Selesai & lulus Gate Fase 0 |
| **D2** | Harness paritas + dump referensi | `scripts/parity/`, `E:\liwet_gguf_build\parity_ref\` | **100%** | Dump referensi 3 model chat **diregenerasi di lingkungan torch `2.9.1+cu128`** dan dipromosikan (`parity_ref_v2 → parity_ref`). `compare.py` masih perlu mode `llamacpp` di Jalur B |
| **D3** | PyTorch fastpath (`LIWET_FASTPATH`) | `liwet-space/liwet_models.py`, `liwet-space/app.py` (`perf/torch-fastpath`) | **100%** | Gate A terpenuhi: `400M` **105.28 tok/s** headless (dari `1.73 tok/s`), `stream_generate` +16.7%, UI +1.2%, P1–P4 `ALL_PASSED` ketiga model, `LIWET_FASTPATH=0` P1 `max_abs 0.0` |
| **D4** | Fork `llama.cpp` + patch MoE + CI CPU wheel | `E:\liwet_gguf_build\llama.cpp` (`liwet-moe`) | **100%** | Binary CUDA SM89 + `llama-cpp-python` wheel terpasang; dispatch capacity (`stable_sort`) & eval-callback selesai (3 commit lokal); workflow CPU wheel ter-commit (belum di-push) |
| **D5** | Skrip ekspor HF-Llama/Mixtral & konversi GGUF | `scripts/gguf/` | **85%** | Gate B2 lulus; 9 GGUF 3 model chat selesai + tokenizer 105/105 exact; tinggal batch 8 varian sisa |
| **D6** | 11 repo `-GGUF` (staging lokal dulu) | `E:\liwet_gguf_build\hf_gguf_staging\` | **0%** | Belum dimulai (menunggu izin upload pemilik; staging lokal disiapkan lebih dulu) |
| **D7** | Web klon 1:1 `liwet-space-cpp/` | `liwet-space-cpp/`, `scripts/run_local_servers.ps1` | **0%** | Belum dimulai (menunggu keputusan threshold paritas MoE) |
| **D8** | Diagnosa Ollama & `Modelfile` dense (`66M`, `163M`) | `scripts/ollama/` | **0%** | Belum dimulai (bonus tahap akhir) |
| **D9** | Laporan akhir seluruh model × backend × dtype | `docs/bench/RESULTS.md` | **60%** | Fase 0, Jalur A, dan **Gate B3** (paritas 9 kombinasi + `llama-bench` + analisis MoE) selesai; 8 varian & Ollama menyusul |

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
  - [x] **Caveat lingkungan tuntas:** instalasi torch lama korup dan sudah di-reinstall ke `2.9.1+cu128`. Karena rounding bf16 antar-build berubah, seluruh dump referensi `parity_ref_v2` (66M, 163M, 400M) diregenerasi di lingkungan saat ini lalu dipromosikan ke `parity_ref`; P1–P4 dijalankan ulang untuk ketiga model dengan fastpath on dan `LIWET_FASTPATH=0` → **semua `ALL_PASSED`**, dan fastpath-off `P1 max_abs = 0.0`.

---

### Jalur B — Konversi GGUF & Fork `llama.cpp` (`§6`) — **GATE B2 LULUS, GATE B3 TERUKUR (~85%)**
- [x] **B0:** Pin `llama-cpp-python` di `v0.3.36` (`1652066`) dan siapkan `E:\liwet_gguf_build\llama.cpp` branch `liwet-moe` tepat di commit submodule vendor `0c1e57098bba43ac29e6e3b677cdceebdd22334f`.
- [x] **B1 & B2.1:** Buat skrip ekspor `scripts/gguf/export_hf_llama.py` (mendukung `66M` `rms_norm_eps=1e-6`, `163M` `rms_norm_eps=1e-5`, dan `400M` `MixtralForCausalLM` + `e_score_correction.bias` + validasi keamanan `F16`) serta verifier `scripts/gguf/verify_hf_llama_parity.py`.
- [x] **B3 (Patch Kode Fork):** Patch `gguf-py/gguf/constants.py`, `conversion/base.py` (termasuk hash `chkhsh` tokenizer Liwet → `"gpt-2"`), `src/llama-model.cpp`, dan `src/models/llama.cpp` untuk mendukung `blk.%d.exp_probs_b`, `SOFTMAX_WEIGHT`, dan `expert_weights_norm=false`. Editable `gguf-py` sudah terpasang di `tubes_venv`.
- [x] **B2.2:** Jalankan `export_hf_llama.py` & `verify_hf_llama_parity.py` untuk 3 model chat prioritas (**Gate Perantara B2 — LULUS**). Dense `66M`/`163M` lulus ketat (`P1 cos = 1.0`, max abs ≤1.14e-05, P2 32/32). `400M` `MixtralForCausalLM`: ditemukan bahwa referensi memakai **capacity-limited dispatch** (`capacity_factor = 1.25`, ~5–6% assignment dibuang saat prefill) — verifier kini meniru dispatch tersebut; hasil layer-wise MoE **3.05e-05** (≈1e-7 relatif) dengan router **100%**, P2 32/32, dan satu-satunya divergensi end-to-end adalah **router near-tie 2.4e-7** di `chat_role_01` (terdokumentasi di `docs/bench/RESULTS.md`). Config ekspor sekarang menyimpan `moe_capacity_factor`.
- [x] **B4 (Build Lokal Windows CUDA `SM89`):**
  - [x] Kompilasi `llama.cpp` di `E:\liwet_gguf_build\llama.cpp` (branch `liwet-moe`, basis `0c1e57098` + 3 commit lokal: `9196caf89`, `f003973d4`, `cae551692`) — `llama-server.exe`, `llama-cli.exe`, `llama-quantize.exe`, `llama-tokenize.exe`, `llama-bench.exe`, `llama-eval-callback.exe`, CUDA `ARCHS=890`.
  - [x] Konversi 3 model chat ke `BF16`, `F16`, dan `Q8_0` di `E:\liwet_gguf_build\gguf\` (9 file), serta verifikasi `llama-tokenize.exe` identik dengan `tokenizers` pada `prompts.jsonl` (**105/105 prompt exact**, termasuk prompt `chat_role` yang butuh BOS eksplisit).
  - [x] Arahkan `llama-cpp-python` ke fork lokal (`vendor/llama.cpp` disinkronkan dari fork), build & install wheel CUDA `SM89` (`llama_cpp_python-0.3.36-py3-none-win_amd64.whl`, `GGML_CUDA=on`, `CMAKE_CUDA_ARCHITECTURES=89`, `-allow-unsupported-compiler`), GPU offload terverifikasi.
- [x] **B5:** `.github/workflows/build-cpu-wheel.yml` (wheel CPU Linux x86_64 AVX2) sudah ada di fork dan ter-commit di branch `liwet-moe` (`9196caf89`); belum di-push ke remote (sesuai batasan: jangan push ke `main`/remote upstream).
- [x] **Gate B3:** `scripts/parity/compare.py --backend llamacpp` menjalankan `P1`–`P4` (router `P3` via `llama-eval-callback`) untuk 9 kombinasi model × `BF16/F16/Q8_0`, plus sweep `llama-bench` GPU/CPU; seluruh hasil & analisis dicatat di [`docs/bench/RESULTS.md`](bench/RESULTS.md):
  - **Dense `66M`:** `ALL_PASSED` di ketiga quant (P1 cos ≥0.9995, top1 100%, P2 median 128, P4 ≤0.036).
  - **Dense `163M`:** P1/P4 lulus (cos ≥0.9999, TVD ≤0.036); `P2-defrep` median 33.5 (target ≥96) **tidak lulus**, tetapi semua divergensi berada pada celah logit referensi ≤0.0625 (1 bf16 ulp; referensi sendiri punya 9 tie + 9 near-tie pada 768 langkah).
  - **MoE `400M`:** **tidak lulus** (P1 cos 0.9686–0.9729, P3 router 87.9–88.8%, P4 TVD 0.082–0.172) akibat flip router pada near-tie ~0.01 (≈1 bf16 ulp) yang menumpuk dari layer 9 ke atas; bukan bug capacity dispatch (diverifikasi `stable_sort`, hasil identik sebelum/sesudah rebuild, juga gagal terhadap referensi fp32).
  - **Benchmark:** CPU `Q8_0` 103–327 tok/s (target ≥25 ✅, 15–30× Jalur A CPU); GPU dense 1.7–2.7× Jalur A; GPU 400M justru ~3× lebih lambat dari Jalur A (30–34 vs 105 tok/s) → rekomendasi pakai Jalur A untuk 400M GPU.
  - **Keputusan tertunda (butuh pemilik):** ambang paritas khusus MoE (mis. P1 cos ≥0.99 + P3 ≥85%) atau implementasi compute bf16 di `llama.cpp`; rencana juga menyebut P2 `Q8_0` sebagai laporan, sedangkan `compare.py` masih menjadikannya blocker.

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
