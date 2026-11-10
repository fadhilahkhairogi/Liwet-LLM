# Previous Agent Session Handoff — Liwet-LLM Inference Optimization

> **Waktu Handoff Awal:** `2026-11-10T04:38+07:00`  
> **Diperbarui:** `2026-11-10` — lihat **§5** untuk status terbaru pasca-reinstall torch & Gate Jalur A.  
> **Dokumen Acuan Utama:** [INFERENCE_OPTIMIZATION_PLAN.md](INFERENCE_OPTIMIZATION_PLAN.md)  
> **Ringkasan Checklist Progres:** [PLAN_PROGRESS_SO_FAR.md](PLAN_PROGRESS_SO_FAR.md)  
> **Log Benchmark & Profiling:** [bench/RESULTS.md](bench/RESULTS.md)

Dokumen ini berisi transfer konteks lengkap tanpa kehilangan informasi (*zero-loss handoff*) dari sesi agen sebelumnya agar agen penerus dapat langsung melanjutkan eksekusi mulai dari titik terakhir tanpa mengulang pekerjaan yang sudah selesai.

---

## 1. Konteks Permintaan & Aturan Pengguna (Wajib Dipatuhi)

1. **Tujuan Proyek (`§0`–`§11` di `INFERENCE_OPTIMIZATION_PLAN.md`):**
   - Mempercepat inferensi ke-11 varian model Liwet-LLM di **Lokal GPU** (`RTX 4050 Laptop 6 GB`), **HF Space ZeroGPU / CPU**, dan **Lokal CPU / Ollama** (khusus `66M` & `163M` dense) **tanpa mengubah perilaku apapun** (bobot, arsitektur matematis, tokenizer, `LIWET_CATALOG`, format prompt, parameter sampling default, maupun tata letak UI).
   - **Dua jalur paralel berurutan:**
     - **Jalur A (`perf/torch-fastpath`):** Optimasi PyTorch internal di balik flag `LIWET_FASTPATH` (default `"1"`, `"0"` mengembalikan perilaku eager asli 100%) pada `liwet-space/liwet_models.py` dan `liwet-space/app.py`.
     - **Jalur B (`liwet-space-cpp/` + fork `llama.cpp@liwet-moe`):** Patch minimal `llama.cpp` untuk routing MoE `400M` (`exp_probs_b` + `SOFTMAX_WEIGHT` + `expert_weights_norm=false`), ekspor ke `LlamaForCausalLM` / `MixtralForCausalLM`, konversi GGUF (`BF16`, `F16`, `Q8_0` — **dilarang memakai Q4 sebagai default**), build `llama-server.exe` & wheel `llama-cpp-python` (CUDA `SM89` lokal + Linux AVX2 CPU via GitHub Actions), serta web klon 1:1 `liwet-space-cpp/`.
   - **Target Kecepatan Decode (`400M-A188M`):** Lokal GPU **`≥ 100 tok/s`** (headless, selisih UI vs headless `≤ 20%`), Lokal CPU **`≥ 25 tok/s`**, HF Space CPU **`≥ 15 tok/s`**.

2. **Batasan Mutlak Tanpa Persetujuan Eksplisit Pemilik (`§11` & Global Rules):**
   - **JANGAN** push ke branch `main`.
   - **JANGAN** mengubah atau menghapus repo Hugging Face safetensors asli (`fadhilahkhairogi/Liwet-*`).
   - **JANGAN** mengunggah repo `-GGUF` ke HF Hub atau membuat/deploy HF Space baru sebelum pemilik memberi konfirmasi eksplisit (siapkan secara lokal di folder staging terlebih dahulu).
   - **JANGAN** menambah/menghapus elemen UI apa pun di `liwet-space/` (dan jangan menambah dropdown opsional di `liwet-space-cpp/` tanpa izin pemilik).
   - **JANGAN** menjalankan skrip Administrator Intel DTT (`C:\Users\Fadhilah\Desktop\Matikan_Intel_Throttling.bat` / `Kembalikan_Intel_Throttling.bat`) sendiri — hanya pemilik yang boleh menjalankannya sebagai Administrator.
   - **Jaga ruang Drive `C:` tetap lega:** seluruh artefak besar (dump paritas, clone/build `llama.cpp`, ekspor HF sementara, file `.gguf`, trace profiler) **wajib ditaruh di `E:\liwet_gguf_build\`**, bukan di `C:\`.

---

## 2. Lingkungan Mesin, Path Penting & Status Git

### 2.1 Hardware & Throttling
- **Laptop:** ASUS Vivobook 16 V3607VU — Intel Core 5 210H (8 physical cores, 12 threads) + NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM, Compute Capability `8.9` / `SM89`, driver `610.88`), 24 GB RAM.
- **Status Throttling Saat Ini:** Intel DTT (`SW Power Cap` & `SW Thermal Slowdown`) terpantau **Active** selama run sejauh ini (lihat `docs/bench/gpu_state_baseline.txt` dan `docs/bench/gpu_state_track_a_before.txt`). Sesuai rencana `§10`, selalu catat `nvidia-smi -q -d PERFORMANCE,POWER,CLOCK` di setiap run benchmark.

### 2.2 Toolchain & Interpreter (Sudah Diverifikasi)
- **Python Venv Utama (`tubes_venv`):**
  ```text
  C:\Users\Fadhilah\.gemini\antigravity\brain\4f1ea565-3211-4335-aaeb-951dcae4662b\scratch\tubes_venv\Scripts\python.exe
  ```
  - Berisi: Python `3.12.14`, `torch 2.9.1+cu128` (**reinstall paksa** — instalasi lama korup), `transformers 5.16.1`, `gradio 5.23.3`, `safetensors`, `tokenizers`, `huggingface_hub`, `psutil`, `sentencepiece`, `protobuf`, serta editable `gguf-py` dari `E:\liwet_gguf_build\llama.cpp\gguf-py`.
  - **Riwayat kerusakan torch:** instalasi lama menjadi paket namespace `7.8 MB` tanpa `torch/nn`, `torch/lib`, dan `__init__.py` (`ModuleNotFoundError: torch.nn`), dengan `RECORD` dist-info dari wheel CPU-only. Diperbaiki dengan `pip install --no-cache-dir --force-reinstall "torch==2.9.1+cu128" --index-url https://download.pytorch.org/whl/cu128` (wheel 2.86 GB, TMP/TEMP/TMPDIR diarahkan ke `E:`). Efek samping: `torchvision` hilang; `starlette` turun ke `0.47.3` saat instalasi gradio. CUDA RTX 4050 terverifikasi jalan.
- **MSVC + CMake + Ninja + CUDA (untuk build Jalur B):**
  - `vcvars64.bat`: `C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat`
  - CUDA Toolkit: `v12.9` (`nvcc.exe` tersedia di sistem; target `-DCMAKE_CUDA_ARCHITECTURES=89`)
  - `cmake` dan `ninja` tersedia. Catatan: CLI `gh` tidak terpasang, gunakan `git` standar.

### 2.3 Lokasi Direktori & Repo
| Komponen | Lokasi Absolut | Branch / Commit Saat Ini |
|---|---|---|
| Repo utama `Liwet-LLM` | `d:\Fadhil\Pelajaran\Kuliah\Semester_7\GENAI\Liwet-LLM` | Branch `perf/torch-fastpath` (base `ab9907b`) |
| Safetensors lokal (11 model) | `d:\Fadhil\Pelajaran\Kuliah\Semester_7\GENAI\TUBES_1_SAFETENSOR` | Read-only (semua 11 varian lengkap) |
| Dump referensi paritas | `E:\liwet_gguf_build\parity_ref\` | **Lengkap & terbaru** untuk 3 model chat (diregenerasi di torch `2.9.1+cu128`, `parity_ref_v2` sudah dipromosikan). Backup lingkungan lama: `parity_ref_backup_20261110_0520` |
| Trace profiler Fase 0 | `E:\liwet_gguf_build\traces\trace_Liwet-400M-A188M-id-chat.json` | Selesai (`79.61 MB`) |
| Fork `llama.cpp` | `E:\liwet_gguf_build\llama.cpp` | Branch `liwet-moe` di atas commit `0c1e57098bba43ac29e6e3b677cdceebdd22334f` (patch belum di-commit, belum di-build) |
| Clone `llama-cpp-python` | `E:\liwet_gguf_build\llama-cpp-python` | Tag `v0.3.36` (`1652066`), submodule `vendor/llama.cpp` menunjuk tepat ke `0c1e57098bba43ac29e6e3b677cdceebdd22334f` |

---

## 3. Apa Saja yang Sudah Dikerjakan & Temuan Teknis Kritis

### 3.1 Fase 0 — Baseline, Profiler & Parity Reference Dump (100% SELESAI)
1. **Benchmark Baseline (`docs/bench/baseline_torch_gpu.json`, `docs/bench/baseline_torch_cpu.json`, `docs/bench/gradio_overhead.json`)** telah selesai dan dicatat lengkap di [bench/RESULTS.md](bench/RESULTS.md):
   - `Liwet-400M-A188M-id-chat` GPU (`bf16`): **`1.73 tok/s`** pure/fixed, **`1.05 tok/s`** natural; CPU (`fp32`): **`0.25 tok/s`**.
   - `Liwet-163M-id-chat` GPU (`bf16`): **`5.07 tok/s`** pure, **`4.20 tok/s`** fixed; CPU (`fp32`): **`0.77 tok/s`**.
   - `Liwet-66M-id-chat` GPU (`bf16`): **`3.10 tok/s`** pure, **`2.86 tok/s`** fixed; CPU (`fp32`): **`1.01 tok/s`**.
2. **Diagnosis `torch.profiler` (`docs/bench/profile_Liwet-400M-A188M-id-chat.json`):**
   - Eager loop me-launch **`1,319` kernel CUDA per token**, membuat GPU idle **`89.72%`** dan tertahan di clock rendah (`288–713 MHz`).
   - Di dalam waktu aktif GPU, **`27%` (`555.62 ms` / 20 tok)** habis di `aten::index` (`self.gate_up_proj[experts]` dan `self.down_proj[experts]` yang menyalin bobot 2 expert ke buffer baru setiap layer setiap token!).
   - Di level `app.py`, `_is_repo_available(..., force_refresh=True)` menambah latensi jaringan **`0.629 s`** per pesan `400M`, dan `chat_send` me-*yield* seluruh riwayat di setiap token (`1.2 tok/s`).
3. **Harness Paritas & Dump Referensi (`scripts/parity/`):**
   - `scripts/parity/reference/liwet_models_ref.py`: salinan murni `liwet_models.py` sebelum disentuh.
   - `scripts/parity/prompts.jsonl`: >30 prompt per `prompt_mode` (`raw`, `raw_bos`, `instruct_plain`, `chat_tags`, `chat_role`), mencakup prompt pendek, multi-turn, non-ASCII/emoji, dan konteks panjang (`1024`/`2048`).
   - `scripts/parity/reference_dump.py`: telah selesai men-generate `meta.json`, `prefill_bf16_cuda.npz`, `prefill_fp32_cpu.npz`, `greedy_no_rep.json`, `greedy_default_rep.json`, dan `sampling_p4.npz` di `E:\liwet_gguf_build\parity_ref\` untuk ke-3 model chat prioritas.
   - `scripts/parity/compare.py`: sudah diperbaiki agar urutan prompt mengikuti urutan stratified di file referensi (`greedy_ids_order`, `prefill_ids_order`, `p4_ids_order`) dan melepas `forward_pre_hook` MoE sebelum capture CUDA Graph pada `run_greedy`.

---

### 3.2 Jalur A — Implementasi Awal, Hasil Run Pertama & Diagnosis Bug Paritas/Kecepatan (PENTING!)

Kode Jalur A sudah ditulis di `liwet-space/liwet_models.py`, `liwet-space/app.py`, dan `liwet-space/requirements.txt` (`torch>=2.5.0`), lalu baru saja diuji melalui `compare.py` dan `bench_torch.py` (`task-1610`).

#### A. Lompatan Kecepatan CUDA Awal (`docs/bench/track_a_cuda.json`, kondisi DTT Throttling Aktif)
| Model | Baseline Pure (tok/s) | Jalur A v1 Pure (tok/s) | Jalur A v1 `stream_generate` Fixed (tok/s) | Peak VRAM (MB) | Avg SM Clock (MHz) |
|---|---:|---:|---:|---:|---:|
| `Liwet-400M-A188M-id-chat` | 1.73 | **85.82** (`p50 = 11.41 ms`) | **58.16** | 1221.4 | 2295.0 |
| `Liwet-163M-id-chat` | 5.07 | **204.16** (`p50 = 4.76 ms`) | **85.76** (natural `110.19`) | 564.9 | 2490.0 |
| `Liwet-66M-id-chat` | 3.10 | **369.89** (`p50 = 2.31 ms`) | **292.41** | 213.2 | 1087.5–2490.0 |

#### B. Hasil Uji Paritas Awal (`docs/bench/parity_track_a_cuda_*.json`) & Akar Masalah yang Harus Diperbaiki Agen Berikutnya

> ⚠️ **KOREKSI (sesi lanjutan):** Diagnosis "penyebab pasti" di bawah **ternyata salah**. `torch.set_float32_matmul_precision("high")`, `allow_tf32=True`, dan fused QKV **semuanya juga ada di referensi pristine** `liwet_models_ref.py`, sehingga bukan penyebab divergensi. Akar masalah sebenarnya adalah **`F.scaled_dot_product_attention(..., enable_gqa=True)`**, yang memilih kernel/reduksi berbeda dari `repeat_interleave` milik referensi. Setelah fastpath memakai `repeat_interleave` (default; `LIWET_SDPA_GQA=1` untuk opt-in) dan tie-break pemilihan token disamakan ke `torch.argmax`, `P1`/`P3` dan `P2` lulus 100%. Jangan mengubah `allow_tf32`/`matmul_precision`/fused QKV hanya demi paritas — itu justru menjauhkan dari referensi. Lihat **§5** untuk hasil final.
1. **Masalah Paritas #1 — `400M-A188M-id-chat` gagal `P1` & `P3` (`min_cosine = 0.958807`, `P3_router = 89.554%`):**
   - **Penyebab pasti di `liwet-space/liwet_models.py`:**
     1. Baris 27 menambahkan `torch.set_float32_matmul_precision("high")` dan baris 1661/2318 menambahkan `torch.backends.cuda.matmul.allow_tf32 = True`! Padahal di referensi (`liwet_models_ref.py`), keduanya default (`"highest"` / `allow_tf32 = False`). Pada `MoE.forward`, operasi float32 / matmul terpengaruh TF32.
     2. Pada `MoEAttention400M` (`forward` & `forward_prefill_static`), `q_proj`, `k_proj`, dan `v_proj` digabung menjadi satu `_fused_qkv_weight()` (`torch.cat([q, k, v], dim=0)`), sedangkan di `liwet_models_ref.py` ketiganya dihitung dengan 3 panggilan `Linear` terpisah, ditambah perubahan GQA dari `repeat_interleave` ke `F.scaled_dot_product_attention(..., enable_gqa=True)`.
     3. Karena routing top-2 MoE di 32 layer bersifat diskontinu (`torch.topk(router_logits + expert_bias, 2)`), pergeseran pembulatan `bfloat16`/TF32 sebesar `~1e-3` di layer awal membalik pilihan expert pada ~10% token-layer yang selisih logit routernya tipis (`P3 = 89.554%`), sehingga di layer ke-32 logit bergeser menjadi `min_cosine = 0.9588`.
   - **Solusi untuk Agen Berikutnya:**
     - Matikan `allow_tf32 = True` dan `torch.set_float32_matmul_precision("high")` (kembalikan ke default PyTorch agar identik dengan referensi).
     - Uji ablasi cepat (10 detik) pada `400M` prefill: gunakan `q_proj`, `k_proj`, `v_proj` terpisah saat prefill (atau cek apakah `enable_gqa=True` vs `repeat_interleave` yang menggeser router `bf16`). Pastikan `P1` (`min_cosine ≥ 0.9999`) dan `P3` (`≥ 99.5%`) lulus 100%!

2. **Masalah Paritas #2 — `163M` & `66M` lulus `P1` (`min_cosine = 0.999954` & `0.999996`, `top1 = 100%`) dan `P4` (`0.036` & `0.0`), serta `non_tie_divergences = []`, tetapi `P2` gagal karena `exact_128_pct` hanya `50%–66.7%`:**
   - **Penyebab pasti:** Di `forward_decode_static`, attention dijalankan di atas panjang statis penuh `max_seq_len` (`2048` / `1024`) menggunakan **mask aditif** `attn_mask` berisi `0` dan `-inf` (`is_causal=False`), sedangkan referensi memakai slice dinamis `k[:, :, :cur_len, :]` dengan `attn_mask=None`. Di PyTorch SDPA, mask aditif float memilih kernel/reduksi softmax berbeda dari `attn_mask=None`, sehingga ketika ada 1 langkah dari 128 langkah yang kebetulan *near-tie* (`logit_gap < 0.05`, terbukti `non_tie_divergences: []` kosong!), langkah itu memilih token ke-2 dan sisa kalimat setelah langkah itu otomatis berbeda.
   - **Catatan penting:** Cek mengapa pada `66M` ada `min_first_div: 0` sementara `P1 top1_match_pct: 100.0%`! Pada `66M` `forward_prefill_static` baris 744: `is_causal=(seq_len > 1)` vs `MiniCausalSelfAttention.forward` asli — cek prompt mana di `greedy_no_rep.json` yang diverge di step 0 (kemungkinan panjang prompt > `block_size` atau selisih <0.05 di prefill). Periksa langsung terhadap `greedy_no_rep.json`!

3. **Cara Menembus Gate Kecepatan `400M ≥ 100 tok/s` (saat ini `85.82 tok/s` pure / `58.16 tok/s` `stream_generate`):**
   - **Untuk Pure Decode (`85.82 → >110+ tok/s`):** Di `MoE.forward` (`liwet_models.py` baris 1140–1152), cabang `num_tokens == 1` **masih menyalin bobot** `selected_gate_up = self.gate_up_proj[experts]` dan `selected_down = self.down_proj[experts]` (`aten::index` menyalin ~4.4 MB × 32 layer = ~140 MB per token!). Ganti bagian tersebut dengan kernel tanpa salin bobot (mis. `torch._grouped_mm` jika tersedia di `torch 2.13`, atau kernel Triton `triton-windows` ringan untuk indexed MoE GEMV `[1, H] @ W[experts]`, atau `torch.compile(mode="reduce-overhead")`).
   - **Untuk Menutup Gap `stream_generate` (`85.82 → 58.16 tok/s`):**
     - Di `stream_generate` (`liwet_models.py` baris 2116–2193), setiap token masih menjalankan `_apply_rep_penalty_mask`, `_sample_next_token`, `.item()`, **dan `bundle.tokenizer.decode(...)` di setiap langkah token** meskipun UI hanya me-yield tiap 8 token / 50 ms!
     - Selain itu, di `bench_gradio_overhead.py` baris 116, `api_name="/chat_send"` mengembalikan `sse_events: 0` pada Gradio `6.31.0` — periksa nama endpoint aktual dengan `client.view_api()` saat mengukur selisih UI vs headless (`≤ 20%`).

---

### 3.3 Jalur B — Patch Fork `llama.cpp` & Ekspor HF-Llama/Mixtral (Kode Siap, Belum Dijalankan/Dikompilasi)
1. **Patch Fork `E:\liwet_gguf_build\llama.cpp` (branch `liwet-moe`, base `0c1e57098`):**
   - `gguf-py/gguf/constants.py`: Menambahkan `MODEL_TENSOR.FFN_EXP_PROBS_B` ke `MODEL_TENSORS[MODEL_ARCH.LLAMA]` dan `SOFTMAX_WEIGHT = 3` ke `ExpertGatingFuncType` (cocok dengan `LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT = 3` di C++).
   - `conversion/base.py`: Menangani `expert_selection_fn == "softmax_weight"`, menulis `expert_weights_norm` & `expert_weights_scale` ke GGUF KV, dan memetakan hash pre-tokenizer `chkhsh` Liwet (`f2beda9d...` untuk 163M/400M dan `86f3edd3...` untuk 66M) ke `"gpt-2"`.
   - `src/models/llama.cpp` & `src/llama-model.cpp`: Membaca KV opsional `expert_weights_scale`, `expert_weights_norm`, `expert_gating_func` (default tetap `true` + `SOFTMAX` agar Mixtral standar tidak berubah), memuat tensor opsional `blk.%d.exp_probs_b`, dan meneruskannya ke `build_moe_ffn(...)`.
   - **Sudah diverifikasi di `src/llama-graph.cpp:2038-2148`:** Saat `SOFTMAX_WEIGHT` dan `norm_w=false`, `build_moe_ffn` menghitung `probs = logits` → `selection_probs = probs + exp_probs_b` → `selected_experts = argsort_top_k(selection_probs, k)` → `weights = get_rows(probs, selected_experts)` → `weights = soft_max(weights)`. Ini **ekuivalen matematis 100%** dengan `MoE.forward` Liwet!
   - **Catatan Verifikasi Saat Konversi GGUF Nanti (`B2.3`):** Di `conversion/llama.py:135-143`, `LlamaModel.set_vocab()` mencoba `_set_vocab_sentencepiece()` lalu `_set_vocab_llama_hf()` sebelum `_set_vocab_gpt2()`. Pastikan untuk folder ekspor Liwet (yang hanya punya `tokenizer.json` ByteLevel BPE tanpa `tokenizer.model`), eksekusi jatuh dengan benar ke `_set_vocab_gpt2()` dan bukan `_set_vocab_llama_hf()`! Jika `_set_vocab_llama_hf()` tidak melempar `TypeError`/`FileNotFoundError`, tambahkan guard kecil di `conversion/llama.py` agar ByteLevel BPE memakai `_set_vocab_gpt2()`.
2. **Skrip Ekspor & Verifikasi Perantara (`scripts/gguf/`):**
   - [scripts/gguf/export_hf_llama.py](../scripts/gguf/export_hf_llama.py): Mengekspor `66M` (`rms_norm_eps=1e-6`) & `163M` (`rms_norm_eps=1e-5`) ke `LlamaForCausalLM`, dan `400M` ke `MixtralForCausalLM` + `e_score_correction.bias` (f32) + pemeriksaan keamanan `F16` (`max|w| < 65504`).
   - [scripts/gguf/verify_hf_llama_parity.py](../scripts/gguf/verify_hf_llama_parity.py): Memverifikasi hasil ekspor dengan `transformers` (`LlamaForCausalLM` & `MixtralForCausalLM` dengan router Liwet) terhadap `prefill_fp32_cpu.npz` (`P1 cosine ≥ 0.99999`, `P2`, `P3`). **Belum dijalankan** — siap langsung dieksekusi begitu Gate Jalur A tuntas.

---

## 4. Urutan Langkah Eksak untuk Agen Berikutnya

Ikuti urutan Gate ketat sesuai [INFERENCE_OPTIMIZATION_PLAN.md](INFERENCE_OPTIMIZATION_PLAN.md):

1. **Tuntaskan Jalur A & Luluskan Gate A (`perf/torch-fastpath`):**
   - Perbaiki isu paritas `400M` (`allow_tf32`, `_fused_qkv_weight` / GQA) dan periksa tie divergence `P2` pada `163M`/`66M`.
   - Optimasi `MoE.forward` (`num_tokens == 1`) agar tidak menyalin `gate_up_proj[experts]` / `down_proj[experts]`, dan pangkas overhead per-token di `stream_generate` hingga `400M` GPU decode headless **`≥ 100 tok/s`** dan gap UI vs headless **`≤ 20%`**.
   - Verifikasi `LIWET_FASTPATH=0` menghasilkan output identik 100% dengan `liwet_models_ref.py`, jalankan benchmark CPU (`bench_torch.py --device cpu`), jalankan checklist paritas UI `P5`, catat semua tabel Jalur A ke `docs/bench/RESULTS.md`, lalu commit perubahan Jalur A di branch `perf/torch-fastpath` (pisahkan file `scripts/gguf/` untuk commit Jalur B bila perlu).
2. **Jalankan Gate B2 (Ekspor HF-Llama/Mixtral & Verifikasi Paritas Perantara):**
   - Jalankan `export_hf_llama.py` ke `E:\liwet_gguf_build\hf_llama_tmp\` untuk 3 model chat prioritas, lalu jalankan `verify_hf_llama_parity.py` hingga `PASSED` (`cosine ≥ 0.99999`).
3. **Build `llama.cpp` Fork, Konversi GGUF, Build `llama-cpp-python`, & Luluskan Gate B3 (`B3`–`B5`):**
   - Kompilasi `E:\liwet_gguf_build\llama.cpp` dengan `vcvars64.bat` + CMake + Ninja + CUDA `SM89`.
   - Konversi 3 model chat ke `E:\liwet_gguf_build\gguf\<Repo>\` (`BF16`, `F16`, `Q8_0`), verifikasi `llama-tokenize.exe` terhadap `prompts.jsonl`.
   - Tambahkan `.github/workflows/build-cpu-wheel.yml` di fork `llama.cpp` (untuk build wheel CPU Linux x86_64 AVX2 `llama-cpp-python`) dan commit di branch `liwet-moe`.
   - Arahkan `E:\liwet_gguf_build\llama-cpp-python\vendor\llama.cpp` ke fork lokal, build & install wheel CUDA `SM89` ke `tubes_venv`.
   - Perluas `scripts/parity/compare.py` untuk backend `llamacpp` (`P1`, `P2`, `P3` via `llama-eval-callback.exe`, `P4`), jalankan untuk `BF16`/`F16`/`Q8_0` (GPU & CPU), dan ukur throughput dengan `llama-bench.exe`.
4. **Bangun Web Klon 1:1 `liwet-space-cpp/` (`§7`) & Luluskan Gate Jalur B:**
   - Buat `app.py`, `liwet_backend.py` (single-source dari `liwet-space/liwet_models.py`), `backends/{torch_backend,llamacpp_inproc,llamacpp_server}.py`, `gguf_catalog.py`, `requirements.txt`, `README.md`, dan `scripts/run_local_servers.ps1`.
   - Verifikasi paritas UI `P5` berdampingan (simpan screenshot tiap tab di `docs/bench/ui/`) dan pastikan target kecepatan tercapai (`400M` GPU `≥ 100 tok/s`, CPU lokal `≥ 25 tok/s`).
5. **Konversi 8 Varian Sisa, Staging 11 Repo `-GGUF`, Bonus Ollama (`§8`), & Laporan Akhir:**
   - Ekspor & konversi ke-8 model sisanya (`BF16`, `F16`, `Q8_0`), jalankan P1/P2 & benchmark ringkas.
   - Siapkan 11 folder staging repo `-GGUF` lengkap (`README.md`, `tokenizer.json`, GGUF, dan `Modelfile` untuk model dense) di `E:\liwet_gguf_build\hf_gguf_staging\`.
   - Diagnosa error startup Ollama lokal (`%LOCALAPPDATA%\Ollama\server.log`), buat `scripts/ollama/` berisi `Modelfile` untuk varian dense (`66M`, `163M`).
   - Lengkapi tabel akhir di `docs/bench/RESULTS.md`, lalu **berhenti dan minta konfirmasi pemilik** sebelum melakukan upload HF Hub / push remote / eksekusi skrip Admin `.bat`.

---

## 5. Pembaruan Sesi Lanjutan — Gate Jalur A Tercapai (2026-11-10)

### 5.1 Ringkasan
Gate A **terpenuhi** di lingkungan torch `2.9.1+cu128` (DTT throttle masih aktif), dengan satu caveat verifikasi paritas 163M/400M yang bergantung pada drive `E:`.

| Gate | Target | Hasil |
|---|---|---|
| `400M` GPU headless pure decode | ≥100 tok/s | **105.28 tok/s** (p50 9.42 ms, p95 9.81 ms) |
| `stream_generate` overhead | ≤20% | 400M **+16.7%**, 163M **+11.3%**, 66M **+6.2%** |
| UI vs headless | ≤20% | direct `chat_send` **+1.2%** (93.7 vs 92.41 tok/s); HTTP/SSE server-side 80.8 tok/s |
| Paritas P1–P4 | lihat plan | `ALL_PASSED` ketiga model (detail di `RESULTS.md`) |
| `LIWET_FASTPATH=0` | identik perilaku lama | P1 `max_abs 0.0`, P2 100% |
| P5 UI checklist | tidak ada elemen UI berubah | diff `app.py` hanya handler/warmup/throttle |

### 5.2 Angka Kunci (headless, post-reinstall)
- **CUDA `bf16`:** `400M` pure **105.28** / fixed **90.18** / natural **92.41** tok/s, prefill {32: 263.6, 512: 3975.6, 1500: 12853.0}, VRAM 1026.8 MB, avg SM 2370 MHz, 24.34 W. `163M` 204.43 / 183.68 / 183.31, prefill {1373.1, 21690.7, 38833.1}, 425.1 MB, 1312.5 MHz. `66M` 384.21 / 361.87 / 373.33, prefill {1328.5, 19499.8, 39703.7}, 215.8 MB, 1275 MHz.
- **CPU `fp32` (16 token):** `66M` 48.46 / 36.70 / 37.39 tok/s; `163M` 7.22 / 6.19 / 4.93; `400M` 3.42 / 3.28 / 3.43. Target CPU ≥25 tok/s hanya tercapai di 66M; 163M/400M diserahkan ke Jalur B (GGUF).
- **Overhead app:** `_is_repo_available(force_refresh)` turun `0.629 → 0.362 s`; `chat_send` direct wall `1.375 s` untuk 128 token; HTTP/SSE `3.355 s` untuk 98 token (biaya SSE ~100 ms/event, bukan model).

### 5.3 Temuan Lingkungan yang Wajib Diingat
1. **Instalasi torch lama korup & sudah di-reinstall.** Cross-build bf16 rounding berubah, sehingga dump referensi lama **tidak lagi cocok numerik** (66M: divergensi pertama di langkah 9/36, **semua tie**, `non_tie_divergences: []`). Direproduksi identik dengan `LIWET_FASTPATH=0` → bukan bug fastpath.
2. **Seluruh dump referensi sudah diregenerasi di lingkungan baru** (`66M`, `163M`, `400M`) dan dipromosikan ke `parity_ref`. P1–P4 dijalankan ulang untuk ketiga model, fastpath on **dan** `LIWET_FASTPATH=0` → semuanya `ALL_PASSED`; fastpath-off `P1 max_abs_diff = 0.0` untuk ketiganya.
3. **Drive `E:` sempat hilang di tengah sesi** (hanya `C:` & `D:` ter-mount) lalu tersambung kembali; dump 400M dilengkapi setelahnya. Artefak besar tetap tidak ditulis ke `C:`.
4. **Bench overhead Gradio perlu warmup 1 generasi** (`bench_gradio_overhead.py` sudah diedit). Tanpa warmup, CUDA-graph capture/cold kernel masuk wall tapi tidak masuk `model_seconds` → overhead palsu 67%.

### 5.4 Titik Lanjut Pertama Agen Berikutnya
1. ✅ **Selesai:** dump 400M dilengkapi, `parity_ref_v2` dipromosikan ke `parity_ref`, dan P1–P4 dijalankan ulang untuk 3 model (fastpath on & `LIWET_FASTPATH=0`). Artefak `docs/bench/parity_*` sudah ditimpa hasil lingkungan saat ini.
2. ✅ Commit Jalur A sudah ada di `perf/torch-fastpath` (`b6d141a`, tanpa push), ditambah commit verifikasi paritas ulang.
3. **Langkah berikutnya:** Gate B2 (`export_hf_llama.py` → `E:\liwet_gguf_build\hf_llama_tmp\`, `verify_hf_llama_parity.py`), lalu B4/B5 dan Gate B3 sesuai urutan §4.
4. Jangan lupa batasan pemilik: tanpa push `main`, tanpa upload `-GGUF`/Space, tanpa mengubah UI, tanpa menjalankan `.bat` Admin, dan tanpa artefak besar di `C:`.
