# Ollama Modelfiles — varian dense 66M & 163M

Folder ini berisi 8 `Modelfile` untuk varian **dense** Liwet-LLM. Varian **400M MoE tidak
didukung** Ollama standar karena memerlukan dispatch/router dari fork `liwet-moe`.

| Modelfile | Nama model yang dipakai | GGUF (`FROM`) |
|---|---|---|
| `Modelfile-Liwet-66M-id-chat` | `liwet-66m-chat` | `Liwet-66M-id-chat-Q8_0.gguf` |
| `Modelfile-Liwet-66M-id-instruct-v1` | `liwet-66m-instruct-v1` | `Liwet-66M-id-instruct-v1-Q8_0.gguf` |
| `Modelfile-Liwet-66M-id-instruct-v2` | `liwet-66m-instruct-v2` | `Liwet-66M-id-instruct-v2-Q8_0.gguf` |
| `Modelfile-Liwet-66M-id-base-v1` | `liwet-66m-base-v1` | `Liwet-66M-id-base-v1-Q8_0.gguf` |
| `Modelfile-Liwet-66M-id-base-v2` | `liwet-66m-base-v2` | `Liwet-66M-id-base-v2-Q8_0.gguf` |
| `Modelfile-Liwet-163M-id-chat` | `liwet-163m-chat` | `Liwet-163M-id-chat-Q8_0.gguf` |
| `Modelfile-Liwet-163M-id-instruct` | `liwet-163m-instruct` | `Liwet-163M-id-instruct-Q8_0.gguf` |
| `Modelfile-Liwet-163M-id-base` | `liwet-163m-base` | `Liwet-163M-id-base-Q8_0.gguf` |

Path `FROM` masih menunjuk ke `E:\liwet_gguf_build\gguf\...` (lokasi build). Sesuaikan sebelum
`ollama create` bila folder build dipindahkan.

```bash
# contoh (jalankan dari folder ini, atau beri path absolut ke Modelfile)
ollama create liwet-66m-chat -f Modelfile-Liwet-66M-id-chat
ollama run liwet-66m-chat
```

File dibuat oleh `E:\liwet_gguf_build\scratch\gen_modelfiles.py` (membaca `LIWET_CATALOG` dari
`liwet-space/liwet_models.py`) dengan `newline="\n"` — lihat diagnosis CRLF di bawah.

## Diagnosis yang ditemukan

### 1. Ollama gagal start (`Unable to init instance`)

`%LOCALAPPDATA%\Ollama\server.log` dan log `OLLAMA_DEBUG=1` menunjukkan:

```text
llama-server discovery: stopped subprocess after collecting GPU info exit=1
llama-server GPU discovery watchdog timed out ... error="context deadline exceeded"
failure during llama-server GPU discovery ... error="context deadline exceeded"
```

Watchdog discovery GPU habis waktu (±1–6 s, bergantung port/instance) di mesin ini. Ollama lalu
melanjutkan dengan nilai VRAM lama dan **fallback ke CPU**. Server tetap dapat melayani
`/api/chat` dan `/api/generate`; model 66M/163M berjalan di CPU. Bila butuh instance terpisah untuk
eksperimen:

```bash
OLLAMA_DEBUG=1 OLLAMA_HOST=127.0.0.1:11435 \
OLLAMA_MODELS="D:/Fadhil/Pelajaran/Atelier/Ollama-Models" ollama serve
```

Instance default (port 11434) juga dipakai untuk validasi akhir. Ini masalah lingkungan Ollama,
bukan bobot/GGUF.

### 2. CRLF di `Modelfile` → prompt `\r\n`

Generator awal menulis file dengan line ending Windows (CRLF). Akibatnya `TEMPLATE` multi-baris
menghasilkan `\r\n`, sehingga prompt `/api/chat` menjadi 134 karakter (bukan 129) dan output
greedy berbeda dari aplikasi, sementara mode `raw` tetap cocok. Perbaikan: generator menulis
dengan `newline="\n"`; semua `Modelfile` di folder ini **LF-only**. Jangan mengeditnya dengan
editor yang mengubah line ending.

### 3. BOS ganda / hilang

- GGUF menulis `tokenizer.ggml.add_bos_token = false` dan `add_eos_token = false`,
  **tanpa `chat_template`**. Ollama karena itu tidak menambah BOS otomatis.
- `TEMPLATE` untuk 163M (`chat_role`/`raw_bos`) menulis `<bos>` eksplisit sekali di awal.
- Tokenizer 66M tidak memiliki token `<bos>`; `TEMPLATE` 66M tidak menambahkan BOS sama sekali.
- System prompt: 66M chat memakai `{{ $.System }}` di dalam blok user; 163M memakai baris
  `System:` di depan; instruct/base memakai `SYSTEM ""`.

## Validasi greedy 8/8

Metode: `temperature=0`, `repeat_penalty=1.0`, `top_k=0`, `top_p=1.0`, `num_predict=16`, prompt
yang sama dengan backend aplikasi; hasil akhir `/api/chat` (setelah template Modelfile) dibandingkan
dengan `liwet-space-cpp` backend `llamacpp` (`LIWET_QUANT=Q8_0`) untuk 8 varian dense.

Hasil: **8/8 keluaran akhir identik** pada mode greedy. Mode sampling tidak diuji untuk kesetaraan
penuh karena urutan sampler Ollama berbeda.

## Keterbatasan yang harus dicantumkan di README `-GGUF`

1. **Urutan sampler** Ollama tidak dapat diatur persis (mis. top-p dihitung sebelum temperature),
   sehingga output sampling tidak identik dengan aplikasi/llama.cpp. Greedy identik.
2. **NFC**: tokenizer 66M memakai normalisasi NFC di `tokenizers` asli; llama.cpp/Ollama tidak
   menerapkannya. Teks dengan karakter non-NFC dapat menghasilkan token berbeda.
3. **400M MoE tidak didukung** Ollama standar (butuh fork `liwet-moe`).
4. GGUF `add_bos_token=false`, jadi BOS harus disusun pemanggil (sudah ditangani `TEMPLATE`).
