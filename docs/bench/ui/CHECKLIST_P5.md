# Checklist P5 — Paritas UI `liwet-space` vs `liwet-space-cpp`

Tanggal: 2026-11-10

Rencana (`docs/INFERENCE_OPTIMIZATION_PLAN.md` §5, baris P5) mewajibkan:
> checklist semua tab, kontrol, preset, label, CSS, telemetri — 100% item sama.

Kedua aplikasi dijalankan berdampingan secara headless (Playwright Chromium,
1600×1000), lalu setiap tab diambil screenshot `full_page` dan inventaris
elemennya dibandingkan secara terstruktur.

| Aplikasi | Backend | Port | Cara jalan |
|---|---|---|---|
| `liwet-space` (lama) | PyTorch safetensors | 7871 | `LIWET_LOCAL_MODELS_DIR=TUBES_1_SAFETENSOR` |
| `liwet-space-cpp` (klon) | GGUF via `llama-cpp-python` | 7872 | `LIWET_BACKEND=llamacpp`, `LIWET_GGUF_DIR=E:\liwet_gguf_build\gguf`, `LIWET_QUANT=Q8_0`, `LIWET_GGUF_NGL=0` |

Skrip reproduksi: `E:\liwet_gguf_build\scratch\ui_shots.py`
(`PLAYWRIGHT_BROWSERS_PATH=E:\liwet_gguf_build\ms-playwright`).

## Hasil inventaris

`ui_inventory_diff.json` — semua kategori identik:

```json
{
  "tabs":    {"identical": true, "only_old": [], "only_cpp": []},
  "buttons": {"identical": true, "only_old": [], "only_cpp": []},
  "labels":  {"identical": true, "only_old": [], "only_cpp": []},
  "headers": {"identical": true, "only_old": [], "only_cpp": []},
  "texts":   {"identical": true, "only_old": [], "only_cpp": []}
}
```

| Butir checklist | Bukti | Status |
|---|---|---|
| Tab | `<div role="tab">`: Percakapan, Arena Perbandingan, Uji Parameter — urutan & teks sama | ✅ |
| Kontrol | seluruh `<button>` sama (teks, urutan) | ✅ |
| Preset | tidak ada dropdown/preset tambahan di klon; katalog & prompt builder di-import dari `liwet-space/liwet_models.py` (single source) | ✅ |
| Label | seluruh `<label>` sama | ✅ |
| Header | seluruh `h1`–`h4` sama | ✅ |
| Input/placeholder | seluruh `textarea`/`input` (`aria-label`/placeholder) sama | ✅ |
| CSS | `app.py` identik modulo satu baris import (lihat bawah); CSS ada di file yang sama (`CUSTOM_CSS`) | ✅ |
| Telemetri | pill status "11 Model Live" ada di kedua versi (lihat screenshot tab 1) | ✅ |
| Screenshot berdampingan | 3 tab × 2 aplikasi, nama sama per tab | ✅ |

## Diff `app.py`

```
$ diff liwet-space/app.py liwet-space-cpp/app.py
34c34
< from liwet_models import (
---
> from liwet_backend import (
```

Satu-satunya perbedaan konten adalah modul backend. Tidak ada elemen,
handler, CSS, preset, atau label UI yang ditambah/dihapus. (Sebelum
normalisasi, file klon memakai CRLF sehingga `diff` mentah menampilkan
seluruh file berbeda; file sudah dinormalisasi ke LF — Python tidak terpengaruh,
tetapi LF membuat diff dapat diaudit.)

## Screenshot

Ketiga pasangan kini **byte-identical** (sha256):

| Tab | File (old / cpp) | sha256 |
|---|---|---|
| Percakapan | `old_1_Percakapan.png` / `cpp_1_Percakapan.png` | `5e558071b0fe4911d3d14337634a34ad8992dba1562ddc70735c6601f892e9b2` |
| Arena Perbandingan | `old_2_Arena_Perbandingan.png` / `cpp_2_Arena_Perbandingan.png` | `ca10479604375db635e69e0c587bdb63c4d56e65b7b5ac378b2d72aee959628d` |
| Uji Parameter | `old_3_Uji_Parameter.png` / `cpp_3_Uji_Parameter.png` | `6935d229d6d64b2f1142e4d3146d20df0ddc81edfd3572aca91ee7974b05f978` |

Catatan: pada percobaan pertama, tab Percakapan berbeda hanya di area hero
(warna lebih pucat). Diferensial piksel menunjukkan hero lama ter-blend putih
α≈0.34, yaitu opacity elemen ~0.66 — Gradio menangkap komponen sedang
*pending update* (fade). Skrip ditambah penantian sampai produk opacity
hero **dan seluruh ancestor-nya** > 0.999 sebelum screenshot; setelah itu
ketiga tab identik byte-per-byte.

## Verdict

**P5 LULUS** — tidak ada elemen UI yang ditambah/dihapus; CSS dan telemetri
identik; ketiga screenshot berdampingan identik.
