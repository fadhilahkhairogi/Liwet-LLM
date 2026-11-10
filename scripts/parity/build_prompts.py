"""Deterministically build scripts/parity/prompts.jsonl with >=30 prompts per prompt_mode."""
from __future__ import annotations

import json
from pathlib import Path

OUT_PATH = Path(__file__).resolve().parent / "prompts.jsonl"

BASE_PROMPTS_ID = [
    "Jelaskan secara singkat apa itu fotosintesis pada tumbuhan.",
    "Apa ibu kota provinsi Jawa Barat dan sebutkan tiga makanan khasnya?",
    "Buatlah pantun jenaka dua bait tentang mahasiswa yang begadang mengerjakan tugas.",
    "Mengapa langit berwarna biru pada siang hari namun jingga saat senja?",
    "Tuliskan tiga langkah praktis menghemat baterai laptop saat kuliah.",
    "Apa perbedaan utama antara RAM dan SSD pada komputer modern?",
    "Berikan ringkasan sejarah singkat berdirinya Kota Bandung.",
    "Bagaimana cara membuat nasi liwet Sunda yang gurih di rice cooker?",
    "Jelaskan konsep dasar jaringan saraf tiruan dengan bahasa sederhana.",
    "Sebutkan lima pulau terbesar di Indonesia beserta satu ciri khasnya.",
    "Apa manfaat utama menggunakan version control seperti Git dalam tim developer?",
    "Tuliskan contoh email singkat berbahasa Indonesia yang sopan untuk izin tidak masuk kuliah.",
    "Bandingkan kelebihan dan kekurangan transportasi kereta cepat dan pesawat jarak menengah.",
    "Jelaskan mengapa air laut terasa asin sementara air sungai umumnya tawar.",
    "Apa arti peribahasa 'berat sama dipikul, ringan sama dijinjing'?",
    "Berikan lima tips menjaga fokus belajar selama ujian akhir semester.",
    "Bagaimana proses terjadinya hujan secara singkat dan jelas?",
    "Tuliskan dialog singkat antara dua teman yang merencanakan liburan ke Yogyakarta.",
    "Apa yang dimaksud dengan inflasi dan bagaimana dampaknya terhadap daya beli?",
    "Sebutkan perbedaan protokol HTTP dan HTTPS dalam keamanan web.",
    "Jelaskan fungsi utama hati dalam sistem metabolisme tubuh manusia.",
    "Mengapa kopi dapat membantu mengurangi rasa kantuk sementara waktu?",
    "Berikan ide judul penelitian skripsi di bidang kecerdasan buatan berbahasa Indonesia.",
    "Apa peran hutan mangrove bagi ekosistem pesisir Indonesia?",
    "Tuliskan puisi pendek empat baris bertema hujan sore di kampus.",
    "Jelaskan secara ringkas perbedaan antara CPU dan GPU untuk komputasi AI.",
    # Non-ASCII, punctuation, code, emoji edge cases:
    "Ubah kalimat berikut ke bahasa baku: 'Gue lagi otw ke kampus nih, macet parah 😅🔥!'",
    "Apa makna simbol matematika ∑, ∫, dan ≈ serta kata serapan 'résumé' / 'naïve'?",
    "Perbaiki bug kecil fungsi Python berikut:\n```python\ndef bagi(a, b):\n    return a / b\n```\nTambahkan penanganan pembagian nol!",
    "Terjemahkan ke bahasa Inggris formal: 'Terima kasih atas kerja samanya selama proyek ini berlangsung 😊🙏.'",
]

MULTI_TURN_PAIRS = [
    [
        {"role": "user", "content": "Halo! Kamu bisa bantu saya belajar struktur data?"},
        {"role": "assistant", "content": "Tentu! Struktur data apa yang sedang kamu pelajari?"},
        {"role": "user", "content": "Jelaskan perbedaan stack dan queue beserta contoh penggunaannya."},
    ],
    [
        {"role": "user", "content": "Saya mau liburan ke Bandung akhir pekan ini."},
        {"role": "assistant", "content": "Menarik! Apakah kamu lebih suka wisata alam, kuliner, atau sejarah?"},
        {"role": "user", "content": "Kuliner dan alam, kasih 3 rekomendasi tempat yang berdekatan."},
    ],
    [
        {"role": "user", "content": "Apa itu Attention dalam arsitektur Transformer?"},
        {"role": "assistant", "content": "Attention adalah mekanisme yang memberi bobot keterkaitan antar token dalam urutan."},
        {"role": "user", "content": "Lalu apa bedanya dengan Grouped-Query Attention (GQA)?"},
    ],
]

# Long context chunk (~1600+ tokens when repeated) for 2048-ctx stress tests
LONG_PARAGRAPH = (
    "Pengembangan model bahasa berskala ringkas berbahasa Indonesia menuntut keseimbangan antara "
    "kualitas korpus pralatih, efisiensi arsitektur seperti Grouped-Query Attention dan Mixture of Experts, "
    "serta strategi penyelarasan instruksi yang cermat agar tetap responsif pada perangkat keras konsumen. "
    "Dalam proses evaluasi lokal, panjang konteks yang mendekati batas maksimum membantu menguji konsistensi "
    "rotary positional embedding, pengelolaan KV-cache, dan kestabilan distribusi logit di berbagai lapisan. "
)
LONG_PROMPT_2048 = (
    "Bacalah teks referensi berikut dengan saksama, lalu rangkum tiga poin utamanya dalam dua kalimat:\n\n"
    + (LONG_PARAGRAPH * 28)
    + "\n\nRingkasan singkat:"
)
LONG_PROMPT_1024 = (
    "Ringkas teks berikut dalam satu kalimat:\n\n"
    + (LONG_PARAGRAPH * 11)
    + "\n\nRingkasan:"
)

MODES = ["raw", "raw_bos", "instruct_plain", "chat_tags", "chat_role"]
DEFAULT_SYS = "Anda adalah asisten AI berbahasa Indonesia yang membantu, ramah, dan ringkas."


def main() -> None:
    records = []
    for mode in MODES:
        # 26 single-turn + 2 multi-turn + 1 non-ascii/long = 30+ prompts per mode
        for idx, text in enumerate(BASE_PROMPTS_ID):
            records.append(
                {
                    "id": f"{mode}_{idx:02d}",
                    "prompt_mode": mode,
                    "kind": "non_ascii" if idx >= 26 else "standard",
                    "system_prompt": DEFAULT_SYS,
                    "messages": [{"role": "user", "content": text}],
                }
            )
        for m_idx, turns in enumerate(MULTI_TURN_PAIRS):
            records.append(
                {
                    "id": f"{mode}_mt_{m_idx:02d}",
                    "prompt_mode": mode,
                    "kind": "multi_turn",
                    "system_prompt": DEFAULT_SYS,
                    "messages": turns,
                }
            )
        records.append(
            {
                "id": f"{mode}_long_2048",
                "prompt_mode": mode,
                "kind": "long_1500plus",
                "system_prompt": DEFAULT_SYS,
                "messages": [{"role": "user", "content": LONG_PROMPT_2048}],
            }
        )
        records.append(
            {
                "id": f"{mode}_long_1024",
                "prompt_mode": mode,
                "kind": "long_700plus",
                "system_prompt": DEFAULT_SYS,
                "messages": [{"role": "user", "content": LONG_PROMPT_1024}],
            }
        )

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} prompt records to {OUT_PATH}")


if __name__ == "__main__":
    main()
