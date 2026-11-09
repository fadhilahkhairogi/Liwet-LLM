# Mini-Transformer Bahasa Indonesia (~66.7M) - Paket 3: Conversational Chat SFT

Paket ini berisi hasil pelatihan lanjutan khusus percakapan (*Conversational Chat SFT*) yang diinisialisasi dari checkpoint terbaik Pretrain 2 Epoch (`pretrain_2epoch_best.pt`), dengan data yang telah dibersihkan dari format kuis pilihan ganda kaku dan diformat menggunakan penanda peran percakapan `<|user|>` dan `<|assistant|>`.

## Ringkasan Metrik
- **Arsitektur Model:** 12 Layer, 8 Attention Heads, Embedding Dimension 512, Context Length 1024, Parameter ~66.7M
- **Tokenizer:** Byte-level BPE, Vocab 32.003 (termasuk `<|user|>`: 32000, `<|assistant|>`: 32001, `<|end|>`: 32002)
- **Pretraining Base:** 2 Epochs (~855,17 Juta Token, Val Loss: 3.1129)
- **Conversational SFT:** 45,31 Juta Token (1.383 Steps, Val Loss: **2.5202**)
- **Loss Masking:** Response-Only Loss Masking (loss 100% dipusatkan pada token asisten)

## Evaluasi
Jalankan evaluator dari folder ini:
```bash
cd eval
python evaluate.py --checkpoint checkpoints/posttrain_chat_best.pt
python compare_epochs.py
python compare_chat.py
```
`evaluate.py` adalah entry point tipis untuk `eval_sentences.py`.
Bobot checkpoint training (`.pt`) dan tokenizer tidak disertakan di repositori ini.
CLI evaluasi membutuhkan checkpoint `.pt`; bobot safetensors untuk inferensi
tersedia di [Hugging Face](https://huggingface.co/fadhilahkhairogi). Letakkan
checkpoint di `eval/checkpoints/` dan tokenizer di `eval/data_id/tokenizer/`.

## Web Demo
Demo web final tersedia di folder `../liwet-space/` (`fadhilahkhairogi/Liwet-Space`).
