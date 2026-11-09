# Liwet-163M (Dense GQA)

Source snapshot of the 163M model from `TUBES_1_160M`.

- Architecture: 16 layers, 12 query heads / 4 KV heads (GQA), 768 hidden, 3,072 FFN, RoPE, RMSNorm, SwiGLU, 2,048 context, 32,000 BPE vocabulary.
- Parameters: approximately 163.0M.
- Training stages: pretraining on about 2.10B tokens, SFT, and SimPO preference alignment.
- Published weights: [base](https://huggingface.co/fadhilahkhairogi/Liwet-163M-id-base), [instruct](https://huggingface.co/fadhilahkhairogi/Liwet-163M-id-instruct), [chat](https://huggingface.co/fadhilahkhairogi/Liwet-163M-id-chat).
- Included: `model/` and `train/` source.
- `eval/` is a placeholder; evaluation scripts will be added after final inference optimization.
- No checkpoints, datasets, logs, tokenizers, or binary weights are committed.
