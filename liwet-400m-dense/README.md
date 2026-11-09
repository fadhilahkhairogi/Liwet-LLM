# Liwet-400M Dense

Source snapshot of the dense 400M model from `TUBES_1_400M`.

- Architecture: 32 layers, 16 query heads / 4 KV heads (GQA), 1,024 hidden, 2,880 SwiGLU FFN, RoPE, RMSNorm, 2,048 context, 32,000 BPE vocabulary.
- Parameters: approximately 399.84M dense.
- Included: `model/`, `train/`, and a sanitized `Colab_A100_Dense_400M_Pipeline.ipynb`.
- The notebook is a GPU training pipeline kept for reproducibility; CPU-only inference uses `model/model.py`.
- The notebook reads `DRIVE_MOUNT` and `WORK_DIR` environment variables; no machine-specific absolute paths are stored.
- This is the dense variant, not the MoE variant. See `../liwet-400m-moe/`.
- No weights, checkpoints, datasets, logs, tokenizers, or binary artifacts are committed.
