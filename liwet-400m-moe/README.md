# Liwet-400M-A188M (Sparse MoE)

Source snapshot of the Sparse Mixture-of-Experts 400M-A188M Indonesian language model (`TUBES_1_400M_MOE`).

## Architecture

- 32 transformer layers (`32L`)
- Hidden size 1,024 (`head_dim = 64`)
- 16 query heads / 4 KV heads (Grouped-Query Attention)
- 8 routed experts with top-2 routing (`moe_intermediate_size = 360`), **no shared expert**
- Auxiliary load-balancing loss (`0.01`) + router z-loss (`0.001`) + DeepSeek-style EMA bias update
- SwiGLU, RoPE (`theta = 10,000`), RMSNorm (`eps = 1e-5`), tied word embeddings
- Context length 2,048 tokens, Byte-Level BPE vocabulary 32,000
- **400,098,304 total parameters (~400.1M)** and **187,761,664 active parameters per token (~187.8M)**

## Contents

- `model/`: `model.py`, `config.json`, `__init__.py`
- `train/`: `train_pretrain.py`, `train_sft.py`, `train_preference.py` (SimPO), `pipeline_a100.py`, `autotune_a100.py`, `benchmark_pretrain.py`, `chat_format.py`, `train_utils.py`
- `web/`: Local Streamlit application (`app.py`, `chat.py`, `arena.py`, `parameter_testing.py`, `analysis.py`, `liwet_models.py`, `common.py`, `MODELS_INFO.md`, `requirements.txt`, `analysis_data/benchmark_results.json`)
- `Colab_A100_MoE_400M_Pipeline.ipynb`: Sanitized end-to-end A100 Colab pipeline (uses `DRIVE_MOUNT` and `WORK_DIR` environment variables; no machine-specific paths or executed outputs)

## Hugging Face model repositories

- `fadhilahkhairogi/Liwet-400M-A188M-id-base`
- `fadhilahkhairogi/Liwet-400M-A188M-id-instruct`
- `fadhilahkhairogi/Liwet-400M-A188M-id-chat`

No model weights, checkpoints, datasets, logs, tokenizers, or binary artifacts are committed to this repository.
