"""Model configuration for MiniTransformer."""

from dataclasses import asdict, dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 8
    n_embd: int = 512
    mlp_hidden: int = 2048
    dropout: float = 0.1
    bias: bool = False
    rope_base: float = 10000.0
    tie_weights: bool = True

    def to_dict(self):
        return asdict(self)
