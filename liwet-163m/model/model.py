import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    hidden_size: int = 768
    num_hidden_layers: int = 16
    num_attention_heads: int = 12
    num_key_value_heads: int = 4
    intermediate_size: int = 3072
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    attention_dropout: float = 0.0
    hidden_act: str = "silu"
    tie_word_embeddings: bool = True
    initializer_range: float = 0.02
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0
    unk_token_id: int = 3

    def __post_init__(self):
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size harus habis dibagi num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads harus habis dibagi num_key_value_heads")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_json(cls, path) -> "ModelConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        names = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in names})

    def save_pretrained(self, save_dir) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "config.json").write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(
            hidden_states, (self.weight.numel(),), self.weight, self.eps
        )


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_position_embeddings: int, theta: float):
        super().__init__()
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.theta = theta
        self.register_buffer(
            "inv_freq",
            1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim)),
            persistent=False,
        )
        self._set_cos_sin_cache(max_position_embeddings, torch.device("cpu"), torch.float32)

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq.to(device=device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
        self.max_seq_len_cached = seq_len

    def forward(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        max_position = int(position_ids.max().item()) + 1
        if max_position > self.max_seq_len_cached:
            self._set_cos_sin_cache(max_position, position_ids.device, torch.float32)
        cos = self.cos_cached[position_ids].to(position_ids.device)
        sin = self.sin_cached[position_ids].to(position_ids.device)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.to(q.dtype).unsqueeze(1)
    sin = sin.to(q.dtype).unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.dropout = config.attention_dropout

        self.q_dim = self.num_heads * self.head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.qkv_proj = nn.Linear(
            self.hidden_size, self.q_dim + 2 * self.kv_dim, bias=False
        )
        self.o_proj = nn.Linear(self.q_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        is_causal: bool = False,
    ):
        batch_size, seq_len, _ = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split((self.q_dim, self.kv_dim, self.kv_dim), dim=-1)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_value is not None:
            k = torch.cat((past_key_value[0], k), dim=2)
            v = torch.cat((past_key_value[1], v), dim=2)

        present = (k, v) if use_cache else None

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.o_proj(attn_output), present


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_up_proj = nn.Linear(
            config.hidden_size, 2 * config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(hidden_states).chunk(2, dim=-1)
        return self.down_proj(self.act_fn(gate) * up)


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        is_causal: bool = False,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present = self.self_attn(
            hidden_states, cos, sin, attention_mask, past_key_value, use_cache, is_causal
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states, present


@dataclass
class CausalLMOutput:
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    past_key_values: Optional[Tuple] = None


class IndonesianChatLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = RotaryEmbedding(
            config.head_dim, config.max_position_embeddings, config.rope_theta
        )
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.apply(self._init_weights)
        self._scale_residual_projections()

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def _scale_residual_projections(self) -> None:
        scale = 1.0 / math.sqrt(2 * self.config.num_hidden_layers)
        for layer in self.layers:
            nn.init.normal_(layer.self_attn.o_proj.weight, mean=0.0, std=self.config.initializer_range * scale)
            nn.init.normal_(layer.mlp.down_proj.weight, mean=0.0, std=self.config.initializer_range * scale)

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False

    def _build_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        batch_size: int,
        q_len: int,
        kv_len: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        past_len = kv_len - q_len

        if attention_mask is None:
            if q_len == 1:
                return None
            causal = torch.ones(q_len, kv_len, dtype=torch.bool, device=device).tril(diagonal=past_len)
            return causal.unsqueeze(0).unsqueeze(0)

        mask = attention_mask[:, None, None, :].bool()
        if q_len == 1:
            return mask

        causal = torch.ones(q_len, kv_len, dtype=torch.bool, device=device).tril(diagonal=past_len)
        return mask & causal

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = False,
        labels: Optional[torch.Tensor] = None,
    ) -> CausalLMOutput:
        batch_size, seq_len = input_ids.shape
        past_len = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        kv_len = past_len + seq_len

        if position_ids is None:
            position_ids = torch.arange(
                past_len, kv_len, device=input_ids.device, dtype=torch.long
            ).unsqueeze(0).expand(batch_size, -1)

        cos, sin = self.rotary_emb(position_ids)

        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.bool)
            attn_mask = self._build_attention_mask(
                attention_mask, batch_size, seq_len, kv_len, input_ids.device
            )
            is_causal = False
        elif past_len == 0 and seq_len > 1:
            attn_mask = None
            is_causal = True
        elif seq_len > 1:
            attn_mask = self._build_attention_mask(
                None, batch_size, seq_len, kv_len, input_ids.device
            )
            is_causal = False
        else:
            attn_mask = None
            is_causal = False

        hidden_states = self.embed_tokens(input_ids)
        presents = () if use_cache else None

        for index, layer in enumerate(self.layers):
            layer_past = past_key_values[index] if past_key_values is not None else None
            if self.gradient_checkpointing and self.training and not use_cache:
                def custom_forward(hidden_states, layer=layer):
                    return layer(
                        hidden_states, cos, sin, attn_mask, None, False, is_causal
                    )[0]

                hidden_states = checkpoint(custom_forward, hidden_states, use_reentrant=True)
            else:
                hidden_states, present = layer(
                    hidden_states, cos, sin, attn_mask, layer_past, use_cache, is_causal
                )
                if use_cache:
                    presents = presents + (present,)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutput(loss=loss, logits=logits, past_key_values=presents)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        eos_token_id = self.config.eos_token_id if eos_token_id is None else eos_token_id
        pad_token_id = self.config.pad_token_id if pad_token_id is None else pad_token_id
        device = input_ids.device
        batch_size = input_ids.shape[0]

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
        position_ids = position_ids[:, -input_ids.shape[1] :]

        output = self.forward(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )
        past_key_values = output.past_key_values
        next_token_logits = output.logits[:, -1, :]

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        generated = input_ids

        for _ in range(max_new_tokens):
            logits = next_token_logits.float()

            if repetition_penalty != 1.0:
                for batch_index in range(batch_size):
                    token_ids = torch.unique(generated[batch_index])
                    token_logits = logits[batch_index, token_ids]
                    token_logits = torch.where(
                        token_logits < 0, token_logits * repetition_penalty, token_logits / repetition_penalty
                    )
                    logits[batch_index, token_ids] = token_logits

            if temperature <= 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature

                if top_k > 0:
                    top_k = min(top_k, logits.size(-1))
                    values, _ = torch.topk(logits, top_k)
                    kth_values = values[:, -1].unsqueeze(-1)
                    logits = torch.where(
                        logits < kth_values, torch.full_like(logits, float("-inf")), logits
                    )

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    sorted_probs = F.softmax(sorted_logits, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = False
                    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
                    indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
                    logits = logits.masked_fill(indices_to_remove, float("-inf"))

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            next_token = next_token.masked_fill(finished.unsqueeze(-1), pad_token_id)
            generated = torch.cat((generated, next_token), dim=1)
            finished = finished | (next_token.squeeze(-1) == eos_token_id)
            if finished.all():
                break

            attention_mask = torch.cat(
                (attention_mask, (~finished).long().unsqueeze(-1)), dim=1
            )
            position_ids = attention_mask.long().cumsum(-1)[:, -1:] - 1

            output = self.forward(
                next_token,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = output.past_key_values
            next_token_logits = output.logits[:, -1, :]

        return generated

    def save_pretrained(self, save_dir) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.config.save_pretrained(save_dir)
        torch.save(self.state_dict(), save_dir / "pytorch_model.bin")

    @classmethod
    def from_pretrained(cls, save_dir, map_location="cpu", dtype=None) -> "IndonesianChatLM":
        save_dir = Path(save_dir)
        config = ModelConfig.from_json(save_dir / "config.json")
        model = cls(config)
        state_dict = torch.load(
            save_dir / "pytorch_model.bin", map_location=map_location, weights_only=True
        )
        model.load_state_dict(state_dict)
        if dtype is not None:
            model.to(dtype=dtype)
        return model


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


if __name__ == "__main__":
    config = ModelConfig.from_json(Path(__file__).with_name("config.json"))
    model = IndonesianChatLM(config)
    total = count_parameters(model)
    print(f"Total parameter: {total:,} ({total / 1e6:.1f}M)")
