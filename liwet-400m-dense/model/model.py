import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def _chunked_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor, chunk_size: int = 256
) -> torch.Tensor:
    total_loss = logits.new_zeros((), dtype=torch.float32)
    total_tokens = torch.zeros((), dtype=torch.long, device=logits.device)
    sequence_length = logits.shape[1]
    for start in range(0, sequence_length - 1, chunk_size):
        end = min(start + chunk_size, sequence_length - 1)
        chunk_logits = logits[:, start:end, :].float()
        chunk_labels = labels[:, start + 1 : end + 1]
        total_loss = total_loss + F.cross_entropy(
            chunk_logits.transpose(1, 2),
            chunk_labels,
            ignore_index=-100,
            reduction="sum",
        )
        total_tokens = total_tokens + (chunk_labels != -100).sum()
    return total_loss / total_tokens.clamp(min=1)


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    hidden_size: int = 1024
    intermediate_size: int = 2880
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 64
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    hidden_act: str = "silu"
    attention_dropout: float = 0.0
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0
    unk_token_id: int = 3
    model_type: str = "indonesian_chat_lm"
    architectures: list = field(
        default_factory=lambda: ["IndonesianChatLM"]
    )
    torch_dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        if self.head_dim * self.num_attention_heads != self.hidden_size:
            raise ValueError(
                "hidden_size harus sama dengan num_attention_heads * head_dim"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads harus habis dibagi num_key_value_heads"
            )
        if self.hidden_act not in {"silu", "swish"}:
            raise ValueError("hidden_act harus 'silu' atau 'swish'")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict) -> "ModelConfig":
        known = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in known})

    @classmethod
    def from_json(cls, path) -> "ModelConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_json(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8"
        )


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (self.weight * hidden_states).to(input_dtype)


def rotate_half(tensor: torch.Tensor) -> torch.Tensor:
    first_half = tensor[..., : tensor.shape[-1] // 2]
    second_half = tensor[..., tensor.shape[-1] // 2 :]
    return torch.cat((-second_half, first_half), dim=-1)


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        return self.down_proj(gate * up)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.attention_dropout = config.attention_dropout
        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, query_length, _ = hidden_states.shape

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = query.view(
            batch_size, query_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = key.view(
            batch_size, query_length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value = value.view(
            batch_size, query_length, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        cos = cos.to(query.dtype)
        sin = sin.to(query.dtype)
        query = (query * cos) + (rotate_half(query) * sin)
        key = (key * cos) + (rotate_half(key) * sin)

        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat((past_key, key), dim=2)
            value = torch.cat((past_value, value), dim=2)

        present_key_value = (key, value) if use_cache else None

        if self.num_key_value_heads != self.num_heads:
            groups = self.num_heads // self.num_key_value_heads
            key = key.repeat_interleave(groups, dim=1)
            value = value.repeat_interleave(groups, dim=1)

        dropout = self.attention_dropout if self.training else 0.0
        if attention_mask is None:
            attn_output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=dropout,
                is_causal=True,
            )
        else:
            attn_output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=dropout,
            )

        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size, query_length, self.hidden_size
        )
        return self.o_proj(attn_output), present_key_value


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = SwiGLU(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            cos,
            sin,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = residual + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states, present_key_value


@dataclass
class CausalLMOutput:
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None


class IndonesianChatLM(nn.Module):
    supports_gradient_checkpointing = True

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.gradient_checkpointing = False
        self._init_weights()
        self.register_buffer(
            "_inv_freq",
            1.0
            / (
                config.rope_theta
                ** (
                    torch.arange(0, config.head_dim, 2, dtype=torch.float32)
                    / config.head_dim
                )
            ),
            persistent=False,
        )

    def _init_weights(self) -> None:
        self.apply(self._initialize_module)
        scaled_std = self.config.initializer_range / math.sqrt(
            2 * self.config.num_hidden_layers
        )
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and (
                name.endswith("o_proj") or name.endswith("down_proj")
            ):
                module.weight.data.normal_(mean=0.0, std=scaled_std)

    def _initialize_module(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range
            )

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False

    def _prepare_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        query_length: int,
        past_length: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        total_length = past_length + query_length
        if attention_mask is None:
            if past_length == 0:
                return None
            causal = torch.ones(
                query_length, total_length, dtype=torch.bool, device=device
            ).tril(diagonal=past_length)
            return causal[None, None]

        mask = attention_mask.to(device=device, dtype=torch.bool)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0)
        if mask.dim() == 2:
            mask = mask[:, None, None, :]
        elif mask.dim() == 3:
            mask = mask[:, None, :, :]

        if past_length > 0 and mask.shape[-1] == query_length:
            prefix = torch.ones(
                mask.shape[0], 1, 1, past_length, dtype=torch.bool, device=device
            )
            mask = torch.cat((prefix, mask), dim=-1)

        if query_length > 1:
            causal = torch.ones(
                query_length, total_length, dtype=torch.bool, device=device
            ).tril(diagonal=past_length)
            mask = mask & causal

        return mask
    def _rotary_embeddings(
        self, position_ids: torch.Tensor, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inverse_frequency = self._inv_freq.to(
            device=position_ids.device, dtype=torch.float32
        )
        frequencies = position_ids.float().unsqueeze(-1) * inverse_frequency
        frequencies = torch.cat((frequencies, frequencies), dim=-1)
        cos = frequencies.cos().to(dtype).unsqueeze(1)
        sin = frequencies.sin().to(dtype).unsqueeze(1)
        return cos, sin

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[
            Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
        ] = None,
        use_cache: bool = False,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        if input_ids.dim() != 2:
            raise ValueError("input_ids harus berbentuk (batch, sequence)")

        batch_size, query_length = input_ids.shape
        past_length = (
            past_key_values[0][0].shape[2]
            if past_key_values is not None
            else 0
        )

        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                past_length + query_length,
                device=input_ids.device,
            ).unsqueeze(0)
        elif position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
        if position_ids.shape[-1] > query_length:
            position_ids = position_ids[:, -query_length:]
        if position_ids.shape[0] == 1 and batch_size > 1:
            position_ids = position_ids.expand(batch_size, -1)

        hidden_states = self.embed_tokens(input_ids)
        cos, sin = self._rotary_embeddings(position_ids, hidden_states.dtype)
        attention_mask = self._prepare_attention_mask(
            attention_mask, query_length, past_length, input_ids.device
        )

        presents = [] if use_cache else None
        for index, layer in enumerate(self.layers):
            layer_past = (
                past_key_values[index] if past_key_values is not None else None
            )
            if self.gradient_checkpointing and self.training:
                hidden_states, present = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    cos,
                    sin,
                    attention_mask,
                    layer_past,
                    use_cache,
                    use_reentrant=False,
                )
            else:
                hidden_states, present = layer(
                    hidden_states,
                    cos,
                    sin,
                    attention_mask=attention_mask,
                    past_key_value=layer_past,
                    use_cache=use_cache,
                )
            if use_cache:
                presents.append(present)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None and logits.shape[1] > 1:
            loss = _chunked_cross_entropy(logits, labels)

        present_key_values = tuple(presents) if use_cache else None

        if not return_dict:
            return (logits, present_key_values) if labels is None else (
                loss,
                logits,
                present_key_values,
            )
        return CausalLMOutput(
            loss=loss, logits=logits, past_key_values=present_key_values
        )

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        repetition_penalty: float = 1.1,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        do_sample: Optional[bool] = None,
    ) -> torch.Tensor:
        if max_new_tokens <= 0:
            return input_ids

        squeezed = input_ids.dim() == 1
        if squeezed:
            input_ids = input_ids.unsqueeze(0)

        device = input_ids.device
        batch_size = input_ids.shape[0]
        eos_token = (
            self.config.eos_token_id if eos_token_id is None else eos_token_id
        )
        pad_token = (
            self.config.pad_token_id if pad_token_id is None else pad_token_id
        )
        fill_token = pad_token if pad_token is not None else eos_token
        if do_sample is None:
            do_sample = temperature is not None and temperature > 0

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = attention_mask.to(device=device)

        generated = input_ids
        past_key_values = None
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            current_ids = (
                generated if past_key_values is None else generated[:, -1:]
            )
            position_ids = attention_mask.long().cumsum(dim=-1) - 1
            position_ids = position_ids[:, -current_ids.shape[1] :]

            output = self.forward(
                current_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = output.past_key_values
            logits = output.logits[:, -1, :].float()

            if repetition_penalty != 1.0:
                for batch_index in range(batch_size):
                    token_ids = torch.unique(generated[batch_index])
                    token_logits = logits[batch_index, token_ids]
                    token_logits = torch.where(
                        token_logits > 0,
                        token_logits / repetition_penalty,
                        token_logits * repetition_penalty,
                    )
                    logits[batch_index, token_ids] = token_logits

            if do_sample:
                if temperature != 1.0:
                    logits = logits / max(temperature, 1e-5)
                if top_k is not None and top_k > 0:
                    k = min(top_k, logits.size(-1))
                    kth_values = torch.topk(logits, k, dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < kth_values, float("-inf"))
                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        logits, descending=True, dim=-1
                    )
                    sorted_probs = torch.softmax(sorted_logits, dim=-1)
                    cumulative_probs = sorted_probs.cumsum(dim=-1)
                    remove = cumulative_probs - sorted_probs > top_p
                    sorted_logits = sorted_logits.masked_fill(
                        remove, float("-inf")
                    )
                    logits = torch.full_like(logits, float("-inf")).scatter(
                        dim=-1, index=sorted_indices, src=sorted_logits
                    )
                probabilities = torch.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probabilities, num_samples=1)
            else:
                next_tokens = torch.argmax(logits, dim=-1, keepdim=True)

            if fill_token is not None:
                next_tokens = torch.where(
                    finished[:, None],
                    torch.full_like(next_tokens, fill_token),
                    next_tokens,
                )
            generated = torch.cat((generated, next_tokens), dim=1)
            attention_mask = torch.cat(
                (
                    attention_mask,
                    (~finished).to(attention_mask.dtype).unsqueeze(1),
                ),
                dim=1,
            )
            if eos_token is not None:
                finished = finished | (next_tokens.squeeze(1) == eos_token)
                if bool(finished.all()):
                    break

        if squeezed:
            generated = generated.squeeze(0)
        return generated

    def save_pretrained(self, save_directory) -> None:
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        self.config.to_json(save_directory / "config.json")
        torch.save(self.state_dict(), save_directory / "pytorch_model.bin")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        map_location="cpu",
        dtype: Optional[torch.dtype] = None,
        strict: bool = True,
    ) -> "IndonesianChatLM":
        model_path = Path(pretrained_model_name_or_path)
        config = ModelConfig.from_json(model_path / "config.json")
        model = cls(config)
        state_path = model_path / "pytorch_model.bin"
        if not state_path.exists():
            raise FileNotFoundError(f"bobot model tidak ditemukan: {state_path}")
        state_dict = torch.load(
            state_path, map_location=map_location, weights_only=False
        )
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=strict)
        if dtype is not None:
            model.to(dtype=dtype)
        return model


def count_parameters(
    model: nn.Module, trainable_only: bool = False
) -> int:
    if trainable_only:
        return sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
    return sum(parameter.numel() for parameter in model.parameters())
