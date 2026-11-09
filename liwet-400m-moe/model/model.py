import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

WEIGHT_FILENAMES = (
    "pytorch_model.bin",
    "model.safetensors",
    "model.safetensors.index.json",
)


def has_model_weights(model_path) -> bool:
    model_path = Path(model_path)
    if model_path.is_file():
        return model_path.suffix in {".bin", ".safetensors"}
    return any((model_path / name).exists() for name in WEIGHT_FILENAMES)


def _load_safetensors(path: Path, map_location="cpu") -> dict:
    try:
        from safetensors.torch import load_file
    except ImportError as error:
        raise ImportError(
            "safetensors belum terpasang, jalankan: pip install safetensors"
        ) from error
    if isinstance(map_location, dict):
        device = "cpu"
    else:
        device = str(map_location)
    return load_file(str(path), device=device)


def _load_state_dict(model_path: Path, map_location="cpu") -> dict:
    if model_path.is_file():
        if model_path.suffix == ".safetensors":
            return _load_safetensors(model_path, map_location)
        state_dict = torch.load(
            model_path, map_location=map_location, weights_only=False
        )
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        return state_dict

    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map", {})
        state_dict = {}
        for shard in sorted(set(weight_map.values())):
            state_dict.update(
                _load_safetensors(model_path / shard, map_location)
            )
        return state_dict

    safe_path = model_path / "model.safetensors"
    if safe_path.exists():
        return _load_safetensors(safe_path, map_location)

    bin_path = model_path / "pytorch_model.bin"
    if bin_path.exists():
        state_dict = torch.load(
            bin_path, map_location=map_location, weights_only=False
        )
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        return state_dict

    raise FileNotFoundError(
        f"bobot model tidak ditemukan di {model_path} "
        f"(dicari: {', '.join(WEIGHT_FILENAMES)})"
    )


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
    use_moe: bool = True
    num_experts: int = 8
    num_experts_per_tok: int = 2
    moe_intermediate_size: int = 360
    moe_capacity_factor: float = 1.25
    router_aux_loss_coef: float = 0.01
    router_z_loss_coef: float = 0.001
    router_jitter_noise: float = 0.0
    router_bias_update_speed: float = 0.001
    router_load_decay: float = 0.01
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
        if self.use_moe:
            if self.num_experts < 1:
                raise ValueError("num_experts harus >= 1")
            if not 1 <= self.num_experts_per_tok <= self.num_experts:
                raise ValueError(
                    "num_experts_per_tok harus di antara 1 dan num_experts"
                )
            if self.moe_intermediate_size < 1:
                raise ValueError("moe_intermediate_size harus > 0")
            if self.moe_capacity_factor <= 0:
                raise ValueError("moe_capacity_factor harus > 0")
            if self.router_aux_loss_coef < 0 or self.router_z_loss_coef < 0:
                raise ValueError("koefisien router loss tidak boleh negatif")
            if not 0.0 <= self.router_load_decay <= 1.0:
                raise ValueError("router_load_decay harus di antara 0 dan 1")
            if self.router_bias_update_speed < 0:
                raise ValueError("router_bias_update_speed tidak boleh negatif")

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
        if not self.training and hasattr(F, "rms_norm"):
            try:
                return F.rms_norm(
                    hidden_states,
                    self.weight.shape,
                    self.weight,
                    self.eps,
                )
            except (RuntimeError, NotImplementedError):
                pass
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


class MoE(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.capacity_factor = config.moe_capacity_factor
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.router_z_loss_coef = config.router_z_loss_coef
        self.jitter_noise = config.router_jitter_noise
        self.bias_update_speed = config.router_bias_update_speed
        self.load_decay = config.router_load_decay
        self.initializer_range = config.initializer_range
        self.router = nn.Linear(
            config.hidden_size, config.num_experts, bias=False
        )
        self.register_buffer(
            "expert_bias", torch.zeros(config.num_experts), persistent=True
        )
        self.register_buffer(
            "load_ema",
            torch.full(
                (config.num_experts,), 1.0 / config.num_experts
            ),
            persistent=True,
        )
        self.gate_up_proj = nn.Parameter(
            torch.empty(
                config.num_experts,
                config.hidden_size,
                2 * config.moe_intermediate_size,
            )
        )
        self.down_proj = nn.Parameter(
            torch.empty(
                config.num_experts,
                config.moe_intermediate_size,
                config.hidden_size,
            )
        )
        self.reset_parameters()

    def _apply(self, fn, *args, **kwargs):
        super()._apply(fn, *args, **kwargs)
        self.expert_bias = self.expert_bias.float()
        self.load_ema = self.load_ema.float()
        return self

    def reset_parameters(self, down_proj_std: Optional[float] = None) -> None:
        std = self.initializer_range
        nn.init.normal_(self.router.weight, mean=0.0, std=std)
        nn.init.normal_(self.gate_up_proj, mean=0.0, std=std)
        nn.init.normal_(
            self.down_proj,
            mean=0.0,
            std=std if down_proj_std is None else down_proj_std,
        )
        self.expert_bias.zero_()
        self.load_ema.fill_(1.0 / self.num_experts)

    def _aux_loss(
        self,
        router_logits: torch.Tensor,
        router_probs: torch.Tensor,
        assignments: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if self.router_aux_loss_coef == 0.0 and self.router_z_loss_coef == 0.0:
            return None
        fraction = assignments.sum(dim=1).mean(dim=0) / self.top_k
        mean_prob = router_probs.mean(dim=0)
        load_balance = self.num_experts * torch.sum(fraction * mean_prob)
        z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
        return (
            self.router_aux_loss_coef * load_balance
            + self.router_z_loss_coef * z_loss
        )

    def forward(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, sequence_length, _ = hidden_states.shape
        num_tokens = batch_size * sequence_length
        flat_states = hidden_states.reshape(num_tokens, self.hidden_size)

        if self.training and self.jitter_noise > 0.0:
            noise = torch.empty_like(flat_states).uniform_(
                1.0 - self.jitter_noise, 1.0 + self.jitter_noise
            )
            router_input = flat_states * noise
        else:
            router_input = flat_states

        router_logits = F.linear(
            router_input, self.router.weight
        ).float()
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        selection_logits = router_logits + self.expert_bias.float()
        top_indices = torch.topk(selection_logits, self.top_k, dim=-1).indices
        top_weights = torch.gather(router_probs, dim=-1, index=top_indices)
        top_weights = top_weights / top_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-9)
        aux_loss = None
        if self.training:
            assignments = F.one_hot(
                top_indices, num_classes=self.num_experts
            ).to(router_probs.dtype)
            aux_loss = self._aux_loss(
                router_logits, router_probs, assignments
            )
            if self.bias_update_speed > 0.0:
                expert_load = (
                    assignments.sum(dim=1).mean(dim=0) / self.top_k
                )
                with torch.no_grad():
                    self.load_ema.mul_(1.0 - self.load_decay).add_(
                        self.load_decay * expert_load
                    )
                    load_error = 1.0 / self.num_experts - self.load_ema
                    self.expert_bias.add_(
                        self.bias_update_speed * torch.sign(load_error)
                    )

        if (
            not self.training
            and num_tokens == 1
            and not getattr(self, "_disable_fast_moe", False)
        ):
            experts = top_indices[0]
            weights = top_weights[0].to(flat_states.dtype)
            selected_gate_up = self.gate_up_proj[experts]
            selected_down = self.down_proj[experts]
            expert_input = flat_states.repeat(
                self.top_k, 1
            ).unsqueeze(1)
            gate_up = torch.bmm(expert_input, selected_gate_up)
            gate, up = gate_up.chunk(2, dim=-1)
            expert_output = torch.bmm(
                F.silu(gate) * up, selected_down
            ).reshape(self.top_k, self.hidden_size)
            output = (
                expert_output * weights.unsqueeze(-1)
            ).sum(dim=0, keepdim=True)
            return (
                output.view(
                    batch_size, sequence_length, self.hidden_size
                ),
                None,
            )

        capacity = max(
            1,
            math.ceil(
                num_tokens * self.top_k / self.num_experts * self.capacity_factor
            ),
        )
        num_assignments = num_tokens * self.top_k
        flat_indices = top_indices.reshape(-1)
        flat_weights = top_weights.reshape(-1).to(flat_states.dtype)

        order = torch.argsort(flat_indices)
        flat_weights = flat_weights[order]
        sorted_indices = flat_indices[order]
        expert_ids = torch.arange(
            self.num_experts, device=flat_states.device
        )
        offsets_end = torch.searchsorted(
            sorted_indices, expert_ids, right=True
        )
        offsets = torch.cat(
            [offsets_end.new_zeros(1), offsets_end[:-1]]
        )
        positions = (
            torch.arange(num_assignments, device=flat_states.device)
            - offsets[sorted_indices]
        )
        valid = positions < capacity
        expert_slots = sorted_indices * capacity + positions
        expert_slots = torch.where(
            valid,
            expert_slots,
            expert_slots.new_full((), self.num_experts * capacity),
        )
        token_indices = order // self.top_k

        expert_input = flat_states.new_zeros(
            self.num_experts * capacity + 1, self.hidden_size
        )
        expert_input[expert_slots] = flat_states[token_indices]
        expert_input = expert_input[: self.num_experts * capacity].view(
            self.num_experts, capacity, self.hidden_size
        )

        gate_up = torch.bmm(expert_input, self.gate_up_proj)
        gate, up = gate_up.chunk(2, dim=-1)
        expert_output = torch.bmm(
            F.silu(gate) * up, self.down_proj
        ).reshape(self.num_experts * capacity, self.hidden_size)

        gather_slots = torch.where(
            valid, expert_slots, expert_slots.new_zeros(())
        )
        combined = expert_output[gather_slots]
        combined = combined * (
            flat_weights * valid.to(flat_weights.dtype)
        ).unsqueeze(-1)
        output = flat_states.new_zeros(num_tokens, self.hidden_size)
        output = output.index_add_(0, token_indices, combined)
        return (
            output.view(batch_size, sequence_length, self.hidden_size),
            aux_loss,
        )


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

    def _fused_qkv_weight(self) -> torch.Tensor:
        weight = self.q_proj.weight
        versions = (
            weight._version,
            self.k_proj.weight._version,
            self.v_proj.weight._version,
        )
        cache = getattr(self, "_qkv_weight_cache", None)
        if (
            cache is not None
            and cache[0] == versions
            and cache[1].device == weight.device
            and cache[1].dtype == weight.dtype
        ):
            return cache[1]
        fused = torch.cat(
            [self.q_proj.weight, self.k_proj.weight, self.v_proj.weight],
            dim=0,
        )
        self._qkv_weight_cache = (versions, fused)
        return fused

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

        if not self.training:
            query_dim = self.num_heads * self.head_dim
            kv_dim = self.num_key_value_heads * self.head_dim
            query, key, value = torch.split(
                F.linear(hidden_states, self._fused_qkv_weight()),
                [query_dim, kv_dim, kv_dim],
                dim=-1,
            )
        else:
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
                is_causal=query_length == key.shape[2],
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
        if config.use_moe:
            self.mlp = MoE(config)
        else:
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
    ) -> Tuple[
        torch.Tensor,
        Optional[Tuple[torch.Tensor, torch.Tensor]],
        Optional[torch.Tensor],
    ]:
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
        if isinstance(self.mlp, MoE):
            mlp_output, aux_loss = self.mlp(
                self.post_attention_layernorm(hidden_states)
            )
            hidden_states = residual + mlp_output
        else:
            hidden_states = residual + self.mlp(
                self.post_attention_layernorm(hidden_states)
            )
            aux_loss = None
        return hidden_states, present_key_value, aux_loss


@dataclass
class CausalLMOutput:
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None
    aux_loss: Optional[torch.Tensor] = None


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
            if isinstance(module, MoE):
                module.reset_parameters(scaled_std)
            elif isinstance(module, nn.Linear) and (
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
            if past_length == 0 or query_length == 1:
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
        self,
        position_ids: torch.Tensor,
        dtype: torch.dtype,
        use_table: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if use_table:
            table = getattr(self, "_rope_table", None)
            size = self.config.max_position_embeddings
            if (
                table is None
                or table["device"] != position_ids.device
                or table["dtype"] != dtype
                or table["size"] < size
            ):
                positions = torch.arange(
                    size, device=position_ids.device, dtype=torch.float32
                )
                inverse_frequency = self._inv_freq.to(
                    device=position_ids.device, dtype=torch.float32
                )
                frequencies = positions.unsqueeze(-1) * inverse_frequency
                frequencies = torch.cat(
                    (frequencies, frequencies), dim=-1
                )
                table = {
                    "device": position_ids.device,
                    "dtype": dtype,
                    "size": size,
                    "cos": frequencies.cos().to(dtype),
                    "sin": frequencies.sin().to(dtype),
                }
                self._rope_table = table
            cos = table["cos"][position_ids].unsqueeze(1)
            sin = table["sin"][position_ids].unsqueeze(1)
            return cos, sin
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

        auto_position_ids = position_ids is None
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
        cos, sin = self._rotary_embeddings(
            position_ids,
            hidden_states.dtype,
            use_table=(
                auto_position_ids
                and not self.training
                and past_length + query_length
                <= self.config.max_position_embeddings
            ),
        )
        attention_mask = self._prepare_attention_mask(
            attention_mask, query_length, past_length, input_ids.device
        )

        presents = [] if use_cache else None
        aux_loss = None
        for index, layer in enumerate(self.layers):
            layer_past = (
                past_key_values[index] if past_key_values is not None else None
            )
            if self.gradient_checkpointing and self.training:
                hidden_states, present, layer_aux_loss = (
                    torch.utils.checkpoint.checkpoint(
                        layer,
                        hidden_states,
                        cos,
                        sin,
                        attention_mask,
                        layer_past,
                        use_cache,
                        use_reentrant=False,
                    )
                )
            else:
                hidden_states, present, layer_aux_loss = layer(
                    hidden_states,
                    cos,
                    sin,
                    attention_mask=attention_mask,
                    past_key_value=layer_past,
                    use_cache=use_cache,
                )
            if use_cache:
                presents.append(present)
            if layer_aux_loss is not None:
                aux_loss = (
                    layer_aux_loss
                    if aux_loss is None
                    else aux_loss + layer_aux_loss
                )

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None and logits.shape[1] > 1:
            loss = _chunked_cross_entropy(logits, labels)
            if self.training and aux_loss is not None:
                loss = loss + aux_loss

        present_key_values = tuple(presents) if use_cache else None

        if not return_dict:
            return (logits, present_key_values) if labels is None else (
                loss,
                logits,
                present_key_values,
            )
        return CausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=present_key_values,
            aux_loss=aux_loss,
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

        provided_attention_mask = attention_mask is not None
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
                attention_mask=(
                    attention_mask if provided_attention_mask else None
                ),
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

    def save_pretrained(
        self, save_directory, safe_serialization: bool = False
    ) -> None:
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        self.config.to_json(save_directory / "config.json")
        state_dict = self.state_dict()
        if safe_serialization:
            try:
                from safetensors.torch import save_file
            except ImportError as error:
                raise ImportError(
                    "safetensors belum terpasang, jalankan: "
                    "pip install safetensors"
                ) from error
            if self.config.tie_word_embeddings:
                state_dict.pop("lm_head.weight", None)
            state_dict = {
                key: value.detach().contiguous()
                for key, value in state_dict.items()
            }
            save_file(state_dict, str(save_directory / "model.safetensors"))
        else:
            torch.save(state_dict, save_directory / "pytorch_model.bin")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        map_location="cpu",
        dtype: Optional[torch.dtype] = None,
        strict: bool = True,
        config_path=None,
    ) -> "IndonesianChatLM":
        model_path = Path(pretrained_model_name_or_path)
        if config_path is None:
            if model_path.is_file():
                config_path = model_path.parent / "config.json"
            else:
                config_path = model_path / "config.json"
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(
                f"config.json tidak ditemukan: {config_path}"
            )
        config = ModelConfig.from_json(config_path)
        model = cls(config)
        state_dict = _load_state_dict(model_path, map_location)
        if state_dict and all(
            key.startswith("model.") for key in state_dict
        ):
            state_dict = {
                key[len("model.") :]: value
                for key, value in state_dict.items()
            }
        if (
            config.tie_word_embeddings
            and "lm_head.weight" not in state_dict
            and "embed_tokens.weight" in state_dict
        ):
            state_dict["lm_head.weight"] = state_dict[
                "embed_tokens.weight"
            ]
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


def count_active_parameters(model: nn.Module) -> int:
    total = count_parameters(model)
    expert_parameters = 0
    routed_parameters = 0
    for module in model.modules():
        if isinstance(module, MoE):
            per_expert = (
                module.gate_up_proj.numel() + module.down_proj.numel()
            ) // module.num_experts
            expert_parameters += per_expert * module.num_experts
            routed_parameters += per_expert * module.top_k
    return total - expert_parameters + routed_parameters
