import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import streamlit as st
import torch
from tokenizers import Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model"
TOKENIZER_PATH = PROJECT_ROOT / "data" / "tokenizer" / "tokenizer.json"

sys.path.insert(0, str(PROJECT_ROOT / "model"))
sys.path.insert(0, str(PROJECT_ROOT / "train"))

from chat_format import render_prompt
from model import (
    IndonesianChatLM,
    ModelConfig,
    count_active_parameters,
    count_parameters,
    has_model_weights,
)

torch.set_float32_matmul_precision("high")
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

CHECKPOINT_CANDIDATES = (
    PROJECT_ROOT / "checkpoints" / "simpo" / "final",
    PROJECT_ROOT / "checkpoints" / "sft" / "final",
    PROJECT_ROOT / "checkpoints" / "pretrain" / "final",
)
DEVICE_CHOICES = ("auto", "cuda", "cpu")
DTYPE_CHOICES = ("auto", "bfloat16", "float16", "float32")
DEFAULT_SYSTEM_PROMPT = (
    "Anda adalah asisten AI berbahasa Indonesia yang membantu, "
    "ramah, dan ringkas."
)


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    generated_tokens: int
    seconds: float
    tokens_per_second: float
    truncated: bool = False


@dataclass
class ModelBundle:
    model: IndonesianChatLM
    tokenizer: Tokenizer
    device: torch.device
    dtype: torch.dtype
    path: Path
    random_init: bool
    total_params: int
    active_params: int
    num_experts: int
    top_k: int
    num_layers: int
    lock: threading.Lock


def resolve_model_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


@st.cache_data(ttl=10)
def discover_checkpoints() -> list:
    found = []
    for stage, label in (
        ("simpo", "SimPO"),
        ("sft", "SFT"),
        ("pretrain", "Pretrain"),
    ):
        base = PROJECT_ROOT / "checkpoints" / stage
        if not base.exists():
            continue
        final = base / "final"
        if (final / "config.json").exists() and has_model_weights(final):
            found.append((f"{label}: final", final))
        steps = sorted(
            base.glob("step_*"), key=lambda path: path.name, reverse=True
        )
        for step in steps:
            if (step / "config.json").exists() and has_model_weights(step):
                found.append((f"{label}: {step.name}", step))
    hf_base = PROJECT_ROOT / "checkpoints" / "hf"
    if hf_base.exists():
        for hub_dir in sorted(hf_base.glob("*")):
            if (hub_dir / "config.json").exists() and has_model_weights(
                hub_dir
            ):
                found.append((f"HF: {hub_dir.name}", hub_dir))
    safetensors_root = PROJECT_ROOT.parent / "TUBES_1_SAFETENSOR"
    if safetensors_root.exists():
        for repo_name, stage_label in (
            ("Liwet-400M-A188M-id-chat", "SafeTensors: 400M Chat (SimPO)"),
            ("Liwet-400M-A188M-id-instruct", "SafeTensors: 400M Instruct (SFT)"),
            ("Liwet-400M-A188M-id-base", "SafeTensors: 400M Base (Pretrain)"),
        ):
            st_dir = safetensors_root / repo_name
            if (st_dir / "config.json").exists() and has_model_weights(st_dir):
                found.append((stage_label, st_dir))
    if (MODEL_DIR / "config.json").exists():
        found.append(("Model acak (tanpa bobot)", MODEL_DIR))
    return found


def default_checkpoint() -> Path:
    for candidate in CHECKPOINT_CANDIDATES:
        if (candidate / "config.json").exists() and has_model_weights(
            candidate
        ):
            return candidate
    discovered = discover_checkpoints()
    return discovered[0][1] if discovered else MODEL_DIR


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    return torch.device(name)


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    supported = torch.cuda.is_bf16_supported()
    return torch.bfloat16 if supported else torch.float16


def parse_hub_reference(value: str) -> tuple:
    value = value.strip()
    if not value:
        raise ValueError("Repo ID Hugging Face kosong")
    if value.startswith("hf://"):
        value = value[len("hf://") :]
    if "huggingface.co/" in value:
        value = value.split("huggingface.co/", 1)[1]
    value = value.strip("/")
    parts = [part for part in value.split("/") if part]
    if len(parts) < 2:
        raise ValueError("Format repo harus 'username/nama-repo'")
    repo = "/".join(parts[:2])
    revision = "main"
    if len(parts) >= 4 and parts[2] in {"tree", "blob", "resolve"}:
        revision = parts[3]
    elif "@" in repo:
        repo, revision = repo.rsplit("@", 1)
    return repo, revision


def download_hub_checkpoint(
    repo_input: str,
    revision: str = "main",
    token: Optional[str] = None,
) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "huggingface_hub belum terpasang, jalankan: "
            "pip install huggingface_hub"
        ) from error
    repo, parsed_revision = parse_hub_reference(repo_input)
    revision = (revision or "main").strip() or "main"
    if revision == "main" and parsed_revision != "main":
        revision = parsed_revision
    slug = repo.replace("/", "__")
    target = (
        PROJECT_ROOT
        / "checkpoints"
        / "hf"
        / f"{slug}@{revision.replace('/', '_')}"
    )
    target.mkdir(parents=True, exist_ok=True)
    common_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]
    safe_patterns = common_files + [
        "*.safetensors",
        "*.safetensors.index.json",
    ]
    bin_patterns = common_files + ["*.bin", "*.bin.index.json"]
    token = token.strip() if token and token.strip() else None
    snapshot_download(
        repo_id=repo,
        revision=revision,
        token=token,
        local_dir=str(target),
        allow_patterns=safe_patterns,
    )
    if not has_model_weights(target):
        snapshot_download(
            repo_id=repo,
            revision=revision,
            token=token,
            local_dir=str(target),
            allow_patterns=bin_patterns,
        )
    if not (target / "config.json").exists():
        raise FileNotFoundError(
            f"config.json tidak ditemukan di repo {repo}"
        )
    if not has_model_weights(target):
        raise FileNotFoundError(
            f"bobot .safetensors atau .bin tidak ditemukan di repo {repo}"
        )
    return target


@st.cache_resource(show_spinner=False, max_entries=1)
def load_bundle(
    model_path: str, device_name: str, dtype_name: str
) -> ModelBundle:
    path = resolve_model_path(model_path)
    config_path = path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json tidak ditemukan: {config_path}")
    device = resolve_device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA dipilih tetapi tidak tersedia")
    dtype = resolve_dtype(dtype_name, device)
    config = ModelConfig.from_json(config_path)
    random_init = not has_model_weights(path)
    if random_init:
        model = IndonesianChatLM(config)
    else:
        model = IndonesianChatLM.from_pretrained(path, map_location="cpu")
    model = model.to(device=device, dtype=dtype)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    tokenizer_path = path / "tokenizer.json"
    if not tokenizer_path.exists():
        tokenizer_path = TOKENIZER_PATH
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"tokenizer.json tidak ditemukan: {tokenizer_path}"
        )
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        path=path,
        random_init=random_init,
        total_params=count_parameters(model),
        active_params=count_active_parameters(model),
        num_experts=config.num_experts if config.use_moe else 1,
        top_k=config.num_experts_per_tok if config.use_moe else 1,
        num_layers=config.num_hidden_layers,
        lock=threading.Lock(),
    )


def model_sidebar() -> Optional[ModelBundle]:
    if "device_choice" not in st.session_state:
        st.session_state.device_choice = "auto"
    if "dtype_choice" not in st.session_state:
        st.session_state.dtype_choice = "auto"
    if "custom_path" not in st.session_state:
        st.session_state.custom_path = False
    if "custom_path_value" not in st.session_state:
        st.session_state.custom_path_value = str(default_checkpoint())
    with st.sidebar:
        st.subheader("Model")
        source = st.radio(
            "Sumber checkpoint",
            ["Lokal", "HF Hub"],
            horizontal=True,
            key="checkpoint_source",
        )
        path_value = None
        if source == "HF Hub":
            repo_value = st.text_input(
                "Repo ID / URL",
                key="hub_repo",
                placeholder="username/nama-repo",
            )
            revision_value = st.text_input(
                "Revision", value="main", key="hub_revision"
            )
            token_value = st.text_input(
                "HF token (opsional)",
                type="password",
                key="hub_token",
            )
            if st.button(
                "Unduh & Muat dari HF", use_container_width=True
            ):
                try:
                    with st.spinner("Mengunduh checkpoint..."):
                        downloaded = download_hub_checkpoint(
                            repo_value, revision_value, token_value
                        )
                    st.session_state.hub_path = str(downloaded)
                    load_bundle.clear()
                except Exception as error:
                    st.error(f"Gagal mengunduh checkpoint: {error}")
            hub_path = st.session_state.get("hub_path")
            if hub_path:
                path_value = hub_path
                st.caption(f"Terunduh: `{Path(hub_path).name}`")
            else:
                st.caption("Masukkan repo `username/repo`.")
        else:
            checkpoints = discover_checkpoints()
            labels = [label for label, _ in checkpoints]
            paths = {label: path for label, path in checkpoints}
            use_custom = st.checkbox("Path manual", key="custom_path")
            if use_custom or not labels:
                path_value = st.text_input(
                    "Path checkpoint",
                    key="custom_path_value",
                )
            else:
                selected_label = st.selectbox(
                    "Checkpoint",
                    labels,
                    key="checkpoint_choice",
                )
                path_value = str(paths[selected_label])
        if path_value is None:
            return None
        device_value = st.selectbox(
            "Device", DEVICE_CHOICES, key="device_choice"
        )
        dtype_value = st.selectbox(
            "Dtype", DTYPE_CHOICES, key="dtype_choice"
        )
        if st.button("Reload model", use_container_width=True):
            load_bundle.clear()
        st.session_state.model_args = {
            "path": path_value,
            "device": device_value,
            "dtype": dtype_value,
        }
        try:
            with st.spinner("Memuat model..."):
                bundle = load_bundle(
                    path_value, device_value, dtype_value
                )
        except Exception as error:
            st.error(f"Gagal memuat model: {error}")
            return None
        st.divider()
        st.markdown(
            f"**{bundle.total_params / 1e6:.1f}M** total · "
            f"**{bundle.active_params / 1e6:.1f}M** aktif"
        )
        st.caption(
            f"{bundle.num_experts} expert · top-{bundle.top_k} · "
            f"{bundle.num_layers} layer"
        )
        st.caption(
            f"{bundle.device.type.upper()} · "
            f"{str(bundle.dtype).replace('torch.', '')}"
        )
        st.caption(f"`{bundle.path.name}`")
        if bundle.random_init:
            st.warning("Bobot acak (belum terlatih).")
        elif "pretrain" in str(bundle.path).lower():
            st.warning("Checkpoint pretrain (belum SFT).")
    return bundle


def build_prompt(messages: list, system_prompt: Optional[str] = None) -> str:
    payload = []
    if system_prompt and system_prompt.strip():
        payload.append(
            {"role": "system", "content": system_prompt.strip()}
        )
    for message in messages:
        payload.append(
            {
                "role": message["role"],
                "content": str(message["content"]),
            }
        )
    return render_prompt(payload)


def count_prompt_tokens(bundle: ModelBundle, prompt: str) -> int:
    ids = bundle.tokenizer.encode(prompt, add_special_tokens=False).ids
    return len(ids) + 1


def trim_messages_for_context(
    bundle: ModelBundle,
    messages: list,
    system_prompt: Optional[str],
    max_new_tokens: int,
) -> tuple:
    budget = (
        bundle.model.config.max_position_embeddings - int(max_new_tokens)
    )
    trimmed = list(messages)
    removed = 0
    while len(trimmed) > 1 and count_prompt_tokens(
        bundle, build_prompt(trimmed, system_prompt)
    ) > budget:
        trimmed = trimmed[2:] if len(trimmed) >= 2 else trimmed[1:]
        removed += 1
    return trimmed, removed


def clean_reply(text: str) -> str:
    text = text.strip()
    if text.startswith("Assistant:"):
        text = text[len("Assistant:") :].lstrip()
    for marker in ("\n\nUser:", "\n\nSystem:", "\nUser:", "\nSystem:"):
        index = text.find(marker)
        if index != -1:
            text = text[:index]
    return text.strip()


def generate(
    bundle: ModelBundle,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.9,
    repetition_penalty: float = 1.1,
    seed: Optional[int] = None,
    do_sample: Optional[bool] = None,
) -> GenerationResult:
    if not prompt.strip():
        raise ValueError("Prompt kosong")
    bos = bundle.tokenizer.token_to_id("<bos>")
    eos = bundle.tokenizer.token_to_id("<eos>")
    pad = bundle.tokenizer.token_to_id("<pad>")
    if bos is None or eos is None:
        raise ValueError("Tokenizer tidak memiliki <bos> atau <eos>")
    ids = [bos] + list(
        bundle.tokenizer.encode(prompt, add_special_tokens=False).ids
    )
    budget = (
        bundle.model.config.max_position_embeddings - int(max_new_tokens)
    )
    truncated = False
    if budget > 0 and len(ids) > budget:
        ids = ids[-budget:]
        truncated = True
    input_ids = torch.tensor(
        [ids], dtype=torch.long, device=bundle.device
    )
    if bundle.device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    with bundle.lock, torch.inference_mode():
        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
        output = bundle.model.generate(
            input_ids,
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_k=(
                None
                if top_k is None or int(top_k) <= 0
                else int(top_k)
            ),
            top_p=(
                None
                if top_p is None or float(top_p) >= 1.0
                else float(top_p)
            ),
            repetition_penalty=float(repetition_penalty),
            eos_token_id=eos,
            pad_token_id=pad,
            do_sample=do_sample,
        )
    if bundle.device.type == "cuda":
        torch.cuda.synchronize()
    seconds = max(time.perf_counter() - started, 1e-6)
    new_ids = output[0, input_ids.shape[1] :].tolist()
    text = clean_reply(
        bundle.tokenizer.decode(new_ids, skip_special_tokens=True)
    )
    generated_tokens = len(new_ids)
    return GenerationResult(
        text=text,
        prompt_tokens=input_ids.shape[1],
        generated_tokens=generated_tokens,
        seconds=seconds,
        tokens_per_second=generated_tokens / seconds,
        truncated=truncated,
    )


def stream_generate(
    bundle: ModelBundle,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.9,
    repetition_penalty: float = 1.1,
    seed: Optional[int] = None,
    do_sample: Optional[bool] = None,
    stats: Optional[dict] = None,
):
    if not prompt.strip():
        raise ValueError("Prompt kosong")
    tokenizer = bundle.tokenizer
    bos = tokenizer.token_to_id("<bos>")
    eos = tokenizer.token_to_id("<eos>")
    if bos is None or eos is None:
        raise ValueError("Tokenizer tidak memiliki <bos> atau <eos>")
    ids = [bos] + list(
        tokenizer.encode(prompt, add_special_tokens=False).ids
    )
    budget = (
        bundle.model.config.max_position_embeddings - int(max_new_tokens)
    )
    truncated = False
    if budget > 0 and len(ids) > budget:
        ids = ids[-budget:]
        truncated = True
    generated = torch.tensor([ids], dtype=torch.long, device=bundle.device)
    prompt_tokens = generated.shape[1]
    generated_ids = []
    previous_text = ""
    past_key_values = None
    if bundle.device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    with bundle.lock, torch.inference_mode():
        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
        for _ in range(int(max_new_tokens)):
            current_ids = (
                generated
                if past_key_values is None
                else generated[:, -1:]
            )
            output = bundle.model.forward(
                current_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = output.past_key_values
            logits = output.logits[:, -1, :].float()
            if repetition_penalty != 1.0:
                token_ids = torch.unique(generated[0])
                token_logits = logits[0, token_ids]
                token_logits = torch.where(
                    token_logits > 0,
                    token_logits / repetition_penalty,
                    token_logits * repetition_penalty,
                )
                logits[0, token_ids] = token_logits
            sampling = (
                do_sample
                if do_sample is not None
                else temperature > 0
            )
            if sampling:
                if temperature != 1.0:
                    logits = logits / max(temperature, 1e-5)
                if top_k is not None and top_k > 0:
                    k = min(int(top_k), logits.size(-1))
                    kth_values = torch.topk(
                        logits, k, dim=-1
                    ).values[:, -1:]
                    logits = logits.masked_fill(
                        logits < kth_values, float("-inf")
                    )
                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        logits, descending=True, dim=-1
                    )
                    sorted_probs = torch.softmax(sorted_logits, dim=-1)
                    cumulative = sorted_probs.cumsum(dim=-1)
                    remove = cumulative - sorted_probs > top_p
                    sorted_logits = sorted_logits.masked_fill(
                        remove, float("-inf")
                    )
                    logits = torch.full_like(
                        logits, float("-inf")
                    ).scatter(
                        dim=-1, index=sorted_indices, src=sorted_logits
                    )
                probabilities = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(
                    probabilities, num_samples=1
                )
            else:
                next_token = torch.argmax(
                    logits, dim=-1, keepdim=True
                )
            token_id = int(next_token.item())
            if eos is not None and token_id == eos:
                break
            generated = torch.cat([generated, next_token], dim=1)
            generated_ids.append(token_id)
            raw_text = tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )
            text = clean_reply(raw_text)
            if text.startswith(previous_text):
                delta = text[len(previous_text) :]
            else:
                delta = text
            previous_text = text
            if delta:
                yield delta
            if any(
                marker in raw_text
                for marker in (
                    "\n\nUser:",
                    "\n\nSystem:",
                    "\nUser:",
                    "\nSystem:",
                )
            ):
                break
    if bundle.device.type == "cuda":
        torch.cuda.synchronize()
    seconds = max(time.perf_counter() - started, 1e-6)
    if stats is not None:
        stats.update(
            {
                "text": previous_text,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": len(generated_ids),
                "seconds": seconds,
                "tokens_per_second": len(generated_ids) / seconds,
                "truncated": truncated,
            }
        )
