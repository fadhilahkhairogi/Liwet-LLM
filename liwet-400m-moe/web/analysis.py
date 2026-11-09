import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent / "analysis_data"
TELEMETRY_FILE = DATA_DIR / "telemetry.json"
BENCHMARK_FILE = DATA_DIR / "benchmark_results.json"

MODEL_COLORS = ["#1565C0", "#2E7D32", "#C62828"]
STAGE_COLORS = ["#6A1B9A", "#00838F", "#EF6C00"]

DECODING_PRESETS = [
    {
        "Preset": "Chat seimbang",
        "Status": "Optimal",
        "Max token": 256,
        "Temperature": 0.7,
        "Top-k": 50,
        "Top-p": 0.9,
        "Repetition penalty": 1.1,
    },
    {
        "Preset": "Kreatif",
        "Status": "Optimal",
        "Max token": 320,
        "Temperature": 1.1,
        "Top-k": 80,
        "Top-p": 0.95,
        "Repetition penalty": 1.05,
    },
    {
        "Preset": "Deterministik",
        "Status": "Optimal",
        "Max token": 256,
        "Temperature": 0.0,
        "Top-k": 0,
        "Top-p": 1.0,
        "Repetition penalty": 1.1,
    },
    {
        "Preset": "Cepat",
        "Status": "Optimal",
        "Max token": 96,
        "Temperature": 0.6,
        "Top-k": 40,
        "Top-p": 0.85,
        "Repetition penalty": 1.1,
    },
    {
        "Preset": "Hindari",
        "Status": "Hindari",
        "Max token": 256,
        "Temperature": 1.2,
        "Top-k": 80,
        "Top-p": 0.95,
        "Repetition penalty": 1.0,
    },
]


@st.cache_data(show_spinner=False, ttl=5)
def load_json(path_value: str):
    path = Path(path_value)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"error": True}


def mean_or_none(values):
    numeric = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    return sum(numeric) / len(numeric) if numeric else None


def aggregate_prompts(prompts):
    prompts = prompts or []
    return {
        "loss": mean_or_none([item.get("loss") for item in prompts]),
        "perplexity": mean_or_none(
            [item.get("perplexity") for item in prompts]
        ),
        "top1": mean_or_none(
            [item.get("top1_accuracy") for item in prompts]
        ),
        "generated_tokens": sum(
            item.get("generated_tokens") or 0 for item in prompts
        ),
        "seconds": sum(item.get("seconds") or 0 for item in prompts),
    }


def build_stage_frame(models, families):
    rows = []
    stage_map = (
        ("pretrain", "Base (Pretrain)"),
        ("sft", "Instruct (SFT)"),
        ("simpo", "Chat (SimPO)"),
    )
    for model in models:
        family = model.get("family")
        if family not in families:
            continue
        for key, label in stage_map:
            stage = model.get(key)
            if not isinstance(stage, dict):
                continue
            rows.append(
                {
                    "Model": family,
                    "Tahap": label,
                    "Val loss": stage.get("val_loss"),
                    "Val PPL": stage.get("val_ppl"),
                    "Val accuracy": stage.get("val_accuracy"),
                    "Test accuracy": stage.get("test_accuracy"),
                }
            )
    return pd.DataFrame(rows)


def build_scale_frame(models):
    rows = []
    for model in models:
        family = model.get("family")
        if family == "Liwet-66M":
            stage = model.get("pretrain_v2", {})
            stage_label = "Pretrain v2"
        else:
            stage = model.get("pretrain", {})
            stage_label = "Pretrain"
        total = model.get("total_params") or 0
        active = model.get("active_params") or 0
        rows.append(
            {
                "Model": family,
                "Tahap acuan": stage_label,
                "Total parameter (M)": total / 1e6,
                "Aktif per token (M)": active / 1e6,
                "Rasio aktif": active / total if total else None,
                "Token latih (B)": (stage.get("tokens") or 0) / 1e9,
                "Val PPL": stage.get("val_ppl"),
            }
        )
    return pd.DataFrame(rows)


def flatten_record(record):
    flat = {}
    for key, value in record.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (str, int, float, bool)) or sub_value is None:
                    flat[f"{key}.{sub_key}"] = sub_value
        elif isinstance(value, (str, int, float, bool)) or value is None:
            flat[key] = value
    return flat


def build_benchmark_frame(payload):
    if payload is None or not isinstance(payload, (dict, list)):
        return pd.DataFrame()
    if isinstance(payload, dict) and payload.get("error"):
        return pd.DataFrame()
    items = None
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        for key in ("results", "benchmarks", "runs", "models", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                items = value
                break
            if isinstance(value, dict):
                items = []
                for name, item in value.items():
                    if isinstance(item, dict):
                        merged = dict(item)
                        merged.setdefault("model", name)
                        items.append(merged)
                break
        if items is None:
            items = [payload]
    rows = []
    for item in items:
        if isinstance(item, dict):
            rows.append(flatten_record(item))
        else:
            rows.append({"value": item})
    return pd.DataFrame(rows)


def find_metric_column(frame, tokens):
    for column in frame.columns:
        lowered = str(column).lower()
        if any(token in lowered for token in tokens):
            if pd.api.types.is_numeric_dtype(frame[column]):
                return column
    return None


def find_label_column(frame):
    for candidate in ("model", "name", "label", "preset", "scale", "repo"):
        for column in frame.columns:
            if candidate in str(column).lower():
                if not pd.api.types.is_numeric_dtype(frame[column]):
                    return column
    for column in frame.columns:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            return column
    return None


def render_architecture(models):
    lookup = {model.get("family"): model for model in models}
    dense = lookup.get("Liwet-163M")
    sparse = lookup.get("Liwet-400M-A188M")
    if not dense or not sparse:
        st.info("Data perbandingan 163M dan 400M-A188M belum tersedia.")
        return

    rows = []
    for label, model in (
        ("Dense 163M", dense),
        ("Sparse MoE 400M-A188M", sparse),
    ):
        total = model.get("total_params") or 0
        active = model.get("active_params") or 0
        experts = (
            f"{model.get('num_experts')} routed · "
            f"top-{model.get('num_experts_per_tok')}"
            if model.get("num_experts")
            else "Dense penuh"
        )
        rows.append(
            {
                "Model": label,
                "Total parameter": f"{total / 1e6:.1f}M",
                "Aktif per token": f"{active / 1e6:.1f}M",
                "Rasio aktif": f"{active / total:.1%}" if total else "-",
                "PPL pretrain": round(
                    model.get("pretrain", {}).get("val_ppl") or 0, 2
                ),
                "Expert": experts,
                "Arsitektur": model.get("arch", "-"),
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    param_rows = []
    for label, model in (
        ("Dense 163M", dense),
        ("Sparse MoE 400M-A188M", sparse),
    ):
        param_rows.append(
            {
                "Model": label,
                "Kategori": "Total parameter",
                "Parameter (M)": (model.get("total_params") or 0) / 1e6,
            }
        )
        param_rows.append(
            {
                "Model": label,
                "Kategori": "Aktif per token",
                "Parameter (M)": (model.get("active_params") or 0) / 1e6,
            }
        )
    param_frame = pd.DataFrame(param_rows)
    param_chart = (
        alt.Chart(param_frame)
        .mark_bar()
        .encode(
            x=alt.X(
                "Kategori:N",
                axis=alt.Axis(labelAngle=0),
                title=None,
                sort=["Total parameter", "Aktif per token"],
            ),
            xOffset=alt.XOffset("Model:N"),
            y=alt.Y("Parameter (M):Q", title="Parameter (juta)"),
            color=alt.Color(
                "Model:N",
                scale=alt.Scale(
                    domain=["Dense 163M", "Sparse MoE 400M-A188M"],
                    range=MODEL_COLORS[:2],
                ),
                title=None,
            ),
            tooltip=[
                "Model",
                "Kategori",
                alt.Tooltip("Parameter (M):Q", format=".1f"),
            ],
        )
    )
    st.altair_chart(param_chart, use_container_width=True)

    total = sparse.get("total_params") or 0
    active = sparse.get("active_params") or 0
    first, second, third, fourth = st.columns(4)
    first.metric("Total 400M-A188M", f"{total / 1e6:.1f}M")
    second.metric(
        "Aktif per token",
        f"{active / 1e6:.1f}M",
        f"{active / total:.1%} dari total" if total else None,
    )
    third.metric("Kedalaman", "32 layer")
    fourth.metric("Atensi GQA", "16Q / 4KV")

    ppl_frame = pd.DataFrame(
        [
            {
                "Model": "Dense 163M",
                "PPL pretrain": dense.get("pretrain", {}).get("val_ppl"),
            },
            {
                "Model": "Sparse MoE 400M-A188M",
                "PPL pretrain": sparse.get("pretrain", {}).get("val_ppl"),
            },
        ]
    )
    ppl_chart = (
        alt.Chart(ppl_frame)
        .mark_bar()
        .encode(
            x=alt.X("Model:N", axis=alt.Axis(labelAngle=0), title=None),
            y=alt.Y("PPL pretrain:Q", title="Validation perplexity"),
            color=alt.Color(
                "Model:N",
                scale=alt.Scale(
                    domain=["Dense 163M", "Sparse MoE 400M-A188M"],
                    range=MODEL_COLORS[:2],
                ),
                title=None,
            ),
            tooltip=[
                "Model",
                alt.Tooltip("PPL pretrain:Q", format=".2f"),
            ],
        )
    )
    st.altair_chart(ppl_chart, use_container_width=True)
    st.caption(
        "Keduanya dilatih pada sekitar 2.10B token. MoE memakai "
        f"{sparse.get('num_experts')} routed expert dengan routing top-"
        f"{sparse.get('num_experts_per_tok')}, sehingga "
        f"hanya {active / total:.1%} parameter yang aktif per token."
    )


def render_epoch_comparison(models, telemetry):
    lookup = {model.get("family"): model for model in models}
    model_66 = lookup.get("Liwet-66M")
    epoch = telemetry.get("epoch_comparison_66m", {})
    chat = telemetry.get("chat_comparison_66m", {})
    if not model_66:
        st.info("Data 66M belum tersedia pada telemetri.")
        return

    version_one = model_66.get("pretrain_v1", {})
    version_two = model_66.get("pretrain_v2", {})
    total_params = model_66.get("total_params") or 0
    tokens_one = version_one.get("tokens") or 0
    tokens_two = version_two.get("tokens") or 0
    ratio_one = tokens_one / total_params if total_params else 0
    ratio_two = tokens_two / total_params if total_params else 0
    chinchilla_tokens = 20 * total_params
    gap = 1 - (tokens_two / chinchilla_tokens) if chinchilla_tokens else 0

    first, second, third, fourth = st.columns(4)
    first.metric("Epoch 1", f"{tokens_one / 1e6:.1f}M token")
    second.metric("Epoch 2", f"{tokens_two / 1e6:.1f}M token")
    third.metric(
        "Token per parameter", f"{ratio_one:.1f}x → {ratio_two:.1f}x"
    )
    fourth.metric("Referensi Chinchilla", "20 token/param")
    st.caption(
        f"Referensi 20 token/parameter setara sekitar "
        f"{chinchilla_tokens / 1e9:.2f}B token. Epoch 2 baru mencapai "
        f"{ratio_two:.1f} token/parameter, sekitar {gap:.0%} di bawah "
        "referensi tersebut, sehingga data belum menunjukkan saturasi."
    )

    ppl_rows = [
        {
            "Tahap": "Pretrain",
            "Varian": "Epoch 1",
            "PPL": version_one.get("val_ppl"),
        },
        {
            "Tahap": "Pretrain",
            "Varian": "Epoch 2",
            "PPL": version_two.get("val_ppl"),
        },
        {
            "Tahap": "SFT",
            "Varian": "Epoch 1",
            "PPL": version_one.get("sft_val_ppl"),
        },
        {
            "Tahap": "SFT",
            "Varian": "Epoch 2",
            "PPL": version_two.get("sft_val_ppl"),
        },
    ]
    ppl_frame = pd.DataFrame(ppl_rows).dropna(subset=["PPL"])
    if not ppl_frame.empty:
        ppl_chart = (
            alt.Chart(ppl_frame)
            .mark_bar()
            .encode(
                x=alt.X(
                    "Tahap:N",
                    axis=alt.Axis(labelAngle=0),
                    title=None,
                ),
                xOffset=alt.XOffset("Varian:N"),
                y=alt.Y("PPL:Q", title="Validation perplexity"),
                color=alt.Color(
                    "Varian:N",
                    scale=alt.Scale(
                        domain=["Epoch 1", "Epoch 2"],
                        range=["#90CAF9", "#1565C0"],
                    ),
                    title=None,
                ),
                tooltip=["Tahap", "Varian", alt.Tooltip("PPL:Q", format=".2f")],
            )
        )
        st.altair_chart(ppl_chart, use_container_width=True)

    group_rows = []
    for group_key, group in epoch.get("groups", {}).items():
        for variant_key, variant in group.get("variants", {}).items():
            metrics = aggregate_prompts(variant.get("prompts"))
            if metrics["perplexity"] is None:
                continue
            group_rows.append(
                {
                    "Grup": group.get("title", group_key),
                    "Varian": variant_key,
                    "PPL prompt": metrics["perplexity"],
                    "Top-1": metrics["top1"],
                    "Token output": metrics["generated_tokens"],
                }
            )
    if group_rows:
        st.markdown("**Evaluasi prompt pada dua epoch**")
        group_frame = pd.DataFrame(group_rows)
        st.dataframe(
            group_frame.round(
                {"PPL prompt": 2, "Top-1": 4}
            ),
            hide_index=True,
            use_container_width=True,
        )
        group_chart = (
            alt.Chart(group_frame)
            .mark_bar()
            .encode(
                x=alt.X("Grup:N", title=None),
                xOffset=alt.XOffset("Varian:N"),
                y=alt.Y("PPL prompt:Q", title="PPL rata-rata prompt"),
                color=alt.Color(
                    "Varian:N",
                    scale=alt.Scale(
                        domain=["1 epoch", "2 epoch", "Pretrain 1 epoch", "Pretrain 2 epoch"],
                        range=["#90CAF9", "#1565C0", "#FFCC80", "#EF6C00"],
                    ),
                    title=None,
                ),
                tooltip=[
                    "Grup",
                    "Varian",
                    alt.Tooltip("PPL prompt:Q", format=".2f"),
                    alt.Tooltip("Top-1:Q", format=".3f"),
                ],
            )
        )
        st.altair_chart(group_chart, use_container_width=True)

    if chat.get("models"):
        st.markdown("**Perbandingan model chat 66M**")
        chat_rows = []
        for name, payload in chat.get("models", {}).items():
            heldout = payload.get("heldout", {})
            chat_rows.append(
                {
                    "Model": name,
                    "Format": heldout.get("format", "-"),
                    "PPL heldout": heldout.get("perplexity"),
                    "Top-1 heldout": heldout.get("top1_accuracy"),
                    "Contoh": heldout.get("examples"),
                }
            )
        chat_frame = pd.DataFrame(chat_rows)
        st.dataframe(
            chat_frame.round(
                {"PPL heldout": 2, "Top-1 heldout": 4}
            ),
            hide_index=True,
            use_container_width=True,
        )
        chat_chart = (
            alt.Chart(chat_frame)
            .mark_bar()
            .encode(
                x=alt.X("Model:N", axis=alt.Axis(labelAngle=0), title=None),
                y=alt.Y("PPL heldout:Q", title="PPL heldout"),
                color=alt.Color(
                    "Model:N",
                    scale=alt.Scale(
                        domain=chat_frame["Model"].tolist(),
                        range=["#1565C0", "#66BB6A"],
                    ),
                    title=None,
                ),
                tooltip=[
                    "Model",
                    "Format",
                    alt.Tooltip("PPL heldout:Q", format=".2f"),
                    alt.Tooltip("Top-1 heldout:Q", format=".3f"),
                ],
            )
        )
        st.altair_chart(chat_chart, use_container_width=True)
        st.caption(
            "Format keluaran kedua model berbeda (`chat` dan `plain`), "
            "sehingga PPL heldout tidak sepenuhnya setara."
        )


def render_stage_progression(models, telemetry):
    stage_frame = build_stage_frame(
        models, {"Liwet-163M", "Liwet-400M-A188M"}
    )
    if stage_frame.empty:
        st.info("Data tahapan model 163M dan 400M belum tersedia.")
        return

    st.dataframe(
        stage_frame.round(
            {
                "Val loss": 3,
                "Val PPL": 2,
                "Val accuracy": 3,
                "Test accuracy": 3,
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    ppl_frame = stage_frame[
        stage_frame["Tahap"].isin(["Base (Pretrain)", "Instruct (SFT)"])
    ].dropna(subset=["Val PPL"])
    if not ppl_frame.empty:
        ppl_chart = (
            alt.Chart(ppl_frame)
            .mark_bar()
            .encode(
                x=alt.X(
                    "Tahap:N",
                    sort=["Base (Pretrain)", "Instruct (SFT)"],
                    axis=alt.Axis(labelAngle=0),
                    title=None,
                ),
                xOffset=alt.XOffset("Model:N"),
                y=alt.Y("Val PPL:Q", title="Validation perplexity"),
                color=alt.Color(
                    "Model:N",
                    scale=alt.Scale(
                        domain=["Liwet-163M", "Liwet-400M-A188M"],
                        range=MODEL_COLORS[1:],
                    ),
                    title=None,
                ),
                tooltip=[
                    "Model",
                    "Tahap",
                    alt.Tooltip("Val PPL:Q", format=".2f"),
                ],
            )
        )
        st.altair_chart(ppl_chart, use_container_width=True)

    accuracy_frame = stage_frame[
        stage_frame["Tahap"] == "Chat (SimPO)"
    ].dropna(how="all", subset=["Val accuracy", "Test accuracy"])
    if not accuracy_frame.empty:
        accuracy_rows = []
        for _, row in accuracy_frame.iterrows():
            accuracy_rows.append(
                {
                    "Model": row["Model"],
                    "Metrik": "Validasi",
                    "Akurasi": row["Val accuracy"],
                }
            )
            accuracy_rows.append(
                {
                    "Model": row["Model"],
                    "Metrik": "Test",
                    "Akurasi": row["Test accuracy"],
                }
            )
        accuracy_plot_frame = pd.DataFrame(accuracy_rows).dropna(
            subset=["Akurasi"]
        )
        accuracy_chart = (
            alt.Chart(accuracy_plot_frame)
            .mark_bar()
            .encode(
                x=alt.X("Model:N", axis=alt.Axis(labelAngle=0), title=None),
                xOffset=alt.XOffset("Metrik:N"),
                y=alt.Y(
                    "Akurasi:Q",
                    axis=alt.Axis(format="%"),
                    scale=alt.Scale(domain=[0, 1]),
                    title="Akurasi preference",
                ),
                color=alt.Color(
                    "Metrik:N",
                    scale=alt.Scale(
                        domain=["Validasi", "Test"],
                        range=["#6A1B9A", "#00838F"],
                    ),
                    title=None,
                ),
                tooltip=[
                    "Model",
                    "Metrik",
                    alt.Tooltip("Akurasi:Q", format=".3f"),
                ],
            )
        )
        st.altair_chart(accuracy_chart, use_container_width=True)

    curves = telemetry.get("curves", {})
    curve_keys = {
        "163m_pretrain": ("163M", "Base (Pretrain)"),
        "163m_sft": ("163M", "Instruct (SFT)"),
        "163m_simpo": ("163M", "Chat (SimPO)"),
        "400m_pretrain": ("400M", "Base (Pretrain)"),
        "400m_sft": ("400M", "Instruct (SFT)"),
        "400m_simpo": ("400M", "Chat (SimPO)"),
    }
    curve_rows = []
    for key, (model_label, stage_label) in curve_keys.items():
        for point in curves.get(key, {}).get("val", []):
            loss = point.get("val_loss")
            if loss is None:
                continue
            curve_rows.append(
                {
                    "Model": model_label,
                    "Tahap": stage_label,
                    "Step": point.get("step"),
                    "Val loss": loss,
                }
            )
    if curve_rows:
        with st.expander("Kurva validasi per tahap"):
            curve_frame = pd.DataFrame(curve_rows)
            curve_columns = st.columns(2)
            for column, model_label in zip(curve_columns, ("163M", "400M")):
                subset = curve_frame[curve_frame["Model"] == model_label]
                curve_chart = (
                    alt.Chart(subset)
                    .mark_line()
                    .encode(
                        x=alt.X("Step:Q", title="Langkah"),
                        y=alt.Y("Val loss:Q", title="Validation loss"),
                        color=alt.Color(
                            "Tahap:N",
                            scale=alt.Scale(
                                domain=[
                                    "Base (Pretrain)",
                                    "Instruct (SFT)",
                                    "Chat (SimPO)",
                                ],
                                range=STAGE_COLORS,
                            ),
                            title=None,
                        ),
                        tooltip=[
                            "Tahap",
                            alt.Tooltip("Step:Q", format=",.0f"),
                            alt.Tooltip("Val loss:Q", format=".3f"),
                        ],
                    )
                    .properties(height=260, title=model_label)
                )
                column.altair_chart(curve_chart, use_container_width=True)
            st.caption(
                "Loss antar tahap tidak sepenuhnya sebanding karena "
                "objektif pretraining, SFT, dan SimPO berbeda."
            )


def render_scalability(models):
    scale_frame = build_scale_frame(models)
    if scale_frame.empty:
        st.info("Data skalabilitas model belum tersedia.")
        return
    display_frame = scale_frame.copy()
    display_frame["Rasio aktif"] = display_frame["Rasio aktif"].apply(
        lambda value: f"{value:.1%}" if pd.notna(value) else "-"
    )
    st.dataframe(
        display_frame.round(
            {
                "Total parameter (M)": 1,
                "Aktif per token (M)": 1,
                "Token latih (B)": 2,
                "Val PPL": 2,
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    scale_chart = (
        alt.Chart(scale_frame)
        .mark_circle(size=320, opacity=0.9)
        .encode(
            x=alt.X(
                "Total parameter (M):Q",
                title="Total parameter (juta)",
                scale=alt.Scale(domain=[0, 450]),
            ),
            y=alt.Y("Val PPL:Q", title="Validation perplexity pretrain"),
            color=alt.Color(
                "Model:N",
                scale=alt.Scale(
                    domain=scale_frame["Model"].tolist(),
                    range=MODEL_COLORS,
                ),
                title=None,
            ),
            tooltip=[
                "Model",
                alt.Tooltip("Total parameter (M):Q", format=".1f"),
                alt.Tooltip("Aktif per token (M):Q", format=".1f"),
                alt.Tooltip("Token latih (B):Q", format=".2f"),
                alt.Tooltip("Val PPL:Q", format=".2f"),
            ],
        )
    )
    labels = scale_chart.mark_text(dy=-14).encode(
        text="Model:N", color=alt.value("#37474F")
    )
    st.altair_chart(scale_chart + labels, use_container_width=True)
    st.caption(
        "66M memakai checkpoint pretrain v2 dengan sekitar 0.85B token, "
        "sedangkan 163M dan 400M-A188M memakai sekitar 2.10B token. "
        "Perbedaan anggaran token perlu diperhitungkan saat membaca PPL."
    )


def render_quality_and_decoding(models, telemetry, benchmark):
    stage_frame = build_stage_frame(
        models, {"Liwet-163M", "Liwet-400M-A188M"}
    )
    lookup = {model.get("family"): model for model in models}
    model_66 = lookup.get("Liwet-66M")
    training_rows = []
    if model_66:
        one = model_66.get("pretrain_v1", {})
        two = model_66.get("pretrain_v2", {})
        chat_sft = model_66.get("chat_sft", {})
        training_rows.extend(
            [
                {
                    "Model": "Liwet-66M",
                    "Tahap": "Base v1",
                    "Val loss": one.get("val_loss"),
                    "Val PPL": one.get("val_ppl"),
                    "Catatan": "1 epoch",
                },
                {
                    "Model": "Liwet-66M",
                    "Tahap": "Base v2",
                    "Val loss": two.get("val_loss"),
                    "Val PPL": two.get("val_ppl"),
                    "Catatan": "2 epoch",
                },
                {
                    "Model": "Liwet-66M",
                    "Tahap": "Chat SFT",
                    "Val loss": chat_sft.get("val_loss"),
                    "Val PPL": chat_sft.get("val_ppl"),
                    "Catatan": "heldout PPL "
                    f"{chat_sft.get('heldout_ppl', 0):.2f}",
                },
            ]
        )
    for _, row in stage_frame.iterrows():
        training_rows.append(
            {
                "Model": row["Model"],
                "Tahap": row["Tahap"],
                "Val loss": row["Val loss"],
                "Val PPL": row["Val PPL"],
                "Catatan": (
                    f"Akurasi val {row['Val accuracy']:.2f}"
                    if pd.notna(row.get("Val accuracy"))
                    else "-"
                ),
            }
        )
    training_frame = pd.DataFrame(training_rows)

    st.markdown("**Ringkasan metrik pelatihan**")
    st.dataframe(
        training_frame.round({"Val loss": 3, "Val PPL": 2}),
        hide_index=True,
        use_container_width=True,
    )

    benchmark_frame = build_benchmark_frame(benchmark)
    human_columns = [
        column
        for column in benchmark_frame.columns
        if any(
            token in str(column).lower()
            for token in ("human", "manual", "rating", "penilai", "kualitatif")
        )
    ]
    st.markdown("**Evaluasi kualitatif manusia**")
    if human_columns:
        label_column = find_label_column(benchmark_frame)
        selected_columns = (
            [label_column] + human_columns
            if label_column and label_column not in human_columns
            else human_columns
        )
        st.dataframe(
            benchmark_frame[selected_columns],
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info(
            "Skor penilaian manusia belum tersedia pada data eksperimen. "
            "Contoh keluaran di bawah dapat digunakan untuk penilaian manual."
        )

    chat = telemetry.get("chat_comparison_66m", {})
    prompt_options = chat.get("prompts", [])
    if prompt_options and chat.get("models"):
        with st.expander("Contoh keluaran untuk penilaian kualitatif"):
            selected_prompt = st.selectbox(
                "Prompt", prompt_options, key="analysis_quality_prompt"
            )
            model_columns = st.columns(len(chat["models"]))
            for column, (name, payload) in zip(
                model_columns, chat["models"].items()
            ):
                with column:
                    st.markdown(f"**{name}**")
                    matching = next(
                        (
                            item
                            for item in payload.get("prompts", [])
                            if item.get("text") == selected_prompt
                        ),
                        None,
                    )
                    generation = (
                        matching.get("generation")
                        if matching
                        else None
                    )
                    st.markdown(generation or "_(kosong)_")

    st.markdown("**Preset konfigurasi decoding**")
    preset_frame = pd.DataFrame(DECODING_PRESETS)
    st.dataframe(
        preset_frame,
        hide_index=True,
        use_container_width=True,
    )
    st.warning(
        "Hindari kombinasi temperature >= 1.1 dengan "
        "repetition_penalty = 1.0 karena berisiko menghasilkan "
        "pengulangan dan keluaran yang tidak stabil."
    )
    optimal_frame = preset_frame[preset_frame["Status"] == "Optimal"]
    avoid_frame = preset_frame[preset_frame["Status"] == "Hindari"]
    preset_chart = (
        alt.Chart(optimal_frame)
        .mark_circle(size=220)
        .encode(
            x=alt.X(
                "Temperature:Q",
                scale=alt.Scale(domain=[0, 1.3]),
                title="Temperature",
            ),
            y=alt.Y(
                "Repetition penalty:Q",
                scale=alt.Scale(domain=[0.98, 1.2]),
                title="Repetition penalty",
            ),
            color=alt.Color(
                "Preset:N",
                scale=alt.Scale(
                    domain=optimal_frame["Preset"].tolist(),
                    range=["#1565C0", "#6A1B9A", "#2E7D32", "#EF6C00"],
                ),
                title=None,
            ),
            tooltip=[
                "Preset",
                "Max token",
                "Temperature",
                "Top-k",
                "Top-p",
                "Repetition penalty",
            ],
        )
    )
    avoid_chart = (
        alt.Chart(avoid_frame)
        .mark_point(size=320, shape="cross", color="#C62828")
        .encode(
            x=alt.X("Temperature:Q"),
            y=alt.Y("Repetition penalty:Q"),
            tooltip=[
                "Preset",
                "Temperature",
                "Repetition penalty",
            ],
        )
    )
    st.altair_chart(preset_chart + avoid_chart, use_container_width=True)
    st.caption(
        "Tanda silang merah menandai konfigurasi yang harus dihindari. "
        "Preset optimal mempertahankan repetition penalty di atas 1.0."
    )

    st.markdown("**Benchmark runtime**")
    if benchmark is None:
        st.info(
            "Hasil benchmark runtime akan muncul otomatis setelah "
            "benchmark dijalankan."
        )
    elif isinstance(benchmark, dict) and benchmark.get("error"):
        st.error("Berkas benchmark_results.json tidak dapat dibaca.")
    elif benchmark_frame.empty:
        st.info("Berkas benchmark_results.json belum berisi data benchmark.")
    else:
        st.dataframe(
            benchmark_frame,
            hide_index=True,
            use_container_width=True,
        )
        speed_column = find_metric_column(
            benchmark_frame,
            ("tok_s", "tokens_per_second", "token/s", "throughput", "speed"),
        )
        latency_column = find_metric_column(
            benchmark_frame,
            ("latency", "seconds", "duration", "waktu"),
        )
        label_column = find_label_column(benchmark_frame)
        if speed_column and label_column:
            speed_chart = (
                alt.Chart(benchmark_frame)
                .mark_bar()
                .encode(
                    x=alt.X(f"{label_column}:N", title=None),
                    y=alt.Y(f"{speed_column}:Q", title="Token per detik"),
                    color=alt.Color(
                        f"{label_column}:N", legend=None
                    ),
                    tooltip=[
                        label_column,
                        alt.Tooltip(f"{speed_column}:Q", format=".2f"),
                    ],
                )
            )
            st.altair_chart(speed_chart, use_container_width=True)
        if latency_column and label_column:
            latency_chart = (
                alt.Chart(benchmark_frame)
                .mark_bar(color="#6A1B9A")
                .encode(
                    x=alt.X(f"{label_column}:N", title=None),
                    y=alt.Y(f"{latency_column}:Q", title="Latensi"),
                    tooltip=[
                        label_column,
                        alt.Tooltip(f"{latency_column}:Q", format=".3f"),
                    ],
                )
            )
            st.altair_chart(latency_chart, use_container_width=True)


telemetry = load_json(str(TELEMETRY_FILE))
benchmark = load_json(str(BENCHMARK_FILE))

if telemetry is None:
    st.error("Data telemetri belum tersedia.")
    st.stop()
if isinstance(telemetry, dict) and telemetry.get("error"):
    st.error("Data telemetri tidak dapat dibaca.")
    st.stop()

models = telemetry.get("models_summary", [])
if not models:
    st.warning("Ringkasan model belum tersedia pada telemetri.")

st.title("Analisis & Evaluasi")
st.caption(
    "Eksperimen tiga skala model Liwet: arsitektur dense versus sparse MoE, "
    "anggaran data, tahapan alignment, skalabilitas parameter, serta "
    "kualitas keluaran dan konfigurasi decoding."
)

tabs = st.tabs(
    [
        "Dense vs MoE",
        "Epoch & Chinchilla",
        "Tahapan Model",
        "Skalabilitas",
        "Kualitas & Decoding",
    ]
)

with tabs[0]:
    render_architecture(models)

with tabs[1]:
    render_epoch_comparison(models, telemetry)

with tabs[2]:
    render_stage_progression(models, telemetry)

with tabs[3]:
    render_scalability(models)

with tabs[4]:
    render_quality_and_decoding(models, telemetry, benchmark)
