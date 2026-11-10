"""Measure Gradio chat_send overhead: HF Hub force_refresh vs stream_generate vs HTTP/SSE yield."""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCAL_MODELS = REPO_ROOT.parent / "TUBES_1_SAFETENSOR"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="fadhilahkhairogi/Liwet-400M-A188M-id-chat")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--local-dir", default=str(DEFAULT_LOCAL_MODELS))
    parser.add_argument("--out", default=str(REPO_ROOT / "docs" / "bench" / "gradio_overhead.json"))
    args = parser.parse_args()

    os.environ["LIWET_LOCAL_MODELS_DIR"] = args.local_dir
    sys.path.insert(0, str(REPO_ROOT / "liwet-space"))
    import liwet_models as lm  # type: ignore

    # Force local weights while keeping _is_repo_on_hub untouched to measure real B6 latency
    orig_dl = lm._download_model_files
    lm._download_model_files = lambda repo_id, token=None: lm._find_local_model_files(repo_id) or orig_dl(repo_id, token=token)

    import app as liwet_app  # type: ignore

    # 1. Measure _is_repo_available(..., force_refresh=True)
    t0 = time.perf_counter()
    avail = liwet_app._is_repo_available(args.model, force_refresh=True)
    hub_check_s = time.perf_counter() - t0

    # Warm load once so model load isn't mixed into inference timing
    _ = lm.load_liwet_model(args.model)

    # Warm one short generation so one-off work (CUDA graph capture, autotune,
    # lazy kernel init) isn't attributed to the measured stream.
    for _ in liwet_app.chat_send(
        "Halo",
        [],
        args.model,
        lm.DEFAULT_SYSTEM_PROMPT,
        8,
        0.0,
        40,
        0.9,
        1.1,
        True,
        42,
    ):
        pass

    # 2. Measure direct Python generator app.chat_send (no HTTP)
    # Simulate 6-turn history (realistic chat payload)
    hist = []
    for i in range(6):
        hist.append({"role": "user", "content": f"Pertanyaan sebelumnya ke-{i+1} tentang sejarah Nusantara."})
        hist.append({"role": "assistant", "content": "Berikut ringkasan singkat mengenai topik tersebut dalam beberapa kalimat yang cukup informatif."})

    # Disable early EOS temporarily to get full max_new_tokens yield stream
    orig_stop = lm._stop_token_ids
    lm._stop_token_ids = lambda tok: set()
    try:
        t_gen_start = time.perf_counter()
        yield_gaps_ms = []
        t_prev = t_gen_start
        last_stats_md = ""
        n_yields = 0
        for display, _msg, stats_md, status_md in liwet_app.chat_send(
            "Jelaskan secara detail tentang arsitektur Transformer dan mekanisme attention.",
            hist,
            args.model,
            lm.DEFAULT_SYSTEM_PROMPT,
            args.max_new_tokens,
            0.0,
            40,
            0.9,
            1.1,
            True,
            42,
        ):
            now = time.perf_counter()
            yield_gaps_ms.append((now - t_prev) * 1000.0)
            t_prev = now
            n_yields += 1
            if stats_md:
                last_stats_md = stats_md
        direct_wall_s = time.perf_counter() - t_gen_start
    finally:
        lm._stop_token_ids = orig_stop

    # 3. Measure full Gradio server HTTP/SSE round-trip via gradio_client if available
    gradio_http_result = {}
    try:
        from gradio_client import Client  # type: ignore
        liwet_app.LAUNCH_KWARGS.update(
            server_name="127.0.0.1",
            server_port=7869,
            share=False,
            prevent_thread_lock=True,
            quiet=True,
        )
        _, local_url, _ = liwet_app.demo.launch(**liwet_app.LAUNCH_KWARGS)
        client = Client(local_url, verbose=False)
        hist_cli = (
            [
                {"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
                for m in hist
            ]
            if getattr(liwet_app, "GRADIO_MAJOR", 5) >= 6
            else hist
        )
        t_cli0 = time.perf_counter()
        job = client.submit(
            "Jelaskan singkat apa itu kecerdasan buatan dalam dua paragraf.",
            hist_cli,
            args.model,
            lm.DEFAULT_SYSTEM_PROMPT,
            args.max_new_tokens,
            0.0,
            40,
            0.9,
            1.1,
            True,
            42,
            api_name="/chat_send",
        )
        events = 0
        final_out = None
        for update in job:
            events += 1
            final_out = update
        http_wall_s = time.perf_counter() - t_cli0
        gradio_http_result = {
            "http_wall_s": round(http_wall_s, 3),
            "sse_events": events,
            "final_stats_md": final_out[2] if isinstance(final_out, (list, tuple)) and len(final_out) > 2 else "",
        }
        liwet_app.demo.close()
    except Exception as err:
        gradio_http_result = {"error": f"{type(err).__name__}: {err}"}

    res = {
        "model": args.model,
        "max_new_tokens": args.max_new_tokens,
        "hub_force_refresh_avail": avail,
        "hub_force_refresh_s": round(hub_check_s, 3),
        "direct_chat_send": {
            "yield_count": n_yields,
            "wall_s": round(direct_wall_s, 3),
            "last_stats_md": last_stats_md,
            "yield_gaps_ms_sum": round(sum(yield_gaps_ms), 1),
        },
        "gradio_http_sse": gradio_http_result,
    }
    Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
