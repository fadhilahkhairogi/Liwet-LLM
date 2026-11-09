# Analysis Data

`benchmark_results.json` is tracked here (< 100 KiB policy).
`telemetry.json` (~463 KB raw curve dump) is omitted from the sterile GitHub release to satisfy `scripts/check_sterile.py` (`MAX_JSON_BYTES = 100 KiB`); `web/analysis.py` falls back gracefully when `telemetry.json` is absent.
