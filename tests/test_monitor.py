import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from src.monitor import evidently_report, ks_drift, load_logs

FEATURES = ["V1", "V2", "V3", "Amount"]


def _frame(n: int, seed: int, shift: dict | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, 3)), columns=FEATURES[:3])
    df["Amount"] = rng.lognormal(3, 1, n)
    for col, delta in (shift or {}).items():
        df[col] = df[col] + delta
    return df


def test_no_drift_on_same_distribution():
    ks = ks_drift(_frame(5000, 0), _frame(1000, 1), FEATURES, 0.01)
    assert ks["drifted"].sum() == 0


def test_detects_shifted_features_only():
    ks = ks_drift(_frame(5000, 0), _frame(1000, 1, {"V1": 1.0, "V3": -1.0}), FEATURES, 0.01)
    assert set(ks.loc[ks["drifted"], "feature"]) == {"V1", "V3"}


def test_amount_scaling_is_detected():
    cur = _frame(1000, 1)
    cur["Amount"] *= 3
    ks = ks_drift(_frame(5000, 0), cur, FEATURES, 0.01)
    assert ks.set_index("feature").loc["Amount", "drifted"]


def test_load_logs_with_lookback(tmp_path):
    now = datetime.now(UTC)
    rows = [{"ts": (now - timedelta(hours=h)).isoformat(), "V1": float(h)} for h in (1, 2, 48)]
    (tmp_path / "predictions-20250101.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    assert len(load_logs(tmp_path)) == 3
    assert len(load_logs(tmp_path, lookback_hours=24)) == 2
    assert load_logs(tmp_path / "empty").empty


def test_evidently_report_written(tmp_path):
    path = tmp_path / "drift.html"
    share = evidently_report(_frame(500, 0), _frame(300, 1, {"V1": 3.0, "V2": 3.0}), path)
    assert path.exists() and path.stat().st_size > 10_000
    assert share is None or 0.0 <= share <= 1.0
