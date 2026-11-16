"""Data-drift monitoring: training distribution (reference) vs. recent live traffic (API logs).

1. Per-feature two-sample Kolmogorov-Smirnov test. A feature "drifts" when p < p_value_threshold.
2. An Evidently HTML report for humans (reports/drift_report.html).
3. If more than drift_share_threshold of the features drift, retraining is recommended — and
   triggered when called with --retrain (or by the Prefect daily-monitoring flow).

    python -m src.monitor             # report only
    python -m src.monitor --retrain   # report, and run the retraining flow on drift
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp

from src.config import REPORTS_DIR, load_params, resolve


# ---------------------------------------------------------------- inputs
def load_logs(log_dir: Path, lookback_hours: float | None = None) -> pd.DataFrame:
    """Read every predictions-*.jsonl file; keep only rows newer than lookback_hours (if given)."""
    files = sorted(Path(log_dir).glob("predictions-*.jsonl"))
    if not files:
        return pd.DataFrame()
    df = pd.concat((pd.read_json(f, lines=True) for f in files), ignore_index=True)
    if lookback_hours is not None and not df.empty:
        ts = pd.to_datetime(df["ts"], utc=True, format="ISO8601")
        df = df[ts >= datetime.now(UTC) - timedelta(hours=lookback_hours)]
    return df.reset_index(drop=True)


# ---------------------------------------------------------------- statistics
def ks_drift(
    reference: pd.DataFrame, current: pd.DataFrame, features: list[str], p_threshold: float = 0.01
) -> pd.DataFrame:
    """One KS test per feature. KS compares the two empirical CDFs; statistic = max gap."""
    rows = []
    for col in features:
        stat, p = ks_2samp(reference[col].dropna(), current[col].dropna())
        rows.append({"feature": col, "ks_stat": float(stat), "p_value": float(p),
                     "drifted": bool(p < p_threshold)})
    return pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)


def evidently_report(reference: pd.DataFrame, current: pd.DataFrame, path: Path) -> float | None:
    """Write Evidently's HTML drift report; return Evidently's own drifted-column share.

    Evidently 0.7 replaced the API (Report([...]).run() -> Snapshot); older versions used
    evidently.report.Report(metrics=[...]).run(). Detect the installed version and branch.
    """
    import evidently

    major, minor = (int(x) for x in evidently.__version__.split(".")[:2])
    if (major, minor) >= (0, 7):
        from evidently import Report
        from evidently.presets import DataDriftPreset

        snapshot = Report([DataDriftPreset()]).run(current_data=current, reference_data=reference)
        snapshot.save_html(str(path))
        for metric in snapshot.dict()["metrics"]:
            if metric["metric_id"].startswith("DriftedColumnsCount"):
                return float(metric["value"]["share"])
        return None

    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    report.save_html(str(path))
    return float(report.as_dict()["metrics"][0]["result"]["share_of_drifted_columns"])


# ---------------------------------------------------------------- orchestration
def run_drift_check(lookback_hours: float | None = None) -> dict:
    params = load_params()
    mp = params["monitor"]
    features: list[str] = mp["features"]
    reference = pd.read_parquet(resolve(params["data"]["processed_dir"]) / "reference.parquet")
    current = load_logs(resolve(mp["log_dir"]), lookback_hours)

    result: dict = {
        "checked_at": datetime.now(UTC).isoformat(),
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "should_retrain": False,
    }
    if len(current) < mp["min_samples"]:
        result["status"] = f"skipped: need >= {mp['min_samples']} logged requests"
        print(json.dumps(result, indent=2))
        return result

    ks = ks_drift(reference, current, features, mp["p_value_threshold"])
    share = float(ks["drifted"].mean())
    REPORTS_DIR.mkdir(exist_ok=True)
    ref_sample = reference[features].sample(
        min(len(reference), mp["evidently_reference_sample"]), random_state=42
    )
    evidently_share = evidently_report(ref_sample, current[features],
                                       REPORTS_DIR / "drift_report.html")
    result.update({
        "status": "checked",
        "drifted_features": ks.loc[ks["drifted"], "feature"].tolist(),
        "n_drifted": int(ks["drifted"].sum()),
        "n_features": len(features),
        "drift_share": round(share, 4),
        "drift_share_threshold": mp["drift_share_threshold"],
        "evidently_drift_share": evidently_share,
        "should_retrain": share > mp["drift_share_threshold"],
        "ks_results": ks.round(6).to_dict(orient="records"),
        "html_report": str(REPORTS_DIR / "drift_report.html"),
    })
    (REPORTS_DIR / "drift.json").write_text(json.dumps(result, indent=2), newline="\n")
    summary = {k: v for k, v in result.items() if k != "ks_results"}
    print(json.dumps(summary, indent=2))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Check data drift between reference and live logs")
    ap.add_argument("--lookback-hours", type=float, default=None,
                    help="only use logs from the last N hours (default: all logs)")
    ap.add_argument("--retrain", action="store_true", help="run the retraining flow on drift")
    args = ap.parse_args()
    result = run_drift_check(args.lookback_hours)
    if result["should_retrain"]:
        print(f"DRIFT: {result['n_drifted']}/{result['n_features']} features drifted "
              f"(> {result['drift_share_threshold']:.0%}) -> retraining recommended")
        if args.retrain:
            from src.flows import retraining_flow  # lazy: avoids importing Prefect unless needed

            retraining_flow()
    else:
        print("No significant drift.")


if __name__ == "__main__":
    main()
