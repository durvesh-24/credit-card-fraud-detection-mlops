"""Chronological train / validation / test split.

Fraud patterns evolve over time, and in production the model always predicts the FUTURE from the
PAST. A random split would leak future information into training and overstate performance, so we
sort by Time and cut: earliest 70% -> train, next 15% -> validation, last 15% -> test.

    python -m src.split
"""

from __future__ import annotations

import json

import pandas as pd

from src.config import REPORTS_DIR, load_params, resolve


def time_split(
    df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0 < train_frac < 1 or not 0 < val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError("need 0 < train_frac, val_frac and train_frac + val_frac < 1")
    df = df.sort_values("Time", kind="stable").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def run() -> dict:
    params = load_params()
    df = pd.read_csv(resolve(params["data"]["raw_path"]))
    train, val, test = time_split(df, params["split"]["train_frac"], params["split"]["val_frac"])

    out = resolve(params["data"]["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out / "train.parquet", index=False)
    val.to_parquet(out / "val.parquet", index=False)
    test.to_parquet(out / "test.parquet", index=False)
    # The training distribution is the "normal" that drift monitoring compares live traffic against.
    train.to_parquet(out / "reference.parquet", index=False)

    stats = {
        name: {
            "rows": int(len(part)),
            "fraud_rate": round(float(part["Class"].mean()), 6),
            "time_min": float(part["Time"].min()),
            "time_max": float(part["Time"].max()),
        }
        for name, part in {"train": train, "val": val, "test": test}.items()
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "split_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    run()
