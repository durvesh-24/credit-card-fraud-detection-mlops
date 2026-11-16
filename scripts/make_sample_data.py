"""Generate a SYNTHETIC dataset with the same schema as Kaggle's creditcard.csv.

Used only for smoke tests / CI so the pipeline can run without the real data.
Numbers produced on this data are meaningless — never report them as results.

    python scripts/make_sample_data.py --rows 100000 --out data/raw/creditcard.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Columns where fraud rows are shifted, loosely mimicking the real dataset's strongest signals
FRAUD_SHIFTS = {"V4": 2.5, "V10": -3.0, "V12": -3.5, "V14": -4.0, "V17": -3.0, "V11": 2.0}


def make_sample(rows: int = 100_000, fraud_rate: float = 0.0035, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    time = np.sort(rng.uniform(0, 172_792, rows)).round()  # two days of seconds, like the original
    v = rng.normal(0, 1, size=(rows, 28))
    df = pd.DataFrame(v, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", time)
    df["Amount"] = np.round(rng.lognormal(mean=3.0, sigma=1.4, size=rows), 2)
    is_fraud = rng.random(rows) < fraud_rate
    for col, shift in FRAUD_SHIFTS.items():
        df.loc[is_fraud, col] += shift + rng.normal(0, 1.5, is_fraud.sum())
    df["Class"] = is_fraud.astype(int)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", type=int, default=100_000)
    ap.add_argument("--fraud-rate", type=float, default=0.0035)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/raw/creditcard.csv")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = make_sample(args.rows, args.fraud_rate, args.seed)
    df.to_csv(out, index=False)
    print(f"wrote {len(df):,} synthetic rows ({df['Class'].sum()} fraud) to {out}")


if __name__ == "__main__":
    main()
