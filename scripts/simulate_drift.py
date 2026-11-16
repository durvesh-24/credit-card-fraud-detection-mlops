"""Send (optionally drifted) test transactions through the running API.

Drift recipe: Amount x3 (e.g. inflation / a new merchant mix) and Gaussian noise on several V
columns (e.g. a changed upstream feature pipeline). The API logs every request, so
`python -m src.monitor` afterwards should detect the shift.

    python scripts/simulate_drift.py --n 500              # drifted traffic
    python scripts/simulate_drift.py --n 500 --no-drift   # clean traffic for comparison
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")  # API_URL may use a non-default port

V_COLUMNS = [f"V{i}" for i in range(1, 29)]
DEFAULT_NOISE_COLS = [f"V{i}" for i in range(1, 13)]  # 12 of 29 monitored features + Amount


def make_drifted(df: pd.DataFrame, amount_factor: float, noise_cols: list[str], noise_std: float,
                 seed: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["Amount"] = out["Amount"] * amount_factor
    for col in noise_cols:
        out[col] = out[col] + rng.normal(0, noise_std, len(out))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--api-url", default=os.getenv("API_URL", "http://localhost:8000"))
    ap.add_argument("--data", default="data/processed/test.parquet")
    ap.add_argument("--amount-factor", type=float, default=3.0)
    ap.add_argument("--noise-std", type=float, default=1.5)
    ap.add_argument("--noise-cols", nargs="*", default=DEFAULT_NOISE_COLS)
    ap.add_argument("--no-drift", action="store_true", help="send the data unchanged")
    # default None = different rows every run. Re-sending identical rows duplicates log entries,
    # which inflates KS significance and makes untouched features look drifted.
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    df = pd.read_parquet(Path(args.data)).sample(n=args.n, random_state=args.seed)
    if not args.no_drift:
        df = make_drifted(df, args.amount_factor, args.noise_cols, args.noise_std, args.seed)

    ok = flagged = errors = 0
    with requests.Session() as s:
        for row in df.itertuples(index=False):
            payload = {"Time": float(row.Time), "Amount": float(row.Amount),
                       "V": [float(getattr(row, c)) for c in V_COLUMNS]}
            r = s.post(f"{args.api_url}/predict", json=payload, timeout=10)
            if r.status_code == 200:
                ok += 1
                flagged += r.json()["is_fraud"]
            else:
                errors += 1
    mode = "clean" if args.no_drift else f"drifted (Amount x{args.amount_factor}, noise on " \
        f"{len(args.noise_cols)} V cols)"
    print(f"sent {len(df)} {mode} transactions: {ok} ok, {errors} errors, "
          f"{flagged} flagged ({flagged / max(ok, 1):.2%})")


if __name__ == "__main__":
    main()
