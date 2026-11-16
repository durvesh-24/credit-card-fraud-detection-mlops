"""Send continuous, realistic traffic to the API so the Grafana dashboard shows live data.

Each request is one of:
- a normal transaction replayed from the test period,
- a real fraud from the test period (--fraud-share, default 2%), so the flagged panels move,
- an invalid request (--invalid-share, default 1%, e.g. a negative amount), so the 4xx panel moves.

    python scripts/demo_traffic.py                       # 10 minutes at ~5 requests/second
    python scripts/demo_traffic.py --minutes 30 --rps 10
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")  # API_URL may use a non-default port

V_COLUMNS = [f"V{i}" for i in range(1, 29)]


def payload(row) -> dict:
    return {"Time": float(row.Time), "Amount": float(row.Amount),
            "V": [float(getattr(row, c)) for c in V_COLUMNS]}


def invalid_payload(rng: np.random.Generator) -> dict:
    """A request the API must reject with HTTP 422."""
    return [
        {"Time": 100.0, "Amount": -5.0, "V": [0.0] * 28},  # negative amount
        {"Time": 100.0, "Amount": 10.0, "V": [0.0] * 27},  # 27 instead of 28 V values
        {"Time": -1.0, "Amount": 10.0, "V": [0.0] * 28},  # negative time
    ][rng.integers(3)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--rps", type=float, default=5, help="target requests per second")
    ap.add_argument("--fraud-share", type=float, default=0.02)
    ap.add_argument("--invalid-share", type=float, default=0.01)
    ap.add_argument("--api-url", default=os.getenv("API_URL", "http://localhost:8000"))
    ap.add_argument("--data", default="data/processed/test.parquet")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    df = pd.read_parquet(args.data)
    frauds = list(df[df["Class"] == 1].itertuples(index=False))
    normals = df[df["Class"] == 0]

    counts = {"normal": 0, "fraud": 0, "invalid": 0, "flagged": 0, "errors": 0}
    interval = 1.0 / args.rps
    end = time.monotonic() + args.minutes * 60
    next_report = time.monotonic() + 30
    print(f"sending to {args.api_url} for {args.minutes:g} min at ~{args.rps:g} req/s "
          f"({args.fraud_share:.0%} fraud, {args.invalid_share:.0%} invalid). Ctrl+C to stop.")

    with requests.Session() as s:
        try:
            while time.monotonic() < end:
                started = time.monotonic()
                roll = rng.random()
                if roll < args.invalid_share:
                    kind, body = "invalid", invalid_payload(rng)
                elif roll < args.invalid_share + args.fraud_share and frauds:
                    kind, body = "fraud", payload(frauds[rng.integers(len(frauds))])
                else:
                    kind = "normal"
                    row = normals.iloc[rng.integers(len(normals))]
                    body = {"Time": float(row["Time"]), "Amount": float(row["Amount"]),
                            "V": [float(row[c]) for c in V_COLUMNS]}
                counts[kind] += 1
                try:
                    r = s.post(f"{args.api_url}/predict", json=body, timeout=10)
                    if r.status_code == 200:
                        counts["flagged"] += r.json()["is_fraud"]
                    elif kind != "invalid":
                        counts["errors"] += 1
                except requests.RequestException:
                    counts["errors"] += 1
                if time.monotonic() >= next_report:
                    print(f"  {counts}")
                    next_report += 30
                time.sleep(max(0.0, interval - (time.monotonic() - started)))
        except KeyboardInterrupt:
            pass
    print(f"done: {counts}")


if __name__ == "__main__":
    main()
