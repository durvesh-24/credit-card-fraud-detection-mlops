"""Validate the raw dataset against a pandera schema before anything else touches it.

A failing check stops the pipeline (non-zero exit) so bad data can never reach training.

    python -m src.validate
"""

from __future__ import annotations

import json
import sys

import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema

from src.config import REPORTS_DIR, load_params, resolve

V_COLUMNS = [f"V{i}" for i in range(1, 29)]
RAW_COLUMNS = ["Time", *V_COLUMNS, "Amount", "Class"]

SCHEMA = DataFrameSchema(
    {
        "Time": Column(float, Check.ge(0), nullable=False, coerce=True),
        **{c: Column(float, nullable=False, coerce=True) for c in V_COLUMNS},
        "Amount": Column(float, Check.ge(0), nullable=False, coerce=True),
        "Class": Column(int, Check.isin([0, 1]), nullable=False, coerce=True),
    },
    strict=True,  # unexpected extra columns are an error too
    ordered=False,
)


def validate_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return the coerced DataFrame, or raise pandera.errors.SchemaErrors listing every failure."""
    return SCHEMA.validate(df, lazy=True)  # lazy=True collects ALL failures, not just the first


def run() -> dict:
    params = load_params()
    raw_path = resolve(params["data"]["raw_path"])
    df = pd.read_csv(raw_path)
    try:
        df = validate_df(df)
    except pa.errors.SchemaErrors as err:
        print("Data validation FAILED:\n", err.failure_cases.head(20).to_string(), file=sys.stderr)
        raise
    summary = {
        "rows": int(len(df)),
        "fraud_count": int(df["Class"].sum()),
        "fraud_rate": round(float(df["Class"].mean()), 6),
        "valid": True,
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "validation.json").write_text(json.dumps(summary, indent=2))
    print(f"Validation passed: {summary}")
    return summary


if __name__ == "__main__":
    run()
