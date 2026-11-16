"""Feature engineering shared by training AND serving.

Keeping one function used in both places prevents training-serving skew: the model sees exactly
the same transformation online as it saw offline. Deliberately depends only on numpy/pandas so the
slim API image can import it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

V_COLUMNS = [f"V{i}" for i in range(1, 29)]
RAW_FEATURE_COLUMNS = ["Time", *V_COLUMNS, "Amount"]
FEATURE_COLUMNS = [*V_COLUMNS, "log_amount", "hour"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw transactions (Time, V1..V28, Amount) -> model features (V1..V28, log_amount, hour).

    - log_amount: log(1 + Amount) tames the heavy right tail of transaction amounts.
    - hour: hour-of-day derived from Time (seconds since the first transaction in the dataset).
      Fraud rates differ between night and day.
    Extra columns (e.g. Class) are ignored; missing required columns raise KeyError.
    """
    missing = [c for c in RAW_FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"missing required columns: {missing}")
    if (df["Amount"] < 0).any():
        raise ValueError("Amount must be >= 0")

    out = df[V_COLUMNS].astype("float64").copy()
    out["log_amount"] = np.log1p(df["Amount"].astype("float64"))
    out["hour"] = ((df["Time"].astype("float64") // 3600) % 24).astype("float64")
    return out[FEATURE_COLUMNS].reset_index(drop=True)
