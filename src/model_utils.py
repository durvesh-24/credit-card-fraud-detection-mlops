"""Helpers shared by promotion, serving and tests (numpy/pandas only — safe for the API image)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def fraud_probability(pred) -> np.ndarray:
    """Normalise a pyfunc prediction to a 1-D array of P(fraud).

    Models are logged with pyfunc_predict_fn="predict_proba", so pyfunc.predict returns an
    (n, 2) array [[p_legit, p_fraud], ...]. Accept 1-D output too, for robustness.
    """
    arr = pred.to_numpy() if isinstance(pred, pd.DataFrame | pd.Series) else np.asarray(pred)
    if arr.ndim == 2:
        return arr[:, -1].astype(float)
    return arr.astype(float)
