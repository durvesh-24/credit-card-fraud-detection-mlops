"""Shared fixtures. Nothing here needs a running MLflow server or the real dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import V_COLUMNS


def make_raw(n: int = 3000, fraud_rate: float = 0.05, seed: int = 1) -> pd.DataFrame:
    """Synthetic transactions with the creditcard.csv schema and a learnable fraud signal."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, 28)), columns=V_COLUMNS)
    df.insert(0, "Time", np.sort(rng.uniform(0, 172_800, n)))
    df["Amount"] = rng.lognormal(3, 1.2, n)
    fraud = rng.random(n) < fraud_rate
    df.loc[fraud, "V14"] -= 4
    df.loc[fraud, "V12"] -= 3
    df["Class"] = fraud.astype(int)
    return df


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    return make_raw()


@pytest.fixture(scope="session")
def fixture_model_dir(tmp_path_factory, raw_df) -> str:
    """A tiny logistic-regression model saved in MLflow format to a LOCAL folder.

    It has the same interface as the registered model (pyfunc.predict -> predict_proba),
    so the API can load it via MODEL_URI=<path> with no tracking server.
    """
    import mlflow.sklearn
    from mlflow.models import infer_signature
    from sklearn.linear_model import LogisticRegression

    from src.features import build_features

    X, y = build_features(raw_df), raw_df["Class"]
    model = LogisticRegression(max_iter=1000).fit(X, y)
    path = tmp_path_factory.mktemp("model") / "fraud_model"
    mlflow.sklearn.save_model(
        model,
        str(path),
        signature=infer_signature(X.head(), model.predict_proba(X.head())),
        pyfunc_predict_fn="predict_proba",
    )
    return str(path)
