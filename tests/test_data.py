import numpy as np
import pandas as pd
import pandera as pa
import pytest

from src.split import time_split
from src.validate import V_COLUMNS, validate_df


def _raw(n: int = 50, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, 28)), columns=V_COLUMNS)
    df.insert(0, "Time", rng.permutation(np.arange(n, dtype=float) * 10))
    df["Amount"] = rng.uniform(0, 500, n)
    df["Class"] = (rng.random(n) < 0.1).astype(int)
    return df


def test_valid_data_passes():
    out = validate_df(_raw())
    assert len(out) == 50


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.assign(Amount=-1.0),
        lambda d: d.assign(Class=2),
        lambda d: d.assign(Time=-5.0),
        lambda d: d.assign(V3=np.nan),
        lambda d: d.drop(columns="V7"),
        lambda d: d.assign(extra=1),
    ],
    ids=["negative_amount", "bad_class", "negative_time", "null", "missing_col", "extra_col"],
)
def test_invalid_data_fails(mutate):
    with pytest.raises(pa.errors.SchemaErrors):
        validate_df(mutate(_raw()))


def test_time_split_is_chronological_and_complete():
    df = _raw(100)
    train, val, test = time_split(df, 0.7, 0.15)
    assert (len(train), len(val), len(test)) == (70, 15, 15)
    # every training row happens before every validation row, which happens before every test row
    assert train["Time"].max() < val["Time"].min()
    assert val["Time"].max() < test["Time"].min()
    assert len(pd.concat([train, val, test]).drop_duplicates()) == len(df)


def test_time_split_rejects_bad_fractions():
    with pytest.raises(ValueError):
        time_split(_raw(), 0.9, 0.2)
