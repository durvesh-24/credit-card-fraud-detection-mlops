import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_COLUMNS, V_COLUMNS, build_features


def _row(time: float = 0.0, amount: float = 0.0, **overrides) -> dict:
    row = {"Time": time, **{c: 0.0 for c in V_COLUMNS}, "Amount": amount}
    row.update(overrides)
    return row


def test_output_columns_and_order():
    out = build_features(pd.DataFrame([_row()]))
    assert list(out.columns) == FEATURE_COLUMNS
    assert len(FEATURE_COLUMNS) == 30


def test_log_amount():
    out = build_features(pd.DataFrame([_row(amount=0.0), _row(amount=np.e - 1)]))
    assert out["log_amount"].tolist() == pytest.approx([0.0, 1.0])


@pytest.mark.parametrize(
    "seconds,hour", [(0, 0), (3599, 0), (3600, 1), (23 * 3600 + 59, 23), (24 * 3600, 0), (90000, 1)]
)
def test_hour_of_day_wraps_every_24h(seconds, hour):
    assert build_features(pd.DataFrame([_row(time=seconds)]))["hour"].iloc[0] == hour


def test_v_columns_pass_through_unchanged():
    out = build_features(pd.DataFrame([_row(V1=1.5, V28=-2.25)]))
    assert out["V1"].iloc[0] == 1.5
    assert out["V28"].iloc[0] == -2.25


def test_extra_columns_ignored_and_index_reset():
    df = pd.DataFrame([_row(), _row()], index=[10, 20]).assign(Class=[0, 1])
    out = build_features(df)
    assert "Class" not in out.columns
    assert list(out.index) == [0, 1]


def test_missing_column_raises():
    with pytest.raises(KeyError):
        build_features(pd.DataFrame([_row()]).drop(columns="V5"))


def test_negative_amount_raises():
    with pytest.raises(ValueError):
        build_features(pd.DataFrame([_row(amount=-1)]))


def test_does_not_mutate_input():
    df = pd.DataFrame([_row(amount=10.0)])
    before = df.copy()
    build_features(df)
    pd.testing.assert_frame_equal(df, before)
