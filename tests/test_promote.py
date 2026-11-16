import mlflow
import mlflow.sklearn
import pytest
from mlflow.tracking import MlflowClient
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from src.features import build_features
from src.promote import decide_promotion, promote

NAME = "fraud-detector-test"


# ---------------------------------------------------------------- pure decision rule
def test_first_model_always_promoted():
    assert decide_promotion(0.10, None, 0.005)[0] is True


@pytest.mark.parametrize(
    "challenger,champion,expected",
    [(0.810, 0.800, True), (0.805, 0.800, False), (0.804, 0.800, False), (0.700, 0.800, False)],
)
def test_requires_strictly_more_than_min_improvement(challenger, champion, expected):
    assert decide_promotion(challenger, champion, 0.005)[0] is expected


# ---------------------------------------------------------------- end-to-end with a real registry
@pytest.fixture
def client(tmp_path):
    """A throwaway SQLite-backed MLflow registry — no server process involved."""
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    exp = mlflow.create_experiment("promote-test", artifact_location=(tmp_path / "art").as_uri())
    mlflow.set_experiment(experiment_id=exp)
    yield MlflowClient(uri)
    mlflow.set_tracking_uri(None)


def _register(model, X, y) -> None:
    with mlflow.start_run():
        mlflow.sklearn.log_model(
            model.fit(X, y),
            artifact_path="model",
            registered_model_name=NAME,
            pyfunc_predict_fn="predict_proba",
        )


def test_champion_challenger_lifecycle(client, raw_df):
    X, y = build_features(raw_df), raw_df["Class"]

    # v1: weak model, no champion exists -> becomes champion
    _register(DummyClassifier(strategy="prior"), X, y)
    r1 = promote(client, NAME, X, y)
    assert r1.promoted and r1.champion_version is None
    assert str(client.get_model_version_by_alias(NAME, "champion").version) == "1"

    # running again with nothing new is a no-op
    assert promote(client, NAME, X, y).promoted is False

    # v2: much better model -> wins by far more than 0.005 -> promoted
    _register(LogisticRegression(max_iter=1000), X, y)
    r2 = promote(client, NAME, X, y)
    assert r2.promoted and r2.challenger_pr_auc > r2.champion_pr_auc + 0.005
    assert str(client.get_model_version_by_alias(NAME, "champion").version) == "2"

    # v3: identical model -> gain 0 -> champion stays v2, but v3 is tagged challenger
    _register(LogisticRegression(max_iter=1000), X, y)
    r3 = promote(client, NAME, X, y)
    assert r3.promoted is False
    assert str(client.get_model_version_by_alias(NAME, "champion").version) == "2"
    assert str(client.get_model_version_by_alias(NAME, "challenger").version) == "3"
