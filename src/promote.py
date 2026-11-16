"""Champion / challenger promotion using MLflow Model Registry aliases.

The newest registered version is the CHALLENGER. It is scored side by side with the current
@champion on the SAME held-out test set. The "champion" alias moves only if the challenger wins
PR-AUC by more than `min_improvement` (params.yaml), so noise-level differences never cause churn.
The API always loads models:/fraud-detector@champion, so moving the alias == deploying.

    python -m src.promote
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import mlflow
import pandas as pd
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sklearn.metrics import average_precision_score

from src.config import REPORTS_DIR, load_params, resolve, tracking_uri
from src.features import build_features
from src.model_utils import fraud_probability


@dataclass
class PromotionResult:
    model_name: str
    challenger_version: str
    champion_version: str | None
    challenger_pr_auc: float | None
    champion_pr_auc: float | None
    promoted: bool
    reason: str


def decide_promotion(
    challenger_score: float, champion_score: float | None, min_improvement: float
) -> tuple[bool, str]:
    """Pure decision rule — easy to unit test."""
    if champion_score is None:
        return True, "no champion yet: first model becomes champion"
    gain = round(challenger_score - champion_score, 9)  # 0.805 - 0.800 is 0.005000000000000004
    if gain > min_improvement:
        return True, f"challenger wins by {gain:+.4f} (> {min_improvement})"
    return False, f"gain {gain:+.4f} is not > {min_improvement}: champion kept"


def latest_version(client: MlflowClient, name: str) -> ModelVersion:
    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        raise RuntimeError(f"no versions registered for '{name}' — run `python -m src.train` first")
    return max(versions, key=lambda v: int(v.version))


def version_by_alias(client: MlflowClient, name: str, alias: str) -> ModelVersion | None:
    try:
        return client.get_model_version_by_alias(name, alias)
    except MlflowException:  # alias not set yet
        return None


def score_version(name: str, version: str, X: pd.DataFrame, y: pd.Series) -> float:
    model = mlflow.pyfunc.load_model(f"models:/{name}/{version}")
    return float(average_precision_score(y, fraud_probability(model.predict(X))))


def promote(
    client: MlflowClient,
    name: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    min_improvement: float = 0.005,
    champion_alias: str = "champion",
    challenger_alias: str = "challenger",
) -> PromotionResult:
    # str(): the REST store returns versions as strings, the SQL store as ints
    challenger_v = str(latest_version(client, name).version)
    client.set_registered_model_alias(name, challenger_alias, challenger_v)
    champion = version_by_alias(client, name, champion_alias)
    champion_v = str(champion.version) if champion is not None else None

    if champion_v == challenger_v:
        return PromotionResult(name, challenger_v, champion_v, None, None, False,
                               "newest version is already champion")

    challenger_score = score_version(name, challenger_v, X_test, y_test)
    champion_score = score_version(name, champion_v, X_test, y_test) if champion_v else None
    promoted, reason = decide_promotion(challenger_score, champion_score, min_improvement)
    if promoted:
        client.set_registered_model_alias(name, champion_alias, challenger_v)
    return PromotionResult(
        name, challenger_v, champion_v, challenger_score, champion_score, promoted, reason
    )


def run() -> dict:
    params = load_params()
    mlflow.set_tracking_uri(tracking_uri())
    pp = params["promote"]
    test = pd.read_parquet(resolve(params["data"]["processed_dir"]) / "test.parquet")
    result = promote(
        MlflowClient(),
        params["mlflow"]["registered_model_name"],
        build_features(test),
        test["Class"].astype(int),
        pp["min_improvement"],
        pp["champion_alias"],
        pp["challenger_alias"],
    )
    out = asdict(result)
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "promotion.json").write_text(json.dumps(out, indent=2), newline="\n")
    print(json.dumps(out, indent=2))
    if result.promoted:
        print("@champion moved. Restart the API or call POST /reload to serve the new model.")
    return out


if __name__ == "__main__":
    run()
