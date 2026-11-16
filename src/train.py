"""Train the fraud model, track everything in MLflow and register it in the Model Registry.

- Main model: XGBoost with scale_pos_weight (= #legit / #fraud in train) to counter the
  ~0.17% class imbalance.
- Baseline:   logistic regression (logged as its own run, not registered) — a model must beat it.
- Tuning:     optional Optuna search scored on the VALIDATION set, each trial a nested MLflow run.
- Threshold:  chosen on validation to maximise F1, then applied unchanged to test.
- Headline metric: PR-AUC (average precision). Accuracy is useless here: predicting "never fraud"
  scores 99.8% accuracy.

    python -m src.train                    # settings from params.yaml
    python -m src.train --tune --n-trials 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
import yaml
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import REPORTS_DIR, ROOT, load_params, resolve, tracking_uri
from src.features import build_features

matplotlib.use("Agg")  # headless: no display needed (CI, Docker, Prefect workers)
TARGET = "Class"


# ----------------------------------------------------------------------------- data
def load_split(processed_dir: Path, name: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(processed_dir / f"{name}.parquet")
    return build_features(df), df[TARGET].astype(int).reset_index(drop=True)


def data_version() -> str:
    """md5 of the raw data from its DVC pointer file: ties each model to the exact data it saw."""
    dvc_file = ROOT / "data/raw/creditcard.csv.dvc"
    if not dvc_file.exists():
        return "untracked"
    return yaml.safe_load(dvc_file.read_text())["outs"][0]["md5"]


# ----------------------------------------------------------------------------- metrics
def best_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.argmax(f1))])


def compute_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, proba)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }


def save_plots(y_true: np.ndarray, proba: np.ndarray, threshold: float, title: str) -> list[Path]:
    REPORTS_DIR.mkdir(exist_ok=True)
    cm_path, pr_path = REPORTS_DIR / "confusion_matrix.png", REPORTS_DIR / "pr_curve.png"

    cm = confusion_matrix(y_true, (proba >= threshold).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["legit", "fraud"]).plot(ax=ax, values_format="d")
    ax.set_title(f"{title} — test @ threshold {threshold:.3f}")
    fig.tight_layout()
    fig.savefig(cm_path, dpi=120)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(y_true, proba)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(recall, precision)
    ax.set(xlabel="Recall", ylabel="Precision", title=f"{title} — test PR curve")
    fig.tight_layout()
    fig.savefig(pr_path, dpi=120)
    plt.close(fig)
    return [cm_path, pr_path]


# ----------------------------------------------------------------------------- models
def make_xgb(hp: dict, scale_pos_weight: float, seed: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=hp["n_estimators"],
        max_depth=hp["max_depth"],
        learning_rate=hp["learning_rate"],
        subsample=hp["subsample"],
        colsample_bytree=hp["colsample_bytree"],
        min_child_weight=hp["min_child_weight"],
        early_stopping_rounds=hp["early_stopping_rounds"],
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
    )


def fit_xgb(hp, X_tr, y_tr, X_val, y_val, seed) -> XGBClassifier:
    spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    model = make_xgb(hp, spw, seed)
    # early stopping watches VALIDATION PR-AUC; the test set is never seen during fitting
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model


def make_logreg(hp: dict, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=hp["C"], max_iter=hp["max_iter"], class_weight="balanced", random_state=seed
                ),
            ),
        ]
    )


def tune_xgb(base_hp, n_trials, X_tr, y_tr, X_val, y_val, seed) -> dict:
    """Optuna search maximising VALIDATION PR-AUC. Each trial is a nested MLflow run."""

    def objective(trial: optuna.Trial) -> float:
        hp = {
            **base_hp,
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }
        with mlflow.start_run(run_name=f"trial-{trial.number:03d}", nested=True):
            model = fit_xgb(hp, X_tr, y_tr, X_val, y_val, seed)
            score = float(average_precision_score(y_val, model.predict_proba(X_val)[:, 1]))
            mlflow.log_params(hp)
            mlflow.log_metric("val_pr_auc", score)
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    mlflow.log_metric("best_trial_val_pr_auc", study.best_value)
    print(f"Optuna best val PR-AUC {study.best_value:.4f} with {study.best_params}")
    return {**base_hp, **study.best_params}


# ----------------------------------------------------------------------------- runs
def evaluate(model, X_val, y_val, X_test, y_test) -> tuple[float, dict, dict, np.ndarray]:
    val_proba = model.predict_proba(X_val)[:, 1]
    threshold = best_f1_threshold(y_val.to_numpy(), val_proba)  # chosen on VALIDATION only
    test_proba = model.predict_proba(X_test)[:, 1]
    return (
        threshold,
        compute_metrics(y_val.to_numpy(), val_proba, threshold),
        compute_metrics(y_test.to_numpy(), test_proba, threshold),
        test_proba,
    )


def log_metric_dicts(val_m: dict, test_m: dict, threshold: float) -> None:
    mlflow.log_metrics({f"val_{k}": v for k, v in val_m.items()})
    mlflow.log_metrics({f"test_{k}": v for k, v in test_m.items()})
    mlflow.log_metric("threshold", threshold)


def run_baseline(params, data) -> dict:
    X_tr, y_tr, X_val, y_val, X_test, y_test = data
    seed = params["train"]["random_state"]
    with mlflow.start_run(run_name="logreg-baseline"):
        mlflow.set_tags({"model_type": "logreg", "role": "baseline", "data_md5": data_version()})
        mlflow.log_params(params["train"]["logreg"])
        model = make_logreg(params["train"]["logreg"], seed).fit(X_tr, y_tr)
        threshold, val_m, test_m, _ = evaluate(model, X_val, y_val, X_test, y_test)
        log_metric_dicts(val_m, test_m, threshold)
    print(f"Baseline logreg: val PR-AUC {val_m['pr_auc']:.4f} | test PR-AUC {test_m['pr_auc']:.4f}")
    return {"threshold": threshold, "val": val_m, "test": test_m}


def run(model_type: str | None = None, tune: bool | None = None, n_trials: int | None = None,
        baseline: bool | None = None) -> dict:
    params = load_params()
    tp = params["train"]
    model_type = model_type or tp["model"]
    tune = tp["tune"] if tune is None else tune
    n_trials = n_trials or tp["n_trials"]
    baseline = tp["run_baseline"] if baseline is None else baseline
    seed = tp["random_state"]
    model_name = params["mlflow"]["registered_model_name"]

    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    processed = resolve(params["data"]["processed_dir"])
    X_tr, y_tr = load_split(processed, "train")
    X_val, y_val = load_split(processed, "val")
    X_test, y_test = load_split(processed, "test")
    data = (X_tr, y_tr, X_val, y_val, X_test, y_test)
    print(
        f"train {len(X_tr):,} rows ({y_tr.mean():.4%} fraud) | "
        f"val {len(X_val):,} | test {len(X_test):,}"
    )

    report: dict = {}
    if baseline:
        report["baseline_logreg"] = run_baseline(params, data)

    with mlflow.start_run(run_name=f"{model_type}{'-tuned' if tune else ''}") as run_:
        mlflow.set_tags({"model_type": model_type, "role": "candidate", "data_md5": data_version()})
        mlflow.log_params(
            {
                "train_rows": len(X_tr),
                "val_rows": len(X_val),
                "test_rows": len(X_test),
                "train_fraud_rate": float(y_tr.mean()),
            }
        )
        if model_type == "xgboost":
            hp = dict(tp["xgboost"])
            if tune:
                hp = tune_xgb(hp, n_trials, X_tr, y_tr, X_val, y_val, seed)
            model = fit_xgb(hp, X_tr, y_tr, X_val, y_val, seed)
            mlflow.log_params({**hp, "scale_pos_weight": model.get_params()["scale_pos_weight"],
                               "best_iteration": model.best_iteration})
        elif model_type == "logreg":
            hp = dict(tp["logreg"])
            model = make_logreg(hp, seed).fit(X_tr, y_tr)
            mlflow.log_params(hp)
        else:
            raise ValueError(f"unknown model type {model_type!r}")

        threshold, val_m, test_m, test_proba = evaluate(model, X_val, y_val, X_test, y_test)
        log_metric_dicts(val_m, test_m, threshold)
        for path in save_plots(y_test.to_numpy(), test_proba, threshold, model_type):
            mlflow.log_artifact(str(path), artifact_path="plots")

        # Signature = the input/output schema the model promises; the registry and serving use it.
        sample = X_val.head(200)
        signature = infer_signature(sample, model.predict_proba(sample))
        info = mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            input_example=X_tr.head(5),
            registered_model_name=model_name,
            pyfunc_predict_fn="predict_proba",  # pyfunc.predict -> [[p_legit, p_fraud], ...]
        )
        version = str(info.registered_model_version)
        client = MlflowClient()
        version_tags = {
            "threshold": f"{threshold:.6f}",
            "model_type": model_type,
            "val_pr_auc": f"{val_m['pr_auc']:.6f}",
            "data_md5": data_version(),
        }
        for key, value in version_tags.items():
            client.set_model_version_tag(model_name, version, key, value)

        report.update({
            "model_type": model_type,
            "tuned": tune,
            "threshold": threshold,
            "val": val_m,
            "test": test_m,
            "mlflow_run_id": run_.info.run_id,
            "registered_model": model_name,
            "registered_version": version,
        })

    REPORTS_DIR.mkdir(exist_ok=True)
    # LF on every OS, so the hash DVC records matches the file git stores
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(report, indent=2), newline="\n")
    print(f"{model_type}: val PR-AUC {val_m['pr_auc']:.4f} | test PR-AUC {test_m['pr_auc']:.4f} | "
          f"F1-optimal threshold {threshold:.4f} -> registered {model_name} v{version}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Train and register the fraud model")
    ap.add_argument("--model", choices=["xgboost", "logreg"])
    ap.add_argument("--tune", action="store_true", default=None, help="Optuna search on validation")
    ap.add_argument("--n-trials", type=int)
    ap.add_argument("--no-baseline", dest="baseline", action="store_false", default=None)
    args = ap.parse_args()
    run(args.model, args.tune, args.n_trials, args.baseline)


if __name__ == "__main__":
    main()
