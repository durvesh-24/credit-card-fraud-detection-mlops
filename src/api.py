"""FastAPI fraud-scoring service.

- Loads MODEL_URI (default models:/fraud-detector@champion) at startup — the alias, not a fixed
  version, so promotion/rollback changes what is served without rebuilding the image.
- Applies the SAME build_features() used in training (no training-serving skew).
- Exposes Prometheus metrics at /metrics and appends every request + prediction to
  PREDICTION_LOG_DIR as JSON lines, which the drift monitor reads later.

    uvicorn src.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator, metrics
from pydantic import BaseModel, ConfigDict, Field

from src.features import V_COLUMNS, build_features
from src.model_utils import fraud_probability

log = logging.getLogger("fraud-api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

DEFAULT_MODEL_URI = "models:/fraud-detector@champion"
RELOAD_COOLDOWN_S = 30  # how often /predict may retry loading when no model is available

# ---------------------------------------------------------------- custom Prometheus metrics
PREDICTIONS = Counter("fraud_predictions_total", "Transactions scored")
FLAGGED = Counter("fraud_flagged_transactions_total", "Transactions flagged as fraud")
SCORES = Histogram(
    "fraud_score", "Predicted fraud probability",
    buckets=(0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
)
MODEL_LOADED = Gauge("fraud_model_loaded", "1 when a model is loaded and serving")


# ---------------------------------------------------------------- request / response schemas
class Transaction(BaseModel):
    """One card transaction. Invalid input is rejected with HTTP 422 before reaching the model."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": {"Time": 406.0, "Amount": 149.62, "V": [0.0] * 28}},
    )
    Time: float = Field(..., ge=0, description="Seconds since the first transaction")
    Amount: float = Field(..., ge=0, description="Transaction amount")
    V: list[float] = Field(..., min_length=28, max_length=28, description="PCA features V1..V28")

    def to_frame(self) -> pd.DataFrame:
        v_values = dict(zip(V_COLUMNS, self.V, strict=True))
        return pd.DataFrame([{"Time": self.Time, **v_values, "Amount": self.Amount}])


class Prediction(BaseModel):
    fraud_probability: float
    is_fraud: bool
    threshold: float
    model_version: str


# ---------------------------------------------------------------- model state
class ModelState:
    def __init__(self) -> None:
        self.reset()
        self.lock = threading.Lock()

    def reset(self) -> None:
        self.model = None
        self.uri = DEFAULT_MODEL_URI
        self.version = "none"
        self.threshold = 0.5
        self.log_dir = Path("data/logs")
        self.last_attempt = 0.0

    def configure(self) -> None:
        self.uri = os.getenv("MODEL_URI", DEFAULT_MODEL_URI)
        self.threshold = float(os.getenv("FRAUD_THRESHOLD", "0.5"))
        self.log_dir = Path(os.getenv("PREDICTION_LOG_DIR", "data/logs"))

    def load(self) -> bool:
        """Load MODEL_URI. On failure the previously loaded model (if any) keeps serving."""
        self.last_attempt = time.monotonic()
        try:
            model = mlflow.pyfunc.load_model(self.uri)
        except Exception as exc:  # registry down, alias missing, bad path...
            log.warning("could not load model from %s: %s", self.uri, exc)
            MODEL_LOADED.set(1 if self.model is not None else 0)
            return False
        self.model, self.version = model, self._resolve_version(model)
        MODEL_LOADED.set(1)
        log.info("loaded %s (version %s), threshold=%.4f", self.uri, self.version, self.threshold)
        return True

    def _resolve_version(self, model) -> str:
        """Turn models:/name@alias into the concrete registry version, for traceability."""
        if self.uri.startswith("models:/") and "@" in self.uri:
            try:
                name, alias = self.uri.removeprefix("models:/").split("@", 1)
                return str(mlflow.MlflowClient().get_model_version_by_alias(name, alias).version)
            except Exception:
                pass
        return f"run:{model.metadata.run_id}" if model.metadata.run_id else self.uri

    def ensure_loaded(self) -> None:
        if self.model is None and time.monotonic() - self.last_attempt > RELOAD_COOLDOWN_S:
            self.load()
        if self.model is None:
            raise HTTPException(status_code=503, detail="no model loaded")

    def append_log(self, txn: Transaction, proba: float, is_fraud: bool) -> None:
        record = {
            "ts": datetime.now(UTC).isoformat(),
            "Time": txn.Time,
            **dict(zip(V_COLUMNS, txn.V, strict=True)),
            "Amount": txn.Amount,
            "fraud_probability": proba,
            "is_fraud": is_fraud,
            "model_version": self.version,
        }
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.log_dir / f"predictions-{datetime.now(UTC):%Y%m%d}.jsonl"
        with self.lock, open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


state = ModelState()


@asynccontextmanager
async def lifespan(_: FastAPI):
    state.reset()  # a fresh process starts with no model
    state.configure()
    state.load()
    yield


app = FastAPI(title="Fraud Detection API", version="1.0.0", lifespan=lifespan)
# Standard RED metrics (Rate, Errors, Duration). The handler-labelled latency histogram gets
# fine-grained buckets so Grafana can compute meaningful p50/p95/p99 for /predict.
LATENCY_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1, 2.5, 5)
Instrumentator(excluded_handlers=["/metrics"]).add(
    metrics.default(latency_lowr_buckets=LATENCY_BUCKETS)
).instrument(app).expose(app, include_in_schema=False)


# ---------------------------------------------------------------- endpoints
@app.get("/health")
def health():
    """200 when a model is loaded (ready to serve), 503 otherwise — used by Docker HEALTHCHECK."""
    body = {"model_loaded": state.model is not None, "model_uri": state.uri,
            "model_version": state.version, "threshold": state.threshold}
    if state.model is None:
        raise HTTPException(status_code=503, detail={"status": "no_model", **body})
    return {"status": "ok", **body}


@app.post("/predict", response_model=Prediction)
def predict(txn: Transaction) -> Prediction:
    state.ensure_loaded()
    proba = float(fraud_probability(state.model.predict(build_features(txn.to_frame())))[0])
    is_fraud = proba >= state.threshold
    PREDICTIONS.inc()
    SCORES.observe(proba)
    if is_fraud:
        FLAGGED.inc()
    state.append_log(txn, proba, is_fraud)
    return Prediction(fraud_probability=proba, is_fraud=is_fraud, threshold=state.threshold,
                      model_version=state.version)


@app.post("/reload")
def reload_model():
    """Re-resolve MODEL_URI — picks up a newly promoted @champion without restarting.
    NOTE: unauthenticated; in production put this behind auth or remove it."""
    state.configure()
    if not state.load():
        raise HTTPException(status_code=503, detail=f"could not load {state.uri}")
    return {"status": "reloaded", "model_version": state.version, "threshold": state.threshold}
