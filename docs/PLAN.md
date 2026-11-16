# Project Plan — Credit-Card Fraud Detection MLOps System

## Folder structure

```
.
├── .dvc/                         # DVC internals (config, cache) — created by `dvc init`
├── .github/workflows/ci.yml      # CI: ruff + pytest on every push; CD: build & push image to GHCR on main
├── data/
│   ├── raw/creditcard.csv        # Kaggle dataset (git-ignored, tracked by DVC via creditcard.csv.dvc)
│   ├── processed/                # train/val/test/reference parquet (outputs of `dvc repro`)
│   └── logs/                     # JSONL request + prediction logs written by the API
├── docs/                         # PLAN.md, PROJECT_GUIDE.md, INTERVIEW_QA.md
├── monitoring/
│   ├── prometheus.yml            # Prometheus scrape config (scrapes api:8000/metrics)
│   └── grafana/
│       ├── provisioning/         # auto-wires the Prometheus datasource + dashboard folder
│       └── dashboards/fraud-api.json   # requests/s, p50/p95/p99, error rate, flagged rate
├── reports/                      # metrics.json, confusion matrix, drift report HTML, promotion.json
├── scripts/
│   ├── make_sample_data.py       # synthetic creditcard-shaped CSV for smoke tests / CI
│   ├── rollback.py               # move @champion alias to a chosen version
│   └── simulate_drift.py         # sends drifted test data through the API
├── src/
│   ├── config.py                 # loads params.yaml + .env, shared paths/constants
│   ├── validate.py               # pandera schema check on raw data
│   ├── split.py                  # time-based 70/15/15 split + reference.parquet
│   ├── features.py               # build_features() — shared by training AND serving
│   ├── train.py                  # XGBoost + LR baseline + optional Optuna, logs to MLflow, registers model
│   ├── promote.py                # champion/challenger comparison, moves @champion alias
│   ├── api.py                    # FastAPI service: /predict /health /metrics /reload
│   ├── monitor.py                # KS tests + Evidently drift report, triggers retraining
│   └── flows.py                  # Prefect flows: fraud-retraining, daily-monitoring
├── tests/                        # pytest suite (no MLflow server needed)
├── Dockerfile                    # slim API image with HEALTHCHECK
├── docker-compose.yml            # mlflow + api + prometheus + grafana
├── dvc.yaml / dvc.lock           # pipeline definition / exact hashes of last run
├── params.yaml                   # every tunable setting lives here
├── requirements.txt              # pinned direct dependencies (full dev/training env)
├── requirements-serve.txt        # minimal pinned deps for the API image
├── .env.example                  # template of environment variables (no secrets in code)
└── README.md
```

## The loop

```
          ┌──────────────────────────────────────────────────────────────────────┐
          │                                                                      │
 data/raw/creditcard.csv (DVC-versioned)                                         │
          │                                                                      │
          ▼                                                                      │
   [validate] pandera schema ──fail──► stop                                       │
          │                                                                      │
          ▼                                                                      │
   [split] by Time: 70% train │ 15% val │ 15% test   (train → reference.parquet) │
          │                                                                      │
          ▼                                                                      │
   [train] build_features() → XGBoost (+LR baseline, +Optuna on val)             │
          │   logs params/metrics/plots/model ──► MLflow Tracking                │
          │   registers new version          ──► MLflow Model Registry           │
          ▼                                        "fraud-detector" vN           │
   [promote] score vN vs @champion on test set                                   │
          │   PR-AUC gain > 0.005 ? move @champion → vN : keep old               │
          ▼                                                                      │
   [serve] FastAPI loads models:/fraud-detector@champion                         │
          │   /predict → build_features() → probability → flag if ≥ threshold    │
          │   /metrics ──► Prometheus ──► Grafana dashboards                     │
          │   every request appended to data/logs/*.jsonl                        │
          ▼                                                                      │
   [monitor] daily (Prefect): KS test per feature, reference vs recent logs      │
          │   + Evidently HTML report                                            │
          │   > 30% of features drifted ? ───────── trigger "fraud-retraining" ──┘
          ▼
        no drift → sleep until tomorrow
```

## Phases
1. Plan (this file) · 2. Setup · 3. Data validation & split · 4. Features · 5. Training ·
6. Promotion & rollback · 7. Serving · 8. Containers · 9. CI/CD · 10. Monitoring ·
11. Orchestration · 12. Docs
