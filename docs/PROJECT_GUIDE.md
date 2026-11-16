# Project Guide: How Everything Works

This guide explains the whole project: what to install, how the pieces fit together, what **every file** does, and why each tool in the stack was chosen. Read it top to bottom once, then use it as a reference.

---

## 1. The project in one paragraph

A bank wants to block fraudulent card transactions in real time. This project trains a model that scores each transaction with a fraud probability, then wraps everything a real company needs around it. The **data is versioned** (DVC) and **validated** (pandera) before use. Every training run is **tracked** (MLflow). New models are registered and only **promoted** to production if they clearly beat the current one. The model is **served** behind an HTTP API (FastAPI) in a **Docker** container. The API is **observed** with Prometheus + Grafana. Live traffic is **checked for drift** against the training data, and **retraining** runs automatically (Prefect) when the data shifts. Every code change is **tested and packaged** by CI/CD (GitHub Actions).

```
data ─► validate ─► split ─► features ─► train ─► registry ─► promote ─► serve ─► monitor ─┐
  ▲                                                                                          │
  └──────────────────────────────── retrain on drift ◄───────────────────────────────────────┘
```

---

## 2. What to download and install

### 2.1 Software (one-time)

| # | What | Why you need it | Where to get it | Notes |
|---|---|---|---|---|
| 1 | **Python 3.11** (64-bit) | All code is Python; dependencies are pinned and tested on 3.11 | python.org/downloads → 3.11.x | On Windows tick *"Add python.exe to PATH"*. Check with `py -3.11 --version`. Newer Pythons (3.13/3.14) will NOT work with these pinned versions. |
| 2 | **Git** | Version control for code, and DVC builds on it | git-scm.com | Comes with *Git Bash* on Windows. |
| 3 | **Docker Desktop** | Runs MLflow, API, Prometheus, Grafana as containers | docker.com/products/docker-desktop | On Windows it needs **WSL 2** (the installer offers to enable it; reboot when asked). Allocate ≥ 4 GB RAM in Settings → Resources. |
| 4 | **VS Code** (optional, recommended) | Editor | code.visualstudio.com | Useful extensions: *Python*, *Ruff*, *Docker*, *DVC*, *YAML*, *Markdown Preview Mermaid Support*. |
| 5 | **GitHub account** | Hosting the repo; CI/CD runs there; images go to GHCR | github.com | No extra secrets needed: CI uses the built-in `GITHUB_TOKEN`. |
| 6 | **Kaggle account** | To download the dataset | kaggle.com | Free. |

### 2.2 The dataset

- Kaggle → search **"Credit Card Fraud Detection"** by *mlg-ulb* (`mlg-ulb/creditcardfraud`) → Download → unzip.
- Put the file at **`data/raw/creditcard.csv`** (exact name and location).
- Facts about it: 284,807 transactions by European cardholders over two days in September 2013; 492 are fraud (**0.172%**). `V1`–`V28` are anonymised PCA components; `Time` is seconds since the first transaction; `Amount` is the transaction amount; `Class` is 1 for fraud.

### 2.3 Python packages

You don't install these one by one. `requirements.txt` lists them with exact versions:

```powershell
pip install -r requirements.txt          # intentional pins (direct + a few problem transitive deps)
# or, for the exact environment this project was tested in:
pip install -r requirements.lock.txt
```

### 2.4 Docker images (pulled automatically by `docker compose up`)

| Image | Used for |
|---|---|
| `python:3.11-slim` | Base of the API image |
| `ghcr.io/mlflow/mlflow:v2.19.0` | MLflow tracking server + model registry |
| `prom/prometheus:v2.55.1` | Metrics database / scraper |
| `grafana/grafana:11.3.1` | Dashboards |

Disk space: plan on roughly **5–8 GB** for Docker images, the Python environment and the data.

### 2.5 First-time setup (Windows PowerShell)

```powershell
cd "C:\path\to\End-to-end MLOps pipeline"

# Virtual environment. Tip: if the project sits inside OneDrive, put the venv OUTSIDE it,
# otherwise OneDrive will try to sync ~1-2 GB of library files:
py -3.11 -m venv $HOME\.venvs\fraud-mlops
& $HOME\.venvs\fraud-mlops\Scripts\Activate.ps1
pip install -r requirements.txt

copy .env.example .env                    # then edit values if you like

# DVC remote (a folder outside the repo, acting as "cloud storage")
dvc remote add --local -d localstorage C:\Users\<you>\dvc-storage\fraud-detection

# Data
#   put creditcard.csv into data\raw\ first
dvc add data/raw/creditcard.csv
git add data/raw/creditcard.csv.dvc data/raw/.gitignore
git commit -m "Track raw data with DVC"
dvc push

# Stack + pipeline
docker compose up -d --build
dvc repro
python -m src.promote
curl.exe -X POST http://localhost:8000/reload
```

> This environment already has a working venv at `C:\Users\durve\.venvs\fraud-mlops` and a DVC
> remote at `C:\Users\durve\dvc-storage\fraud-detection`.

---

## 3. How it works, step by step

### 3.1 The training side (batch, offline)

1. **Versioning the data (DVC).** `dvc add data/raw/creditcard.csv` computes the file's MD5 hash and copies it into `.dvc/cache`. It also writes a tiny text file, `creditcard.csv.dvc`, containing that hash. Git commits the *pointer*, not the 150 MB file. `dvc push` uploads the real file to the remote. Anyone can later `git checkout <commit>` + `dvc pull` and get exactly the data used at that commit.
2. **`dvc repro`** reads `dvc.yaml`, which defines three stages. Each stage lists its dependencies (`deps`), the `params.yaml` keys it uses, and its outputs. DVC re-runs only the stages whose inputs changed and records the hashes in `dvc.lock`.
3. **validate** (`src/validate.py`): pandera checks every column. 31 expected columns and nothing extra, numeric types, `Time ≥ 0`, `Amount ≥ 0`, `Class ∈ {0,1}`, no nulls. `lazy=True` collects *all* problems in one report. Any failure → non-zero exit → the pipeline stops, so bad data never reaches a model.
4. **split** (`src/split.py`): sort by `Time`. The first 70% of rows is train, the next 15% validation, and the last 15% test. That mirrors reality, where a model trained on the past predicts the future. A random split would leak future patterns into training and inflate scores. `train` is also saved as `reference.parquet`, the "normal" distribution that drift monitoring compares against later.
5. **features** (`src/features.py`): `build_features()` keeps V1–V28, adds `log_amount = log(1 + Amount)` (squashes the long tail of amounts) and `hour = (Time // 3600) % 24` (time-of-day pattern). The **same function** is imported by the API, so training and serving can never compute features differently (this is how training-serving skew is avoided).
6. **train** (`src/train.py`):
   - Logs a **logistic-regression baseline** run first (a sanity bar the main model must beat).
   - Trains **XGBoost** with `scale_pos_weight = #legit / #fraud` (≈ 580 on the real data). This makes each fraud example count as much as all the legit ones combined, so the model can't simply ignore fraud. Early stopping watches **validation** PR-AUC.
   - Optional `--tune`: **Optuna** tries N hyper-parameter sets, each scored on validation. Every trial is a *nested* MLflow run under one parent.
   - Picks the **threshold** that maximises F1 on validation, then applies that same threshold to test.
   - Logs to MLflow: parameters, val + test metrics (PR-AUC headline, ROC-AUC, precision, recall, F1), confusion-matrix and PR-curve plots, and the model itself with a **signature** (input/output schema) and an **input example**.
   - **Registers** the model as a new version of `fraud-detector` and tags it with `threshold`, `model_type`, `val_pr_auc` and `data_md5` (the DVC hash, so you know exactly which data produced the model).
   - Writes `reports/metrics.json`, which DVC tracks as a metrics file (`dvc metrics show`, `dvc metrics diff`).
7. **promote** (`src/promote.py`): the newest version gets the alias **`@challenger`**. It and the current **`@champion`** are scored on the *same* test set. `@champion` moves to the challenger only if its PR-AUC is higher by **more than 0.005**, so noise-level differences don't cause churn. If no champion exists yet, the first model becomes champion. The result goes to `reports/promotion.json`.
8. **rollback** (`scripts/rollback.py --version N`): repoints `@champion` at any older version in about a second, with no retraining or rebuild.

### 3.2 The serving side (online, real time)

1. `docker compose up` starts four containers on one private network: `mlflow`, `api`, `prometheus` and `grafana`. Containers reach each other by service name (`http://mlflow:5000`).
2. The API (`src/api.py`) starts and loads **`models:/fraud-detector@champion`**. MLflow resolves the alias to a version number and downloads that model's files. Because the API asks for the *alias* rather than a fixed version, promoting or rolling back changes what is served. `POST /reload` (or a restart) picks up the change.
3. A client sends `POST /predict` with `{"Time": ..., "Amount": ..., "V": [28 numbers]}`.
4. **pydantic** validates the body: `Time ≥ 0`, `Amount ≥ 0`, exactly 28 V values, numbers only, no unknown fields. Anything invalid gets an automatic **HTTP 422**, and the model never sees it.
5. The request is turned into a one-row DataFrame → `build_features()` (the same code as training) → `model.predict()` returns `[p_legit, p_fraud]` → `is_fraud = p_fraud ≥ FRAUD_THRESHOLD`.
6. Side effects:
   - Prometheus counters update: `fraud_predictions_total`, `fraud_flagged_transactions_total`, plus the `fraud_score` histogram. The instrumentator also records request counts, status codes and latency automatically.
   - The full request and its prediction are appended as one JSON line to `data/logs/predictions-YYYYMMDD.jsonl`. That folder is mounted from your machine, so the monitor can read it.
7. **Prometheus** scrapes `api:8000/metrics` every 5 seconds and stores the time series. **Grafana** queries Prometheus and draws the dashboard: requests/s, p50/p95/p99 latency, error rate, flagged rate, score distribution and model-loaded status.
8. `GET /health` returns 200 when a model is loaded and 503 otherwise. Docker's `HEALTHCHECK` calls it every 30 s and marks the container *healthy* or *unhealthy*.

### 3.3 The monitoring + retraining loop

1. `python -m src.monitor` (or the Prefect `daily-monitoring` flow) loads `reference.parquet` and the recent API logs.
2. For each of 29 features (V1–V28 + Amount) it runs a **two-sample Kolmogorov–Smirnov test**: "do these two samples come from the same distribution?" A p-value below 0.01 means the feature has drifted.
3. It builds an **Evidently** HTML report (`reports/drift_report.html`) showing per-feature distributions for humans.
4. If **more than 30%** of features drifted, `should_retrain = true`. The Prefect flow then runs **`fraud-retraining`**: validate → split → train → promote (each a task with automatic retries). If a new champion is promoted, it calls `POST /reload` on the API.
5. The cycle repeats daily (`python -m src.flows serve`, cron `0 6 * * *`).

### 3.4 The CI/CD side (on GitHub)

1. **On every push / pull request**, GitHub Actions runs `ruff check .` (lint) and `pytest` (49 tests). It then runs a **pipeline smoke test**: synthetic data → validate → split → train → promote against a throwaway SQLite MLflow.
2. **On pushes to `main` only**, after the tests pass: build the Docker image and push it to **GitHub Container Registry** as `ghcr.io/<owner>/<repo>/fraud-api:<commit-sha>` (and `:latest`). The SHA tag means every image traces back to the exact code that built it.

---

## 4. Every file explained

### Root

| File | What it does |
|---|---|
| `README.md` | Project front page: summary, Mermaid architecture, results table (placeholders until real runs), quick start, drift demo, drift theory, limitations, next steps. |
| `params.yaml` | **Single source of truth for settings**: data paths, split fractions, model hyper-parameters, tuning switch, MLflow names, promotion margin, monitoring thresholds. DVC watches the keys each stage uses. |
| `dvc.yaml` | Defines the pipeline stages `validate → split → train`: command, inputs, params, outputs, metrics and plots for each. |
| `dvc.lock` | *(created by `dvc repro`)* Exact hashes of every input and output of the last run, i.e. the reproducibility record. Commit it after running on real data. |
| `requirements.txt` | Pinned direct dependencies, plus pins for transitive packages whose newer releases broke things (`pathspec`, `sqlalchemy`, `multimethod`, `cloudpickle`). |
| `requirements.lock.txt` | `pip freeze` of the fully tested environment (every package, exact version). |
| `requirements-serve.txt` | Minimal deps for the API image only (no DVC, Prefect, Optuna, Evidently...). Uses `mlflow-skinny` and `xgboost-cpu` to keep the image small. |
| `Dockerfile` | Builds the API image: `python:3.11-slim`, installs `libgomp1` (needed by XGBoost), copies only 4 source files, runs as a non-root user, defines `HEALTHCHECK`, starts uvicorn. |
| `.dockerignore` | Keeps data, tests, `.git` etc. out of the image build context (faster builds, smaller and safer images). |
| `docker-compose.yml` | Starts `mlflow`, `api`, `prometheus` and `grafana` together, with ports, volumes, health checks and startup order (`api` waits until `mlflow` is healthy). Reads secrets from `.env`. |
| `.env.example` | Template of every environment variable (tracking URI, model URI, threshold, log dir, API URL, Grafana login). Copy to `.env`, which is git-ignored. |
| `.gitignore` | Keeps data, logs, MLflow state, generated reports, venvs and `.env` out of git. |
| `.gitattributes` | Forces LF line endings so scripts work in Linux containers even when edited on Windows. |
| `.dvcignore` | Like `.gitignore`, for DVC. |
| `pyproject.toml` | pytest settings (`pythonpath = .`, test folder) and ruff lint rules (line length 100; pyflakes, pycodestyle, isort, bugbear, pyupgrade). |

### `src/` — application code

| File | What it does | Key functions |
|---|---|---|
| `config.py` | Shared config: project `ROOT`, `REPORTS_DIR`, loads `params.yaml` (cached) and `.env`, resolves relative paths, gives the MLflow tracking URI. Also forces UTF-8 console output, because MLflow prints emoji that crash Windows' cp1252 console. | `load_params()`, `resolve()`, `tracking_uri()` |
| `validate.py` | pandera schema for the raw CSV; writes `reports/validation.json`. | `SCHEMA`, `validate_df()`, `run()` |
| `split.py` | Chronological 70/15/15 split; writes train/val/test/reference parquet and `reports/split_stats.json`. | `time_split()`, `run()` |
| `features.py` | **The one feature function** used by training and serving. Only numpy/pandas, so the slim API image can import it. | `build_features()`, `FEATURE_COLUMNS` |
| `model_utils.py` | Turns any pyfunc output into a 1-D array of fraud probabilities. | `fraud_probability()` |
| `train.py` | Baseline + XGBoost (+ Optuna), threshold selection, metrics, plots, MLflow logging and registration, `reports/metrics.json`. CLI flags: `--model`, `--tune`, `--n-trials`, `--no-baseline`. | `best_f1_threshold()`, `compute_metrics()`, `fit_xgb()`, `tune_xgb()`, `run()` |
| `promote.py` | Champion/challenger comparison and alias moves. | `decide_promotion()` (pure rule), `promote()`, `run()` |
| `api.py` | FastAPI app: request schema, model loading by alias, `/predict`, `/health`, `/reload`, `/metrics`, custom Prometheus metrics, JSONL request logging. | `Transaction`, `ModelState`, `predict()` |
| `monitor.py` | Loads logs, runs KS tests, writes the Evidently report and `reports/drift.json`, decides `should_retrain`. Adapts to the installed Evidently version. CLI: `--lookback-hours`, `--retrain`. | `load_logs()`, `ks_drift()`, `evidently_report()`, `run_drift_check()` |
| `flows.py` | Prefect tasks (with retries) and the flows `fraud-retraining` and `daily-monitoring`. CLI: `retrain`, `monitor`, `serve`. | `retraining_flow()`, `daily_monitoring()` |

### `scripts/` — operator tools

| File | What it does |
|---|---|
| `make_sample_data.py` | Generates a **synthetic** CSV with the real schema, for smoke tests and CI. Its metrics are meaningless. |
| `rollback.py` | `--list` shows versions with aliases, val PR-AUC and threshold. `--version N` repoints `@champion`. |
| `simulate_drift.py` | Samples test transactions, multiplies `Amount` ×3, adds Gaussian noise to V1–V12, and POSTs them to the API. `--no-drift` sends clean data for comparison. |

### `tests/` — 49 automated tests (no MLflow server, no real data)

| File | What it proves |
|---|---|
| `conftest.py` | Fixtures: a synthetic dataset generator, and a tiny logistic-regression model saved in MLflow format to a temp folder, so the API can be tested without a server. |
| `test_data.py` | Valid data passes; negative amount, bad class, negative time, nulls, missing or extra columns all fail. The split is strictly chronological, complete, and rejects bad fractions. |
| `test_features.py` | Column order, `log_amount` maths, `hour` wraps every 24 h, V columns pass through, extra columns are ignored, errors on missing columns and negative amounts, the input isn't mutated. |
| `test_train.py` | The F1-optimal threshold is correct; metrics are in range; no crash when nothing is predicted positive. |
| `test_promote.py` | The decision rule (including the floating-point edge case 0.805 − 0.800). The full lifecycle on a throwaway SQLite registry: first champion → no-op → better model promoted → equal model rejected but tagged challenger. |
| `test_api.py` | Health 200; valid prediction; fraud-like input scores higher; requests are logged; **seven bad inputs → 422**; `/metrics` exposes counters and the flagged counter increments; missing model → 503. |
| `test_monitor.py` | KS finds no drift on identical distributions, finds exactly the shifted features, catches Amount ×3; log lookback filtering; the Evidently report is written. |

### `monitoring/` — observability config

| File | What it does |
|---|---|
| `prometheus.yml` | Scrape `api:8000/metrics` every 5 s (plus Prometheus itself). |
| `grafana/provisioning/datasources/prometheus.yml` | Auto-creates the Prometheus datasource (uid `prometheus`) when Grafana starts. |
| `grafana/provisioning/dashboards/dashboards.yml` | Tells Grafana to load dashboards from `/var/lib/grafana/dashboards`. |
| `grafana/dashboards/fraud-api.json` | The dashboard: model-loaded, total predictions, total flagged, flag rate, requests/s, p50/p95/p99 latency, 4xx/5xx error rate, flagged-transaction rate, fraud-score distribution. |

### Other folders

| Path | What it holds |
|---|---|
| `.github/workflows/ci.yml` | The CI/CD pipeline described in §3.4. |
| `.dvc/` | DVC's config (`config` is committed; `config.local` with your machine's remote path is not) and its cache. |
| `data/raw/` | `creditcard.csv` (DVC-tracked, git-ignored) and its `.dvc` pointer (committed). |
| `data/processed/` | `train/val/test/reference.parquet`, outputs of `dvc repro`. |
| `data/logs/` | API request/prediction logs (JSONL), the monitor's input. |
| `reports/` | `validation.json`, `split_stats.json`, `metrics.json`, `confusion_matrix.png`, `pr_curve.png` (DVC-tracked), plus `promotion.json`, `drift.json`, `drift_report.html` (git-ignored). |
| `mlflow_data/` | MLflow's SQLite database and model artifacts (mounted into the MLflow container). |
| `docs/` | `PLAN.md` (phase-1 plan), this guide, and `INTERVIEW_QA.md`. |

---

## 5. The tech stack: what, why, where, alternatives

| Tool | What it is | Why here | Where in code | Alternatives |
|---|---|---|---|---|
| **Python 3.11** | Language | Ecosystem standard for ML | everywhere | — |
| **pandas / numpy / pyarrow** | DataFrames, arrays, Parquet I/O | Data handling; Parquet is typed, compressed, fast | all `src/` | Polars |
| **scikit-learn** | Classic ML library | Logistic-regression baseline, metrics, scaler | `train.py`, `promote.py` | — |
| **XGBoost** | Gradient-boosted decision trees | Top performer on tabular data; handles imbalance via `scale_pos_weight` | `train.py` | LightGBM, CatBoost |
| **Optuna** | Hyper-parameter optimisation (TPE Bayesian search) | Smarter than grid search; integrates easily with MLflow nested runs | `train.py --tune` | Hyperopt, Ray Tune |
| **DVC** | Git for data and pipelines | Versions the 150 MB CSV, makes the pipeline reproducible (`dvc repro`) | `dvc.yaml`, `.dvc/` | LakeFS, Delta Lake, Pachyderm |
| **pandera** | DataFrame schema validation | Declarative, readable checks; fails fast | `validate.py` | Great Expectations, pydantic |
| **MLflow** | Experiment tracking + model registry | Logs every run; versioned models with aliases; standard model format (pyfunc) | `train.py`, `promote.py`, `api.py` | Weights & Biases, Neptune, SageMaker |
| **FastAPI + pydantic + uvicorn** | Web framework + validation + ASGI server | Type-driven validation (auto 422), auto OpenAPI docs at `/docs`, fast | `api.py` | Flask, BentoML, Seldon, KServe |
| **prometheus-fastapi-instrumentator** | Auto-instruments FastAPI | Request count, status, latency metrics with one line | `api.py` | OpenTelemetry |
| **Prometheus** | Time-series DB that *pulls* metrics | Industry standard for service metrics; PromQL queries | `monitoring/prometheus.yml` | Datadog, CloudWatch |
| **Grafana** | Dashboards | Visualises Prometheus; provisioned as code | `monitoring/grafana/` | Kibana, Datadog |
| **scipy (KS test)** | Statistics | Simple, assumption-light drift test per feature | `monitor.py` | PSI, Wasserstein, chi-square |
| **Evidently** | ML monitoring reports | Rich HTML drift report for humans | `monitor.py` | WhyLabs, NannyML, Arize |
| **Prefect** | Workflow orchestration | Python-native flows/tasks, retries, schedules, UI | `flows.py` | Airflow, Dagster, Kubeflow |
| **Docker / Compose** | Containers / multi-container runner | Same environment everywhere; one command starts the stack | `Dockerfile`, `docker-compose.yml` | Podman, Kubernetes |
| **GitHub Actions + GHCR** | CI/CD + container registry | Free for public repos; built-in token auth | `.github/workflows/ci.yml` | GitLab CI, Jenkins, Docker Hub, ECR |
| **pytest** | Testing framework | Fixtures, parametrisation | `tests/` | unittest |
| **ruff** | Linter | Very fast, replaces flake8 + isort + pyupgrade | `pyproject.toml` | flake8, pylint |
| **python-dotenv** | Loads `.env` | Config from env vars, no secrets in code | `config.py` | — |

---

## 6. Configuration reference

### `params.yaml`

| Key | Meaning |
|---|---|
| `data.raw_path`, `data.processed_dir` | Where the raw CSV and processed parquet live |
| `split.train_frac` / `val_frac` | 0.70 / 0.15 (test = the rest) |
| `train.model` | `xgboost` or `logreg` |
| `train.run_baseline` | Also log a logistic-regression baseline run |
| `train.tune`, `train.n_trials` | Optuna on/off and number of trials |
| `train.xgboost.*` | Trees, depth, learning rate, subsampling, early-stopping rounds |
| `train.logreg.*` | Regularisation `C`, `max_iter` |
| `mlflow.experiment_name`, `registered_model_name` | `fraud-detection`, `fraud-detector` |
| `promote.min_improvement` | 0.005: the PR-AUC margin a challenger must beat |
| `promote.champion_alias` / `challenger_alias` | Alias names |
| `monitor.features` | The 29 features tested for drift |
| `monitor.p_value_threshold` | 0.01 |
| `monitor.drift_share_threshold` | 0.30 |
| `monitor.min_samples` | 100: don't judge drift on less traffic than this |
| `monitor.evidently_reference_sample` | Reference rows sampled for the HTML report (keeps it fast) |

### Environment variables (`.env`)

| Variable | Used by | Default |
|---|---|---|
| `MLFLOW_TRACKING_URI` | train/promote/rollback/flows (host) | `http://localhost:5000` (inside compose the API uses `http://mlflow:5000`) |
| `MODEL_URI` | API | `models:/fraud-detector@champion` (can be a local folder path) |
| `FRAUD_THRESHOLD` | API | `0.5`. Set it to the F1-optimal threshold printed by training, then `docker compose up -d api` |
| `PREDICTION_LOG_DIR` | API | `data/logs` |
| `API_URL` | simulate_drift, flows (reload) | `http://localhost:8000` |
| `GRAFANA_ADMIN_USER` / `GRAFANA_ADMIN_PASSWORD` | Grafana | `admin` / `admin` if unset |
| `PREFECT_API_URL` | Prefect | unset → temporary local server; set it to `http://127.0.0.1:4200/api` after `prefect server start` |

### Ports

| URL | Service |
|---|---|
| http://localhost:5000 | MLflow UI (experiments, runs, registry) |
| http://localhost:8000/docs | API interactive docs (Swagger UI) |
| http://localhost:8000/metrics | Raw Prometheus metrics |
| http://localhost:9090 | Prometheus (try the query `fraud_predictions_total`) |
| http://localhost:3000 | Grafana → *Fraud Detection* folder → *Fraud Detection API* |
| http://localhost:4200 | Prefect UI (after `prefect server start`) |

---

## 7. Command cheat sheet

```powershell
docker compose up -d --build        # start everything        | docker compose down   # stop
docker compose ps                   # status + health         | docker compose logs -f api
dvc repro                           # rerun changed stages    | dvc dag  # show the graph
dvc metrics show                    # current metrics         | dvc metrics diff  # vs last commit
dvc push / dvc pull                 # sync data with remote
python -m src.train --tune --n-trials 30
python -m src.promote
python scripts/rollback.py --list
python scripts/rollback.py --version 3
curl.exe -X POST http://localhost:8000/reload
python scripts/simulate_drift.py --n 500 [--no-drift]
python -m src.monitor [--retrain] [--lookback-hours 24]
python -m src.flows retrain | monitor | serve
ruff check .  ;  pytest
```

---

## 8. Troubleshooting (all of these really happened while building this project)

| Symptom | Cause | Fix (already applied) |
|---|---|---|
| `dvc add` → `cannot import name '_DIR_MARK' from pathspec` | An unpinned transitive dependency (`pathspec` 1.x) broke DVC 3.58 | Pinned `pathspec==0.12.1`. Lesson: pin transitive deps / use a lockfile. |
| pandera → `cannot import name 'overload' from multimethod` | `multimethod` 2.x removed `overload` | Pinned `multimethod==1.12` |
| `UnicodeEncodeError: 'charmap' codec can't encode '\U0001f3c3'` | MLflow prints 🏃; Windows piped console is cp1252 | `config.py` reconfigures stdout/stderr to UTF-8 |
| Promotion test: 0.805 vs 0.800 counted as "> 0.005" | Floating point: 0.805 − 0.800 = 0.005000000000000004 | Round the gain to 9 decimals before comparing |
| `version == "1"` fails in the test, passes against the server | SQL store returns versions as `int`, REST store as `str` | Normalise with `str()` |
| `rollback.py --list` shows no aliases | `search_model_versions()` doesn't return aliases in MLflow 2.19 | Read them from `get_registered_model(name).aliases` |
| API image was **2.11 GB** | The Linux `xgboost` wheel bundles ~700 MB of CUDA/NCCL GPU libs | Serving uses `xgboost-cpu` → **876 MB** |
| `docker compose up` → `server gave HTTP response to HTTPS client` / `EOF` | Docker Desktop's proxy had transient network errors | Retry `docker pull <image>`; check Docker Desktop → Settings → Proxies |
| Drift flagged features that were never changed | The simulator re-sent the *same* rows twice; duplicates inflate KS significance | Simulator now samples fresh rows by default (`--seed` optional) |
| `docker compose up` → `ports are not available ... bind: Only one usage of each socket address` | Another program already uses that host port (here: a Windows-installed Grafana on 3000 and another uvicorn app on 8000) | Set `API_PORT` / `GRAFANA_PORT` (etc.) in `.env`, e.g. 8001 / 3001, and keep `API_URL` in sync |
| `curl localhost:8000/health` → `{"detail":"Not Found"}` (404) | You reached a *different* app on port 8000, not this API | Same fix: use the port from `API_PORT` |
| Docker Desktop: *"An unexpected error occurred"*, log says `remove ...engine.sock` / `...run\dockerInference: The file cannot be accessed by the system` | Stale Unix-socket files left by a previous Docker run that Windows can't delete | Quit Docker Desktop, rename `%LOCALAPPDATA%\docker-secrets-engine` and `%LOCALAPPDATA%\Dockerun` (e.g. append `.stale`), start Docker again |
| API `/health` returns 503 | No `@champion` yet, or MLflow unreachable | Train + promote, then `POST /reload` |
| Prefect schedule disappears | `serve` started a *temporary* Prefect server | `prefect server start` + set `PREFECT_API_URL` |

---

## 9. What has been verified, and what is still to do

Verified while building, on a **synthetic** dataset with the real schema, since the Kaggle file wasn't on this machine yet:
- `dvc add` / `dvc push` to the local remote; `dvc repro` runs validate → split → train.
- Training registers versions in MLflow, and nested Optuna runs appear under their parent.
- First promotion sets `@champion`; re-running is a no-op; rollback moves the alias both ways.
- The full Docker stack comes up healthy; the API serves `@champion` from the MLflow container; Prometheus scrapes it; Grafana provisions the dashboard and every panel query returns data.
- Clean traffic → no drift; drifted traffic → exactly the 13 manipulated features flagged; the Prefect monitoring flow triggers the retraining subflow, which registers a new version and correctly **keeps** the champion (same data → no real improvement).
- 49 tests pass; ruff is clean; the CI smoke job was replayed locally.

**To do with the real data** (the synthetic run's numbers are meaningless and were not committed):
1. Put the real `creditcard.csv` in `data/raw/`, then `dvc add` + commit the `.dvc` pointer + `dvc push`.
2. `docker compose up -d`, `dvc repro`, `python -m src.promote`, then commit `dvc.lock` and `reports/metrics.json`.
3. Copy the numbers from `reports/metrics.json` into the README results table.
4. Push to GitHub and watch the Actions tab go green.
