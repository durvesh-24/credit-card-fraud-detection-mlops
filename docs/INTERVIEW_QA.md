# Interview Prep: Fraud-Detection MLOps Project

Four parts: **(1)** how to pitch the project, **(2)** the topics it covers and why each matters in industry, **(3)** about 110 interview questions with answers grounded in *this* codebase, and **(4)** a full A–Z glossary of every term.

> Rule for interviews: never quote metrics you haven't produced. Until you run on the real Kaggle data, say *"the README table is filled from `reports/metrics.json` after a real run."* Interviewers respect that far more than invented numbers.

---

## Part 1 — Pitching the project

**30-second version**
> "I built an end-to-end fraud-detection system on the Kaggle credit-card dataset, where only 0.17% of transactions are fraud. The model is XGBoost, but the focus is everything around it. The data is versioned with DVC and validated with pandera. Experiments are tracked in MLflow, and a new model only replaces the production 'champion' if it beats it by a margin on the same test set. It's served by a Dockerized FastAPI service with Prometheus and Grafana monitoring. A daily Prefect flow runs KS drift tests on live traffic and triggers retraining when more than 30% of features drift. GitHub Actions lints, tests and smoke-runs the pipeline on every push, and publishes the image to GHCR on main."

**2-minute version — walk the loop:** data (DVC + pandera) → time-based split (why not random) → shared `build_features` (training-serving skew) → XGBoost with `scale_pos_weight`, PR-AUC as the headline, threshold tuned on validation → MLflow registry with `@champion`/`@challenger` aliases → FastAPI with pydantic validation, `/metrics`, request logging → Prometheus/Grafana (RED metrics + flag rate) → KS + Evidently drift → Prefect retraining → CI/CD. Close with **one bug story** (Part 3, section Q) and **one limitation** you'd fix next (concept drift needs labels).

---

## Part 2 — Topics covered, and how each is useful

| Topic | Where in this project | Why it matters in a real job |
|---|---|---|
| Imbalanced classification | `scale_pos_weight`, PR-AUC, threshold tuning | Fraud, churn, defects, medical diagnosis, and anomaly detection are all imbalanced |
| Evaluation design | Time split, val-only tuning, held-out test | Prevents shipping models that only looked good offline |
| Data leakage prevention | Chronological split, threshold from validation | The most common reason ML projects fail in production |
| Data versioning | DVC pointers, remote, `dvc.lock` | Audits and regulators ask "which data trained this model?" |
| Data validation / contracts | pandera schema | Upstream schema changes silently break models |
| Feature engineering parity | `build_features()` shared by train + API | Training-serving skew is a classic production bug |
| Gradient boosting | XGBoost, early stopping | The default strong model for tabular business data |
| Hyper-parameter optimisation | Optuna TPE, nested runs | Efficient tuning, with traceable trials |
| Experiment tracking | MLflow params/metrics/artifacts | Reproducibility, collaboration, comparing runs |
| Model registry & lifecycle | Versions, aliases, tags | Governance; separates "trained" from "deployed" |
| Safe deployment & rollback | Champion/challenger gate, alias rollback | Cheap, instant rollback reduces the risk of every release |
| API design for ML | FastAPI, pydantic, 422/503, `/health` | Most models are consumed through an API |
| Containerisation | Slim image, non-root, healthcheck, compose | "Works on my machine" → works everywhere |
| CI/CD | GitHub Actions, GHCR, SHA tags | Every change tested; every artifact traceable |
| Observability | Prometheus counters/histograms, Grafana | You can't operate what you can't see (latency SLOs, error budgets) |
| Drift detection | KS tests, Evidently, prediction-drift panels | Models decay silently as the world changes |
| Orchestration | Prefect flows, retries, schedules, subflows | Automates retraining and recovers from transient failures |
| Testing ML systems | Unit, API, registry-integration, smoke tests | ML code breaks in ways notebooks never reveal |
| Config & secrets hygiene | `params.yaml`, `.env.example`, env vars | Security reviews; the same code across dev/stage/prod |
| Dependency management | Pins, lockfile, transitive-dep breakages | Reproducible builds; avoiding "it broke overnight" |

**Roles this maps to:** ML Engineer, MLOps / ML Platform Engineer, Data Scientist (production-minded), Data Engineer, Backend engineer on ML products.

---

## Part 3 — Questions & answers

### A. Project overview & design

**1. What problem does the project solve?**
Real-time scoring of card transactions for fraud, plus the operational system that keeps that model trustworthy over time: versioning, validation, tracking, controlled promotion, serving, monitoring and retraining.

**2. Why call it "production-style" if it runs on a laptop?**
It has the same moving parts as production systems: data and model versioning, a gated release process, containerised serving, metrics/dashboards, drift detection, scheduled retraining, CI/CD and rollback. Only the infrastructure scale differs (SQLite instead of Postgres, Compose instead of Kubernetes).

**3. Walk me through the architecture.**
DVC-tracked CSV → pandera validation → chronological split → `build_features` → XGBoost training logged to MLflow → registered version → promotion gate moves `@champion` → FastAPI loads `@champion` → Prometheus scrapes `/metrics` → Grafana. The API logs requests → the daily Prefect flow runs KS tests vs the training reference → on drift it runs the retraining flow → which re-enters the promotion gate.

**4. What's the most important design decision?**
Serving the *alias* `models:/fraud-detector@champion` instead of a fixed version. That decouples deployment from code: promotion and rollback are metadata changes, with no rebuild or redeploy.

**5. What would you change for a real bank?**
A label feedback loop (chargebacks) to monitor concept drift and live precision/recall; business-cost-based thresholds; shadow/canary testing of challengers; authentication on the API; Postgres + object storage for MLflow; Kubernetes with autoscaling; async logging via Kafka; explainability (SHAP) for analysts and regulators.

### B. Data

**6. Describe the dataset.**
284,807 European card transactions over two days (September 2013); 492 frauds = 0.172%. `V1`–`V28` are PCA components (anonymised), plus `Time` (seconds since the first transaction), `Amount` and `Class`.

**7. Why split by time instead of randomly?**
In production the model always predicts the future from the past. A random split puts later transactions in training, which leaks future fraud patterns and inflates scores. A chronological split (70/15/15) simulates deployment honestly.

**8. Why three splits?**
Train fits the model. Validation is for every decision: early stopping, Optuna, threshold. Test is touched only for final reporting and the champion comparison. If you tune on test, the test score is no longer an unbiased estimate.

**9. Why save the training split as `reference.parquet`?**
Drift is defined relative to what the model learned from. The training distribution is the baseline that live traffic is compared against.

**10. What does pandera validate, and why fail hard?**
Exact columns (`strict=True`), numeric types (with coercion), `Time ≥ 0`, `Amount ≥ 0`, `Class ∈ {0,1}`, and no nulls. It fails hard because a silently-trained model on corrupt data is worse than a stopped pipeline. `lazy=True` reports all failures at once.

**11. What is data leakage? Give examples relevant here.**
Information unavailable at prediction time leaking into training. Here: random splitting (future → past), choosing the threshold on test, scaling with statistics computed on all data, or features derived from the label.

**12. How do you handle class imbalance?**
`scale_pos_weight = #neg/#pos` in XGBoost (`class_weight="balanced"` in logistic regression), which reweights the loss. Also PR-AUC as the metric and a tuned threshold instead of 0.5. I avoided SMOTE/oversampling: it can create unrealistic synthetic frauds, and reweighting achieves a similar effect more simply.

### C. Features

**13. What features did you engineer?**
`log_amount = log1p(Amount)` to compress the heavy tail, and `hour = (Time // 3600) % 24` for time-of-day effects, plus V1–V28 as-is.

**14. What is training-serving skew and how did you prevent it?**
When features are computed differently in training and serving (e.g. log vs log1p, different hour logic), the live model sees inputs it never learned from. I prevent it with a single `build_features()` in `src/features.py`, imported by both `train.py` and `api.py` and covered by unit tests.

**15. Why does `features.py` depend only on numpy/pandas?**
So the slim serving image can import it without the training stack (DVC, Optuna...).

**16. Any caveat with the `hour` feature?**
`Time` counts from the first transaction, not midnight, so `hour` is an offset from an unknown start time, not the true clock hour. It still captures the day/night cycle.

### D. Modeling & metrics

**17. Why XGBoost?**
Gradient-boosted trees are the strongest general-purpose models for tabular data. They capture non-linear interactions, need no scaling, are fast, and handle imbalance via `scale_pos_weight`.

**18. Why a logistic-regression baseline?**
A model is only "good" relative to something simple. If XGBoost can't beat a linear model, the complexity isn't justified. The baseline also catches pipeline bugs: if both are terrible, suspect the data.

**19. Why is accuracy useless here?**
Predicting "not fraud" for everything gives 99.83% accuracy and catches zero fraud.

**20. PR-AUC vs ROC-AUC — which and why?**
ROC-AUC uses the false-positive *rate*, which stays tiny when negatives are huge, so it looks great even for weak fraud models. PR-AUC (average precision) focuses on the positive class: of what we flag, how much is fraud (precision), and how much fraud we catch (recall). It's the headline metric. ROC-AUC is reported as secondary.

**21. How did you pick the decision threshold?**
I computed the precision-recall curve on **validation**, chose the threshold that maximises F1, and applied it unchanged to test. It's logged as a metric and as a tag on the model version, and the API reads it from `FRAUD_THRESHOLD`.

**22. Is max-F1 the right threshold for a bank?**
Not necessarily. F1 weights precision and recall equally. A bank should weigh the cost of missed fraud (chargeback amount) against false alarms (analyst time, customer friction) and pick the threshold that minimises expected cost. That's listed as a next step.

**23. Precision vs recall in plain words?**
Precision: when we flag fraud, how often are we right. Recall: of all actual frauds, how many we caught.

**24. What does `scale_pos_weight` do mathematically?**
It multiplies the gradient/hessian contribution of positive examples by that factor, which is like duplicating each fraud row that many times in the loss.

**25. What is early stopping and what does it watch?**
Training stops adding trees once validation PR-AUC (`eval_metric="aucpr"`) hasn't improved for `early_stopping_rounds`. This prevents overfitting and picks the number of trees automatically. It uses validation, never test.

**26. How does Optuna work?**
Each trial samples hyper-parameters, trains, and returns validation PR-AUC. The TPE sampler models which regions of the search space gave good vs bad scores and samples more from the good ones. That's more efficient than grid or random search. Seeded for reproducibility.

**27. Why nested MLflow runs for tuning?**
One parent run per tuning session, with each trial as a child. The UI groups them, so you can compare trials without cluttering the experiment, and the parent holds the final model.

**28. Doesn't using validation for early stopping AND tuning AND threshold bias it?**
Yes, validation scores become optimistic. That's exactly why the separate test set exists, and why reported numbers come from test.

### E. Experiment tracking & registry (MLflow)

**29. What is experiment tracking?**
Automatically recording every training run's parameters, metrics, artifacts (plots, model files), code version and data version, so any result can be compared and reproduced.

**30. What does each run log here?**
Params (hyper-parameters, row counts, fraud rate, `scale_pos_weight`, best iteration), val/test metrics, threshold, confusion matrix and PR curve, the model with signature and input example. Tags include `model_type`, `role` and `data_md5`. MLflow adds the git commit automatically.

**31. What is a model signature, and why log an input example?**
The signature is the declared input schema (30 named float columns) and output schema. MLflow enforces it at prediction time, catching column or type mismatches. The input example documents real input and lets MLflow validate that the model can actually be served.

**32. What is a model registry?**
A central catalogue of named models with numbered, immutable versions, plus metadata (tags, descriptions, lineage to the run). It separates "a model was trained" from "this model is in production".

**33. What are aliases, and why use them instead of stages?**
An alias is a mutable named pointer to a version (`@champion` → v7). MLflow deprecated the fixed stages (Staging/Production) in favour of aliases, which are flexible: any number, any names, one version per alias. Consumers load `models:/name@alias`, so moving an alias redeploys.

**34. What's the difference between `@champion` and `@challenger`?**
Champion is what production serves. Challenger is the newest candidate under evaluation. The promotion script always tags the newest version as challenger, then decides whether it takes the champion alias.

**35. What is pyfunc and why `pyfunc_predict_fn="predict_proba"`?**
pyfunc is MLflow's generic model interface (`load_model(uri).predict(df)`), so the same loading code works for any flavour. By default sklearn's pyfunc calls `predict` (labels). Setting it to `predict_proba` returns probabilities, which we need for thresholding and PR-AUC.

**36. How does the MLflow server store things?**
SQLite for the backend store (runs, params, metrics, registry) and a folder for artifacts, served through the server (`--serve-artifacts`). Clients upload and download via HTTP, so the host scripts and the API container share one store.

**37. How do you link a model to the exact data it saw?**
`train.py` reads the MD5 from `creditcard.csv.dvc` and tags both the run and the model version with `data_md5`. With the git commit (auto-tagged), that gives full lineage: code + data + params → model.

### F. Data versioning (DVC)

**38. What is DVC and why not just commit the CSV?**
Git is bad at large binary files: the repo bloats and diffs are meaningless. DVC stores the file in a content-addressed cache and remote, and commits a small pointer file (hash + size) to git. You get git-style versioning of data without the bloat.

**39. What happens on `dvc add`, `dvc push`, `dvc pull`?**
`add`: hash the file, move it into `.dvc/cache`, write `file.dvc`, and git-ignore the original. `push`: upload cache objects to the remote. `pull`: download the objects matching the current pointer files and restore the workspace.

**40. What is `dvc.yaml` vs `dvc.lock`?**
`dvc.yaml` is the pipeline *definition* you write: stages, commands, deps, params, outs. `dvc.lock` is generated: exact hashes of every dep, param value and output from the last run. Commit both and anyone can reproduce or verify the run.

**41. How does `dvc repro` decide what to rerun?**
It compares current hashes of deps/params to `dvc.lock`. Change only `train.xgboost.max_depth` and only `train` reruns. Change the raw CSV and everything reruns.

**42. What is a DVC remote? How would you switch to S3?**
The storage backend for the cache (like a git remote for data). Here it's a local folder in `.dvc/config.local`. For S3: `pip install "dvc[s3]"`, `dvc remote add -d storage s3://bucket/path`, with credentials from AWS env vars or `~/.aws`, never committed. Google Drive works the same way with `dvc[gdrive]`.

**43. Why is the remote configured with `--local`?**
The path is machine-specific, so it goes to the git-ignored `.dvc/config.local`. A shared cloud remote would go in the committed `.dvc/config`.

### G. Promotion, deployment strategies, rollback

**44. Explain your promotion logic.**
Newest version = challenger. If there's no champion, it becomes champion. Otherwise both are scored on the same test set, and the alias moves only if challenger PR-AUC − champion PR-AUC > 0.005.

**45. Why a margin of 0.005 instead of "any improvement"?**
Small differences are within noise (random seeds, sampling). Promoting on noise causes churn and risk for no real gain. A margin acts as a cheap significance guard. A bootstrap confidence interval would be more rigorous.

**46. Isn't comparing on the same test set every time a problem?**
Yes. Repeatedly selecting models on one test set slowly overfits it ("test-set reuse"). Better: a fresh, recent holdout each cycle, or shadow evaluation on live traffic once labels arrive.

**47. How do you roll back?**
`python scripts/rollback.py --version N` points `@champion` at version N, then `POST /reload` (or restart the API). There's no retraining, rebuild or redeploy, and old versions are immutable.

**48. Champion/challenger vs A/B test vs shadow vs canary?**
- *Champion/challenger (offline)*: compare on historical data before any traffic. This project does this.
- *Shadow*: the challenger scores live traffic in parallel, but its outputs are ignored and only logged. Zero user risk.
- *Canary*: send a small % of traffic to the new model, watch metrics, then ramp up.
- *A/B test*: split traffic between models to measure business impact with statistical tests.

**49. Blue-green deployment?**
Two identical environments. Switch the router from blue (old) to green (new) at once, and switch back to roll back. Alias-based loading is a lightweight model-level version of the same idea.

### H. Serving (FastAPI)

**50. Why FastAPI?**
Type hints drive request validation (pydantic), automatic OpenAPI/Swagger docs at `/docs`, async support and high performance on uvicorn (ASGI).

**51. What does pydantic validate, and what happens on bad input?**
`Time ≥ 0`, `Amount ≥ 0`, `V` has exactly 28 floats, and `extra="forbid"` rejects unknown fields. Invalid requests get **HTTP 422 Unprocessable Entity** with a precise error, before the model runs. Seven such cases are tested.

**52. Why does `/health` return 503 when no model is loaded?**
It's a readiness signal: the process is alive but can't serve. Docker's HEALTHCHECK marks the container unhealthy, and an orchestrator wouldn't route traffic to it. (Kubernetes splits this into liveness and readiness probes.)

**53. How does the API get a newly promoted model?**
`POST /reload` re-resolves `models:/fraud-detector@champion`. If loading fails, the old model keeps serving, so a bad reload never takes the service down. The retraining flow calls `/reload` automatically after a promotion.

**54. What if MLflow is down when the API starts?**
The API still starts, `/health` says 503, and `/predict` retries loading at most every 30 s. It recovers on its own once MLflow is back.

**55. Why log every request to JSONL?**
The logs are the input to drift monitoring, and later, joined with chargeback labels, the data for measuring live performance and retraining. JSONL is append-friendly and easy to read with pandas.

**56. Weaknesses of that logging approach?**
It's synchronous file I/O on the request path, single-node, and has no rotation or retention policy. In production you'd emit to a queue (Kafka/Kinesis) asynchronously and land the data in a warehouse.

**57. How would you scale the API?**
Multiple uvicorn workers / replicas behind a load balancer (Kubernetes HPA on CPU or latency), model loaded once per worker, batch endpoints for bulk scoring, and possibly ONNX/Treelite to speed up tree inference.

**58. Sync `def` vs `async def` endpoints here?**
`predict` is CPU-bound (model inference), so a plain `def` runs in FastAPI's threadpool and doesn't block the event loop. `async def` with CPU-heavy code would block all requests.

### I. Containers

**59. Docker image vs container?**
An image is an immutable, layered template (filesystem + metadata). A container is a running instance of an image with its own process space and writable layer.

**60. How did you keep the image slim?**
`python:3.11-slim` base; a separate `requirements-serve.txt` without the training stack; `mlflow-skinny`; `--no-cache-dir`; `.dockerignore`; copying only 4 source files. The biggest win: the default Linux `xgboost` wheel bundles ~700 MB of CUDA libraries, and switching to `xgboost-cpu` cut the image from 2.11 GB to 876 MB.

**61. Why copy `requirements` before the source code?**
Layer caching: the slow `pip install` layer is reused until the requirements change, so code edits rebuild in seconds.

**62. Why a non-root user?**
If the app is compromised, the attacker doesn't get root in the container, which reduces the blast radius. It's a standard security-scan requirement.

**63. What does HEALTHCHECK do?**
Docker runs the command periodically (here: GET `/health` must return 200) and reports the container as healthy or unhealthy. Compose uses health to order startup (`api` waits for `mlflow` to be `service_healthy`).

**64. How do containers talk to each other in Compose?**
Compose creates a private network with DNS by service name: the API uses `http://mlflow:5000`, and Prometheus scrapes `api:8000`. From your laptop you use the published ports on `localhost`.

**65. Volumes vs bind mounts here?**
Bind mounts map host folders (`./mlflow_data`, `./data/logs`, monitoring configs) so data is visible on the host. Named volumes (`prometheus_data`, `grafana_data`) are Docker-managed persistence.

**66. Why must sklearn/xgboost/cloudpickle versions match between training and serving?**
Models are pickled. Unpickling with a different library version can fail or silently change behaviour. Those pins are deliberately identical in both requirements files.

### J. CI/CD

**67. CI vs CD?**
Continuous Integration: every push is automatically linted, tested and verified, so integration problems surface immediately. Continuous Delivery/Deployment: every passing change on main is automatically packaged (here, the Docker image pushed to GHCR) and is ready to deploy, or gets deployed.

**68. What does your workflow do?**
On every push and PR: ruff + pytest, then a pipeline smoke job (synthetic data → validate → split → train → promote on a SQLite MLflow). On main only: build and push `ghcr.io/<owner>/<repo>/fraud-api:<sha>` and `:latest`.

**69. Why tag images with the commit SHA?**
Immutable, unique and traceable: from a running container you can find the exact code. `:latest` is mutable and ambiguous, so it's fine for convenience but never for deploys.

**70. How does the workflow authenticate to GHCR without secrets?**
The built-in `GITHUB_TOKEN`, with `packages: write` permission granted only to that job (least privilege).

**71. Why a smoke test on synthetic data?**
Unit tests check functions. The smoke test checks that the stages actually run together (file paths, MLflow registration, promotion). It catches integration breaks without needing the real 150 MB dataset in CI.

**72. What's `concurrency: cancel-in-progress` for?**
If you push twice quickly, the older run for that branch is cancelled. That saves CI minutes and avoids racing image pushes.

### K. Observability (Prometheus & Grafana)

**73. Monitoring vs observability?**
Monitoring watches known failure signals (dashboards, alerts). Observability is the ability to ask new questions about system behaviour from its outputs (metrics, logs, traces).

**74. How does Prometheus collect metrics?**
Pull model: it scrapes HTTP `/metrics` endpoints on an interval (5 s here) and stores time series labelled by name and labels.

**75. Counter vs gauge vs histogram?**
Counter: only goes up (`fraud_predictions_total`); use `rate()`. Gauge: goes up and down (`fraud_model_loaded`). Histogram: counts observations into buckets (`http_request_duration_seconds`, `fraud_score`), which lets you compute percentiles.

**76. How are p95/p99 latency computed?**
`histogram_quantile(0.95, sum by (le) (rate(http_request_duration_seconds_bucket{handler="/predict"}[5m])))`, which interpolates within buckets. I set fine buckets (1 ms–5 s), because the default 0.1/0.5/1 s buckets make the percentiles meaningless.

**77. Why percentiles instead of average latency?**
Averages hide tail latency. p99 tells you what your slowest 1% of users experience, and SLOs are written on percentiles.

**78. What are RED metrics?**
Rate (requests/s), Errors (error rate), Duration (latency). They're the standard health signals for a request-driven service. The dashboard covers all three, plus ML-specific panels.

**79. Which ML-specific metrics did you add, and why?**
Flagged-transaction rate and the fraud-score distribution. They detect *prediction drift* immediately, without labels. A sudden jump in the flag rate means either an attack or a broken input pipeline.

**80. What is Grafana provisioning?**
Datasources and dashboards defined as files and loaded at startup. That's dashboards-as-code: versioned, reviewable and reproducible, with nothing clicked together by hand.

**81. How would you alert?**
Grafana alert rules or Prometheus Alertmanager: e.g. 5xx rate > 1% for 5 min, p99 > 200 ms, `fraud_model_loaded == 0`, flag rate outside its normal band, routed to Slack or PagerDuty.

### L. Drift

**82. Data drift vs concept drift vs prediction drift?**
Data drift: P(X) changes (amounts triple). Concept drift: P(y|X) changes (fraud starts looking like normal spending). Prediction drift: the model's output distribution changes. Data and prediction drift are detectable without labels. Concept drift needs labels.

**83. Does data drift always hurt the model?**
No. A shift in a feature the model barely uses may not matter, and concept drift can happen with no data drift at all. That's why the final arbiter is labelled performance.

**84. Explain the KS test.**
The two-sample Kolmogorov–Smirnov statistic is the maximum vertical distance between the two empirical CDFs. The null hypothesis is that both samples come from the same distribution. A small p-value (< 0.01 here) means the difference is unlikely to be chance, so the feature is flagged as drifted.

**85. Problems with KS for drift?**
(1) With large samples, tiny harmless shifts become "significant". (2) It's univariate, so it misses changes in correlations. (3) Testing 29 features at α=0.01 gives about 0.29 false alarms per check even with no drift; I observed 1/29 on clean traffic. Mitigations: effect-size thresholds (KS statistic, PSI), Bonferroni/FDR correction, and requiring a *share* of features (> 30%) rather than any single one.

**86. What is PSI?**
Population Stability Index: bin the feature, then sum (cur% − ref%) · ln(cur% / ref%). Rules of thumb: < 0.1 stable, 0.1–0.25 moderate, > 0.25 major shift. It's popular in credit risk.

**87. Why both KS and Evidently?**
KS gives a simple, testable, programmatic decision rule. Evidently gives a rich HTML report for humans to investigate *what* changed. Evidently's own drift share agreed with mine in the demo.

**88. How did you simulate drift, and what did you observe?**
Amount ×3 plus Gaussian noise (σ=1.5) on V1–V12, sent through the real API. The monitor flagged exactly those 13 features (Amount + V1–V12, 44.8% > 30%) and triggered retraining. Clean traffic flagged 1/29.

**89. Why does the retrained model not get promoted in the demo?**
The raw labelled data didn't change, so retraining reproduces essentially the same model. The gate correctly refuses a −0.0003 "improvement". In reality you retrain on newly labelled recent data, which is how the model adapts.

**90. Why `min_samples: 100` before judging drift?**
With very few requests, distribution tests are unreliable and noisy, which would cause false retrains.

### M. Orchestration (Prefect)

**91. Why an orchestrator instead of cron + scripts?**
Retries, dependency ordering, state tracking, logs per task, a UI with run history, parameters, scheduling, and subflows. Failures are visible and recoverable, not silent.

**92. Flow vs task in Prefect?**
A flow is the workflow function (the container of logic). Tasks are units of work inside it, each with its own state, retries, caching and logs.

**93. Where are retries, and why there?**
Validate and split: 2 retries (file/IO hiccups). Promote: 2 retries (MLflow briefly unreachable). Train: 1 retry with a longer delay (it's expensive). Retries are for *transient* failures. A data-validation failure that repeats will still fail, which is correct.

**94. How is the daily schedule set up?**
`daily_monitoring.serve(name=..., cron="0 6 * * *")` registers a deployment with a cron schedule and polls for runs. For persistence, run `prefect server start` and set `PREFECT_API_URL`. Otherwise Prefect spins up a temporary server.

**95. What is a subflow?**
A flow called from inside another flow. `fraud-retraining` runs as a child of `daily-monitoring`, with its own state and visible nesting in the UI.

**96. Prefect vs Airflow?**
Airflow is mature and DAG-centric (static DAGs, scheduler + workers, heavier to run). Prefect is Python-native (decorate functions, dynamic control flow like `if drift: retrain()`) and lighter for small teams. Dagster emphasises data assets and lineage.

### N. Testing

**97. How do you test an ML system?**
Layers: unit tests (features, metric helpers, decision rules), data tests (schema pass/fail, split ordering), integration tests (promotion against a real SQLite registry, API with a real MLflow-format model), and a pipeline smoke test in CI. Plus model-quality gates (the promotion margin).

**98. How do the API tests avoid needing an MLflow server?**
A session fixture trains a tiny logistic regression on synthetic data and saves it in MLflow format to a temp folder. The test sets `MODEL_URI` to that path, and the API loads it exactly as it would a registry model.

**99. What bugs did tests catch?**
(1) The floating-point promotion edge case. (2) int-vs-str version types between MLflow stores. (3) A failed model load leaving a previous model in memory across app restarts (global state), which was fixed by resetting state on startup while keeping the old model on a failed `/reload`.

**100. What's a "behavioural" test for a model?**
Checking expected directional behaviour: `test_fraud_like_transaction_scores_higher` asserts that strongly fraud-like V14/V12 values score higher than neutral input.

### O. Config, secrets & security

**101. How are secrets handled?**
Nothing sensitive in code. Values live in environment variables, documented in `.env.example`; the real `.env` is git-ignored. Compose reads `.env`. CI uses the built-in `GITHUB_TOKEN`. Cloud credentials for DVC remotes come from the environment or credential files.

**102. Why both `params.yaml` and env vars?**
`params.yaml` holds *experiment* settings that should be versioned and tracked by DVC. Env vars hold *deployment* settings that differ per environment (URLs, thresholds, passwords). That follows the twelve-factor app principle.

**103. Security gaps you'd close?**
Auth on `/reload` (and ideally `/predict`: API keys/OAuth), TLS, rate limiting, image vulnerability scanning (Trivy) in CI, a non-SQLite MLflow behind auth, a secret manager (Vault / AWS Secrets Manager), and PII handling for logged transactions.

### P. Reproducibility & dependencies

**104. How is a result reproducible here?**
Git commit (code) + DVC pointer and `dvc.lock` (data and pipeline) + `params.yaml` (settings) + pinned requirements and lockfile (environment) + seeds (`random_state=42`, seeded Optuna sampler) + MLflow run (everything logged).

**105. Why pin transitive dependencies?**
While building, unpinned transitive packages broke the project twice: `pathspec` 1.x broke DVC, and `multimethod` 2.x broke pandera. Pinning direct deps isn't enough. A lockfile captures the whole tested environment.

### Q. Behavioural / "tell me about a bug" stories (true, from this build)

**106. Hardest bug?**
*Drift on features I never touched.* The drift simulator flagged V15/V20/V25/V27, which I hadn't modified. Root cause: two runs used the same random seed, so the same 500 rows were sent twice. Duplicates double n without adding information, which inflates KS significance. Fix: the simulator samples fresh rows by default. Lesson: statistical tests assume independent samples; check your data-generating process before trusting a p-value.

**107. A performance/size win?**
The API image was 2.11 GB. Profiling `site-packages` showed 469 MB of `nvidia` libraries plus a large `xgboost` package, from GPU support bundled into the default Linux wheel. Switching serving to `xgboost-cpu` (same module, same pickles) brought it to 876 MB, with faster pulls, a smaller attack surface and quicker scaling.

**108. A subtle correctness bug?**
The promotion rule said "promote if gain > 0.005". The test with 0.805 vs 0.800 promoted, because 0.805 − 0.800 = 0.005000000000000004 in floating point. Fixed by rounding the gain before comparing. The unit test is what caught it.

**109. A platform-specific issue?**
MLflow prints an emoji when logging runs. On Windows with piped output (cp1252), that raised `UnicodeEncodeError` and crashed training mid-run. Fixed centrally in `config.py` by reconfiguring stdout/stderr to UTF-8.

### R. Curveballs

**110. Your PR-AUC is 0.85 — is that good?**
Compare it to the baseline (the no-skill PR-AUC equals the positive rate, ≈ 0.0017) and to the logistic-regression baseline. Then translate it to business terms: at the chosen threshold, how many frauds are caught and how many false alarms analysts must review per day.

**111. Labels arrive 30 days late. How does that change your design?**
Monitor data and prediction drift in the meantime as leading indicators. Store predictions with IDs, join labels when they arrive, compute delayed performance, and retrain on matured windows. Avoid training on recent data whose labels aren't final yet (that's label leakage from incomplete outcomes).

**112. How would this run on Kubernetes?**
The API as a Deployment with readiness/liveness probes (`/health`), an HPA, and a ConfigMap/Secret for env. MLflow with Postgres + S3. Prometheus Operator `ServiceMonitor`. Prefect workers as pods. CI pushes images, and Argo CD/Helm deploys the SHA tag (GitOps).

**113. Online vs batch inference?**
Online: per-request, low latency (this API: authorisation-time fraud checks). Batch: score many records on a schedule (nightly risk reports). The same registered model can serve both.

**114. Why not retrain on a fixed schedule instead of on drift?**
Scheduled retraining is simpler and often fine. Drift-triggered retraining saves compute and reacts faster to sudden shifts. Many teams combine them: scheduled retraining, plus drift alerts for emergencies. Either way, the promotion gate protects production.

---

## Part 4 — Glossary (A–Z)

| Term | Meaning |
|---|---|
| **422 Unprocessable Entity** | HTTP status for a well-formed request with invalid content; FastAPI returns it on validation failure. |
| **503 Service Unavailable** | HTTP status meaning "can't serve right now"; returned when no model is loaded. |
| **A/B test** | Splitting live traffic between variants to measure impact statistically. |
| **Accuracy** | Correct predictions / all predictions; misleading on imbalanced data. |
| **Alias (MLflow)** | Mutable named pointer to a model version, e.g. `@champion`. |
| **Artifact** | Any file logged with a run: model, plot, report. |
| **Artifact store** | Where MLflow keeps artifacts (folder, S3...). |
| **ASGI** | Async server gateway interface; the standard uvicorn uses to run FastAPI. |
| **AUC** | Area under a curve (ROC or PR); summarises ranking quality across thresholds. |
| **Average precision (AP)** | A step-wise estimate of PR-AUC; what `average_precision_score` computes. |
| **Backend store** | MLflow's database of runs, params, metrics and registry (SQLite here). |
| **Baseline model** | A simple reference model a complex one must beat. |
| **Batch inference** | Scoring many records at once on a schedule. |
| **Bind mount** | Mapping a host folder into a container. |
| **Blue-green deployment** | Two environments; switch traffic all at once; switch back to roll back. |
| **Bonferroni correction** | Dividing α by the number of tests to control false positives across many tests. |
| **Bucket (histogram)** | A range boundary (`le`) used to count observations for percentile estimation. |
| **Build context** | Files sent to Docker when building; trimmed by `.dockerignore`. |
| **Cache (DVC)** | Local content-addressed store (`.dvc/cache`) of data files by hash. |
| **Canary release** | Sending a small share of traffic to a new version before full rollout. |
| **CD (Continuous Delivery/Deployment)** | Automatically packaging (and optionally deploying) every passing change. |
| **CDF** | Cumulative distribution function; KS compares two empirical CDFs. |
| **Champion** | The model currently serving production. |
| **Challenger** | A candidate model competing to replace the champion. |
| **CI (Continuous Integration)** | Automatically building and testing every change. |
| **Class imbalance** | One class far rarer than the other (0.17% fraud). |
| **cloudpickle** | Library that serialises Python objects (models) to bytes. |
| **Concept drift** | Change in P(y\|X): the input-label relationship changes. |
| **Confusion matrix** | 2×2 table of TP, FP, FN, TN at a threshold. |
| **Container** | A running, isolated instance of an image. |
| **Covariate shift** | Another name for data drift, P(X) changing. |
| **Cron** | Schedule syntax, e.g. `0 6 * * *` = every day at 06:00. |
| **DAG** | Directed acyclic graph: tasks with dependencies, no cycles (`dvc dag`). |
| **Data contract** | An agreed schema/quality spec between data producer and consumer. |
| **Data drift** | Change in the input distribution P(X) over time. |
| **Data leakage** | Training on information unavailable at prediction time. |
| **Data lineage** | Traceability of which data/code produced which model. |
| **Deployment (Prefect)** | A flow registered with a schedule/parameters so it can run remotely. |
| **Docker Compose** | A tool to define and run multi-container apps from one YAML file. |
| **Dockerfile** | Recipe for building an image, layer by layer. |
| **Drift share** | Fraction of monitored features flagged as drifted (threshold 30%). |
| **DVC** | Data Version Control: git-like versioning for data, plus pipelines. |
| **`dvc.lock`** | Recorded hashes of the last pipeline run. |
| **`dvc repro`** | Re-run pipeline stages whose inputs changed. |
| **Early stopping** | Stop training when validation stops improving. |
| **Endpoint** | A URL path + method the API exposes (`POST /predict`). |
| **Environment variable** | A key-value setting provided by the OS/container, used for config and secrets. |
| **Evidently** | Open-source library for ML data/model monitoring reports. |
| **Experiment (MLflow)** | A named group of runs (`fraud-detection`). |
| **Experiment tracking** | Recording params, metrics, artifacts and versions of every run. |
| **F1 score** | Harmonic mean of precision and recall. |
| **False negative (FN)** | Fraud predicted as legit (missed fraud). |
| **False positive (FP)** | Legit predicted as fraud (false alarm). |
| **FastAPI** | Python web framework built on type hints and pydantic. |
| **Feature** | A model input variable. |
| **Feature engineering** | Creating model inputs from raw data (`log_amount`, `hour`). |
| **Feature store** | A system serving consistent features to training and serving (e.g. Feast). |
| **Fixture (pytest)** | Reusable setup object injected into tests. |
| **Flow (Prefect)** | A decorated Python function representing a workflow. |
| **Gauge** | Prometheus metric that can go up or down. |
| **GHCR** | GitHub Container Registry (`ghcr.io`). |
| **GitHub Actions** | GitHub's CI/CD automation platform. |
| **GitOps** | Deploying by committing desired state to git, applied automatically. |
| **Gradient boosting** | Ensemble that adds trees sequentially, each correcting previous errors. |
| **Grafana** | Dashboarding tool that queries data sources such as Prometheus. |
| **Ground truth** | The true label (was it really fraud?). |
| **Health check** | Endpoint/command reporting whether a service is ready. |
| **Histogram (Prometheus)** | Metric counting observations into buckets for percentiles. |
| **Holdout set** | Data kept aside for final evaluation (the test set). |
| **HPA** | Kubernetes Horizontal Pod Autoscaler. |
| **Hyper-parameter** | A setting chosen before training (depth, learning rate). |
| **Idempotent** | Safe to run repeatedly with the same result (promote twice → no-op). |
| **Image (Docker)** | Immutable layered template for containers. |
| **Inference** | Using a trained model to make predictions. |
| **Input example** | A sample input logged with the model for docs/validation. |
| **Instrumentation** | Adding code that emits metrics/traces. |
| **JSONL** | JSON Lines: one JSON object per line; append-friendly logs. |
| **KS test** | Kolmogorov–Smirnov test comparing two distributions. |
| **Label delay** | Time between a prediction and knowing its true outcome. |
| **Latency** | Time to serve one request. |
| **Layer caching** | Docker reusing unchanged image layers to speed up builds. |
| **Lazy validation** | Collect all schema errors before raising (`lazy=True`). |
| **Least privilege** | Grant only the permissions needed (`packages: write` only in the push job). |
| **Lint / linter** | Static code checks for errors and style (ruff). |
| **Liveness / readiness probe** | Kubernetes checks: is the process alive / can it take traffic. |
| **Lockfile** | Exact versions of every dependency (`requirements.lock.txt`). |
| **log1p** | log(1 + x); safe for zero amounts. |
| **MD5 hash** | Fingerprint of file contents; DVC uses it to identify data versions. |
| **MLflow** | Platform for experiment tracking, model packaging and registry. |
| **mlflow-skinny** | Lightweight MLflow client package without the server/UI. |
| **MLOps** | Practices for reliably building, deploying and operating ML in production. |
| **Model registry** | Catalogue of named, versioned models with metadata and aliases. |
| **Model signature** | Declared input/output schema of a logged model. |
| **Multiple testing** | Running many tests inflates false positives. |
| **Nested run** | An MLflow run created inside a parent run (Optuna trials). |
| **Non-root user** | Running the container process without admin rights. |
| **Observability** | Understanding system behaviour from metrics, logs and traces. |
| **Online inference** | Real-time, per-request predictions. |
| **OpenAPI / Swagger** | Machine-readable API spec; FastAPI serves interactive docs at `/docs`. |
| **Optuna** | Hyper-parameter optimisation framework. |
| **Orchestration** | Coordinating multi-step workflows with dependencies, retries and schedules. |
| **Overfitting** | Model memorises training data and generalises poorly. |
| **p-value** | Probability of seeing a result at least this extreme if there were no real difference. |
| **p50 / p95 / p99** | Latency percentiles: 50%/95%/99% of requests are faster than this. |
| **pandera** | DataFrame schema validation library. |
| **Parquet** | Columnar, compressed, typed file format. |
| **PCA** | Principal Component Analysis; produced the anonymised V1–V28. |
| **Pin (dependency)** | Fixing an exact package version (`==`). |
| **Pipeline (DVC)** | Ordered stages defined in `dvc.yaml`. |
| **PR-AUC** | Area under the precision-recall curve; the headline metric for rare positives. |
| **Precision** | TP / (TP + FP): how many flagged are truly fraud. |
| **Prediction drift** | Change in the distribution of model outputs. |
| **Prefect** | Python workflow orchestration framework. |
| **Prometheus** | Pull-based time-series metrics database. |
| **PromQL** | Prometheus query language (`rate`, `histogram_quantile`). |
| **Promotion** | Making a model version the production one (moving `@champion`). |
| **Provisioning (Grafana)** | Loading datasources/dashboards from files at startup. |
| **PSI** | Population Stability Index, a binned drift measure. |
| **pydantic** | Data validation using Python type hints. |
| **pyfunc** | MLflow's generic model interface (`load_model().predict()`). |
| **rate()** | PromQL function: per-second increase of a counter over a window. |
| **Recall** | TP / (TP + FN): how much fraud is caught. |
| **RED metrics** | Rate, Errors, Duration: core service health signals. |
| **Reference data** | Baseline distribution (training set) for drift comparison. |
| **Registry URI** | Address of the model registry (same as the tracking server here). |
| **Remote (DVC)** | Storage location for data (local folder, S3, GDrive). |
| **Reproducibility** | Ability to recreate a result exactly from code + data + config + env. |
| **Retry** | Automatically re-running a failed task after a delay. |
| **ROC-AUC** | Area under the TPR-vs-FPR curve; optimistic on imbalanced data. |
| **Rollback** | Reverting production to a previous version. |
| **ruff** | Very fast Python linter. |
| **Run (MLflow)** | One execution of training code with its logged data. |
| **`scale_pos_weight`** | XGBoost parameter weighting positive examples (#neg/#pos). |
| **Schema** | Expected structure: columns, types, constraints. |
| **Scrape** | Prometheus pulling metrics from an endpoint. |
| **Secrets** | Sensitive values (passwords, tokens) kept out of code. |
| **Shadow deployment** | New model scores live traffic silently, without affecting decisions. |
| **SHA (commit)** | Unique git commit hash; used as an immutable image tag. |
| **SLO / SLA** | Service level objective (internal target) / agreement (contractual). |
| **Slim image** | Minimal base image with fewer packages. |
| **Smoke test** | Quick end-to-end run checking that nothing is fundamentally broken. |
| **SMOTE** | Synthetic oversampling of the minority class (not used here). |
| **Stage (DVC)** | One step of a DVC pipeline (validate, split, train). |
| **Subflow** | A Prefect flow called inside another flow. |
| **Task (Prefect)** | A unit of work inside a flow, with its own state and retries. |
| **Test set** | Final, untouched evaluation data (latest 15% by time). |
| **Threshold** | Probability cutoff above which a transaction is flagged. |
| **Time-based split** | Splitting data chronologically to mimic deployment. |
| **TPE** | Tree-structured Parzen Estimator, Optuna's default Bayesian sampler. |
| **Tracking server** | MLflow service that receives and stores run data. |
| **Training-serving skew** | Mismatch between features at training time and serving time. |
| **Transitive dependency** | A package your dependencies depend on. |
| **True positive (TP)** | Fraud correctly flagged. |
| **Twelve-factor app** | Methodology for cloud apps (config in env, stateless processes...). |
| **uvicorn** | ASGI server that runs FastAPI. |
| **Validation set** | Data for tuning decisions (early stopping, Optuna, threshold). |
| **Version (model)** | An immutable numbered entry of a registered model. |
| **Virtual environment (venv)** | An isolated Python package installation per project. |
| **Volume (Docker)** | Docker-managed persistent storage. |
| **Wasserstein distance** | "Earth-mover" distance between distributions; an alternative drift measure. |
| **Worker** | A process that picks up and executes scheduled flow runs. |
| **XGBoost** | Optimised gradient-boosted-trees library. |
| **xgboost-cpu** | CPU-only XGBoost wheel without bundled CUDA libraries. |
