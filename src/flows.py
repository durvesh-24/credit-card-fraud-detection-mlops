"""Prefect orchestration.

fraud-retraining : validate -> split -> train -> promote (each a task with retries), then asks the
                   API to reload if a new champion was promoted.
daily-monitoring : drift check on the last 24h of API logs; on drift, runs fraud-retraining as a
                   subflow. Scheduled daily with `python -m src.flows serve`.

    python -m src.flows retrain             # run the retraining flow once
    python -m src.flows monitor             # run the monitoring flow once
    python -m src.flows serve               # long-running: daily-monitoring every day at 06:00
"""

from __future__ import annotations

import argparse
import os

import requests
from prefect import flow, get_run_logger, task

from src import monitor, promote, split, train, validate


# ---------------------------------------------------------------- tasks
@task(name="validate-data", retries=2, retry_delay_seconds=10)
def validate_data() -> dict:
    return validate.run()


@task(name="split-data", retries=2, retry_delay_seconds=10)
def split_data() -> dict:
    return split.run()


@task(name="train-model", retries=1, retry_delay_seconds=60)
def train_model(tune: bool = False) -> dict:
    return train.run(tune=tune)


@task(name="promote-model", retries=2, retry_delay_seconds=15)  # e.g. MLflow briefly unreachable
def promote_model() -> dict:
    return promote.run()


@task(name="reload-api", retries=2, retry_delay_seconds=5)
def reload_api() -> None:
    """Best effort: tell the running API to pick up the new @champion."""
    url = os.getenv("API_URL", "http://localhost:8000")
    try:
        r = requests.post(f"{url}/reload", timeout=30)
        get_run_logger().info("API reload: %s %s", r.status_code, r.text)
    except requests.RequestException as exc:
        get_run_logger().warning("API not reachable at %s (%s) — restart it manually", url, exc)


@task(name="check-drift", retries=1, retry_delay_seconds=10)
def check_drift(lookback_hours: float | None) -> dict:
    return monitor.run_drift_check(lookback_hours)


# ---------------------------------------------------------------- flows
@flow(name="fraud-retraining", log_prints=True)
def retraining_flow(tune: bool = False) -> dict:
    validate_data()  # raises -> flow fails -> nothing downstream runs on bad data
    split_data()
    trained = train_model(tune)
    promotion = promote_model()
    if promotion["promoted"]:
        reload_api()
    return {"registered_version": trained["registered_version"], "promotion": promotion}


@flow(name="daily-monitoring", log_prints=True)
def daily_monitoring(lookback_hours: float | None = 24, retrain_on_drift: bool = True) -> dict:
    logger = get_run_logger()
    drift = check_drift(lookback_hours)
    if drift["should_retrain"] and retrain_on_drift:
        logger.warning("drift share %.1f%% > threshold: starting fraud-retraining",
                       100 * drift["drift_share"])
        drift["retraining"] = retraining_flow()  # runs as a subflow, visible in the Prefect UI
    else:
        logger.info("no retraining needed (%s)", drift.get("status"))
    return drift


def main() -> None:
    ap = argparse.ArgumentParser(description="Run or schedule the Prefect flows")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("retrain")
    r.add_argument("--tune", action="store_true")
    m = sub.add_parser("monitor")
    m.add_argument("--lookback-hours", type=float, default=24)
    s = sub.add_parser("serve")
    s.add_argument("--cron", default="0 6 * * *", help="cron schedule (default 06:00 daily)")
    args = ap.parse_args()

    if args.cmd == "retrain":
        retraining_flow(tune=args.tune)
    elif args.cmd == "monitor":
        daily_monitoring(lookback_hours=args.lookback_hours)
    else:
        # Registers a deployment with a cron schedule and keeps a worker process polling for runs.
        daily_monitoring.serve(name="daily-monitoring", cron=args.cron)


if __name__ == "__main__":
    main()
