"""Roll back (or forward) by pointing the @champion alias at a chosen model version.

No retraining and no redeploy: the API resolves models:/fraud-detector@champion, so after this
runs, restart the API container or call POST /reload.

    python scripts/rollback.py --list
    python scripts/rollback.py --version 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from src.config import load_params, tracking_uri  # noqa: E402


def list_versions(client: MlflowClient, name: str) -> None:
    # search_model_versions() omits aliases; read them from the registered model (alias -> version)
    alias_map = client.get_registered_model(name).aliases
    for v in sorted(client.search_model_versions(f"name='{name}'"), key=lambda v: int(v.version)):
        aliases = ",".join(a for a, ver in alias_map.items() if str(ver) == str(v.version)) or "-"
        tags = v.tags
        print(
            f"v{v.version:<4} aliases={aliases:<22} "
            f"val_pr_auc={tags.get('val_pr_auc', '?'):<10} threshold={tags.get('threshold', '?')}"
        )


def main() -> None:
    params = load_params()
    name = params["mlflow"]["registered_model_name"]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", help="model version that should become the alias target")
    ap.add_argument("--alias", default=params["promote"]["champion_alias"])
    ap.add_argument("--list", action="store_true", help="list versions and their aliases")
    args = ap.parse_args()

    mlflow.set_tracking_uri(tracking_uri())
    client = MlflowClient()
    if args.list or not args.version:
        list_versions(client, name)
        return

    client.get_model_version(name, args.version)  # raises if the version does not exist
    try:
        previous = client.get_model_version_by_alias(name, args.alias).version
    except mlflow.exceptions.MlflowException:
        previous = None
    client.set_registered_model_alias(name, args.alias, args.version)
    print(f"@{args.alias}: v{previous} -> v{args.version}. Restart the API or POST /reload.")


if __name__ == "__main__":
    main()
