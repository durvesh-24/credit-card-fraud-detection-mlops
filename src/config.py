"""Shared configuration: project paths, params.yaml loading and .env loading."""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import yaml

# Windows consoles default to cp1252; MLflow prints emoji and would crash with UnicodeEncodeError.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
PARAMS_PATH = ROOT / "params.yaml"
REPORTS_DIR = ROOT / "reports"

try:  # python-dotenv is optional at runtime (the slim API image does not ship it)
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
except ImportError:  # pragma: no cover
    pass


@lru_cache(maxsize=1)
def load_params(path: str | os.PathLike = PARAMS_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str | os.PathLike) -> Path:
    """Resolve a path from params.yaml relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def tracking_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
