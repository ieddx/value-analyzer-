"""Local disk cache for API responses.

All fetched data lands here before the calling function returns, so repeated
analysis runs never hit the network for the same ticker.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".value_analyzer" / "cache"


def _ensure() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def df_path(key: str) -> Path:
    _ensure()
    return CACHE_DIR / f"{key}.parquet"


def json_path(key: str) -> Path:
    _ensure()
    return CACHE_DIR / f"{key}.json"


def save_df(df: pd.DataFrame, key: str) -> None:
    p = df_path(key)
    df.to_parquet(p)
    logger.debug("cached %s → %s (%d rows)", key, p, len(df))


def load_df(key: str) -> pd.DataFrame | None:
    p = df_path(key)
    if p.exists():
        logger.debug("cache hit %s", key)
        return pd.read_parquet(p)
    return None


def save_json(data: dict | list, key: str) -> None:
    json_path(key).write_text(json.dumps(data, default=str))


def load_json(key: str) -> dict | list | None:
    p = json_path(key)
    if p.exists():
        return json.loads(p.read_text())
    return None


def invalidate(key: str) -> None:
    """Remove a cached entry (forces re-fetch on next access)."""
    for p in [df_path(key), json_path(key)]:
        if p.exists():
            p.unlink()
            logger.info("invalidated cache key %s", key)
