"""
Utility: predict head-to-head outcome with the XGBoost model
-----------------------------------------------------------
Exposes `predict_match(player1_id, player2_id, surface=…, best_of=…, draw_size=…)`
and returns a human-readable line such as:

    “Carlos Alcaraz is predicted to win — Carlos Alcaraz 58.4% | Jannik Sinner 41.6%”
"""
from __future__ import annotations

import datetime as _dt
import functools
from pathlib import Path
from typing import Dict

import numpy as np               # noqa: F401 (imported for completeness)
import pandas as pd
import requests
from tqdm import tqdm
from xgboost import XGBClassifier

# ────────────────────────────────────────────────────────────────────
# Configuration (most paths now come from tennisbot.config)
# ────────────────────────────────────────────────────────────────────
from tennisbot.config import get_settings
from utils.updateStats import createStats, updateStats, getStats  # noqa: E402  keep original order

cfg = get_settings()

API_BASE: str          = "http://localhost:5000/api/tennis"
RANKINGS_ENDPOINT: str = f"{API_BASE}/rankings/atp/db?limit=500&page=1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]          # ← NEW
CLEAN_DATA_PATH = cfg.DATA_PATH                             # unchanged
MODEL_PATH      = PROJECT_ROOT / "models" / "xgb_model.json"  # ← NEW

# ────────────────────────────────────────────────────────────────────
# Helper functions
# ────────────────────────────────────────────────────────────────────
def _age_from_iso(birth_iso: str) -> int:
    try:
        b = _dt.datetime.fromisoformat(birth_iso.replace("Z", ""))
    except Exception:
        return 0
    today = _dt.datetime.today()
    return today.year - b.year - ((today.month, today.day) < (b.month, b.day))


@functools.lru_cache(maxsize=1)
def _rankings_map() -> Dict[int, int]:
    """Fetch latest ATP rankings once per session → {player_id: points}."""
    resp = requests.get(RANKINGS_ENDPOINT, timeout=10)
    resp.raise_for_status()
    return {p["player_id"]: p["points"] for p in resp.json().get("players", [])}


def _fetch_player(player_id: int) -> dict:
    """Return minimal info dict required for feature generation."""
    d1 = requests.get(f"{API_BASE}/player/{player_id}", timeout=10).json()
    api_id = d1.get("player_id_api")
    d2 = requests.get(f"{API_BASE}/player/rapid/{api_id}", timeout=10).json()

    return {
        "Name":        (d1.get("row_name") or d2.get("fullName") or f"Player {player_id}").strip(),
        "ID":          player_id,
        "ATP_POINTS":  _rankings_map().get(player_id, 0),
        "ATP_RANK":    d2.get("ranking", 0),
        "AGE":         _age_from_iso(d2.get("birthDateUTC", "")),
        "HEIGHT":      int(round((d2.get("heightMeters") or 0) * 100)),
    }


# ────────────────────────────────────────────────────────────────────
# Cached loaders
# ────────────────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=1)
def _base_stats() -> dict:
    """Load the cleanedDataset.csv once and build cumulative stats dict."""
    df = pd.read_csv(CLEAN_DATA_PATH)
    stats = createStats()
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building stats"):
        stats = updateStats(row, stats)
    return stats


@functools.lru_cache(maxsize=1)
def _model() -> XGBClassifier:
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    return model


# ────────────────────────────────────────────────────────────────────
# Public entry-point
# ────────────────────────────────────────────────────────────────────
def predict_match(
    player1_id: int,
    player2_id: int,
    *,
    surface: str = "Hard",
    best_of: int = 3,
    draw_size: int = 128,
) -> str:
    """Return human-readable prediction line for the given matchup."""
    p1 = _fetch_player(player1_id)
    p2 = _fetch_player(player2_id)
    meta = {"BEST_OF": best_of, "DRAW_SIZE": draw_size, "SURFACE": surface}

    feats = getStats(p1, p2, meta, _base_stats())
    X = pd.DataFrame([dict(sorted(feats.items()))]).to_numpy(dtype=object)

    # XGBoost order: class 0 => player2, class 1 => player1
    p2_prob, p1_prob = _model().predict_proba(X)[0]
    winner = p1["Name"] if p1_prob >= p2_prob else p2["Name"]
    loser  = p2["Name"] if p1_prob >= p2_prob else p1["Name"]

    if p1_prob == p2_prob:
        return f"Draw! Both players have equal chances ({p1_prob:.1%})"

    return (f"{winner} is predicted to win — "
            f"{p1['Name']} {p1_prob:.1%} | {p2['Name']} {p2_prob:.1%}")
