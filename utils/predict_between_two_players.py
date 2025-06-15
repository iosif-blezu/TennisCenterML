from __future__ import annotations

import datetime as _dt
import functools
import pickle
from datetime import timedelta
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm
from xgboost import XGBClassifier

from tennisbot.config import get_settings
from utils.updateStats import createStats, updateStats, getStats


cfg = get_settings()
API_BASE = "http://localhost:5000/api/tennis"
RANKINGS_ENDPOINT = f"{API_BASE}/rankings/atp/db?limit=500&page=1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_DATA_PATH = cfg.DATA_PATH
MODEL_PATH = PROJECT_ROOT / "models" / "xgb_model.json"

# Caching for stats
STATS_CACHE_PATH = Path(CLEAN_DATA_PATH).parent / "stats_cache.pkl"
CACHE_TTL = timedelta(days=1)


def _age_from_iso(birth_iso: str) -> int:
    try:
        b = _dt.datetime.fromisoformat(birth_iso.replace("Z", ""))
    except Exception:
        return 0
    today = _dt.datetime.today()
    return today.year - b.year - ((today.month, today.day) < (b.month, b.day))

@functools.lru_cache(maxsize=1)
def _rankings_map() -> Dict[int, int]:
    resp = requests.get(RANKINGS_ENDPOINT, timeout=10)
    resp.raise_for_status()
    return {p["player_id"]: p["points"] for p in resp.json().get("players", [])}

@functools.lru_cache(maxsize=1)
def _model() -> XGBClassifier:
    model = XGBClassifier()
    model.load_model(str(MODEL_PATH))
    return model


def _fetch_player(player_id: int) -> dict:
    d1 = requests.get(f"{API_BASE}/player/{player_id}", timeout=10).json()
    api_id = d1.get("player_id_api")
    d2 = requests.get(f"{API_BASE}/player/rapid/{api_id}", timeout=10).json()
    return {
        "Name":       (d1.get("row_name") or d2.get("fullName") or f"Player {player_id}").strip(),
        "ID":         player_id,
        "ATP_POINTS": _rankings_map().get(player_id, 0),
        "ATP_RANK":   d2.get("ranking", 0),
        "AGE":        _age_from_iso(d2.get("birthDateUTC", "")),
        "HEIGHT":     int(round((d2.get("heightMeters") or 0) * 100)),
    }


def _build_stats() -> dict:
    # Always rebuild from raw CSV
    df = pd.read_csv(CLEAN_DATA_PATH)
    stats = createStats()
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building stats"):
        stats = updateStats(row, stats)
    return stats

@functools.lru_cache(maxsize=1)
def _base_stats() -> dict:
    # Load from cache if fresh
    try:
        if STATS_CACHE_PATH.exists():
            mtime = STATS_CACHE_PATH.stat().st_mtime
            if _dt.datetime.now() - _dt.datetime.fromtimestamp(mtime) < CACHE_TTL:
                with STATS_CACHE_PATH.open('rb') as f:
                    return pickle.load(f)
    except Exception:
        pass
    # Rebuild and cache
    stats = _build_stats()
    try:
        with STATS_CACHE_PATH.open('wb') as f:
            pickle.dump(stats, f)
    except Exception:
        pass
    return stats

def predict_match(
    player1_id: int,
    player2_id: int,
    *,
    surface: str = "Hard",
    best_of: int = 3,
    draw_size: int = 128,
) -> str:
    # Return human-readable prediction line.
    p1 = _fetch_player(player1_id)
    p2 = _fetch_player(player2_id)
    meta = {"BEST_OF": best_of, "DRAW_SIZE": draw_size, "SURFACE": surface}
    feats = getStats(p1, p2, meta, _base_stats())
    X = pd.DataFrame([dict(sorted(feats.items()))]).to_numpy(dtype=object)
    p2_prob, p1_prob = _model().predict_proba(X)[0]
    if p1_prob == p2_prob:
        return f"Draw! Both players have equal chances ({p1_prob:.1%})"
    winner = p1["Name"] if p1_prob > p2_prob else p2["Name"]
    return (f"{winner} is predicted to win — "
            f"{p1['Name']} {p1_prob:.1%} | {p2['Name']} {p2_prob:.1%}")


def predict_between_two_players(
    player1_id: int,
    player2_id: int,
    *,
    surface: str = "Hard",
    best_of: int = 3,
    draw_size: int = 128,
) -> dict:
    # Return structured dict with percentages and winner metadata.
    p1 = _fetch_player(player1_id)
    print(p1)
    p2 = _fetch_player(player2_id)
    print(p2)
    meta = {"BEST_OF": best_of, "DRAW_SIZE": draw_size, "SURFACE": surface}
    feats = getStats(p1, p2, meta, _base_stats())
    X = pd.DataFrame([dict(sorted(feats.items()))]).to_numpy(dtype=object)
    p2_prob, p1_prob = _model().predict_proba(X)[0]
    p1_pct = round(float(p1_prob) * 100, 1)
    p2_pct = round(float(p2_prob) * 100, 1)
    winner_id = player1_id if p1_prob >= p2_prob else player2_id
    winner = p1["Name"] if p1_prob >= p2_prob else p2["Name"]
    return {
        "player1_id":    player1_id,
        "player2_id":    player2_id,
        "surface":       surface,
        "best_of":       best_of,
        "draw_size":     draw_size,
        "p1_percentage": p1_pct,
        "p2_percentage": p2_pct,
        "winner_id":     winner_id,
        "winner":        winner,
    }
