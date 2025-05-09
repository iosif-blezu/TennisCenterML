import datetime as _dt
import functools
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from xgboost import XGBClassifier

# ────────────────────────────────────────────────────────────────────
# Configuration – change these if your paths or host differ
# ────────────────────────────────────────────────────────────────────
API_BASE          = "http://localhost:5000/api/tennis"
RANKINGS_ENDPOINT = f"{API_BASE}/rankings/atp/db?limit=500&page=1"
CLEAN_DATA_PATH   = "../data/cleanedDataset.csv"
MODEL_PATH        = "../models/xgb_model.json"

def _age_from_iso(birth_iso: str) -> int:
    try:
        b = _dt.datetime.fromisoformat(birth_iso.replace("Z", ""))
    except Exception:
        return 0
    today = _dt.datetime.today()
    return today.year - b.year - ((today.month, today.day) < (b.month, b.day))

@functools.lru_cache(maxsize=1)
def _rankings_map() -> dict[int, int]:
    """Fetch latest ATP rankings once per session → {player_id: points}."""
    resp = requests.get(RANKINGS_ENDPOINT, timeout=10)
    resp.raise_for_status()
    return {p["player_id"]: p["points"] for p in resp.json().get("players", [])}

def _fetch_player(player_id: int) -> dict:
    """Build minimal feature dict for one player (name, rank, points, etc.)."""
    d1 = requests.get(f"{API_BASE}/player/{player_id}", timeout=10).json()
    api_id = d1.get("player_id_api")
    d2 = requests.get(f"{API_BASE}/player/rapid/{api_id}", timeout=10).json()

    return {
        "Name":     (d1.get("row_name") or d2.get("fullName") or f"Player {player_id}").strip(),
        "ID":       player_id,
        "ATP_POINTS": _rankings_map().get(player_id, 0),
        "ATP_RANK": d2.get("ranking", 0),
        "AGE":       _age_from_iso(d2.get("birthDateUTC", "")),
        "HEIGHT":    int(round((d2.get("heightMeters") or 0) * 100)),
    }

# ────────────────────────────────────────────────────────────────────
# Stats / model loaders (cached)
# ────────────────────────────────────────────────────────────────────
from utils.updateStats import getStats, updateStats, createStats    # noqa: E402

@functools.lru_cache(maxsize=1)
def _base_stats() -> dict:
    df = pd.read_csv(CLEAN_DATA_PATH)
    stats = createStats()
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building stats"):
        stats = updateStats(row, stats)
    return stats

@functools.lru_cache(maxsize=1)
def _model() -> XGBClassifier:
    m = XGBClassifier()
    m.load_model(MODEL_PATH)
    return m

def predict_match(
    player1_id: int,
    player2_id: int,
    *,
    surface: str = "Hard",
    best_of: int = 3,
    draw_size: int = 128,
) -> str:

    p1 = _fetch_player(player1_id)
    p2 = _fetch_player(player2_id)
    meta = {"BEST_OF": best_of, "DRAW_SIZE": draw_size, "SURFACE": surface}

    feats = getStats(p1, p2, meta, _base_stats())
    X = pd.DataFrame([dict(sorted(feats.items()))]).to_numpy(dtype=object)

    p2_prob, p1_prob = _model().predict_proba(X)[0]   # class 0 → P2, class 1 → P1
    winner = p1["Name"] if p1_prob >= p2_prob else p2["Name"]
    loser = p2["Name"] if p1_prob >= p2_prob else p1["Name"]
    if p1_prob == p2_prob:
        return f"Draw! Both players have equal chances ({p1_prob:.1%})"

    return f"{winner} is predicted to win — {p1['Name']} {p1_prob:.1%} | {p2['Name']} {p2_prob:.1%}"