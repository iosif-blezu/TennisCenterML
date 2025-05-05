"""
mergePlayers.py
Combine API ranking data with master player list.

Output columns (in this order):
player_id, name_first, name_last, row_name, player_id_api, hand, dob, ioc, height, slug
"""

import json
import pandas as pd
from pathlib import Path

# ---------- file paths ----------
RANKINGS_JSON = "../data/all/atp_matches_1991.csv"
PLAYERS_CSV   = "../data/atp_players.csv"
OUT_CSV       = "../data/all/merged_players.csv"

# ---------- helpers ----------
def split_name(full_name: str) -> tuple[str, str]:
    """
    Split 'Carlos Alcaraz' → ('Carlos', 'Alcaraz').
    If the surname contains spaces (e.g. 'van de Zandschulp') it keeps them.
    """
    parts = full_name.strip().split()
    return parts[0], " ".join(parts[1:]) if len(parts) > 1 else ""

# ---------- load files ----------
with RANKINGS_JSON.open(encoding="utf‑8") as f:
    rankings_raw = json.load(f)["rankings"]    # API object has a key called "rankings"

# flatten the JSON into a DataFrame with only what we need
api_df = pd.json_normalize(
    rankings_raw,
    sep="_"
)[["rowName", "id", "team_slug"]]                # id = player_id_api

api_df.rename(
    columns={"rowName": "row_name",
             "id": "player_id_api",
             "team_slug": "slug"},
    inplace=True
)

# break ‘row_name’ into first / last for joining
api_df[["name_first", "name_last"]] = api_df["row_name"].apply(
    lambda s: pd.Series(split_name(s))
)

# make everything lower‑case for a robust merge
api_df["key"] = (api_df["name_first"] + " " + api_df["name_last"]).str.lower()

# ---- load second file & build the same join key ----
players_df = pd.read_csv(PLAYERS_CSV, dtype={"ioc": "string"})
players_df["key"] = (players_df["name_first"] + " " +
                     players_df["name_last"]).str.lower()

# ---- merge: inner keeps only overlaps; change to 'left' if you want all 500 ----
merged = pd.merge(
    players_df,
    api_df[["key", "row_name", "player_id_api", "slug"]],
    on="key",
    how="inner",
    validate="one_to_one"          # will raise if duplicates appear
)

# arrange columns + drop the helper key
merged = merged[["player_id", "name_first", "name_last",
                 "row_name", "player_id_api",
                 "hand", "dob", "ioc", "height", "slug"]]

merged.to_csv(OUT_CSV, index=False)
print(f"✓ Wrote {len(merged)} rows to {OUT_CSV.resolve()}")
