# utils/head_to_head.py

import os
import pandas as pd
from functools import lru_cache
from typing import Dict, Any

# --------------------------------------------------------------------------- #
# 1.  ONE-TIME DATA LOAD (cached with lru_cache)                               #
# --------------------------------------------------------------------------- #
DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "data", "cleanedDataset.csv"
)

@lru_cache(maxsize=1)
def _dataset() -> pd.DataFrame:
    """Read cleanedDataset.csv once and keep it in memory."""
    print(f"HEAD_TO_HEAD: loading {DATA_PATH}…")
    return pd.read_csv(DATA_PATH)


# --------------------------------------------------------------------------- #
# 2.  PUBLIC API: head_to_head                                                #
# --------------------------------------------------------------------------- #
def head_to_head(player1_id: int, player2_id: int) -> Dict[str, Any]:
    """
    Return every match between the two players plus a win tally.

    Response schema
    ---------------
    {
        "player1_id": 134770,
        "player2_id": 126203,
        "total_matches": 7,
        "head_to_head": {"134770": 5, "126203": 2},
        "matches": [ …dicts for each match… ]
    }
    """
    df = _dataset()

    # All H2H rows
    matches = df[
        ((df["p1_id"] == player1_id) & (df["p2_id"] == player2_id)) |
        ((df["p1_id"] == player2_id) & (df["p2_id"] == player1_id))
    ].copy()

    # Initialize win counts
    wins = {str(player1_id): 0, str(player2_id): 0}

    # If no matches, return zeroed response
    if matches.empty:
        return {
            "player1_id": player1_id,
            "player2_id": player2_id,
            "total_matches": 0,
            "head_to_head": wins,
            "matches": []
        }

    result_col = matches["RESULT"]

    # Case A: numeric 0/1 → 1 means player1 won, 0 means player2 won
    if pd.api.types.is_numeric_dtype(result_col) and set(result_col.dropna().unique()) <= {0, 1}:
        wins[str(player1_id)] = int((result_col == 1).sum())
        wins[str(player2_id)] = int((result_col == 0).sum())

    # Case B: numeric and values are actual player IDs
    elif pd.api.types.is_numeric_dtype(result_col):
        wins[str(player1_id)] = int((result_col == player1_id).sum())
        wins[str(player2_id)] = int((result_col == player2_id).sum())

    # Case C: “Player 1” / “Player 2” string labels
    else:
        wins[str(player1_id)] = int(
            ((result_col == "Player 1") & (matches["p1_id"] == player1_id)).sum() +
            ((result_col == "Player 2") & (matches["p2_id"] == player1_id)).sum()
        )
        wins[str(player2_id)] = int(
            ((result_col == "Player 1") & (matches["p1_id"] == player2_id)).sum() +
            ((result_col == "Player 2") & (matches["p2_id"] == player2_id)).sum()
        )

    return {
        "player1_id": player1_id,
        "player2_id": player2_id,
        "total_matches": len(matches),
        "head_to_head": wins,
        "matches": matches.to_dict(orient="records")
    }
