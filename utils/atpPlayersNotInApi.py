
from pathlib import Path
import pandas as pd

MERGED_CSV = Path("../data/all/merged_players.csv")
ATP_CSV    = Path("../data/atp_players.csv")
OUT_CSV    = Path("../data/atp_players_not_in_api.csv")

merged_ids = pd.read_csv(MERGED_CSV, usecols=["player_id"], dtype={"player_id": "int64"})
atp_df     = pd.read_csv(ATP_CSV,    dtype={"player_id": "int64"})

missing = atp_df[~atp_df["player_id"].isin(merged_ids["player_id"])].copy()

missing["row_name"]      = "unknown"
missing["player_id_api"] = "unknown"
missing["slug"]          = "unknown"

missing = missing[[
    "player_id", "name_first", "name_last",
    "row_name", "player_id_api",
    "hand", "dob", "ioc", "height", "slug"
]]

missing.to_csv(OUT_CSV, index=False)
print(f" Wrote {len(missing)} missing players to {OUT_CSV.resolve()}")
