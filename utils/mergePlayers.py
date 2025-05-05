from pathlib import Path
import json
import pandas as pd

RANKINGS_JSON = Path("../data/top500.json")
PLAYERS_CSV   = Path("../data/atp_players.csv")
OUT_CSV       = Path("../data/all/merged_players.csv")


def split_name(full_name: str) -> tuple[str, str]:
    parts = full_name.strip().split()
    return parts[0], " ".join(parts[1:]) if len(parts) > 1 else ""

with RANKINGS_JSON.open(encoding="utf‑8") as f:
    rankings_raw = json.load(f)["rankings"]

api_df = (
    pd.json_normalize(rankings_raw, sep="_")
      .loc[:, ["rowName", "team_id", "team_slug"]]
      .rename(columns={
          "rowName": "row_name",
          "team_id": "player_id_api",
          "team_slug": "slug",
      })
)

api_df[["name_first", "name_last"]] = api_df["row_name"].apply(
    lambda s: pd.Series(split_name(s))
)
api_df["key"] = (api_df["name_first"] + " " + api_df["name_last"]).str.lower()

dtype_map = {
    "player_id": "int64",
    "name_first": "string",
    "name_last": "string",
    "hand": "string",
    "dob": "Int64",
    "ioc": "string",
    "height": "Int64",
    "wikidata_id": "string",
}
players_df = pd.read_csv(PLAYERS_CSV, dtype=dtype_map, na_values=[""])

players_df["key"] = (
    players_df["name_first"].str.strip() + " " +
    players_df["name_last"].str.strip()
).str.lower()

dupes = players_df[players_df.duplicated("key", keep=False)]
if not dupes.empty:
    print("Duplicate names in atp_players.csv (showing first 10 rows):")
    print(dupes, "\n")
    print(len(dupes), "duplicates found")

merged = pd.merge(
    players_df,
    api_df[["key", "row_name", "player_id_api", "slug"]],
    on="key",
    how="inner",
    validate="many_to_one"
)

merged = merged[
    [
        "player_id", "name_first", "name_last",
        "row_name", "player_id_api",
        "hand", "dob", "ioc", "height", "slug"
    ]
]

merged.to_csv(OUT_CSV, index=False)
print(f"Wrote {len(merged)} rows to {OUT_CSV.resolve()}")