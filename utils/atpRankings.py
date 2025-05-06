import json
from pathlib import Path
from datetime import datetime, timezone
import csv

INFILE  = Path("../data/top500.json")
OUTFILE = Path("../data/atpRankings.csv")

def flatten(player: dict) -> dict:
    team    = player["team"]
    country = player.get("country") or team.get("country", {})
    return {
        "player_id":          team["id"],
        "name":               player["rowName"],
        "slug":               team["slug"],
        "rank":               player["ranking"],
        "prev_rank":          player.get("previousRanking"),
        "points":             player["points"],
        "prev_points":        player.get("previousPoints"),
        "country":            country.get("alpha3"),
        "tournaments_played": player.get("tournamentsPlayed"),
        "best_rank":          player.get("bestRanking"),
        "gender":             team.get("gender"),
    }

def main() -> None:
    data = json.loads(INFILE.read_text(encoding="utf-8"))["rankings"]
    rows = [flatten(p) for p in data]
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for row in rows:
        row["date_extracted"] = timestamp

    fieldnames = list(rows[0].keys())
    with OUTFILE.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUTFILE.resolve()}")

if __name__ == "__main__":
    main()