from pathlib import Path
import pandas as pd

CSV_IN  = Path("../data/atp_players_not_in_api.csv")
CSV_OUT = Path("../data/atp_players_not_in_api_full.csv")

df = pd.read_csv(CSV_IN, dtype=str)
df = df.replace("", "unknown").fillna("unknown")

df.to_csv(CSV_OUT, index=False)
print(f"✓ All blanks filled. Wrote {len(df)} rows to {CSV_OUT.resolve()}")