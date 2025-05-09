import os, re
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()
tavily = TavilyClient(os.getenv("TAVILY_API_KEY"))

BUCKETS = {
    "day":   "day",   "d": "day",
    "week":  "week",  "w": "week",
    "month": "month", "m": "month",
    "year":  "year",  "y": "year",
}

def _validate_bucket(b: str) -> str:
    if b.lower() not in BUCKETS:
        raise ValueError(f"time_range must be one of {list(BUCKETS)}")
    return BUCKETS[b.lower()]

def build_query(player: str) -> str:
    return f'What are the latest news for tennis player {player}?'

def news_for_player(
    player: str,
    *,
    time: str = "week", # "day" / "week" / "month" / "year"
    max_results: int = 10,
    score_cut: float = 0.4
) -> list[dict]:
    """
    1) Tavily search with templated query
    2) keep URLs above score_cut
    3) Tavily extract full text
    """
    bucket = _validate_bucket(time)
    search = tavily.search(
        query           = build_query(player),
        search_depth    = "advanced",
        max_results     = max_results,
        time_range      = bucket,
        include_answer  = "basic"
    )

    urls = [r["url"] for r in search["results"] if r.get("score", 0) >= score_cut]
    if not urls:
        return []

    extracted = tavily.extract(urls=urls)
    return extracted["results"]

if __name__ == "__main__":
    time_range='week'
    docs = news_for_player("Carlos Alcaraz", time=time_range, max_results=10, score_cut=0.4)
    print(f"[{time_range}] → {len(docs)} articles")
