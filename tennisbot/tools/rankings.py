from typing import Optional, TypedDict, List
import requests, requests_cache
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun

from tennisbot.config import get_settings

cfg = get_settings()
# Initialize cache and capture cache object
tmp_session = requests_cache.install_cache(
    "rankings_cache",
    expire_after=cfg.CACHE_TTL["rankings"]
)
# Ensure we have the CacheController via get_cache()
cache = requests_cache.get_cache()
# Print cache configuration
print(f"[RankingsTool] Cache backend: {cache.__class__.__name__}")
print(f"[RankingsTool] Cache name: {cache.cache_name}.sqlite")
print(f"[RankingsTool] TTL (seconds): {cfg.CACHE_TTL['rankings']}")

class _Input(TypedDict, total=False):
    limit: int          # default 10
    page: int           # default 1
    country: str        # optional ISO-3 filter

class _Player(TypedDict):
    rank: int
    player_id: int
    name: str
    country: str
    points: int

class RankingsTool(BaseTool):
    name: str = "rankings"
    description: str = (
        "Fetch current ATP rankings. Input keys: {limit:int=10, page:int=1, country?:str}. "
        "Returns list of players [{rank, player_id, name, country, points}]."
    )

    def _run(
        self,
        limit: Optional[int] = 10,
        page: Optional[int] = 1,
        country: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> List[_Player] | str:
        params = {"limit": limit, "page": page}
        try:
            resp = requests.get(cfg.endpoint_rankings, params=params, timeout=8)
            data = resp.json()
            # Log cache usage
            from_cache = getattr(resp, 'from_cache', False)
            print(f"[RankingsTool] Request URL: {resp.url}")
            print(f"[RankingsTool] Response from cache: {from_cache}")
        except Exception as e:
            return f"Rankings API error: {e}"

        players = data.get("players", [])
        if country:
            players = [p for p in players if p.get("country", "").lower() == country.lower()]

        return [
            {
                "rank": p["rank"],
                "player_id": p["player_id"],
                "name": p["name"],
                "country": p["country"],
                "points": p["points"],
            }
            for p in players
        ]

    async def _arun(self, *_, **__):
        raise NotImplementedError
