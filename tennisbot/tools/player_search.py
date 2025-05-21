"""
Fuzzy search for a tennis player.

Usage:
    from tennisbot.tools.player_search import PlayerSearchTool

    tool = PlayerSearchTool()
    result = tool.invoke({"query": "Alcaraz"})
    # -> {"player_id": 207989, "name": "Carlos Alcaraz"}
"""
from typing import Optional, TypedDict

import requests
import requests_cache
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun

from tennisbot.config import get_settings

cfg = get_settings()
requests_cache.install_cache(
    "player_search_cache",
    expire_after=cfg.CACHE_TTL["players_search"]
)


class _InputSchema(TypedDict):
    query: str
    limit: Optional[int]


class _OutputSchema(TypedDict):
    player_id: int
    name: str
    score: float


class PlayerSearchTool(BaseTool):
    """Find the best-matching player_id for a fuzzy query (surname, full name…)."""

    name: str = "player_search"
    description: str = (
        "Use when you need to resolve an arbitrary player name to the canonical "
        "player_id integer used by the Tennis API. "
        "Input keys: { query: string, limit?: integer }. "
        "Returns JSON { player_id, name, score }."
    )

    def _run(               # sync version (fine for short HTTP call)
        self,
        query: _InputSchema | str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> _OutputSchema | str:
        # allow LLM to pass bare string
        if isinstance(query, str):
            query = {"query": query}

        limit = query.get("limit", 5)
        params = {"q": query["query"], "limit": limit}
        url = f"{cfg.endpoint_player_search}"

        try:
            resp = requests.get(url, params=params, timeout=8)
            resp.raise_for_status()
        except Exception as e:
            return f"Search API error: {e}"

        data = resp.json()
        if data["total"] == 0:
            return "No matching player found."

        # pick highest-score hit
        best = max(data["players"], key=lambda p: p.get("score", 0))
        return {
            "player_id": best["player_id"],
            "name": f"{best['name_first']} {best['name_last']}",
            "score": best.get("score", 0.0),
        }

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError("Use sync _run for now (fast HTTP request).")
