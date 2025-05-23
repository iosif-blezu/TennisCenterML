from typing import Optional, TypedDict
import requests, requests_cache
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun

from tennisbot.config import get_settings

cfg = get_settings()
# Install cache and retrieve controller
tmp_session = requests_cache.install_cache(
    "tournament_cache",
    expire_after=cfg.CACHE_TTL["tournament"]
)
cache = requests_cache.get_cache()
# Print cache configuration
print(f"[TournamentInfoTool] Cache backend: {cache.__class__.__name__}")
print(f"[TournamentInfoTool] Cache name: {cache.cache_name}.sqlite")
print(f"[TournamentInfoTool] TTL (seconds): {cfg.CACHE_TTL['tournament']}")

class _Input(TypedDict):
    tournament_id: int

class TournamentInfoTool(BaseTool):
    name: str = "tournament_info"
    description: str =  (
        "Fetch surface, prize money, holder, etc. for a tournament. "
        "Input {tournament_id:int}. Returns JSON."
    )

    def _run(
        self,
        tournament_id: int,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ):
        url = f"{cfg.endpoint_tournament}/{tournament_id}/info"
        try:
            resp = requests.get(url, timeout=8)
            info = resp.json()
            # Log cache usage
            from_cache = getattr(resp, 'from_cache', False)
            print(f"[TournamentInfoTool] Request URL: {resp.url}")
            print(f"[TournamentInfoTool] Response from cache: {from_cache}")
        except Exception as e:
            return f"Tournament info API error: {e}"
        return info

    async def _arun(self, *_, **__):
        raise NotImplementedError
