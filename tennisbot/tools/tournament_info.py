# tennisbot/tools/tournament_info.py
from typing import Optional, TypedDict

import requests, requests_cache
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun

from tennisbot.config import get_settings

cfg = get_settings()
requests_cache.install_cache(
    "tournament_cache",
    expire_after=cfg.CACHE_TTL["tournament"]
)


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
            info = requests.get(url, timeout=8).json()
        except Exception as e:
            return f"Tournament info API error: {e}"
        return info

    async def _arun(self, *_, **__):
        raise NotImplementedError
