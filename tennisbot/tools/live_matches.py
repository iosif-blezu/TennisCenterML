# tennisbot/tools/live_matches.py
from typing import Optional, TypedDict, List

import requests, requests_cache
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun

from tennisbot.config import get_settings

cfg = get_settings()
requests_cache.install_cache(
    "live_cache",
    expire_after=cfg.CACHE_TTL["live"]
)


class _Input(TypedDict, total=False):
    player_id: int
    tournament_id: int
    surface: str


class _Match(TypedDict):
    eventId: int
    tournamentName: str
    roundName: str
    status: str
    home: str
    away: str
    score: str


class LiveMatchesTool(BaseTool):
    name: str = "live_matches"
    description: str = (
        "Get live matches. Optional filters: player_id, tournament_id, surface. "
        "Returns list of brief match objects."
    )

    def _run(
        self,
        player_id: Optional[int] = None,
        tournament_id: Optional[int] = None,
        surface: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> List[_Match] | str:
        try:
            matches = requests.get(cfg.endpoint_live, timeout=10).json()
        except Exception as e:
            return f"Live API error: {e}"

        out: List[_Match] = []
        for m in matches:
            if player_id and player_id not in (m["home"]["teamId"], m["away"]["teamId"]):
                continue
            if tournament_id and tournament_id != m["tournamentId"]:
                continue
            if surface and surface.lower() not in m.get("surface", "").lower():
                continue

            scoreline = " / ".join(map(str, m["home"]["sets"][:m["currentSet"]]))
            out.append(
                {
                    "eventId": m["eventId"],
                    "tournamentName": m["tournamentName"],
                    "roundName": m["roundName"],
                    "status": m["status"]["description"],
                    "home": m["home"]["name"],
                    "away": m["away"]["name"],
                    "score": scoreline,
                }
            )
        return out or "No live matches match the filters."

    async def _arun(self, *_, **__):
        raise NotImplementedError
