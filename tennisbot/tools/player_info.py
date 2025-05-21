"""
Return a unified bio/ranking object for a given player_id.

Depends on PlayerSearchTool to resolve names before calling if needed.
"""
from typing import Optional, TypedDict

import requests
import requests_cache
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun

from tennisbot.config import get_settings

cfg = get_settings()
requests_cache.install_cache(
    "player_info_cache",
    expire_after=cfg.CACHE_TTL["player_info"]
)


class _InputSchema(TypedDict):
    player_id: int


class PlayerInfoTool(BaseTool):
    """Combine /player and /player/rapid endpoints into one JSON."""

    name: str = "player_info"
    description: str = (
        "Use to fetch detailed information (bio, ranking, handedness, height, "
        "prize money, residence…) about a player when you already know the player_id. "
        "Input: { player_id: integer }. Returns a JSON dictionary."
    )


    def _run(  # <-- note the parameter name
            self,
            player_id: int,  # <-- this must match the dict key you pass
            run_manager: Optional[CallbackManagerForToolRun] = None,
    ):
        pid = player_id  # keep the variable name simple

        # ---------- First call: core bio ----------------------------------- #
        core_url = f"{cfg.endpoint_player}/{pid}"
        try:
            core = requests.get(core_url, timeout=8).json()
        except Exception as e:
            return f"Bio API error: {e}"

        api_pid = core.get("player_id_api")
        if not api_pid or str(api_pid).lower() == "unknown":
            return core

        # ---------- Second call: rapid profile ----------------------------- #
        rapid_url = f"{cfg.endpoint_player}/rapid/{api_pid}"
        try:
            rapid = requests.get(rapid_url, timeout=8).json()
        except Exception as e:
            return f"Rapid API error: {e}"

        merged = {**core, **rapid}
        return merged

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError
