from typing import Optional, TypedDict
import requests, requests_cache
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun
from tennisbot.config import get_settings

cfg = get_settings()
# Install cache and retrieve controller
tmp_session = requests_cache.install_cache(
    "player_info_cache",
    expire_after=cfg.CACHE_TTL["player_info"]
)
cache = requests_cache.get_cache()
# Print cache configuration
print(f"[PlayerInfoTool] Cache backend: {cache.__class__.__name__}")
print(f"[PlayerInfoTool] Cache name: {cache.cache_name}.sqlite")
print(f"[PlayerInfoTool] TTL (seconds): {cfg.CACHE_TTL['player_info']}")

class _InputSchema(TypedDict):
    player_id: int

class PlayerInfoTool(BaseTool):
    # Combine /player and /player/rapid endpoints into one JSON.

    name: str = "player_info"
    description: str = (
        "Use to fetch detailed information (bio, ranking, handedness, height, "
        "prize money, residence…) about a player when you already know the player_id. "
        "Input: { player_id: integer }. Returns a JSON dictionary."
    )

    def _run(
        self,
        player_id: int,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ):
        pid = player_id

        # First API call
        core_url = f"{cfg.endpoint_player}/{pid}"
        try:
            resp_core = requests.get(core_url, timeout=8)
            core = resp_core.json()
            from_cache = getattr(resp_core, 'from_cache', False)
            print(f"[PlayerInfoTool] Core request URL: {resp_core.url}")
            print(f"[PlayerInfoTool] Core response from cache: {from_cache}")
        except Exception as e:
            return f"Bio API error: {e}"

        api_pid = core.get("player_id_api")
        if not api_pid or str(api_pid).lower() == "unknown":
            return core

        # Second API call
        rapid_url = f"{cfg.endpoint_player}/rapid/{api_pid}"
        try:
            resp_rapid = requests.get(rapid_url, timeout=8)
            rapid = resp_rapid.json()
            from_cache = getattr(resp_rapid, 'from_cache', False)
            print(f"[PlayerInfoTool] Rapid request URL: {resp_rapid.url}")
            print(f"[PlayerInfoTool] Rapid response from cache: {from_cache}")
        except Exception as e:
            return f"Rapid API error: {e}"

        merged = {**core, **rapid}
        return merged

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError("Use sync _run for now.")
