from typing import Optional, TypedDict, List
import datetime as _dt

import requests, requests_cache
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun

from tennisbot.config import get_settings

cfg = get_settings()
requests_cache.install_cache(
    "calendar_cache",
    expire_after=cfg.CACHE_TTL["calendar"]
)


class _Input(TypedDict, total=False):
    month: int     # default current month
    year: int      # default current year
    day: int       # optional, returns only that day


class _Day(TypedDict):
    date: str
    uniqueTournamentIds: List[int]


class CalendarTool(BaseTool):
    name: str = "tournament_calendar"
    description: str = (
        "Get tournament IDs scheduled for a given month (and optionally day). "
        "Input {month:int, year:int, day?:int}. Returns list of {date, ids}."
    )

    def _run(
        self,
        month: Optional[int] = None,
        year: Optional[int] = None,
        day: Optional[int] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> List[_Day] | str:
        today = _dt.date.today()
        month = month or today.month
        year = year or today.year

        url = f"{cfg.endpoint_calendar}"
        params = {"month": month, "year": year}
        try:
            data = requests.get(url, params=params, timeout=10).json()
        except Exception as e:
            return f"Calendar API error: {e}"

        days = data.get("dailyUniqueTournaments", [])
        if day:
            wanted = f"{year:04d}-{month:02d}-{day:02d}"
            days = [d for d in days if d["date"] == wanted]
            if not days:
                return f"No tournaments found on {wanted}."

        return days

    async def _arun(self, *_, **__):
        raise NotImplementedError
