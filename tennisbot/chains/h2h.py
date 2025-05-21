from __future__ import annotations
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional, TypedDict

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import Runnable
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun

from tennisbot.config import get_settings
from tennisbot.tools.player_search import PlayerSearchTool

# Initialize logging
tt_logger = logging.getLogger(__name__)

load_dotenv()
cfg = get_settings()
DATA_PATH: Path = cfg.DATA_PATH


# dataset utils
@lru_cache(maxsize=1)
def _load_matches() -> pd.DataFrame:
    tt_logger.info("Loading matches dataset from %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH, parse_dates=["tourney_date"])
    df["surface"] = df["surface"].str.title()
    tt_logger.info("Loaded %d matches", len(df))
    return df


def _get_h2h_df(p1_id: int, p2_id: int) -> pd.DataFrame:
    all_df = _load_matches()
    h2h_df = all_df[
        ((all_df["p1_id"] == p1_id) & (all_df["p2_id"] == p2_id)) |
        ((all_df["p1_id"] == p2_id) & (all_df["p2_id"] == p1_id))
    ]
    tt_logger.debug(
        "Found %d H2H matches for p1_id=%s, p2_id=%s",
        len(h2h_df), p1_id, p2_id
    )
    return h2h_df


def _summary_all(df: pd.DataFrame, p1: str, p2: str) -> str:
    if df.empty:
        tt_logger.debug("No meetings found between %s and %s", p1, p2)
        return f"{p1} and {p2} have never met on tour."
    wins = {p1: 0, p2: 0}
    for _, r in df.iterrows():
        winner = r["p1_name"] if r["RESULT"] == 1 else r["p2_name"]
        wins[winner] += 1
    last3 = df.sort_values("tourney_date", ascending=False).head(3)
    lines = []
    for _, r in last3.iterrows():
        w, l = (r["p1_name"], r["p2_name"]) if r["RESULT"] == 1 else (r["p2_name"], r["p1_name"])
        lines.append(
            f"{r['tourney_date'].date()} | {r['surface']} | {r['tourney_name']} | "
            f"{r['round']} | {w} d. {l} {r['score']}"
        )
    tt_logger.debug(
        "Summary all: %d meetings, wins: %s vs %s", len(df), wins[p1], wins[p2]
    )
    return (
        f"Total meetings: {len(df)} — {p1} {wins[p1]} W vs {p2} {wins[p2]} W\n"
        "Last 3 meetings:\n" + "\n".join(lines)
    )


def _summary_surface(df: pd.DataFrame, p1: str, p2: str, surface: str) -> str:
    sf = df[df["surface"].str.lower() == surface.lower()]
    if sf.empty:
        tt_logger.debug("No meetings on surface %s between %s and %s", surface, p1, p2)
        return f"No prior meetings on {surface}."
    wins = {p1: 0, p2: 0}
    for _, r in sf.iterrows():
        winner = r["p1_name"] if r["RESULT"] == 1 else r["p2_name"]
        wins[winner] += 1
    last = sf.sort_values("tourney_date", ascending=False).iloc[0]
    w, l = (last["p1_name"], last["p2_name"]) if last["RESULT"] == 1 else (last["p2_name"], last["p1_name"])
    last_line = (
        f"{last['tourney_date'].date()} | {last['tourney_name']} {last['round']} | "
        f"{w} d. {l} {last['score']}"
    )
    tt_logger.debug(
        "Summary surface %s: wins %s-%s, last meeting %s", surface, wins[p1], wins[p2], last_line
    )
    return f"{surface} H2H: {p1} {wins[p1]} W – {p2} {wins[p2]} W\nLast meeting: {last_line}"


# prompt template
_polish_prompt = PromptTemplate.from_template(
    "Turn the raw head-to-head stats below into a concise, punchy paragraph that "
    "a TV commentator might say on air (max 90 words).\n\nRAW:\n{stats}\n\nCOMMENTATOR:"
)

def _build_llm() -> ChatOpenAI:
    tt_logger.debug("Building LLM chain with model %s", cfg.OPENAI_MODEL_CHAT or cfg.HF_LOCAL_MODEL)
    if cfg.OPENAI_API_KEY or cfg.OPENAI_BASE_URL:
        return ChatOpenAI(
            model_name=cfg.OPENAI_MODEL_CHAT,
            temperature=cfg.LLM_TEMPERATURE_CHAT,
            base_url=cfg.OPENAI_BASE_URL,
            api_key=cfg.OPENAI_API_KEY or "EMPTY",
        )
    return ChatOpenAI(
        model_name=cfg.HF_LOCAL_MODEL,
        base_url="http://localhost:1234/v1",
        api_key="local",
        temperature=cfg.LLM_TEMPERATURE_CHAT,
    )

_polish_chain: Runnable = _polish_prompt | _build_llm()


# tool wrapper
class _Input(TypedDict, total=False):
    p1_id: int
    p2_id: int
    player1_name: str
    player2_name: str
    surface: str

class H2HTool(BaseTool):
    name: str = "h2h_chain"
    description: str = (
        "Summarise head-to-head history between two players. "
        "Input either p1_id & p2_id, or player1_name & player2_name, "
        "and optional surface:str. Returns a short narrative paragraph."
    )

    _polish_chain: Runnable | None = None

    def _get_chain(self) -> Runnable:
        if self._polish_chain is None:
            self._polish_chain = _polish_chain
        return self._polish_chain

    def _run(
        self,
        p1_id: Optional[int] = None,
        p2_id: Optional[int] = None,
        player1_name: Optional[str] = None,
        player2_name: Optional[str] = None,
        surface: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        tt_logger.info(
            "H2HTool invoked with p1_id=%s, p2_id=%s, player1_name=%s, player2_name=%s, surface=%s",
            p1_id, p2_id, player1_name, player2_name, surface
        )
        # Resolve names to IDs if needed
        if (p1_id is None or p2_id is None) and player1_name and player2_name:
            try:
                search = PlayerSearchTool()
                p1_id = p1_id or search.invoke({"query": player1_name})["player_id"]
                p2_id = p2_id or search.invoke({"query": player2_name})["player_id"]
                tt_logger.info("Resolved names to IDs: %s -> %s, %s -> %s", player1_name, p1_id, player2_name, p2_id)
            except Exception as e:
                tt_logger.error("Error resolving player IDs: %s", e)
        if p1_id is None or p2_id is None:
            tt_logger.error("Could not resolve both player IDs (p1_id=%s, p2_id=%s)", p1_id, p2_id)
            return "Could not resolve both player IDs."

        df = _get_h2h_df(p1_id, p2_id)
        if df.empty:
            tt_logger.info("No H2H history found for IDs %s and %s", p1_id, p2_id)
            return "These players have never met on tour."

        # Determine player display names
        first_row = df.iloc[0]
        p1_name = first_row["p1_name"] if first_row["p1_id"] == p1_id else first_row["p2_name"]
        p2_name = first_row["p2_name"] if first_row["p2_id"] == p2_id else first_row["p1_name"]
        tt_logger.debug("Determined player names: p1_name=%s, p2_name=%s", p1_name, p2_name)

        raw_stats = (
            _summary_surface(df, p1_name, p2_name, surface)
            if surface
            else _summary_all(df, p1_name, p2_name)
        )
        tt_logger.debug("Generated raw stats: %s", raw_stats.replace('\n', ' | '))
        result = self._get_chain().invoke({"stats": raw_stats})
        tt_logger.info("Generated polished commentator paragraph")
        return result

    async def _arun(self, *_, **__):
        raise NotImplementedError("H2HTool is synchronous.")