from __future__ import annotations
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional, TypedDict

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun

from tennisbot.config import get_settings
from tennisbot.tools.player_search import PlayerSearchTool
from tennisbot.chains.prompt_judge import PromptJudgeTool

# Configure logging
logger = logging.getLogger(__name__)

load_dotenv()
cfg = get_settings()
DATA_PATH: Path = cfg.DATA_PATH

# helpers
@lru_cache(maxsize=1)
def _load_matches() -> pd.DataFrame:
    logger.info("Loading matches dataset from %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH, parse_dates=["tourney_date"])
    df["surface"] = df["surface"].str.title()
    logger.info("Loaded %d matches", len(df))
    return df


def _summary_all(df: pd.DataFrame, p1: str, p2: str) -> str:
    """Overall H2H summary regardless of surface."""
    if df.empty:
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

    return (
        f"Total meetings: {len(df)} — {p1} {wins[p1]} W vs {p2} {wins[p2]} W\n"
        "Last 3 meetings:\n" + "\n".join(lines)
    )


def _summary_surface(df: pd.DataFrame, p1: str, p2: str, surface: str) -> str:
    """Surface‑specific H2H summary."""
    sf = df[df["surface"].str.lower() == surface.lower()]
    if sf.empty:
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

    return (
        f"{surface.title()} H2H: {p1} {wins[p1]} W – {p2} {wins[p2]} W\n"
        f"Last meeting: {last_line}"
    )

# tool definition
class _Input(TypedDict, total=False):
    p1_id: int
    p2_id: int
    player1_name: str
    player2_name: str
    surface: str

class H2HTool(BaseTool):
    name: str = "h2h_chain"
    description: str = (
        "Return a short commentator‑style paragraph summarising the head‑to‑head "
        "between two players. Accepts p1_id & p2_id **or** player names, plus an "
        "optional `surface` filter."
    )

    def _run(
        self,
        p1_id: Optional[int] = None,
        p2_id: Optional[int] = None,
        player1_name: Optional[str] = None,
        player2_name: Optional[str] = None,
        surface: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        logger.info("H2HTool called: p1_id=%s p2_id=%s p1=%s p2=%s surface=%s", p1_id, p2_id, player1_name, player2_name, surface)

        # Resolve fuzzy names → IDs
        if (p1_id is None or p2_id is None) and player1_name and player2_name:
            search = PlayerSearchTool()
            p1_id = p1_id or search.invoke({"query": player1_name})["player_id"]
            p2_id = p2_id or search.invoke({"query": player2_name})["player_id"]
        if p1_id is None or p2_id is None:
            return "Could not resolve both player IDs."

        # Fetch H2H rows
        df = _load_matches()
        mask = ((df["p1_id"] == p1_id) & (df["p2_id"] == p2_id)) | ((df["p1_id"] == p2_id) & (df["p2_id"] == p1_id))
        h2h = df[mask]
        if h2h.empty:
            return "These players have never met on tour."

        first_row = h2h.iloc[0]
        p1_name = first_row["p1_name"] if first_row["p1_id"] == p1_id else first_row["p2_name"]
        p2_name = first_row["p2_name"] if first_row["p2_id"] == p2_id else first_row["p1_name"]

        raw_stats = _summary_surface(h2h, p1_name, p2_name, surface) if surface else _summary_all(h2h, p1_name, p2_name)
        print(f"[H2HTool] RAW STATS:\n{raw_stats}\n")

        # Generate prompt variants & choose best
        best_prompt_raw = PromptJudgeTool().invoke({"task": raw_stats, "n": 5})
        best_prompt = best_prompt_raw.content if hasattr(best_prompt_raw, "content") else str(best_prompt_raw)
        best_prompt = best_prompt.strip()
        print(f"[H2HTool] CHOSEN PROMPT:\n{best_prompt}\n")

        # Escape { } so PromptTemplate doesn't treat them as variables
        safe_prompt = best_prompt.replace("{", "{{").replace("}", "}}")
        template_text = f"{safe_prompt}\n\nRAW:\n{{stats}}\n\nCOMMENTATOR:"
        custom_template = PromptTemplate.from_template(template_text)

        llm = ChatOpenAI(
            model_name=cfg.OPENAI_MODEL_CHAT,
            temperature=cfg.LLM_TEMPERATURE_CHAT,
            api_key=cfg.OPENAI_API_KEY or "EMPTY",
            base_url=cfg.OPENAI_BASE_URL,
        )
        polished_msg = (custom_template | llm).invoke({"stats": raw_stats})
        polished_text = polished_msg.content if hasattr(polished_msg, "content") else str(polished_msg)
        print(f"[H2HTool] POLISHED OUTPUT:\n{polished_text}\n")
        return polished_text

    async def _arun(self, *_, **__):
        raise NotImplementedError
