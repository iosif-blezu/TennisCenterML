from __future__ import annotations

import logging

import json
import re
from typing import Optional, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import Runnable
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun

from tennisbot.config import get_settings
from tennisbot.tools.player_search import PlayerSearchTool
from utils.predict_between_two_players import predict_match  # your ML model

tt_logger = logging.getLogger(__name__)

tt_logger.debug("Loaded PredictionTool module")

load_dotenv()
cfg = get_settings()

_polish_prompt = PromptTemplate.from_template(
    "You are a seasoned tennis analyst. The model prediction is:\n\n"
    "{model_pred}\n\n"
    "Write a 2–3 sentence rationale (max 70 words) that explains the key "
    "tactical factors behind the expected outcome. Do **not** restate the "
    "probabilities. Respond with only the paragraph."
)

def _build_llm() -> ChatOpenAI:
    tt_logger.debug("Building LLM for PredictionTool: model=%s", cfg.OPENAI_MODEL_CHAT or cfg.HF_LOCAL_MODEL)
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

_PROB_RE = re.compile(r"([A-Za-zÀ-ÿ'. -]+?)\s+(\d+(?:\.\d+)?)%")

def _parse_raw_pred(raw: str) -> tuple[str, int]:
    tt_logger.debug("Parsing raw prediction: %s", raw)
    pairs = _PROB_RE.findall(raw)
    if len(pairs) >= 2:
        parsed = [(n.strip(), float(p)) for n, p in pairs]
        winner, conf = max(parsed, key=lambda x: x[1])
        tt_logger.info("Parsed prediction winner=%s confidence=%s", winner, conf)
        return winner, int(round(conf))
    tt_logger.warning("Failed to parse prediction, raw output: %s", raw)
    return "Unknown", 0

class _Input(TypedDict, total=False):
    player1_id: int
    player2_id: int
    player1_name: str
    player2_name: str
    surface: str
    best_of: int
    draw_size: int

class PredictionTool(BaseTool):
    name: str = "prediction_chain"
    description: str = (
        "Predict the winner between two players on a given surface. "
        "Input either player1_id & player2_id or player1_name & player2_name, "
        "plus optional surface:str, best_of:int, draw_size:int. "
        "Returns JSON {winner, confidence, narrative}."
    )

    def _run(
        self,
        player1_id: Optional[int] = None,
        player2_id: Optional[int] = None,
        player1_name: Optional[str] = None,
        player2_name: Optional[str] = None,
        surface: str = "Hard",
        best_of: int = 3,
        draw_size: int = 128,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        tt_logger.info(
            "PredictionTool invoked with player1_id=%s, player2_id=%s, player1_name=%s, player2_name=%s, surface=%s, best_of=%d, draw_size=%d",
            player1_id, player2_id, player1_name, player2_name, surface, best_of, draw_size
        )
        # Resolve names to IDs if needed
        if (player1_id is None or player2_id is None) and (player1_name and player2_name):
            tt_logger.debug("Resolving player names to IDs: %s, %s", player1_name, player2_name)
            search = PlayerSearchTool()
            if player1_id is None:
                res1 = search.invoke({"query": player1_name})
                player1_id = res1.get("player_id")
                tt_logger.info("Resolved %s to ID %s", player1_name, player1_id)
            if player2_id is None:
                res2 = search.invoke({"query": player2_name})
                player2_id = res2.get("player_id")
                tt_logger.info("Resolved %s to ID %s", player2_name, player2_id)

        if not player1_id or not player2_id:
            tt_logger.error(
                "Could not resolve both player IDs (player1_id=%s, player2_id=%s)",
                player1_id, player2_id
            )
            return json.dumps(
                {"winner": "Unknown", "confidence": 0, "narrative": "Could not resolve player IDs."},
                ensure_ascii=False,
            )

        # Call the ML model
        tt_logger.info("Calling ML model predict_match for IDs %s vs %s", player1_id, player2_id)
        raw_pred: str = predict_match(
            player1_id=player1_id,
            player2_id=player2_id,
            surface=surface,
            best_of=best_of,
            draw_size=draw_size,
        )
        tt_logger.debug("Received raw prediction: %s", raw_pred)

        # Extract numeric winner & confidence
        winner, conf = _parse_raw_pred(raw_pred)

        # Ask LLM for a short rationale
        tt_logger.info("Requesting narrative rationale from LLM")
        rationale_msg = _polish_chain.invoke({"model_pred": raw_pred})
        rationale = (
            rationale_msg.content.strip()
            if hasattr(rationale_msg, "content")
            else str(rationale_msg).strip()
        )
        tt_logger.debug("Generated rationale: %s", rationale)

        # Return robust JSON
        result = json.dumps(
            {"winner": winner, "confidence": conf, "narrative": rationale},
            ensure_ascii=False,
        )
        tt_logger.info("PredictionTool result: %s", result)
        return result

    async def _arun(self, *_, **__):
        raise NotImplementedError("PredictionTool is synchronous.")
