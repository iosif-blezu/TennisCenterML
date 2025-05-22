from __future__ import annotations

from functools import lru_cache

from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

from tennisbot.tools.player_search import PlayerSearchTool
from tennisbot.tools.player_info import PlayerInfoTool
from tennisbot.tools.rankings import RankingsTool
from tennisbot.tools.live_matches import LiveMatchesTool
from tennisbot.tools.calendar import CalendarTool
from tennisbot.tools.tournament_info import TournamentInfoTool
from tennisbot.chains.news_rag import NewsRAGTool
from tennisbot.chains.h2h import H2HTool
from tennisbot.chains.prediction import PredictionTool
from tennisbot.chains.prompt_judge import PromptJudgeTool


from tennisbot.config import get_settings

cfg = get_settings()


def _build_router_llm() -> ChatOpenAI:
    """Return the chat model used for tool-selection & final answer."""
    if cfg.OPENAI_API_KEY or cfg.OPENAI_BASE_URL:
        return ChatOpenAI(
            model_name=cfg.OPENAI_MODEL_CHAT,
            temperature=0.2,
            base_url=cfg.OPENAI_BASE_URL,
            api_key=cfg.OPENAI_API_KEY or "EMPTY",
        )
    return ChatOpenAI(
        model_name=cfg.HF_LOCAL_MODEL,
        base_url="http://localhost:1234/v1",
        api_key="local",
        temperature=0.2,
    )

SYSTEM_PROMPT = """You are TennisBot, a professional tennis assistant.

Use a function whenever it will help answer accurately.
When the user asks:
  – “Who will win ... ?” then call prediction_chain …
  – “Latest news ...”    then news_rag
  – “Head-to-head ...”   then h2h_chain
  – “Is anyone playing live?” then live_matches
  – “Where is X ranked?” then rankings …
  – “Tell me about tournament Y” then tournament_info or calendar
If no function adds value, answer directly.
If you don’t know, say so.  If the topic is not tennis,
explain that you’re a tennis-focused assistant."""


TOOLS = [
    PlayerSearchTool(),
    PlayerInfoTool(),
    RankingsTool(),
    LiveMatchesTool(),
    CalendarTool(),
    TournamentInfoTool(),
    NewsRAGTool(),
    H2HTool(),
    PredictionTool(),
    PromptJudgeTool(),
]


@lru_cache(maxsize=1)
def get_router_agent():
    """Return a singleton LangChain agent with memory."""
    # 1) create a single ConversationBufferMemory
    memory = ConversationBufferMemory(
        memory_key="chat_history",     # this key is what your agent code will see
        return_messages=True           # so we get full message objects
    )

    # 2) pass it into initialize_agent
    agent = initialize_agent(
        tools=TOOLS,
        llm=_build_router_llm(),
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        memory=memory,                 # ← enable memory here
        system_message=SYSTEM_PROMPT,
    )
    return agent
