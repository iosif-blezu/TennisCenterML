from functools import lru_cache
from typing import Dict

from pydantic_settings import BaseSettings
from pydantic import Field, validator
from pydantic.networks import AnyHttpUrl
from pathlib import Path

ENV_PATH = Path(__file__).parent / ".env"

# Settings class
class Settings(BaseSettings):
    # Core service URLs
    TENNIS_API_BASE: AnyHttpUrl = Field(
        "http://localhost:5000/api/tennis", description="Root of tennis REST API"
    )
    PLAYERS_SEARCH_BASE: AnyHttpUrl = Field(
        "http://localhost:5000/api/players", description="Root for fuzzy player search"
    )

    # LLM & embedding back-ends
    OPENAI_MODEL_CHAT: str = Field("gpt-4o-mini", description="Main chat model")
    OPENAI_MODEL_EMBED: str = Field("text-embedding-3-small", description="Embedding model")
    OPENAI_BASE_URL: AnyHttpUrl | None = Field(
        default=None,
        description="Override for self-hosted OpenAI-compatible gateway (LM Studio, Groq, etc.)",
    )
    OPENAI_API_KEY: str | None = None
    HF_LOCAL_MODEL: str = Field("phi-4@q8_0", description="Local chat model (LM Studio)")

    TAVILY_API_KEY: str | None = None

    LLM_TEMPERATURE_CHAT: float = 0.2
    LLM_TEMPERATURE_TOOLS: float = 0.0

    # Vector DB
    CHROMA_PERSIST_DIR: Path = Path("chroma_store")
    CHROMA_COLLECTION_NEWS: str = "news_articles"

    DATA_PATH: Path = Field(
        default_factory=lambda:
        Path(__file__).resolve().parent.parent / "data" / "cleanedDataset.csv",
        description="Path to the cleaned matches CSV"
    )

    # Caching TTLs (seconds)
    CACHE_TTL: Dict[str, int] = {
        "players_search": 24 * 3600,   # 24 h
        "player_info":    3600,        # 1 h
        "rankings":       24 * 3600,   # 24 h
        "calendar":       12 * 3600,   # 12 h
        "tournament":     12 * 3600,
        "live":           60,          # 1 minute
    }


    # Misc
    LANGSMITH_PROJECT: str | None = None
    LOG_LEVEL: str = Field("INFO", pattern="DEBUG|INFO|WARNING|ERROR|CRITICAL")

    # Build full endpoint URLs
    @property
    def endpoint_player(self) -> str:
        return f"{self.TENNIS_API_BASE}/player"

    @property
    def endpoint_player_search(self) -> str:
        return f"{self.PLAYERS_SEARCH_BASE}/search"

    @property
    def endpoint_rankings(self) -> str:
        return f"{self.TENNIS_API_BASE}/rankings/atp/db"

    @property
    def endpoint_live(self) -> str:
        return f"{self.TENNIS_API_BASE}/live"

    @property
    def endpoint_calendar(self) -> str:
        return f"{self.TENNIS_API_BASE}/calendar"

    @property
    def endpoint_tournament(self) -> str:
        return f"{self.TENNIS_API_BASE}/tournament"

    # Validators
    @validator("OPENAI_API_KEY", always=True)
    def _warn_if_missing_key(cls, v):
        if v in (None, "", "YOUR_API_KEY"):
            print("[tennisbot.config] OPENAI_API_KEY not set – "
                  "only local models will work.")
        return v

    class Config:
        env_file = str(ENV_PATH)
        env_file_encoding = "utf-8"
        case_sensitive = False


# Public helper (import-once singleton)
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
