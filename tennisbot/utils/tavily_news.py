from __future__ import annotations
import os, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

from tavily import TavilyClient
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from tennisbot.config import get_settings

# prep
cfg = get_settings()
tavily = TavilyClient(api_key=cfg.TAVILY_API_KEY)

# strip junk from raw text
_NOISE_RE = re.compile(
    r"(Advertising|Subscribe|Trending|Newsletter|Free Newsletters"
    r"|Follow (Us|us)|Latest News|Popular Topics).*",
    re.I,
)

def _pre_clean(text: str) -> str:
    return "\n".join(
        ln for ln in text.splitlines() if ln.strip() and not _NOISE_RE.match(ln)
    )

# LLM cleaner
_llm = ChatOpenAI(
    model_name="gpt-3.5-turbo-0125",
    temperature=0,
    api_key=cfg.OPENAI_API_KEY,
)

_clean_prompt = PromptTemplate(
    input_variables=["article"],
    template=(
        "You are a meticulous copy-editor. Given a raw scrape of a news article, "
        "remove everything that is not the core story text (ads, navigation links, "
        "repeated lists, social-media prompts, quizzes, etc.). "
        "Return *only* well-formatted paragraphs of the article in its original "
        "language—no commentary, no summary.\n\n"
        "RAW ARTICLE:\n----------------\n{article}\n----------------\nCLEAN ARTICLE:"
    ),
)

_clean_chain = _clean_prompt | _llm

def _clean_single(raw_text: str) -> str:
    result = _clean_chain.invoke({"article": _pre_clean(raw_text)})
    return result["text"] if isinstance(result, dict) else result

def get_tavily_results(
    player_name: str,
    *,
    time_range: str = "week",    # day | week | month | year
    max_results: int = 10,
    score_cut: float = 0.4,
    max_workers: int = 6,        # parallel LLM clean jobs
    rpm_cap: int = 80,           # OpenAI requests/min cap
) -> Tuple[List[dict], List[str]]:
    # search
    query = f"latest news about tennis player {player_name}"
    search = tavily.search(
        query=query,
        search_depth="advanced",
        time_range=time_range,
        max_results=max_results,
        include_answer="basic",
    )

    urls = [r["url"] for r in search["results"] if r.get("score", 0) >= score_cut]
    if not urls:
        return [], []

    # extract raw content
    extract_res = tavily.extract(urls=urls)
    raw_articles = extract_res["results"]  # list of dicts (url, raw_content…)

    # clean in parallel
    delay = 60 / rpm_cap if rpm_cap else 0
    cleaned: List[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_clean_single, art.get("raw_content", "")): idx
            for idx, art in enumerate(raw_articles)
            if art.get("raw_content")
        }
        for fut in as_completed(futures):
            try:
                cleaned.append(fut.result())
            except Exception as exc:
                print("LLM clean error:", exc)
            if delay:
                time.sleep(delay)

    # Preserve original order
    cleaned_sorted = [cleaned[futures[fut]] for fut in futures]
    return raw_articles, cleaned_sorted
