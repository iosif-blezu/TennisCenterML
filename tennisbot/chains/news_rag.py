# tennisbot/chains/news_rag.py
from __future__ import annotations
import logging
from hashlib import sha256
from typing import Sequence, TypedDict, Optional, List

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.tools import BaseTool
from langchain.schema import Document
from langchain.callbacks.manager import CallbackManagerForToolRun

from tennisbot.config import get_settings
from tennisbot.utils.tavily_news import get_tavily_results

tt_logger = logging.getLogger(__name__)
cfg = get_settings()

# ────────────────────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────────────────────
def _canonical_url(url: str) -> str:
    return url.split("?", 1)[0].split("#", 1)[0]

def _make_id(url: str) -> str:
    return sha256(_canonical_url(url).encode()).hexdigest()

def _safe_meta(meta: dict) -> dict:
    return {k: v for k, v in meta.items() if v is not None}

# ────────────────────────────────────────────────────────────────────────────
# ingest into Chroma
# ────────────────────────────────────────────────────────────────────────────
def ingest_news(
    raw_articles: Sequence[dict],
    clean_texts: Sequence[str],
    *,
    player_name: str,
) -> None:
    if not raw_articles or not clean_texts:
        tt_logger.info("[NewsRAG] Nothing to ingest for %s", player_name)
        return
    if len(raw_articles) != len(clean_texts):
        tt_logger.error(
            "[NewsRAG] raw vs clean length mismatch: %d vs %d",
            len(raw_articles), len(clean_texts),
        )
        raise ValueError("raw_articles and clean_texts length mismatch")

    docs: List[Document] = []
    for art, body in zip(raw_articles, clean_texts):
        url = art.get("url", "").strip()
        # coerce whatever came back into a plain string
        if isinstance(body, str):
            body_text = body
        elif hasattr(body, "text"):  # e.g. dict-like {"text": "..."}
            body_text = str(body.text)
        elif hasattr(body, "content"):  # guard for AIMessage/content attr
            body_text = str(body.content)
        else:
            body_text = str(body)  # last-chance fallback

        if not url or not body_text.strip():
            tt_logger.debug("Skipping empty URL/body for %s", player_name)
            continue

        doc_id = _make_id(url)
        docs.append(
            Document(
                id=doc_id,
                page_content=body_text,
                metadata=_safe_meta({
                    "id": doc_id,
                    "url": url,
                    "player_name": player_name,
                    "published": art.get("published"),
                }),
            )
        )

    tt_logger.info(
        "[NewsRAG] Prepared %d docs for ingestion for %s",
        len(docs), player_name,
    )

    vectordb = Chroma(
        collection_name=cfg.CHROMA_COLLECTION_NEWS,
        embedding_function=OpenAIEmbeddings(model=cfg.OPENAI_MODEL_EMBED),
        persist_directory=str(cfg.CHROMA_PERSIST_DIR),
    )
    before = vectordb._collection.count()
    vectordb.add_documents(docs)  # upsert by ID
    after = vectordb._collection.count()
    tt_logger.info(
        "[NewsRAG] Upserted %d docs for %s (total %d→%d)",
        len(docs), player_name, before, after,
    )

# ────────────────────────────────────────────────────────────────────────────
# NewsRAG retrieval‐QA tool
# ────────────────────────────────────────────────────────────────────────────
class _Input(TypedDict, total=False):
    player_name: str
    k: int

class NewsRAGTool(BaseTool):
    name: str = "news_rag"
    description: str = (
        "Fetch & answer recent news about a tennis player. "
        "Auto‐scrapes & ingests if none found."
    )
    _chain: RetrievalQA | None = None

    @property
    def chain(self) -> RetrievalQA:
        tt_logger.debug("Building NewsRAG RetrievalQA chain")
        vectordb = Chroma(
            collection_name=cfg.CHROMA_COLLECTION_NEWS,
            embedding_function=OpenAIEmbeddings(model=cfg.OPENAI_MODEL_EMBED),
            persist_directory=str(cfg.CHROMA_PERSIST_DIR),
        )
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are an expert tennis news assistant.\n"
                "Use ONLY the context to answer in ≤3 sentences. "
                "If you don't know, say so. Always cite URLs.\n\n"
                "{context}\n\nQuestion: {question}\nAnswer:"
            ),
        )
        llm = ChatOpenAI(
            model_name=cfg.OPENAI_MODEL_CHAT,
            temperature=cfg.LLM_TEMPERATURE_CHAT,
            base_url=cfg.OPENAI_BASE_URL,
            api_key=cfg.OPENAI_API_KEY,
        )
        self._chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectordb.as_retriever(),
            return_source_documents=False,
            chain_type_kwargs={"prompt": prompt},
        )
        return self._chain

    def _run(
        self,
        player_name: str,
        k: int = 3,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        tt_logger.info("NewsRAGTool invoked for %s (k=%d)", player_name, k)

        # check existing docs
        vectordb = Chroma(
            collection_name=cfg.CHROMA_COLLECTION_NEWS,
            embedding_function=OpenAIEmbeddings(model=cfg.OPENAI_MODEL_EMBED),
            persist_directory=str(cfg.CHROMA_PERSIST_DIR),
        )
        docs = vectordb._collection.get( where={"player_name": player_name},
                                         include=["metadatas"], )
        existing = docs.get("metadatas", [])
        tt_logger.debug("Found %d existing docs for %s", len(existing), player_name)

        # if none, scrape+clean+ingest
        if not existing:
            tt_logger.info("No docs; fetching from Tavily for %s", player_name)
            raw, clean = get_tavily_results(player_name)
            ingest_news(raw, clean, player_name=player_name)
            self._chain = None  # rebuild chain with new docs

        # finally run the QA
        chain = self.chain
        chain.retriever.search_kwargs = {
            "k": k,
            "filter": {"player_name": player_name},
        }
        question = f"Latest news about {player_name}"
        tt_logger.info("Running QA chain for question: %s", question)
        answer = chain.invoke({"query": question})
        tt_logger.info("NewsRAG answer for %s: %s", player_name, answer)
        return answer

    async def _arun(self, *_, **__):
        raise NotImplementedError("NewsRAGTool is synchronous.")

# ────────────────────────────────────────────────────────────────────────────
# quick CLI test
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, json
    player = sys.argv[1] if len(sys.argv)>1 else "Alexander Zverev"
    out = NewsRAGTool().invoke({"player_name": player, "k": 4})
    print(json.dumps({"answer": out}, indent=2, ensure_ascii=False))
