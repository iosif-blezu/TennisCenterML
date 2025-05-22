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

# logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
tt_logger = logging.getLogger(__name__)

cfg = get_settings()

# helpers
def _canonical_url(url: str) -> str:
    return url.split("?", 1)[0].split("#", 1)[0]

def _make_id(url: str) -> str:
    return sha256(_canonical_url(url).encode()).hexdigest()

def _safe_meta(meta: dict) -> dict:
    return {k: v for k, v in meta.items() if v is not None}

# news ingestion
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
        msg = (f"[NewsRAG] raw vs clean length mismatch: "
               f"{len(raw_articles)} vs {len(clean_texts)}")
        tt_logger.error(msg); raise ValueError(msg)

    docs: List[Document] = []
    seen: set[str] = set()
    skipped = 0

    for art, body in zip(raw_articles, clean_texts):
        url = art.get("url", "").strip()
        if isinstance(body, str):
            body_text = body
        elif hasattr(body, "text"):
            body_text = str(body.text)
        elif hasattr(body, "content"):
            body_text = str(body.content)
        else:
            body_text = str(body)

        if not url or not body_text.strip():
            continue

        doc_id = _make_id(url)
        if doc_id in seen:
            skipped += 1
            continue
        seen.add(doc_id)

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

    if skipped:
        print(f"[NewsRAG] Skipped {skipped} duplicate URLs for {player_name}")

    tt_logger.info("[NewsRAG] Prepared %d docs for %s", len(docs), player_name)

    vectordb = Chroma(
        collection_name=cfg.CHROMA_COLLECTION_NEWS,
        embedding_function=OpenAIEmbeddings(model=cfg.OPENAI_MODEL_EMBED),
        persist_directory=str(cfg.CHROMA_PERSIST_DIR),
    )
    before = vectordb._collection.count()
    vectordb.add_documents(docs)              # upsert by unique ID
    after = vectordb._collection.count()
    print(f"[NewsRAG] Added {len(docs)} docs — total {before} → {after}")

# tool wrapper
class _Input(TypedDict, total=False):
    player_name: str
    k: int

class NewsRAGTool(BaseTool):
    name: str = "news_rag"
    description: str = (
        "Fetch & answer recent news about a tennis player. "
        "If no articles are cached, the tool scrapes with Tavily and ingests them."
    )
    _chain: RetrievalQA | None = None

    def _get_chain(self) -> RetrievalQA:
        if self._chain is not None:
            return self._chain
        vectordb = Chroma(
            collection_name=cfg.CHROMA_COLLECTION_NEWS,
            embedding_function=OpenAIEmbeddings(model=cfg.OPENAI_MODEL_EMBED),
            persist_directory=str(cfg.CHROMA_PERSIST_DIR),
        )
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are an expert tennis news assistant.\n"
                "Use ONLY the context to answer in no more than 3 sentences. "
                "If the answer is not in the context, say you don't know. "
                "Always cite the source URL in parentheses.\n\n"
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
        tt_logger.info("[NewsRAG] query for %s (k=%d)", player_name, k)

        vectordb = Chroma(
            collection_name=cfg.CHROMA_COLLECTION_NEWS,
            embedding_function=OpenAIEmbeddings(model=cfg.OPENAI_MODEL_EMBED),
            persist_directory=str(cfg.CHROMA_PERSIST_DIR),
        )
        existing = vectordb._collection.get(
            where={"player_name": player_name},
            include=["metadatas"],
        ).get("metadatas", [])

        print(f"[NewsRAG] {len(existing)} cached docs for {player_name}")

        if not existing:
            print(f"[NewsRAG] Scraping Tavily for {player_name} …")
            raw, clean = get_tavily_results(player_name)
            ingest_news(raw, clean, player_name=player_name)
            self._chain = None  # refresh chain

        chain = self._get_chain()
        chain.retriever.search_kwargs = {
            "k": k,
            "filter": {"player_name": player_name},
        }
        question = f"Latest news about {player_name}"
        answer = chain.invoke({"query": question})
        return answer

    async def _arun(self, *_, **__):
        raise NotImplementedError("NewsRAGTool is synchronous.")

if __name__ == "__main__":
    import sys, json
    who = sys.argv[1] if len(sys.argv) > 1 else "Alexander Zverev"
    result = NewsRAGTool().invoke({"player_name": who, "k": 3})
    print(json.dumps({"answer": result}, indent=2, ensure_ascii=False))
