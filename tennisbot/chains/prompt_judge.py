"""PromptJudgeTool — generate N prompt variants and pick the best one via an LLM judge.
Fixes:
• Safely read `.content` from ChatMessage objects instead of `str()` so we only keep raw text.
• Robust candidate extraction and judge‑selection parsing.
"""
from __future__ import annotations

import logging
from typing import List, Optional, TypedDict, Union

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain_openai import ChatOpenAI

from tennisbot.config import get_settings

load_dotenv()
logger = logging.getLogger(__name__)
cfg = get_settings()

# prompt templates
_generate_tpl = PromptTemplate.from_template(
    """
You are a prompt engineer. Given a task description, generate {n} alternative
prompts that are clear, concise, and likely to produce accurate and useful answers.

Task: {task}

Provide each prompt on its own line, numbered 1 to {n}.
"""
)

_judge_tpl = PromptTemplate.from_template(
    """
You are a prompt‑optimization judge. Given a task and a list of candidate prompts,
evaluate each on clarity, specificity, and likelihood of eliciting a correct and detailed answer.
Rank them from best to worst.

Task: {task}

Candidates:
{candidates}

Output only the number of the best prompt followed by the prompt text.
"""
)

# LLMs
_llm_gen = ChatOpenAI(
    model_name=cfg.OPENAI_MODEL_CHAT,
    temperature=0.7,
    api_key=cfg.OPENAI_API_KEY,
    base_url=cfg.OPENAI_BASE_URL,
)

_llm_judge = ChatOpenAI(
    model_name=cfg.OPENAI_MODEL_CHAT,
    temperature=0,
    api_key=cfg.OPENAI_API_KEY,
    base_url=cfg.OPENAI_BASE_URL,
)

_generate_chain = _generate_tpl | _llm_gen
_judge_chain = _judge_tpl | _llm_judge

logger.info("PromptJudgeTool – gen & judge chains initialised (model %s)", cfg.OPENAI_MODEL_CHAT)


class _Input(TypedDict, total=False):
    task: str
    n: int  # number of variants

class PromptJudgeTool(BaseTool):
    name: str = "prompt_judge"
    description: str = (
        "Generate {n} prompt variants for a given task, then judge and return the best one. "
        "Input keys: task:str, n:int (default 5)."
    )

    def _run(
        self,
        task: str,
        n: int = 5,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        logger.info("PromptJudgeTool invoked (task=%.40s… n=%d)", task, n)


        gen_out = _generate_chain.invoke({"task": task, "n": n})
        gen_text = gen_out.content if hasattr(gen_out, "content") else str(gen_out)
        logger.debug("Generated variants raw:\n%s", gen_text)

        # Extract numbered prompts "1. text…"
        candidates: List[str] = []
        for line in gen_text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line[0].isdigit():
                # split on first dot/paren then strip
                sep_idx = line.find(".")
                if sep_idx == -1:
                    sep_idx = line.find(")")
                prompt = line[sep_idx + 1 :].strip() if sep_idx != -1 else line
                if prompt:
                    candidates.append(prompt)
        if not candidates:
            candidates = [gen_text.strip()]  # fallback to whole text
        logger.debug("Parsed %d candidates", len(candidates))

        # Join candidates for judge input
        joined = "\n".join(f"{i+1}. {p}" for i, p in enumerate(candidates))
        judge_out = _judge_chain.invoke({"task": task, "candidates": joined})
        judge_text = judge_out.content if hasattr(judge_out, "content") else str(judge_out)
        logger.debug("Judge raw output: %s", judge_text)

        first_line = judge_text.strip().splitlines()[0] if judge_text else ""
        # Expected format: "X. best‑prompt" or "X) best‑prompt"
        if first_line and (". " in first_line or ") " in first_line):
            selected = first_line.split(". ", 1)[1] if ". " in first_line else first_line.split(") ", 1)[1]
            logger.info("Selected prompt: %s", selected)
            return selected.strip()

        logger.warning("Judge response not in expected format; returning entire first line")
        return first_line.strip()

    async def _arun(self, *args, **kwargs):
        return self._run(*args, **kwargs)

if __name__ == "__main__":
    import sys
    example_task = sys.argv[1] if len(sys.argv) > 1 else "Translate English to French: How are you?"
    print(PromptJudgeTool().invoke({"task": example_task, "n": 5}))
