from __future__ import annotations

import logging

from typing import Optional, TypedDict, List, Union

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
# removed deprecated import:
# from langchain.chains import LLMChain
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun

from tennisbot.config import get_settings
from tennisbot.tools.player_search import PlayerSearchTool

# Initialize logging
tt_logger = logging.getLogger(__name__)
tt_logger.debug("Loaded PromptJudgeTool module")

load_dotenv()
cfg = get_settings()

# Prompt Variant Generation Chain
generate_template = PromptTemplate.from_template(
    """
You are a prompt engineer. Given a task description, generate {n} alternative
prompts that are clear, concise, and likely to produce accurate and useful answers.

Task: {task}

Provide each prompt on its own line, numbered 1 to {n}.
"""
)
llm_gen = ChatOpenAI(
    model_name=cfg.OPENAI_MODEL_CHAT,
    temperature=0.7,
    api_key=cfg.OPENAI_API_KEY,
    base_url=cfg.OPENAI_BASE_URL,
)

generate_chain = generate_template | llm_gen
tt_logger.info("Initialized generate_chain with model %s", cfg.OPENAI_MODEL_CHAT)


judge_template = PromptTemplate.from_template(
    """
You are a prompt optimization judge. Given a task and a list of candidate prompts,
evaluate each on the criteria of clarity, specificity, and likelihood of eliciting
correct and detailed responses from an LLM. Rank them from best to worst.

Task: {task}

Candidates:
{candidates}

Output only the number of the best prompt followed by the prompt text.
"""
)
llm_judge = ChatOpenAI(
    model_name=cfg.OPENAI_MODEL_CHAT,
    temperature=0,
    api_key=cfg.OPENAI_API_KEY,
    base_url=cfg.OPENAI_BASE_URL,
)
judge_chain = judge_template | llm_judge
tt_logger.info("Initialized judge_chain with model %s", cfg.OPENAI_MODEL_CHAT)

class _Input(TypedDict, total=False):
    task: str
    n: int  # number of variants to generate

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
        tt_logger.info("PromptJudgeTool invoked for task='%s' with n=%d", task, n)
        # 1. Generate variants
        gen_result: Union[str, dict] = generate_chain.invoke({"task": task, "n": n})
        gen_text = gen_result.get("text") if isinstance(gen_result, dict) else str(gen_result)
        tt_logger.debug("Raw generate_chain output: %s", gen_text.replace('\n', ' | '))

        raw_lines = gen_text.replace("\r", "").splitlines()
        # extract numbered prompts
        candidates: List[str] = []
        for line in raw_lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if parts[0].rstrip('.)').isdigit() and len(parts) > 1:
                prompt_text = parts[1].strip()
                candidates.append(prompt_text)
                tt_logger.debug("Extracted candidate prompt: %s", prompt_text)
        if not candidates:
            candidates = [gen_text.strip()]
            tt_logger.warning("No numbered candidates found, using full text as single candidate")

        # 2. Judge the best prompt
        joined = "\n".join(f"{i+1}. {p}" for i, p in enumerate(candidates))
        tt_logger.debug("Joining candidates for judge_chain: %s", joined)
        judge_input = {"task": task, "candidates": joined}
        judge_result: Union[str, dict] = judge_chain.invoke(judge_input)
        judge_text = judge_result.get("text") if isinstance(judge_result, dict) else str(judge_result)
        tt_logger.debug("Raw judge_chain output: %s", judge_text)

        # parse judge output: expecting "X. prompt text"
        first_line = judge_text.strip().splitlines()[0] if judge_text else ""
        tt_logger.info("Judge selected line: %s", first_line)
        if first_line and ('. ' in first_line or ') ' in first_line):
            selected = first_line.split('. ', 1)[1] if '. ' in first_line else first_line.split(') ', 1)[1]
            tt_logger.info("Selected best prompt: %s", selected)
            return selected
        tt_logger.warning("Judge output not in expected format, returning raw first line")
        return first_line

    async def _arun(self, *args, **kwargs) -> str:
        return self._run(*args, **kwargs)

# CLI demo
if __name__ == "__main__":
    import sys
    task_desc = sys.argv[1] if len(sys.argv) > 1 else "Translate English to French: How are you?"
    print(PromptJudgeTool().invoke({"task": task_desc, "n": 5}))
