# tennisbot/eval/evaluate_responses.py
"""
Evaluation harness for TennisBot outputs.
Supports classical metrics (ROUGE, BLEU) and LLM-based evaluations via DeepEval.
"""
import os
import json
from pathlib import Path
from typing import List, Dict, Any

from rouge_score import rouge_scorer
import evaluate
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval import assert_test

# Load test cases from a JSON file (to be created) under eval/ directory
# Ensure OPENAI_API_KEY is available for DeepEval
import os
from tennisbot.config import get_settings
cfg = get_settings()
if cfg.OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = cfg.OPENAI_API_KEY

# Optionally configure DeepEval local model if not using OpenAI
# from deepeval import set_local_model
# set_local_model(
#     model_name=cfg.DEEPEVAL_LOCAL_MODEL,
#     base_url=cfg.DEEPEVAL_BASE_URL,
#     api_key=cfg.OPENAI_API_KEY or "",
# )
TEST_CASES_PATH = Path(__file__).parent / "test_cases.json"

# Classical metrics
rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
bleu = evaluate.load("bleu")

# LLM-based correctness metric
correctness_metric = GEval(
    name="Correctness",
    criteria="Determine if actual output matches expected output.",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    threshold=0.7,
)

def evaluate_classical(predicted: str, expected: str) -> Dict[str, float]:
    """
    Compute ROUGE-1 F1, ROUGE-L F1, and BLEU score for a single output pair.
    """
    r = rouge.score(expected, predicted)
    b = bleu.compute(predictions=[predicted], references=[[expected]])
    return {
        "rouge1_f1": r["rouge1"].fmeasure,
        "rougeL_f1": r["rougeL"].fmeasure,
        "bleu": b["bleu"],
    }


def evaluate_llm(actual: str, expected: str, context: List[str] = None) -> None:
    """
    Use DeepEval to assert if actual matches expected under LLM-based correctness.
    Raises AssertionError on failure.
    """
    tc = LLMTestCase(
        input="",
        actual_output=actual,
        expected_output=expected,
        retrieval_context=context or [],
    )
    assert_test(tc, [correctness_metric])


def load_test_cases() -> List[Dict[str, Any]]:
    """
    Expects test_cases.json with entries:
    [
      {
        "chain": "h2h",
        "input": { ... },
        "expected": "..."
      },
      ...
    ]
    """
    with open(TEST_CASES_PATH, encoding="utf-8") as f:
        return json.load(f)


def run_evaluation():
    cases = load_test_cases()
    results = []
    for case in cases:
        chain_name = case["chain"]
        inp = case.get("input")
        expected = case.get("expected")

        # Dynamically import and run the chain/tool
        from tennisbot.agent.router import get_router_agent
        agent = get_router_agent()
        # invoke with function call syntax
        res = agent.invoke({"input": None, **inp})
        output = res.get("output") if isinstance(res, dict) else str(res)

        classical = evaluate_classical(output, expected)
        llm_pass = None
        try:
            evaluate_llm(output, expected)
            llm_pass = True
        except AssertionError:
            llm_pass = False

        results.append({
            "chain": chain_name,
            "classical": classical,
            "llm_pass": llm_pass,
        })
        print(f"Case {chain_name}: ROUGE1={classical['rouge1_f1']:.2f}, BLEU={classical['bleu']:.2f}, LLM pass={llm_pass}")

    # Optionally, write results to a file
    out_path = Path(__file__).parent / "results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Evaluation complete. Results saved to {out_path}")

if __name__ == "__main__":
    run_evaluation()
