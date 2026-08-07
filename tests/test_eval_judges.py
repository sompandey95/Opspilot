"""Judge tests: tool-accuracy matching rules, deterministic hallucination
grounding, and the LLM judges' parsing/failure behaviour (fake LLM)."""
import pytest

from app.llm.client import LLMResponse, Usage

from evals.judges.faithfulness import FaithfulnessJudge
from evals.judges.hallucination import check_hallucination
from evals.judges.relevance import RelevanceJudge
from evals.judges.tool_accuracy import score_tool_calls


# --------------------------------------------------------------------- #
# Tool accuracy                                                           #
# --------------------------------------------------------------------- #

def call(tool, **args):
    return {"tool": tool, "args": args}


def test_exact_match_scores_one():
    result = score_tool_calls(
        [call("check_order_status", order_id="ORD-2024-55001")],
        [call("check_order_status", order_id="ORD-2024-55001")],
    )
    assert result.exact == 1 and result.score == 1.0


def test_extra_actual_args_still_exact():
    """Ground truth lists only the args that matter."""
    result = score_tool_calls(
        [call("process_refund", order_id="ORD-2024-78432")],
        [call("process_refund", order_id="ORD-2024-78432", reason="delivery_delayed")],
    )
    assert result.exact == 1 and result.score == 1.0


def test_wrong_arg_is_partial():
    result = score_tool_calls(
        [call("check_order_status", order_id="ORD-2024-55001")],
        [call("check_order_status", order_id="ORD-2024-54000")],
    )
    assert result.partial == 1
    assert result.score == 0.5


def test_missing_and_extra():
    result = score_tool_calls(
        [call("check_refund_eligibility", order_id="ORD-2024-55001")],
        [call("cancel_order", order_id="ORD-2024-55001")],
    )
    assert result.missing == 1 and result.extra == 1
    assert result.score == 0.0


def test_both_empty_is_perfect():
    assert score_tool_calls([], []).score == 1.0


def test_unexpected_business_call_penalised():
    result = score_tool_calls([], [call("process_refund", order_id="ORD-2024-55001")])
    assert result.extra == 1 and result.score == 0.0


def test_search_knowledge_ignored_on_both_sides():
    result = score_tool_calls(
        [call("search_knowledge", query="return window")],
        [call("search_knowledge", query="different words")],
    )
    assert result.score == 1.0
    assert result.exact == result.partial == result.missing == result.extra == 0


def test_duplicate_expected_calls_matched_individually():
    expected = [call("get_delivery_eta", order_id="ORD-2024-55001"),
                call("get_delivery_eta", order_id="ORD-2024-54000")]
    actual = [call("get_delivery_eta", order_id="ORD-2024-54000"),
              call("get_delivery_eta", order_id="ORD-2024-55001")]
    result = score_tool_calls(expected, actual)
    assert result.exact == 2 and result.score == 1.0


# --------------------------------------------------------------------- #
# Hallucination                                                           #
# --------------------------------------------------------------------- #

CONTEXT = (
    'Where is my order ORD-2024-55001?\n'
    '{"order_id": "ORD-2024-55001", "amount_inr": 1499.0, "status": "delayed", '
    '"delivery_eta": "2026-08-05"}'
)


def test_grounded_answer_passes():
    result = check_hallucination(
        "Your order ORD-2024-55001 (₹1,499) is delayed; new ETA 2026-08-05.", CONTEXT
    )
    assert not result.hallucinated
    assert result.total_claims >= 3


def test_fabricated_order_id_is_hard_fail():
    result = check_hallucination("Your other order ORD-2024-11111 is also delayed.", CONTEXT)
    assert result.hallucinated
    assert result.fabricated_ids == ["ORD-2024-11111"]


def test_invented_amount_flagged():
    result = check_hallucination("You'll receive ₹9,999 as compensation.", CONTEXT)
    assert result.hallucinated
    assert "₹9,999" in result.ungrounded_claims


def test_number_grounded_via_digits_in_json():
    result = check_hallucination("The amount was ₹1,499.", CONTEXT)
    assert not result.hallucinated


def test_no_claims_is_clean():
    result = check_hallucination("Happy to help with anything else!", CONTEXT)
    assert not result.hallucinated
    assert result.total_claims == 0


# --------------------------------------------------------------------- #
# LLM judges (fake LLM)                                                   #
# --------------------------------------------------------------------- #

class FakeLLM:
    def __init__(self, content=None, error=None):
        self.content = content
        self.error = error
        self.calls = []

    async def complete(self, role, messages, **kwargs):
        self.calls.append({"role": role, "messages": messages})
        if self.error:
            raise self.error
        return LLMResponse(content=self.content, usage=Usage(200, 60), model="gpt-4o")


async def test_faithfulness_parses_and_clamps():
    llm = FakeLLM(content='{"score": 1.7, "unsupported_statements": ["x"], "reasoning": "r"}')
    judge = FaithfulnessJudge(llm)
    result = await judge.score("q", "a", "ctx")
    assert result == {"score": 1.0, "unsupported_statements": ["x"]}
    sent = llm.calls[0]["messages"][1]["content"]
    assert "CONTEXT" in sent and "ANSWER" in sent


async def test_faithfulness_returns_none_on_judge_failure():
    judge = FaithfulnessJudge(FakeLLM(error=TimeoutError("azure down")))
    assert await judge.score("q", "a", "ctx") is None


async def test_faithfulness_returns_none_on_garbage_output():
    judge = FaithfulnessJudge(FakeLLM(content="not json at all"))
    assert await judge.score("q", "a", "ctx") is None


async def test_relevance_score_parsed():
    judge = RelevanceJudge(FakeLLM(content='{"score": 0.75, "reasoning": "ok"}'))
    assert await judge.score("q", "a", reference="ref") == 0.75


async def test_relevance_none_on_failure():
    judge = RelevanceJudge(FakeLLM(error=ConnectionError("down")))
    assert await judge.score("q", "a") is None
