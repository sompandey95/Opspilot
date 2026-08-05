"""Confidence scorer tests — pure heuristic, fully deterministic."""
from app.agent.confidence import ConfidenceScorer
from app.config import Settings
from app.observability.trace import Trace

ANSWER = "Your order ORD-2024-55001 is delayed; the new ETA is 5 Aug 2026."


def scorer(**overrides) -> ConfidenceScorer:
    return ConfidenceScorer(Settings(_env_file=None, **overrides))


def trace_with(steps: list[dict], intent: str = "action_simple") -> Trace:
    trace = Trace(query="q", intent=intent)
    trace.steps = steps
    return trace


def llm_step():
    return {"type": "llm", "step": 0}


def tool_ok(scores=None):
    step = {"type": "tool_result", "tool": "t", "success": True}
    if scores is not None:
        step["retrieval_scores"] = scores
    return step


def tool_fail():
    return {"type": "tool_result", "tool": "t", "success": False}


def test_all_successful_action_flow_clears_threshold():
    trace = trace_with([llm_step(), tool_ok(), llm_step()])
    score = scorer().score("q", ANSWER, trace)
    assert score >= 0.7  # 0.5 + 0.3 tool bonus


def test_faq_with_good_retrieval_scores_high():
    trace = trace_with(
        [llm_step(), tool_ok(scores=[2.4, 1.3]), llm_step()], intent="faq"
    )
    score = scorer().score("q", ANSWER, trace)
    assert score > 0.9


def test_faq_without_retrieval_is_penalised():
    trace = trace_with([llm_step()], intent="faq")
    assert scorer().score("q", ANSWER, trace) < 0.5


def test_tool_failures_pull_below_threshold():
    trace = trace_with([llm_step(), tool_fail(), llm_step()])
    assert scorer().score("q", ANSWER, trace) < 0.7


def test_partial_failure_stays_conservative():
    trace = trace_with([llm_step(), tool_fail(), llm_step(), tool_ok(), llm_step()])
    assert scorer().score("q", ANSWER, trace) < 0.7


def test_validation_failures_penalised_and_capped():
    base = scorer().score("q", ANSWER, trace_with([llm_step(), tool_ok()]))
    one = scorer().score(
        "q", ANSWER, trace_with([llm_step(), {"type": "validation_error"}, tool_ok()])
    )
    many_steps = [llm_step()] + [{"type": "validation_error"}] * 10 + [tool_ok()]
    many = scorer().score("q", ANSWER, trace_with(many_steps))
    assert one == base - 0.05
    assert many == base - 0.15  # capped


def test_near_max_steps_penalised():
    steps = [llm_step() for _ in range(10)]
    trace = trace_with(steps + [tool_ok()])
    capped = scorer().score("q", ANSWER, trace)
    short = scorer().score("q", ANSWER, trace_with([llm_step(), tool_ok()]))
    assert capped == short - 0.10


def test_short_answer_penalised():
    trace = trace_with([llm_step(), tool_ok()])
    full = scorer().score("q", ANSWER, trace)
    terse = scorer().score("q", "ok", trace)
    assert terse == full - 0.10


def test_clarifying_question_on_action_query_clears_threshold():
    trace = trace_with([llm_step()], intent="action_simple")
    question = "Could you share your order ID (format ORD-YYYY-NNNNN)?"
    assert scorer().score("where is my order?", question, trace) >= 0.7


def test_imperative_info_request_also_counts_as_clarifying():
    trace = trace_with([llm_step()], intent="action_simple")
    imperative = "Please share your ShopEasy order ID in the format ORD-YYYY-NNNNN."
    assert scorer().score("where is my order?", imperative, trace) >= 0.7


def test_clarifying_bonus_not_applied_after_tool_use():
    trace = trace_with([llm_step(), tool_ok(), llm_step()], intent="action_simple")
    with_q = scorer().score("q", ANSWER + " Anything else?", trace)
    without_q = scorer().score("q", ANSWER, trace)
    assert with_q == without_q


def test_score_clamped_to_unit_interval():
    bad_steps = (
        [llm_step() for _ in range(10)]
        + [{"type": "validation_error"}] * 5
        + [tool_fail(), tool_fail()]
    )
    assert scorer().score("q", "x", trace_with(bad_steps, intent="faq")) >= 0.0
    good = [llm_step(), tool_ok(scores=[10.0])]
    assert scorer().score("q", ANSWER, trace_with(good)) <= 1.0
