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


def tool_not_found():
    return {"type": "tool_result", "tool": "t", "success": False, "not_found": True}


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


def test_faq_tagged_order_lookup_grounded_by_tool_not_penalised():
    # Regression: the intent classifier sometimes tags an order-status
    # question "faq" even though check_order_status (not search_knowledge) is
    # the right tool. That answer is grounded via the tool result, not
    # retrieval — it must not eat the FAQ-no-retrieval penalty on top of the
    # tool-success bonus, or a fully correct, fully grounded answer scores
    # 0.65 and gets discarded for a canned escalation message.
    trace = trace_with([llm_step(), tool_ok(), llm_step()], intent="faq")
    assert scorer().score("q", ANSWER, trace) >= 0.7


def test_tool_failures_pull_below_threshold():
    trace = trace_with([llm_step(), tool_fail(), llm_step()])
    assert scorer().score("q", ANSWER, trace) < 0.7


def test_not_found_lookup_is_not_treated_as_a_tool_failure():
    # A valid-format order ID that doesn't exist: the tool worked and returned a
    # definitive answer, so telling the customer must not be swapped for a
    # canned escalation the way a genuine tool malfunction is.
    trace = trace_with([llm_step(), tool_not_found(), llm_step()])
    honest = "I couldn't find an order with that ID in our system."
    assert scorer().score("where is ORD-2024-99999?", honest, trace) >= 0.7


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


def test_clarifying_question_on_bare_faq_query_clears_threshold():
    # A one-word query like "refund" classifies as faq. Asking what the customer
    # actually needs is the right move, not grounds for paging a human.
    trace = trace_with([llm_step()], intent="faq")
    question = "Happy to help with a refund — could you tell me which order it's for?"
    assert scorer().score("refund", question, trace) >= 0.7


def test_faq_answer_asserting_a_fact_is_still_penalised_despite_a_question_mark():
    trace = trace_with([llm_step()], intent="faq")
    ungrounded = "Electronics can be returned within 30 days. Anything else?"
    assert scorer().score("return window?", ungrounded, trace) < 0.7


def test_imperative_id_request_without_a_question_mark_counts_as_clarifying():
    # Observed phrasings vary ("please paste it", "please send your full order
    # ID") and often carry no question mark, so the exemption must not hinge on
    # one fixed wording.
    trace = trace_with([llm_step()], intent="action_simple")
    for phrasing in (
        "I can help with that, but I need the full order ID first. "
        "Please paste it in the format ORD-YYYY-NNNNN exactly as shown.",
        "Sure — I can help. Please send your full order ID in this format: ORD-YYYY-NNNNN.",
    ):
        assert scorer().score("my order is #78432, where is it?", phrasing, trace) >= 0.7


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
