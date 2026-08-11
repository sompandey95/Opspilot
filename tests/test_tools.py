"""Tests for the tool layer: registry, schemas, validation, execution."""
import json

import httpx
import pytest

from app.config import Settings
from app.guardrails.schemas import SchemaValidator
from app.tools.base import RiskLevel, ToolResult
from app.tools.registry import build_default_registry
from mock_services.order_service import main as svc
from mock_services.order_service.seed import build_store

EXPECTED_TOOLS = {
    "search_knowledge": (RiskLevel.NONE, False),
    "check_order_status": (RiskLevel.LOW, False),
    "get_delivery_eta": (RiskLevel.LOW, False),
    "check_refund_eligibility": (RiskLevel.LOW, False),
    "search_customer": (RiskLevel.LOW, False),
    "process_refund": (RiskLevel.HIGH, True),
    "cancel_order": (RiskLevel.HIGH, True),
    "create_jira_ticket": (RiskLevel.MEDIUM, True),
    "update_jira_ticket": (RiskLevel.MEDIUM, True),
    "send_slack_summary": (RiskLevel.LOW, True),
    "escalate_to_manager": (RiskLevel.HIGH, True),
}


@pytest.fixture
def settings():
    return Settings(_env_file=None)


@pytest.fixture(autouse=True)
def fresh_store():
    customers, orders = build_store()
    svc.CUSTOMERS.clear()
    svc.CUSTOMERS.update(customers)
    svc.ORDERS.clear()
    svc.ORDERS.update(orders)
    yield


@pytest.fixture
async def order_client():
    transport = httpx.ASGITransport(app=svc.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
def registry(settings, order_client):
    return build_default_registry(settings, retriever=None, order_client=order_client)


# --------------------------------------------------------------------- #
# Registry + schemas                                                      #
# --------------------------------------------------------------------- #

def test_registry_has_all_eleven_tools(registry):
    assert len(registry) == 11
    for name, (risk, state_changing) in EXPECTED_TOOLS.items():
        tool = registry.get(name)
        assert tool is not None, name
        assert tool.risk_level == risk, name
        assert tool.is_state_changing == state_changing, name


def test_openai_schema_format(registry):
    schemas = registry.get_tool_schemas()
    assert len(schemas) == 11
    for schema in schemas:
        assert schema["type"] == "function"
        fn = schema["function"]
        assert fn["name"] and fn["description"]
        assert fn["parameters"]["type"] == "object"


# --------------------------------------------------------------------- #
# Argument validation                                                     #
# --------------------------------------------------------------------- #

def test_validator_accepts_valid_args(registry):
    validator = SchemaValidator(registry)
    assert validator.validate("check_order_status", {"order_id": "ORD-2024-55001"}).valid
    assert validator.validate(
        "process_refund", {"order_id": "ORD-2024-55001", "reason": "delivery_delayed"}
    ).valid
    assert validator.validate("search_customer", {"email": "a@b.com"}).valid


def test_validator_rejects_bad_args(registry):
    validator = SchemaValidator(registry)

    missing = validator.validate("check_order_status", {})
    assert not missing.valid and "order_id" in missing.error

    bad_pattern = validator.validate("check_order_status", {"order_id": "12345"})
    assert not bad_pattern.valid

    bad_enum = validator.validate(
        "process_refund", {"order_id": "ORD-2024-55001", "reason": "because"}
    )
    assert not bad_enum.valid

    no_identifier = validator.validate("search_customer", {})
    assert not no_identifier.valid


def test_validator_unknown_tool(registry):
    result = SchemaValidator(registry).validate("teleport_order", {"x": 1})
    assert not result.valid
    assert "Unknown tool" in result.error
    assert "check_order_status" in result.error  # lists available tools


# --------------------------------------------------------------------- #
# Execution against the mock order service                                #
# --------------------------------------------------------------------- #

async def test_check_order_status_success_and_error(registry):
    tool = registry.get("check_order_status")

    ok = await tool.execute(order_id="ORD-2024-55001")
    assert ok.success
    assert ok.data["status"] == "delayed"

    missing = await tool.execute(order_id="ORD-9999-00000")
    assert not missing.success
    assert "404" in missing.error


async def test_refund_execution_and_ineligible(registry):
    refund = await registry.get("process_refund").execute(
        order_id="ORD-2024-78432", reason="delivery_delayed"
    )
    assert refund.success
    assert refund.data["amount_inr"] == 1299.00

    ineligible = await registry.get("process_refund").execute(
        order_id="ORD-2024-54000", reason="delivery_delayed"
    )
    assert not ineligible.success


async def test_tool_result_message_roundtrip(registry):
    result = await registry.get("get_delivery_eta").execute(order_id="ORD-2024-55001")
    message = json.loads(result.to_message())
    assert message["success"] is True
    assert message["data"]["delayed"] is True

    restored = ToolResult.from_json(result.to_json())
    assert restored.success and restored.from_cache


# --------------------------------------------------------------------- #
# Knowledge tool                                                          #
# --------------------------------------------------------------------- #

class _FakeRetrievalResult:
    def __init__(self, chunk_id, content, doc_type):
        self.chunk_id = chunk_id
        self.content = content
        self.score = 0.9
        self.metadata = {"doc_type": doc_type, "source_file": "x.md"}


class _FakeRetriever:
    async def retrieve(self, query, top_k=None):
        return [
            _FakeRetrievalResult("changelog_x_001", "window is 10 days", "changelog"),
            _FakeRetrievalResult("faq_x_001", "window is 7 days", "faq"),
        ]


async def test_knowledge_tool_with_doc_type_filter(settings):
    from app.tools.knowledge_tool import SearchKnowledgeTool

    tool = SearchKnowledgeTool(_FakeRetriever())
    result = await tool.execute(query="return window", doc_type="changelog")
    assert result.success
    assert [r["chunk_id"] for r in result.data] == ["changelog_x_001"]


async def test_knowledge_tool_without_retriever():
    from app.tools.knowledge_tool import SearchKnowledgeTool

    result = await SearchKnowledgeTool(None).execute(query="anything")
    assert not result.success


# --------------------------------------------------------------------- #
# Unconfigured integrations fail gracefully                               #
# --------------------------------------------------------------------- #

async def test_jira_and_slack_unconfigured(registry):
    jira = await registry.get("create_jira_ticket").execute(
        summary="Payment stuck", description="UPI payment debited but order not confirmed"
    )
    assert not jira.success and "not configured" in jira.error

    slack = await registry.get("send_slack_summary").execute(text="daily summary here")
    assert not slack.success and "not configured" in slack.error


async def test_slack_tool_sends_when_configured(settings):
    from app.tools.slack_tool import SendSlackSummaryTool

    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return httpx.Response(200, text="ok")

    settings_with_hook = settings.model_copy(
        update={"SLACK_WEBHOOK_URL": "https://hooks.slack.test/services/T/B/X"}
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        tool = SendSlackSummaryTool(settings_with_hook, client=client)
        result = await tool.execute(text="resolved 3 delayed-order refunds today")

    assert result.success
    assert calls == [{"text": "resolved 3 delayed-order refunds today"}]


async def test_escalate_tool_returns_queued(settings):
    from app.tools.escalate_tool import EscalateToManagerTool

    tool = EscalateToManagerTool(settings)
    result = await tool.execute(
        reason="customer_requested_human",
        summary="Customer insists on speaking to a supervisor about a refund.",
    )
    assert result.success
    assert result.data["status"] == "queued"
    assert result.data["escalation_id"].startswith("ESC-")
