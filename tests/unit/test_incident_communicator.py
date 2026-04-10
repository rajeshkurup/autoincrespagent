"""Unit tests for the incident_communicator agent node."""

import json
import pytest
from langchain_core.messages import AIMessage
from unittest.mock import AsyncMock, MagicMock

from autoincrespagent.agents.incident_communicator import make_incident_communicator


# ── Helpers ───────────────────────────────────────────────────────────

def _state(**overrides) -> dict:
    base = {
        "phase": "communicate",
        "session_id": "test-session",
        "incident_id": "INC-TESTCOMM",
        "severity": "SEV2",
        "anomaly_nodes": [
            {"id": "ANO-001", "type": "latency_spike", "severity": "high", "status": "active"}
        ],
        "root_causes": [
            {"node_id": "db-001", "node_type": "Storage",
             "hypothesis": "Connection pool exhausted", "evidence": ["ANO-001"], "confidence": 0.9}
        ],
        "mitigation_workflows": [
            {"score": 0.85, "payload": {
                "workflow_id": "WF-001", "title": "Restart connection pool",
                "target_node_type": "Storage",
                "steps": ["Check pool metrics", "Restart pooler"],
            }}
        ],
        "mitigation_confidence": 0.85,
        "communication_event": "incident_detected",
        "next_phase": "root_cause",
        "feedback_iteration": 0,
        "communications_sent": [],
        "messages": [],
        "feedback_request": None,
        "incident_summary": None,
    }
    base.update(overrides)
    return base


def _mock_comms_tool(name: str, response: dict | None = None) -> MagicMock:
    """Build a mock LangChain comms tool returning a JSON-encoded response."""
    tool = MagicMock()
    tool.name = name
    default = {
        "channel": name.replace("send_", "").replace("page_", "pagerduty"),
        "message_id": f"{name.upper()[:5]}-ABC123",
        "sent_at": "2026-04-10T00:00:00Z",
        "stub": True,
    }
    tool.ainvoke = AsyncMock(return_value=json.dumps(response or default))
    return tool


def _all_comms_tools() -> list:
    return [
        _mock_comms_tool("send_email"),
        _mock_comms_tool("send_slack"),
        _mock_comms_tool("send_teams"),
        _mock_comms_tool("send_sms"),
        _mock_comms_tool("page_oncall", {"page_id": "PD-ABC123", "channel": "pagerduty", "stub": True}),
        _mock_comms_tool("update_ticket_comms", {"incident_id": "INC-TESTCOMM", "event": "incident_detected", "stub": True}),
    ]


# ── Phase routing ──────────────────────────────────────────────────────

class TestPhaseRouting:
    async def test_advances_to_next_phase(self):
        agent = make_incident_communicator()
        result = await agent(_state(communication_event="incident_detected", next_phase="root_cause"))
        assert result["phase"] == "root_cause"

    async def test_advances_to_mitigate_after_rca(self):
        agent = make_incident_communicator()
        result = await agent(_state(communication_event="root_cause_found", next_phase="mitigate"))
        assert result["phase"] == "mitigate"

    async def test_advances_to_summarize_after_mitigation(self):
        agent = make_incident_communicator()
        result = await agent(_state(communication_event="mitigation_complete", next_phase="summarize"))
        assert result["phase"] == "summarize"

    async def test_clears_communication_fields(self):
        agent = make_incident_communicator()
        result = await agent(_state())
        assert result["communication_event"] is None
        assert result["next_phase"] is None

    async def test_message_added(self):
        agent = make_incident_communicator()
        result = await agent(_state())
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)


# ── Channel selection by severity ──────────────────────────────────────

class TestChannelSelection:
    async def test_sev1_pages_oncall(self):
        tools = _all_comms_tools()
        page_tool = next(t for t in tools if t.name == "page_oncall")
        agent = make_incident_communicator(comms_tools=tools)
        await agent(_state(severity="SEV1"))
        page_tool.ainvoke.assert_called_once()

    async def test_sev1_sends_slack(self):
        tools = _all_comms_tools()
        slack_tool = next(t for t in tools if t.name == "send_slack")
        agent = make_incident_communicator(comms_tools=tools)
        await agent(_state(severity="SEV1"))
        slack_tool.ainvoke.assert_called_once()

    async def test_sev1_sends_email(self):
        tools = _all_comms_tools()
        email_tool = next(t for t in tools if t.name == "send_email")
        agent = make_incident_communicator(comms_tools=tools)
        await agent(_state(severity="SEV1"))
        email_tool.ainvoke.assert_called_once()

    async def test_sev1_does_not_send_teams(self):
        tools = _all_comms_tools()
        teams_tool = next(t for t in tools if t.name == "send_teams")
        agent = make_incident_communicator(comms_tools=tools)
        await agent(_state(severity="SEV1"))
        teams_tool.ainvoke.assert_not_called()

    async def test_sev2_sends_slack_teams_email(self):
        tools = _all_comms_tools()
        slack_tool = next(t for t in tools if t.name == "send_slack")
        teams_tool = next(t for t in tools if t.name == "send_teams")
        email_tool = next(t for t in tools if t.name == "send_email")
        agent = make_incident_communicator(comms_tools=tools)
        await agent(_state(severity="SEV2"))
        slack_tool.ainvoke.assert_called_once()
        teams_tool.ainvoke.assert_called_once()
        email_tool.ainvoke.assert_called_once()

    async def test_sev2_does_not_page_oncall(self):
        tools = _all_comms_tools()
        page_tool = next(t for t in tools if t.name == "page_oncall")
        agent = make_incident_communicator(comms_tools=tools)
        await agent(_state(severity="SEV2"))
        page_tool.ainvoke.assert_not_called()

    async def test_sev3_sends_email_and_teams_only(self):
        tools = _all_comms_tools()
        email_tool = next(t for t in tools if t.name == "send_email")
        teams_tool = next(t for t in tools if t.name == "send_teams")
        slack_tool = next(t for t in tools if t.name == "send_slack")
        page_tool  = next(t for t in tools if t.name == "page_oncall")
        agent = make_incident_communicator(comms_tools=tools)
        await agent(_state(severity="SEV3"))
        email_tool.ainvoke.assert_called_once()
        teams_tool.ainvoke.assert_called_once()
        slack_tool.ainvoke.assert_not_called()
        page_tool.ainvoke.assert_not_called()

    async def test_sev4_sends_email_only(self):
        tools = _all_comms_tools()
        email_tool = next(t for t in tools if t.name == "send_email")
        slack_tool = next(t for t in tools if t.name == "send_slack")
        page_tool  = next(t for t in tools if t.name == "page_oncall")
        agent = make_incident_communicator(comms_tools=tools)
        await agent(_state(severity="SEV4"))
        email_tool.ainvoke.assert_called_once()
        slack_tool.ainvoke.assert_not_called()
        page_tool.ainvoke.assert_not_called()


# ── Tool argument correctness ──────────────────────────────────────────

class TestToolArguments:
    async def test_email_receives_incident_id(self):
        tools = _all_comms_tools()
        email_tool = next(t for t in tools if t.name == "send_email")
        agent = make_incident_communicator(comms_tools=tools)
        await agent(_state(incident_id="INC-XYZ", severity="SEV2"))
        call_args = email_tool.ainvoke.call_args[0][0]
        assert call_args["incident_id"] == "INC-XYZ"

    async def test_email_receives_severity(self):
        tools = _all_comms_tools()
        email_tool = next(t for t in tools if t.name == "send_email")
        agent = make_incident_communicator(comms_tools=tools)
        await agent(_state(severity="SEV2"))
        call_args = email_tool.ainvoke.call_args[0][0]
        assert call_args["severity"] == "SEV2"

    async def test_slack_receives_title_and_body(self):
        tools = _all_comms_tools()
        slack_tool = next(t for t in tools if t.name == "send_slack")
        agent = make_incident_communicator(comms_tools=tools)
        await agent(_state(severity="SEV2", communication_event="incident_detected"))
        call_args = slack_tool.ainvoke.call_args[0][0]
        assert "title" in call_args
        assert "body" in call_args
        assert len(call_args["title"]) > 0

    async def test_page_oncall_receives_summary_and_severity(self):
        tools = _all_comms_tools()
        page_tool = next(t for t in tools if t.name == "page_oncall")
        agent = make_incident_communicator(comms_tools=tools)
        await agent(_state(severity="SEV1"))
        call_args = page_tool.ainvoke.call_args[0][0]
        assert call_args["severity"] == "SEV1"
        assert "summary" in call_args

    async def test_update_ticket_comms_called_with_event(self):
        tools = _all_comms_tools()
        ticket_tool = next(t for t in tools if t.name == "update_ticket_comms")
        agent = make_incident_communicator(comms_tools=tools)
        await agent(_state(communication_event="root_cause_found", severity="SEV2"))
        call_args = ticket_tool.ainvoke.call_args[0][0]
        assert call_args["event"] == "root_cause_found"
        assert call_args["incident_id"] == "INC-TESTCOMM"

    async def test_update_ticket_includes_channels_notified(self):
        tools = _all_comms_tools()
        ticket_tool = next(t for t in tools if t.name == "update_ticket_comms")
        agent = make_incident_communicator(comms_tools=tools)
        await agent(_state(severity="SEV2"))
        call_args = ticket_tool.ainvoke.call_args[0][0]
        # SEV2 channels: slack, teams, email
        assert "send_slack" in call_args["channels_notified"]
        assert "send_email" in call_args["channels_notified"]


# ── communications_sent state ──────────────────────────────────────────

class TestCommunicationsSent:
    async def test_appends_record_to_communications_sent(self):
        tools = _all_comms_tools()
        agent = make_incident_communicator(comms_tools=tools)
        result = await agent(_state(severity="SEV2"))
        assert len(result["communications_sent"]) == 1
        record = result["communications_sent"][0]
        assert record["incident_id"] == "INC-TESTCOMM"
        assert record["event"] == "incident_detected"

    async def test_accumulates_across_calls(self):
        tools = _all_comms_tools()
        agent = make_incident_communicator(comms_tools=tools)
        state1 = _state(communications_sent=[{"event": "prior", "incident_id": "INC-TESTCOMM", "channels": [], "sent_at": "x"}])
        result = await agent(state1)
        assert len(result["communications_sent"]) == 2

    async def test_dispatched_channels_in_record(self):
        tools = _all_comms_tools()
        agent = make_incident_communicator(comms_tools=tools)
        result = await agent(_state(severity="SEV1"))
        record = result["communications_sent"][0]
        assert "page_oncall" in record["channels"]
        assert "send_slack" in record["channels"]
        assert "send_email" in record["channels"]


# ── Graceful degradation ───────────────────────────────────────────────

class TestGracefulDegradation:
    async def test_no_comms_tools_does_not_raise(self):
        agent = make_incident_communicator()
        result = await agent(_state())
        assert result["phase"] == "root_cause"

    async def test_tool_failure_does_not_raise(self):
        tools = _all_comms_tools()
        for t in tools:
            t.ainvoke = AsyncMock(side_effect=Exception("network error"))
        agent = make_incident_communicator(comms_tools=tools)
        result = await agent(_state(severity="SEV2"))
        # Phase still advances even if all tools fail
        assert result["phase"] == "root_cause"

    async def test_partial_tool_failure_dispatches_remaining(self):
        tools = _all_comms_tools()
        # Make slack fail, others succeed
        slack_tool = next(t for t in tools if t.name == "send_slack")
        slack_tool.ainvoke = AsyncMock(side_effect=Exception("slack down"))
        agent = make_incident_communicator(comms_tools=tools)
        result = await agent(_state(severity="SEV2"))
        record = result["communications_sent"][0]
        # slack failed, but email and teams should still be in dispatched
        assert "send_email" in record["channels"]
        assert "send_teams" in record["channels"]
        assert "send_slack" not in record["channels"]

    async def test_prints_mcp_unavailable_when_no_tools(self, capsys):
        agent = make_incident_communicator()
        await agent(_state())
        out = capsys.readouterr().out
        assert "mcp unavailable" in out.lower() or "print only" in out.lower()


# ── Stdout output ──────────────────────────────────────────────────────

class TestOutput:
    async def test_prints_incident_detected(self, capsys):
        agent = make_incident_communicator()
        await agent(_state(communication_event="incident_detected"))
        out = capsys.readouterr().out
        assert "INCIDENT DECLARED" in out
        assert "INC-TESTCOMM" in out
        assert "SEV2" in out

    async def test_prints_root_cause_found(self, capsys):
        agent = make_incident_communicator()
        await agent(_state(communication_event="root_cause_found"))
        out = capsys.readouterr().out
        assert "ROOT CAUSE" in out
        assert "Connection pool exhausted" in out
        assert "90%" in out

    async def test_prints_mitigation_complete(self, capsys):
        agent = make_incident_communicator()
        await agent(_state(communication_event="mitigation_complete"))
        out = capsys.readouterr().out
        assert "MITIGATION EXECUTED" in out
        assert "WF-001" in out
        assert "0.85" in out

    async def test_prints_dispatched_channels(self, capsys):
        tools = _all_comms_tools()
        agent = make_incident_communicator(comms_tools=tools)
        await agent(_state(severity="SEV2"))
        out = capsys.readouterr().out
        assert "Channels" in out

    async def test_unknown_event_prints_generic(self, capsys):
        agent = make_incident_communicator()
        await agent(_state(communication_event="unknown_event"))
        out = capsys.readouterr().out
        assert "INC-TESTCOMM" in out


# ── Qdrant rolling summary ─────────────────────────────────────────────

class TestQdrantUpsert:
    async def test_upsert_called_when_client_provided(self):
        mock_client = MagicMock()
        mock_client.upsert = AsyncMock(return_value=None)
        mock_embeddings = MagicMock()
        mock_embeddings.aembed_query = AsyncMock(return_value=[0.1] * 768)

        agent = make_incident_communicator(qdrant_client=mock_client, embeddings=mock_embeddings)
        await agent(_state(communication_event="incident_detected"))

        mock_client.upsert.assert_called_once()
        assert mock_client.upsert.call_args[1]["collection_name"] == "incident_summaries"

    async def test_no_upsert_without_client(self):
        agent = make_incident_communicator()
        result = await agent(_state())
        assert result["phase"] == "root_cause"

    async def test_upsert_failure_does_not_raise(self):
        mock_client = MagicMock()
        mock_client.upsert = AsyncMock(side_effect=Exception("qdrant down"))
        mock_embeddings = MagicMock()
        mock_embeddings.aembed_query = AsyncMock(return_value=[0.1] * 768)

        agent = make_incident_communicator(qdrant_client=mock_client, embeddings=mock_embeddings)
        result = await agent(_state())
        assert result["phase"] == "root_cause"
