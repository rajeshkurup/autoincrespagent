"""Unit tests for the incident_detector agent node."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage

from autoincrespagent.agents.incident_detector import make_incident_detector


# ── Helpers ───────────────────────────────────────────────────────────

def _make_tool(name: str, return_value: str) -> MagicMock:
    """Create a mock MCP tool that returns a fixed string on ainvoke."""
    tool = MagicMock()
    tool.name = name
    tool.ainvoke = AsyncMock(return_value=return_value)
    return tool


def _make_llm(json_response: dict) -> MagicMock:
    """Create a mock LLM that returns a fixed JSON AIMessage."""
    llm = MagicMock()
    msg = AIMessage(content=json.dumps(json_response))
    llm.ainvoke = AsyncMock(return_value=msg)
    return llm


def _make_tools(anomaly_payload: str = "[]", ticket_response: str = '{"id": "INC-TESTTEST"}'):
    list_tool   = _make_tool("list_anomalies", anomaly_payload)
    create_tool = _make_tool("create_incident_ticket", ticket_response)
    return [list_tool, create_tool]


# ── No anomalies ──────────────────────────────────────────────────────

class TestNoAnomalies:
    async def test_returns_done_phase_when_empty(self, minimal_state):
        tools   = _make_tools(anomaly_payload="[]")
        llm     = _make_llm({"is_incident": False})
        detector = make_incident_detector(tools, llm=llm)

        result = await detector(minimal_state)

        assert result["phase"] == "done"
        assert result["anomaly_nodes"] == []

    async def test_llm_not_called_when_no_anomalies(self, minimal_state):
        llm = _make_llm({"is_incident": False})
        detector = make_incident_detector(_make_tools("[]"), llm=llm)

        await detector(minimal_state)

        llm.ainvoke.assert_not_called()

    async def test_create_ticket_not_called_when_no_anomalies(self, minimal_state):
        tools = _make_tools("[]")
        detector = make_incident_detector(tools, llm=_make_llm({}))

        await detector(minimal_state)

        create_tool = next(t for t in tools if t.name == "create_incident_ticket")
        create_tool.ainvoke.assert_not_called()


# ── Anomalies present but LLM says no incident ────────────────────────

class TestNoIncident:
    async def test_returns_done_phase(self, minimal_state, anomaly_payload):
        tools   = _make_tools(anomaly_payload)
        llm     = _make_llm({"is_incident": False, "reasoning": "all minor"})
        detector = make_incident_detector(tools, llm=llm)

        result = await detector(minimal_state)

        assert result["phase"] == "done"

    async def test_anomaly_nodes_populated(self, minimal_state, anomaly_payload):
        detector = make_incident_detector(_make_tools(anomaly_payload), llm=_make_llm({"is_incident": False}))
        result = await detector(minimal_state)

        assert len(result["anomaly_nodes"]) == 2

    async def test_create_ticket_not_called(self, minimal_state, anomaly_payload):
        tools = _make_tools(anomaly_payload)
        detector = make_incident_detector(tools, llm=_make_llm({"is_incident": False}))

        await detector(minimal_state)

        create_tool = next(t for t in tools if t.name == "create_incident_ticket")
        create_tool.ainvoke.assert_not_called()


# ── Incident declared ─────────────────────────────────────────────────

class TestIncidentDeclared:
    async def test_returns_root_cause_phase(self, minimal_state, anomaly_payload):
        tools = _make_tools(anomaly_payload)
        llm   = _make_llm({"is_incident": True, "severity": "SEV2", "reasoning": "high error rate"})
        detector = make_incident_detector(tools, llm=llm)

        result = await detector(minimal_state)

        assert result["phase"] == "root_cause"

    async def test_incident_id_is_set(self, minimal_state, anomaly_payload):
        tools = _make_tools(anomaly_payload)
        llm   = _make_llm({"is_incident": True, "severity": "SEV2"})
        detector = make_incident_detector(tools, llm=llm)

        result = await detector(minimal_state)

        assert result["incident_id"] is not None
        assert result["incident_id"].startswith("INC-")

    async def test_severity_propagated(self, minimal_state, anomaly_payload):
        tools = _make_tools(anomaly_payload)
        llm   = _make_llm({"is_incident": True, "severity": "SEV1"})
        detector = make_incident_detector(tools, llm=llm)

        result = await detector(minimal_state)

        assert result["severity"] == "SEV1"

    async def test_anomaly_nodes_set(self, minimal_state, anomaly_payload):
        tools = _make_tools(anomaly_payload)
        llm   = _make_llm({"is_incident": True, "severity": "SEV2"})
        detector = make_incident_detector(tools, llm=llm)

        result = await detector(minimal_state)

        assert len(result["anomaly_nodes"]) == 2

    async def test_create_ticket_called_with_correct_args(self, minimal_state, anomaly_payload):
        tools = _make_tools(anomaly_payload)
        llm   = _make_llm({"is_incident": True, "severity": "SEV3"})
        detector = make_incident_detector(tools, llm=llm)

        result = await detector(minimal_state)

        create_tool = next(t for t in tools if t.name == "create_incident_ticket")
        create_tool.ainvoke.assert_called_once()
        call_args = create_tool.ainvoke.call_args[0][0]
        assert call_args["id"] == result["incident_id"]
        assert call_args["severity"] == "SEV3"
        assert call_args["status"] == "open"
        assert "startTime" in call_args

    async def test_message_added_to_state(self, minimal_state, anomaly_payload):
        tools = _make_tools(anomaly_payload)
        llm   = _make_llm({"is_incident": True, "severity": "SEV2"})
        detector = make_incident_detector(tools, llm=llm)

        result = await detector(minimal_state)

        assert len(result.get("messages", [])) == 1
        assert isinstance(result["messages"][0], AIMessage)

    async def test_default_severity_when_missing(self, minimal_state, anomaly_payload):
        # LLM omits severity — should default to SEV3
        tools = _make_tools(anomaly_payload)
        llm   = _make_llm({"is_incident": True})
        detector = make_incident_detector(tools, llm=llm)

        result = await detector(minimal_state)

        assert result["severity"] == "SEV3"


# ── Error handling ────────────────────────────────────────────────────

class TestErrorHandling:
    async def test_graphserv_unreachable_returns_done(self, minimal_state):
        list_tool = MagicMock()
        list_tool.name = "list_anomalies"
        list_tool.ainvoke = AsyncMock(side_effect=Exception("connection refused"))
        create_tool = _make_tool("create_incident_ticket", "{}")
        detector = make_incident_detector([list_tool, create_tool], llm=_make_llm({}))

        result = await detector(minimal_state)

        assert result["phase"] == "done"
        assert result["anomaly_nodes"] == []

    async def test_llm_returns_malformed_json(self, minimal_state, anomaly_payload):
        tools = _make_tools(anomaly_payload)
        bad_llm = MagicMock()
        bad_llm.ainvoke = AsyncMock(return_value=AIMessage(content="not json at all"))
        detector = make_incident_detector(tools, llm=bad_llm)

        result = await detector(minimal_state)

        assert result["phase"] == "done"

    async def test_ticket_creation_failure_does_not_raise(self, minimal_state, anomaly_payload):
        list_tool  = _make_tool("list_anomalies", anomaly_payload)
        create_tool = MagicMock()
        create_tool.name = "create_incident_ticket"
        create_tool.ainvoke = AsyncMock(side_effect=Exception("graphserv error"))
        llm = _make_llm({"is_incident": True, "severity": "SEV2"})
        detector = make_incident_detector([list_tool, create_tool], llm=llm)

        # Should not raise; incident is still detected despite ticket failure
        result = await detector(minimal_state)
        assert result["phase"] == "root_cause"


# ── Factory validation ────────────────────────────────────────────────

class TestFactory:
    def test_raises_if_list_anomalies_missing(self):
        create_tool = _make_tool("create_incident_ticket", "{}")
        with pytest.raises(ValueError, match="list_anomalies"):
            make_incident_detector([create_tool])

    def test_raises_if_create_ticket_missing(self):
        list_tool = _make_tool("list_anomalies", "[]")
        with pytest.raises(ValueError, match="create_incident_ticket"):
            make_incident_detector([list_tool])
