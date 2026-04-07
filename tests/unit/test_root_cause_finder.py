"""Unit tests for the root_cause_finder agent node."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage

from autoincrespagent.agents.root_cause_finder import make_root_cause_finder


# ── Helpers ───────────────────────────────────────────────────────────

def _make_tool(name: str, return_value) -> MagicMock:
    tool = MagicMock()
    tool.name = name
    if isinstance(return_value, Exception):
        tool.ainvoke = AsyncMock(side_effect=return_value)
    else:
        payload = return_value if isinstance(return_value, str) else json.dumps(return_value)
        tool.ainvoke = AsyncMock(return_value=payload)
    return tool


def _make_llm(hypotheses: list, summary: str = "Test summary") -> MagicMock:
    llm = MagicMock()
    content = json.dumps({"hypotheses": hypotheses, "summary": summary})
    llm.ainvoke = AsyncMock(return_value=AIMessage(content=content))
    return llm


def _make_all_tools(
    relationships=None,
    rca_result=None,
    blast_result=None,
    change_tickets=None,
    rca_tickets=None,
) -> list:
    rels_payload = json.dumps(relationships or [])
    rca_payload  = json.dumps(rca_result  or {"origins": [], "candidates": 0})
    blast_payload = json.dumps(blast_result or {"dependentCount": 0, "dependents": []})
    ct_payload   = json.dumps(change_tickets or [])
    rcatix_payload = json.dumps(rca_tickets or [])

    return [
        _make_tool("get_relationships",    rels_payload),
        _make_tool("root_cause_analysis",  rca_payload),
        _make_tool("blast_radius",         blast_payload),
        _make_tool("get_change_tickets",   ct_payload),
        _make_tool("get_rca_tickets",      rcatix_payload),
    ]


def _state(anomaly_nodes=None, severity="SEV2", feedback_request=None, feedback_iteration=0):
    return {
        "phase": "root_cause",
        "session_id": "test-session",
        "incident_id": "INC-TEST001",
        "severity": severity,
        "feedback_iteration": feedback_iteration,
        "anomaly_nodes": anomaly_nodes or [
            {"id": "ANO-001", "type": "latency_spike", "severity": "high", "status": "active"}
        ],
        "root_causes": [],
        "mitigation_workflows": [],
        "mitigation_confidence": 0.0,
        "communications_sent": [],
        "messages": [],
        "feedback_request": feedback_request,
        "incident_summary": None,
    }


# ── Factory validation ────────────────────────────────────────────────

class TestFactory:
    def test_raises_if_tool_missing(self):
        # Provide only 4 of the 5 required tools
        tools = _make_all_tools()
        tools = [t for t in tools if t.name != "blast_radius"]
        with pytest.raises(ValueError, match="blast_radius"):
            make_root_cause_finder(tools)

    def test_accepts_all_required_tools(self):
        tools = _make_all_tools()
        agent = make_root_cause_finder(tools, llm=_make_llm([]))
        assert callable(agent)


# ── Phase routing ─────────────────────────────────────────────────────

class TestPhaseRouting:
    async def test_returns_mitigate_phase(self):
        tools = _make_all_tools()
        llm   = _make_llm([{"node_id": "s1", "node_type": "Storage",
                             "hypothesis": "DB OOM", "evidence": [], "confidence": 0.9}])
        agent = make_root_cause_finder(tools, llm=llm)

        result = await agent(_state())

        assert result["phase"] == "communicate"
        assert result["next_phase"] == "mitigate"

    async def test_feedback_request_cleared(self):
        tools = _make_all_tools()
        agent = make_root_cause_finder(tools, llm=_make_llm([]))
        state = _state(feedback_request="dig deeper into storage")

        result = await agent(state)

        assert result["feedback_request"] is None


# ── Root causes population ────────────────────────────────────────────

class TestRootCauses:
    async def test_root_causes_populated_from_llm(self):
        hypotheses = [
            {"node_id": "db-001", "node_type": "Storage",
             "hypothesis": "Connection pool exhausted", "evidence": ["ANO-001"], "confidence": 0.9},
            {"node_id": "app-001", "node_type": "Application",
             "hypothesis": "Memory leak in payment service", "evidence": [], "confidence": 0.6},
        ]
        tools = _make_all_tools()
        agent = make_root_cause_finder(tools, llm=_make_llm(hypotheses))

        result = await agent(_state())

        assert len(result["root_causes"]) == 2
        assert result["root_causes"][0]["node_id"] == "db-001"
        assert result["root_causes"][0]["confidence"] == 0.9

    async def test_empty_hypotheses_on_llm_failure(self):
        bad_llm = MagicMock()
        bad_llm.ainvoke = AsyncMock(return_value=AIMessage(content="not valid json {{"))
        tools = _make_all_tools()
        agent = make_root_cause_finder(tools, llm=bad_llm)

        result = await agent(_state())

        assert result["root_causes"] == []
        assert result["phase"] == "communicate"
        assert result["next_phase"] == "mitigate"   # still advances to mitigate

    async def test_mitigation_workflows_empty_without_qdrant(self):
        # No qdrant_client → mitigation_workflows must be empty
        tools = _make_all_tools()
        agent = make_root_cause_finder(tools, llm=_make_llm([
            {"node_id": "x", "node_type": "Storage", "hypothesis": "h",
             "evidence": [], "confidence": 0.8}
        ]))

        result = await agent(_state())

        assert result["mitigation_workflows"] == []


# ── Graph traversal ───────────────────────────────────────────────────

class TestGraphTraversal:
    async def test_get_relationships_called_per_anomaly(self):
        tools = _make_all_tools()
        agent = make_root_cause_finder(tools, llm=_make_llm([]))
        anomalies = [
            {"id": "ANO-001", "status": "active"},
            {"id": "ANO-002", "status": "active"},
        ]

        await agent(_state(anomaly_nodes=anomalies))

        rels_tool = next(t for t in tools if t.name == "get_relationships")
        assert rels_tool.ainvoke.call_count == 2

    async def test_rca_called_when_detected_on_node_found(self):
        rels = [{"to": {"label": "Storage", "id": "db-001"}, "type": "DETECTED_ON"}]
        tools = _make_all_tools(relationships=rels)
        agent = make_root_cause_finder(tools, llm=_make_llm([]))

        await agent(_state())

        rca_tool = next(t for t in tools if t.name == "root_cause_analysis")
        rca_tool.ainvoke.assert_called_once()
        call_args = rca_tool.ainvoke.call_args[0][0]
        assert call_args["startLabel"] == "Storage"
        assert call_args["startId"] == "db-001"

    async def test_blast_radius_called_when_detected_on_node_found(self):
        rels = [{"to": {"label": "Storage", "id": "db-001"}, "type": "DETECTED_ON"}]
        tools = _make_all_tools(relationships=rels)
        agent = make_root_cause_finder(tools, llm=_make_llm([]))

        await agent(_state())

        blast_tool = next(t for t in tools if t.name == "blast_radius")
        blast_tool.ainvoke.assert_called_once()

    async def test_graph_tool_failure_does_not_raise(self):
        tools = _make_all_tools()
        # Make get_relationships raise
        rels_tool = next(t for t in tools if t.name == "get_relationships")
        rels_tool.ainvoke = AsyncMock(side_effect=Exception("graphserv down"))
        agent = make_root_cause_finder(tools, llm=_make_llm([]))

        result = await agent(_state())   # must not raise

        assert result["phase"] == "communicate"
        assert result["next_phase"] == "mitigate"


# ── Change + RCA ticket fetching ──────────────────────────────────────

class TestTicketFetching:
    async def test_change_tickets_fetched(self):
        change_tickets = [{"id": "CHG-001", "description": "deploy v2.1", "status": "completed"}]
        tools = _make_all_tools(change_tickets=change_tickets)
        agent = make_root_cause_finder(tools, llm=_make_llm([]))

        await agent(_state())

        ct_tool = next(t for t in tools if t.name == "get_change_tickets")
        ct_tool.ainvoke.assert_called_once_with({"limit": 20})

    async def test_rca_tickets_fetched(self):
        tools = _make_all_tools()
        agent = make_root_cause_finder(tools, llm=_make_llm([]))

        await agent(_state())

        rca_tool = next(t for t in tools if t.name == "get_rca_tickets")
        rca_tool.ainvoke.assert_called_once_with({"limit": 10})

    async def test_change_ticket_failure_does_not_raise(self):
        tools = _make_all_tools()
        ct_tool = next(t for t in tools if t.name == "get_change_tickets")
        ct_tool.ainvoke = AsyncMock(side_effect=Exception("timeout"))
        agent = make_root_cause_finder(tools, llm=_make_llm([]))

        result = await agent(_state())

        assert result["phase"] == "communicate"
        assert result["next_phase"] == "mitigate"


# ── Qdrant integration ────────────────────────────────────────────────

class TestQdrantIntegration:
    async def test_qdrant_search_called_when_client_provided(self):
        mock_response = MagicMock()
        mock_response.points = []
        mock_client = MagicMock()
        mock_client.query_points = AsyncMock(return_value=mock_response)
        mock_embeddings = MagicMock()
        mock_embeddings.aembed_query = AsyncMock(return_value=[0.1] * 768)

        tools = _make_all_tools()
        agent = make_root_cause_finder(
            tools, llm=_make_llm([]),
            qdrant_client=mock_client, embeddings=mock_embeddings,
        )

        await agent(_state())

        # Should have searched rca_documents, change_context, and mitigation_workflows
        # (mitigation_workflows only if there are hypotheses — no hypotheses here so 2 calls)
        assert mock_client.query_points.call_count >= 2

    async def test_mitigation_workflow_search_uses_top_hypothesis(self):
        mock_response = MagicMock()
        mock_response.points = []
        mock_client = MagicMock()
        mock_client.query_points = AsyncMock(return_value=mock_response)
        mock_embeddings = MagicMock()
        mock_embeddings.aembed_query = AsyncMock(return_value=[0.1] * 768)

        hypotheses = [{"node_id": "db-1", "node_type": "Storage",
                       "hypothesis": "DB OOM kill", "evidence": [], "confidence": 0.9}]
        tools = _make_all_tools()
        agent = make_root_cause_finder(
            tools, llm=_make_llm(hypotheses),
            qdrant_client=mock_client, embeddings=mock_embeddings,
        )

        await agent(_state(severity="SEV1"))

        # 5 calls: rca_documents, change_context, incident_summaries, feedback_history, mitigation_workflows
        assert mock_client.query_points.call_count == 5

    async def test_message_added_to_state(self):
        tools = _make_all_tools()
        agent = make_root_cause_finder(tools, llm=_make_llm([]))

        result = await agent(_state())

        assert len(result.get("messages", [])) == 1
        assert isinstance(result["messages"][0], AIMessage)
