"""Unit tests for the incident_summarizer agent node."""

import pytest
from langchain_core.messages import AIMessage
from unittest.mock import AsyncMock, MagicMock

from autoincrespagent.agents.incident_summarizer import make_incident_summarizer


def _state(**overrides) -> dict:
    base = {
        "phase": "summarize",
        "session_id": "test-session",
        "incident_id": "INC-TESTSUM",
        "severity": "SEV2",
        "feedback_iteration": 0,
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
                "steps": ["Check metrics", "Restart pooler"],
            }}
        ],
        "mitigation_confidence": 0.85,
        "communications_sent": [],
        "messages": [],
        "feedback_request": None,
        "communication_event": None,
        "next_phase": None,
        "incident_summary": None,
    }
    base.update(overrides)
    return base


class TestPhaseRouting:
    async def test_sets_phase_done(self):
        agent = make_incident_summarizer()
        result = await agent(_state())
        assert result["phase"] == "done"

    async def test_incident_summary_populated(self):
        agent = make_incident_summarizer()
        result = await agent(_state())
        assert result["incident_summary"] is not None
        assert len(result["incident_summary"]) > 0

    async def test_message_added(self):
        agent = make_incident_summarizer()
        result = await agent(_state())
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)


class TestSummaryContent:
    async def test_summary_contains_incident_id(self, capsys):
        agent = make_incident_summarizer()
        result = await agent(_state())
        assert "INC-TESTSUM" in result["incident_summary"]

    async def test_summary_contains_severity(self):
        agent = make_incident_summarizer()
        result = await agent(_state())
        assert "SEV2" in result["incident_summary"]

    async def test_summary_contains_root_cause(self):
        agent = make_incident_summarizer()
        result = await agent(_state())
        assert "Connection pool exhausted" in result["incident_summary"]

    async def test_summary_contains_workflow(self):
        agent = make_incident_summarizer()
        result = await agent(_state())
        assert "WF-001" in result["incident_summary"]

    async def test_summary_handles_empty_state(self):
        agent = make_incident_summarizer()
        result = await agent(_state(root_causes=[], mitigation_workflows=[], anomaly_nodes=[]))
        assert result["phase"] == "done"
        assert result["incident_summary"] is not None

    async def test_prints_to_stdout(self, capsys):
        agent = make_incident_summarizer()
        await agent(_state())
        out = capsys.readouterr().out
        assert "INCIDENT SUMMARY" in out
        assert "INC-TESTSUM" in out


class TestQdrantPersistence:
    async def test_upsert_called_when_client_provided(self):
        mock_client = MagicMock()
        mock_client.upsert = AsyncMock(return_value=None)
        mock_embeddings = MagicMock()
        mock_embeddings.aembed_query = AsyncMock(return_value=[0.1] * 768)

        agent = make_incident_summarizer(qdrant_client=mock_client, embeddings=mock_embeddings)
        await agent(_state())

        mock_client.upsert.assert_called_once()
        call_kwargs = mock_client.upsert.call_args[1]
        assert call_kwargs["collection_name"] == "incident_summaries"

    async def test_point_payload_has_expected_fields(self):
        mock_client = MagicMock()
        mock_client.upsert = AsyncMock(return_value=None)
        mock_embeddings = MagicMock()
        mock_embeddings.aembed_query = AsyncMock(return_value=[0.1] * 768)

        agent = make_incident_summarizer(qdrant_client=mock_client, embeddings=mock_embeddings)
        await agent(_state())

        points = mock_client.upsert.call_args[1]["points"]
        assert len(points) == 1
        payload = points[0].payload
        assert payload["incident_id"] == "INC-TESTSUM"
        assert payload["severity"] == "SEV2"
        assert "summary" in payload
        assert "completed_at" in payload

    async def test_no_upsert_without_client(self):
        agent = make_incident_summarizer()
        result = await agent(_state())   # must not raise
        assert result["phase"] == "done"

    async def test_upsert_failure_does_not_raise(self):
        mock_client = MagicMock()
        mock_client.upsert = AsyncMock(side_effect=Exception("qdrant down"))
        mock_embeddings = MagicMock()
        mock_embeddings.aembed_query = AsyncMock(return_value=[0.1] * 768)

        agent = make_incident_summarizer(qdrant_client=mock_client, embeddings=mock_embeddings)
        result = await agent(_state())   # must not raise
        assert result["phase"] == "done"

    async def test_same_point_id_as_communicator(self):
        """Summarizer must use same deterministic ID so it overwrites partial summaries."""
        import uuid
        from autoincrespagent.agents.incident_communicator import _point_id as comm_id
        from autoincrespagent.agents.incident_summarizer import _point_id as summ_id

        assert comm_id("INC-TEST") == summ_id("INC-TEST")
