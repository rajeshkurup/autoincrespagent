"""Shared pytest fixtures."""

import pytest

from autoincrespagent.agents.state import AgentState


@pytest.fixture
def minimal_state() -> AgentState:
    """A fully-populated AgentState with safe default values."""
    return {
        "phase": "detect",
        "session_id": "test-session-001",
        "feedback_iteration": 0,
        "incident_id": None,
        "severity": None,
        "anomaly_nodes": [],
        "root_causes": [],
        "mitigation_workflows": [],
        "mitigation_confidence": 0.0,
        "communications_sent": [],
        "incident_summary": None,
        "messages": [],
        "feedback_request": None,
    }


@pytest.fixture
def anomaly_payload() -> str:
    """Canned JSON string mimicking a list_anomalies response."""
    return (
        '[{"id": "ANO-001", "type": "latency_spike", "severity": "high", "status": "active",'
        ' "startTime": "2026-04-05T09:00:00Z"},'
        ' {"id": "ANO-002", "type": "error_rate_spike", "severity": "critical", "status": "active",'
        ' "startTime": "2026-04-05T09:05:00Z"}]'
    )
