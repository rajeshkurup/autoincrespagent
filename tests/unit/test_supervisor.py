"""Unit tests for the supervisor routing function."""

import pytest
from langgraph.graph import END

from autoincrespagent.agents.supervisor import supervisor


def _state(phase: str) -> dict:
    return {"phase": phase}


class TestSupervisorRouting:
    def test_detect_routes_to_incident_detector(self):
        assert supervisor(_state("detect")) == "incident_detector"

    def test_root_cause_routes_to_root_cause_finder(self):
        assert supervisor(_state("root_cause")) == "root_cause_finder"

    def test_mitigate_routes_to_incident_mitigator(self):
        assert supervisor(_state("mitigate")) == "incident_mitigator"

    def test_feedback_routes_to_root_cause_finder(self):
        # Feedback loop sends back to root_cause_finder
        assert supervisor(_state("feedback")) == "root_cause_finder"

    def test_communicate_routes_to_incident_communicator(self):
        assert supervisor(_state("communicate")) == "incident_communicator"

    def test_summarize_routes_to_incident_summarizer(self):
        assert supervisor(_state("summarize")) == "incident_summarizer"

    def test_done_returns_end(self):
        assert supervisor(_state("done")) == END

    def test_unknown_phase_returns_end(self):
        assert supervisor(_state("something_unexpected")) == END

    def test_empty_phase_returns_end(self):
        assert supervisor(_state("")) == END

    def test_missing_phase_key_returns_end(self):
        # state without 'phase' key — get() returns None → falls back to END
        assert supervisor({}) == END
