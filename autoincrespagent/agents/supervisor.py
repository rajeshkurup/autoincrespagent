"""Deterministic supervisor — maps phase → next node name.

No LLM calls. Used as a conditional edge function in the StateGraph so
every agent node routes through here after it completes.
"""

from langgraph.graph import END

from autoincrespagent.agents.state import AgentState

# Phase → node name routing table
_ROUTING: dict[str, str] = {
    "detect":      "incident_detector",
    "root_cause":  "root_cause_finder",
    "mitigate":    "incident_mitigator",
    "feedback":    "root_cause_finder",   # loops back
    "communicate": "incident_communicator",
    "summarize":   "incident_summarizer",
}


def supervisor(state: AgentState) -> str:
    """Return the name of the next node to execute, or END.

    Called by LangGraph as the conditional edge function after each
    agent node completes and updates the shared state.
    """
    phase = state.get("phase", "done")
    return _ROUTING.get(phase, END)
