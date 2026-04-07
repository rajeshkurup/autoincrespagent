"""LLM factory — returns a per-agent ChatOllama instance."""

from langchain_ollama import ChatOllama

from autoincrespagent.config import settings

# Model assigned to each agent role
_MODEL_MAP: dict[str, str] = {
    "incident_detector":  "qwen2.5:7b",
    "root_cause_finder":  "llama3.1:8b",
    "incident_mitigator": "qwen2.5:7b",
    "incident_communicator": "mistral:7b",
    "incident_summarizer":   "llama3.1:8b",
}

# Agents that need deterministic JSON output
_JSON_AGENTS: frozenset[str] = frozenset({"incident_detector", "root_cause_finder", "incident_mitigator"})


def get_llm(agent_name: str, temperature: float = 0.0) -> ChatOllama:
    """Return a ChatOllama configured for the given agent.

    Args:
        agent_name: Key from _MODEL_MAP; falls back to llama3.1:8b.
        temperature: Override default temperature (0.0 for structured agents).
    """
    model = _MODEL_MAP.get(agent_name, "llama3.1:8b")
    kwargs: dict = {
        "model": model,
        "temperature": temperature,
        "base_url": settings.ollama_base_url,
    }
    if agent_name in _JSON_AGENTS:
        kwargs["format"] = "json"
    return ChatOllama(**kwargs)
