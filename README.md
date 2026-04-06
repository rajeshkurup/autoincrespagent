# autoincrespagent

AI-driven incident response agent built with LangGraph and Ollama. Monitors infrastructure anomalies via the [graphserv](../graphserv/README.md) topology graph, classifies them into incidents, and orchestrates a multi-agent pipeline to root-cause, mitigate, communicate, and summarize them.

---

## Table of Contents

- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Agents](#agents)
- [Dependencies](#dependencies)
- [Configuration](#configuration)
- [Setup](#setup)
- [Running](#running)
- [Testing](#testing)
- [Deployment (Docker)](#deployment-docker)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        autoincrespagent                             │
│                                                                     │
│  trigger.py ──► StateGraph                                         │
│                     │                                              │
│            ┌────────▼────────┐                                     │
│            │ incident_detector│ ◄── Qwen 2.5 7B (JSON)            │
│            └────────┬────────┘                                     │
│                     │ supervisor (phase routing)                   │
│            ┌────────▼────────┐                                     │
│            │root_cause_finder│ ◄── Llama 3.1 8B      [Phase 2]   │
│            └────────┬────────┘                                     │
│                     │                                              │
│            ┌────────▼────────┐                                     │
│            │incident_mitigator│ ◄── Qwen 2.5 7B       [Phase 3]  │
│            └────────┬────────┘                                     │
│               ▲     │ (feedback loop if confidence < 0.75)        │
│               └─────┘                                              │
│            ┌────────▼──────────┐                                   │
│            │incident_communicator│ ◄── Mistral 7B      [Phase 4] │
│            └────────┬──────────┘                                   │
│            ┌────────▼──────────┐                                   │
│            │incident_summarizer │ ◄── Llama 3.1 8B    [Phase 5]  │
│            └────────┬──────────┘                                   │
│                     │ END                                          │
└─────────────────────────────────────────────────────────────────────┘
         │                          │
         ▼                          ▼
  Graph DB MCP              MySQL (checkpoints)
  (graphmcpserv)
         │
         ▼
  graphserv :8080
  (Neo4j topology)
```

### Control Flow

1. `trigger.py` polls graphserv every `POLL_INTERVAL_SECONDS` for active anomalies
2. **Incident Detector** fetches anomalies → asks LLM if they constitute an incident → creates `IncidentTicket` in Neo4j
3. **Root Cause Finder** *(Phase 2)* — graph traversal + Qdrant RAG → root cause hypotheses
4. **Incident Mitigator** *(Phase 3)* — semantic workflow search; loops back to Root Cause Finder if confidence < threshold
5. **Incident Communicator** *(Phase 4)* — drafts and dispatches messages to Slack, email, PagerDuty
6. **Incident Summarizer** *(Phase 5)* — writes final summary to Qdrant and MySQL; marks session complete

### Shared State

All agents read and write a single `AgentState` TypedDict:

| Field | Type | Set by |
|-------|------|--------|
| `phase` | `str` | Every agent — supervisor uses it to route |
| `session_id` | `str` | trigger.py |
| `feedback_iteration` | `int` | Mitigator (feedback loop counter) |
| `incident_id` | `str` | Incident Detector |
| `severity` | `str` | Incident Detector |
| `anomaly_nodes` | `list[dict]` | Incident Detector |
| `root_causes` | `list[dict]` | Root Cause Finder |
| `mitigation_workflows` | `list[dict]` | Incident Mitigator |
| `mitigation_confidence` | `float` | Incident Mitigator |
| `communications_sent` | `list[dict]` | Incident Communicator |
| `incident_summary` | `str` | Incident Summarizer |
| `messages` | `Annotated[list, add_messages]` | All agents (accumulates) |
| `feedback_request` | `str` | Mitigator → Root Cause Finder |

### Supervisor

The supervisor is a **deterministic routing function** (no LLM calls). It reads the `phase` field and returns the next node name:

| phase | next node |
|-------|-----------|
| `detect` | `incident_detector` |
| `root_cause` | `root_cause_finder` |
| `mitigate` | `incident_mitigator` |
| `feedback` | `root_cause_finder` |
| `communicate` | `incident_communicator` |
| `summarize` | `incident_summarizer` |
| anything else | `END` |

---

## Project Structure

```
autoincrespagent/
├── Dockerfile                          # Build context: incident-response/
├── pyproject.toml                      # Package + dependency config
├── .env.example                        # Environment variable template
├── trigger.py                          # CLI entry point
│
├── autoincrespagent/                   # Python package
│   ├── config.py                       # Settings (reads .env via pydantic-settings)
│   ├── llm/
│   │   └── factory.py                  # get_llm(agent_name) → ChatOllama
│   ├── agents/
│   │   ├── state.py                    # AgentState TypedDict
│   │   ├── supervisor.py               # Deterministic phase → node router
│   │   └── incident_detector.py        # Phase 1 agent ✓
│   ├── graph/
│   │   ├── mcp_client.py               # MultiServerMCPClient factory
│   │   └── workflow.py                 # StateGraph assembly
│   └── memory/
│       └── mysql_saver.py              # LangGraph BaseCheckpointSaver → MySQL
│
├── sql/
│   └── schema.sql                      # MySQL DDL for all 4 tables
│
└── tests/
    ├── conftest.py                     # Shared fixtures
    └── unit/
        ├── test_supervisor.py          # 10 routing tests
        └── test_detector.py            # 18 incident detector tests
```

---

## Agents

### ✅ Phase 1 — Incident Detector (`agents/incident_detector.py`)

**Model:** Qwen 2.5 7B (`qwen2.5:7b`, `format=json`)

**MCP tools used:**
- `list_anomalies` — fetches active anomalies from Neo4j via graphserv
- `create_incident_ticket` — writes an `IncidentTicket` node to Neo4j

**Logic:**
1. Calls `list_anomalies(status="active", limit=50)` directly
2. Passes anomalies to LLM with a classification prompt
3. LLM returns `{"is_incident": bool, "severity": "SEV1-4", "reasoning": "..."}`
4. If `is_incident=true`, generates `INC-<uuid>` and calls `create_incident_ticket`
5. Sets `phase="root_cause"` to continue, or `phase="done"` to exit

**Output state fields:** `phase`, `incident_id`, `severity`, `anomaly_nodes`, `messages`

---

### 🔲 Phase 2 — Root Cause Finder *(not yet implemented)*

**Model:** Llama 3.1 8B

**Planned tools:** `root_cause_analysis`, `blast_radius`, `get_change_tickets` (Graph DB MCP) + Qdrant `rca_documents` search

**Output:** `root_causes` list with hypothesis, evidence, confidence per candidate

---

### 🔲 Phase 3 — Incident Mitigator *(not yet implemented)*

**Model:** Qwen 2.5 7B

**Planned tools:** `search_mitigation_workflows`, `execute_mitigation_step` (Mitigation MCP)

**Feedback loop:** if top workflow score < `CONFIDENCE_THRESHOLD`, sets `phase="feedback"` to re-run Root Cause Finder (capped at `MAX_FEEDBACK_ITERATIONS`)

---

### 🔲 Phase 4 — Incident Communicator *(not yet implemented)*

**Model:** Mistral 7B

**Planned tools:** `send_slack`, `send_email`, `page_oncall`, `update_ticket_comms` (Comms MCP)

---

### 🔲 Phase 5 — Incident Summarizer *(not yet implemented)*

**Model:** Llama 3.1 8B

**Planned:** writes `incident_summary` to Qdrant `incident_summaries` collection and updates the MySQL `sessions` row to `completed`

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `langgraph` | ≥1.0 | StateGraph orchestration + checkpointing |
| `langchain` | ≥1.0 | Tool calling, prompts, message types |
| `langchain-core` | ≥1.0 | Base types (BaseTool, messages, RunnableConfig) |
| `langchain-ollama` | ≥1.0 | ChatOllama integration |
| `langchain-mcp-adapters` | ≥0.2 | Converts MCP tools → LangChain BaseTool |
| `mcp` | ≥1.0 | MCP protocol SDK (spawns graphmcpserv subprocess) |
| `aiomysql` | ≥0.2 | Async MySQL driver for checkpoint saver |
| `pydantic-settings` | ≥2.0 | Environment variable loading |
| `python-dotenv` | ≥1.0 | `.env` file support |

**Also requires** `graphmcpserv` installed in the same Python environment — see [Setup](#setup).

---

## Configuration

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `GRAPHSERV_URL` | `http://localhost:8080` | graphserv REST API base URL |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API base URL |
| `MYSQL_HOST` | `127.0.0.1` | MySQL host |
| `MYSQL_PORT` | `3306` | MySQL port |
| `MYSQL_USER` | `ir_user` | MySQL username |
| `MYSQL_PASSWORD` | *(empty)* | MySQL password |
| `MYSQL_DATABASE` | `incident_response` | MySQL database name |
| `CONFIDENCE_THRESHOLD` | `0.75` | Mitigation confidence below which feedback loop triggers |
| `MAX_FEEDBACK_ITERATIONS` | `3` | Max Mitigator ↔ Root Cause Finder cycles |
| `POLL_INTERVAL_SECONDS` | `60` | Seconds between anomaly polls |

---

## Setup

### Prerequisites

| Service | Purpose | Check |
|---------|---------|-------|
| Neo4j 5.x | Topology graph | `curl http://localhost:7474` |
| graphserv | REST API for Neo4j | `curl http://localhost:8080/health` |
| Ollama | Local LLM runtime | `ollama list` |
| MySQL 8 | Session checkpoints | `mysql -u ir_user -p` |
| `qwen2.5:7b` model | Incident Detector LLM | `ollama list` |

### Install

```bash
cd autoincrespagent

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install agent + graphmcpserv (MCP subprocess must be in same venv)
pip install -e .
pip install -e ../graphmcpserv

# Configure environment
cp .env.example .env
# edit .env with your MySQL credentials
```

### Apply MySQL schema

```bash
mysql -u root -p incident_response < sql/schema.sql
```

### Pull required Ollama models

```bash
ollama pull qwen2.5:7b          # Incident Detector + Mitigator
ollama pull llama3.1:8b         # Root Cause Finder + Summarizer
ollama pull mistral:7b          # Communicator
```

---

## Running

```bash
source .venv/bin/activate

# Run once (single anomaly poll)
python trigger.py

# Poll continuously (every POLL_INTERVAL_SECONDS)
python trigger.py --poll
```

**Example output (incident found):**

```
2026-04-05 22:11:09 INFO trigger: loaded 10 MCP tools
2026-04-05 22:11:09 INFO incident_detector: {"event": "anomalies_found", "count": 1}
2026-04-05 22:11:14 INFO incident_detector: {"event": "incident_declared", "incident_id": "INC-362A3B55", "severity": "SEV2"}
2026-04-05 22:11:14 INFO workflow: root_cause_finder: not yet implemented — ending workflow
{
  "phase": "done",
  "incident_id": "INC-362A3B55",
  "severity": "SEV2",
  "anomaly_nodes": [...]
}
```

**Example output (no incident):**

```
2026-04-05 22:11:09 INFO incident_detector: {"event": "no_anomalies"}
{ "phase": "done", "anomaly_nodes": [] }
```

**Fallback behaviour:**
- MySQL unavailable → falls back to in-memory checkpointer (no persistence)
- graphserv unavailable → agent logs error and exits cleanly with `phase=done`
- LLM returns malformed JSON → treated as no-incident, logs error

---

## Testing

```bash
source .venv/bin/activate
pytest tests/unit/ -v
```

### Test coverage

| Test file | Tests | What's covered |
|-----------|-------|----------------|
| `tests/unit/test_supervisor.py` | 10 | All phase → node routing, unknown/empty phase → END |
| `tests/unit/test_detector.py` | 18 | No anomalies, no incident, incident declared, error handling, factory validation |

### Unit test approach

- MCP tools are mocked with `unittest.mock.AsyncMock` — no running graphserv needed
- LLM is injected via `make_incident_detector(tools, llm=fake_llm)` — no Ollama needed
- All tests run offline in < 1 second

### Adding tests for new agents

Follow the same pattern in `tests/unit/`:
1. Create `test_<agent_name>.py`
2. Use `conftest.py`'s `minimal_state` fixture as the base state
3. Mock tools with `_make_tool(name, return_value)` helper
4. Inject a fake LLM via the agent factory's `llm=` parameter

---

## Deployment (Docker)

The Dockerfile and `docker-compose.yml` live at the project root (`incident-response/`).

### Build the agent image

```bash
# From incident-response/ (parent directory — required as build context)
docker build -f autoincrespagent/Dockerfile -t autoincrespagent:latest .
```

### Run the full stack

```bash
cd incident-response/

# First run: pull Ollama models into the named volume
docker compose run --rm ollama ollama pull qwen2.5:7b
docker compose run --rm ollama ollama pull llama3.1:8b

# Start all services
docker compose up -d

# Watch agent logs
docker compose logs -f agent

# Stop everything
docker compose down
```

### Services

| Service | Image | Ports | Memory |
|---------|-------|-------|--------|
| `neo4j` | neo4j:5-community | 7474, 7687 | 1 GB |
| `graphserv` | rajeshkurup77/graphserv:latest | 8080 | 256 MB |
| `mysql` | mysql:8 | 3306 | 512 MB |
| `qdrant` | qdrant/qdrant | 6333 | 512 MB |
| `ollama` | ollama/ollama | 11434 | 8 GB |
| `agent` | local build | — | 2 GB |

> **Note:** Default password is `changeme` in `docker-compose.yml`. Change before any non-local use.
