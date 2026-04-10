# autoincrespagent

AI-driven incident response agent built with LangGraph and Ollama. Monitors infrastructure anomalies via the [graphserv](../graphserv/README.md) topology graph, classifies them into incidents, and orchestrates a multi-agent pipeline to root-cause, mitigate, communicate, and summarize them.

---

## Table of Contents

- [Architecture](#architecture)
- [MCP Servers](#mcp-servers)
- [Agent Pipeline](#agent-pipeline)
- [Data Flow](#data-flow)
- [Qdrant Vector Collections](#qdrant-vector-collections)
- [Shared State](#shared-state)
- [Supervisor Routing](#supervisor-routing)
- [Agents](#agents)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Configuration](#configuration)
- [Setup](#setup)
- [Running](#running)
- [Testing](#testing)
- [Deployment (Docker)](#deployment-docker)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            autoincrespagent                                 │
│                                                                             │
│  trigger.py ──► StateGraph (LangGraph)                                     │
│                     │                                                       │
│            ┌────────▼──────────┐                                           │
│            │ incident_detector  │ ◄── Qwen 2.5 7B (JSON)     [Phase 1]    │
│            └────────┬──────────┘                                           │
│                     │ phase="communicate" (incident_detected)              │
│            ┌────────▼──────────┐                                           │
│            │incident_communicator│ ◄── commsmcpserv           [Comms]     │
│            └────────┬──────────┘                                           │
│                     │ next_phase="root_cause"                              │
│            ┌────────▼──────────┐                                           │
│            │ root_cause_finder  │ ◄── Llama 3.1 8B + Qdrant [Phase 2]    │
│            └────────┬──────────┘                                           │
│                     │ phase="communicate" (root_cause_found)               │
│            ┌────────▼──────────┐                                           │
│            │incident_communicator│ ◄── commsmcpserv                       │
│            └────────┬──────────┘                                           │
│                     │ next_phase="mitigate"                                │
│            ┌────────▼──────────┐                                           │
│            │ incident_mitigator │ ◄── mitigationmcpserv      [Phase 3]    │
│            └────────┬──────────┘                                           │
│               ▲     │                                                       │
│               │     │ [confidence < threshold & iter < max]               │
│               │     │ phase="feedback" ──────────────────────┐            │
│               │     │                                         │            │
│               └─────┘ root_cause_finder (feedback pass)      │            │
│                     │                                         │            │
│                     │ [confidence >= threshold OR iter >= max]│            │
│                     │ phase="communicate" (mitigation_complete)            │
│            ┌────────▼──────────┐                             │            │
│            │incident_communicator│◄───────────────────────────┘           │
│            └────────┬──────────┘                                           │
│                     │ next_phase="summarize"                               │
│            ┌────────▼──────────┐                                           │
│            │incident_summarizer │ ◄── deterministic + Qdrant [Phase 5]   │
│            └────────┬──────────┘                                           │
│                     │ END                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
  graphmcpserv         mitigationmcpserv      commsmcpserv
  (Neo4j topology)     (workflow execution)   (notifications)
         │                    │                    │
         ▼                    ▼                    ▼
  graphserv :8080        Qdrant                log files
  (Neo4j)          mitigation_workflows     email/slack/teams/
                   feedback_history         sms/pagerduty
```

---

## MCP Servers

Three MCP servers are spawned automatically as subprocesses when the agent starts. Each exposes typed tools over stdio transport — agents call them via `langchain-mcp-adapters` exactly like any other LangChain tool.

```
trigger.py
  └── MultiServerMCPClient.get_tools()
        ├── spawns: python -m mcp_servers.graph_db.server     (graphmcpserv)
        ├── spawns: python -m mcp_servers.mitigation.server   (mitigationmcpserv)
        └── spawns: python -m mcp_servers.comms.server        (commsmcpserv)
```

### graphmcpserv — Graph DB MCP Server

**Location:** `../graphmcpserv`  **Tools:** 10

Wraps the graphserv REST API so agents never touch Neo4j directly.

| Tool | Purpose |
|------|---------|
| `list_anomalies` | Fetch active anomalies |
| `get_node` | Fetch a single node by label + ID |
| `root_cause_analysis` | Downstream graph traversal |
| `blast_radius` | Dependent service count |
| `get_relationships` | Traverse edges from a node |
| `create_incident_ticket` | Write IncidentTicket to Neo4j |
| `link_incident_to_node` | Create IMPACTS relationship |
| `get_rca_tickets` | Historical RCA records |
| `get_change_tickets` | Recent change events |
| `update_node_status` | PATCH node status |

**Used by:** incident_detector, root_cause_finder

---

### mitigationmcpserv — Mitigation Services MCP Server

**Location:** `../mitigationmcpserv`  **Tools:** 4

Provides semantic workflow search (via Qdrant) and stub step execution. All tools work offline — Qdrant search falls back to a built-in mock catalogue of 5 workflows when the collection is empty or unreachable.

| Tool | Purpose |
|------|---------|
| `search_mitigation_workflows` | Semantic search in Qdrant `mitigation_workflows` — returns top-k workflows with steps and confidence scores |
| `execute_mitigation_step` | Stub executor — logs each step to `logs/mitigation_runs.log` and tracks run state in memory |
| `check_mitigation_status` | Returns current execution status of a workflow run |
| `store_mitigation_feedback` | Upserts outcome feedback into Qdrant `feedback_history` |

**Used by:** incident_mitigator

**Execution flow:**
```
incident_mitigator
  │  top workflow selected (score=0.87)
  ├── execute_mitigation_step(step_index=0, "Check active connection count")  → run_id=abc123
  ├── execute_mitigation_step(step_index=1, "Increase max_connections", run_id=abc123)
  ├── execute_mitigation_step(step_index=2, "Restart application pods", run_id=abc123)
  ├── check_mitigation_status(run_id=abc123)  → 3/3 steps completed
  └── store_mitigation_feedback(outcome="resolved", run_id=abc123)   [if feedback loop was used]
```

---

### commsmcpserv — Communication Services MCP Server

**Location:** `../commsmcpserv`  **Tools:** 6

All tools are stubs that write structured JSON to log files and print to stdout. Swap any stub for a real backend (SMTP, Slack webhook, Twilio, PagerDuty Events v2) without changing tool schemas or agent code.

| Tool | Channel | Log file |
|------|---------|----------|
| `send_email` | Email (SMTP) | `logs/email_outbox.log` |
| `send_slack` | Slack webhook | `logs/slack_outbox.log` |
| `send_teams` | MS Teams webhook | `logs/teams_outbox.log` |
| `send_sms` | SMS (Twilio) | `logs/sms_outbox.log` |
| `page_oncall` | PagerDuty | `logs/pagerduty_outbox.log` |
| `update_ticket_comms` | Incident ticket (ITSM) | `logs/ticket_comms.log` |

**Used by:** incident_communicator

**Channel selection by severity:**

| Severity | Channels dispatched |
|----------|---------------------|
| SEV1 | `page_oncall` + `send_slack` + `send_email` |
| SEV2 | `send_slack` + `send_teams` + `send_email` |
| SEV3 | `send_email` + `send_teams` |
| SEV4 | `send_email` |

---

## Agent Pipeline

### Full happy-path flow

```
trigger.py
    │
    ▼
incident_detector
    │  • polls graphserv for active anomalies via list_anomalies
    │  • LLM classifies anomalies → incident / no-incident
    │  • creates IncidentTicket in Neo4j via create_incident_ticket
    │  • sets: incident_id, severity, anomaly_nodes
    │  phase="communicate", event="incident_detected"
    ▼
incident_communicator  (event: incident_detected)
    │  • SEV2 → send_slack + send_teams + send_email via commsmcpserv
    │  • update_ticket_comms(event="incident_detected", channels=[...])
    │  • upserts partial summary → Qdrant incident_summaries
    │  • prints 🚨 INCIDENT DECLARED banner
    │  next_phase="root_cause"
    ▼
root_cause_finder
    │  • graph traversal: get_relationships → root_cause_analysis
    │  •                  blast_radius, get_change_tickets, get_rca_tickets
    │  • Qdrant search: rca_documents, change_context,
    │                   incident_summaries, feedback_history
    │  • LLM synthesises → ranked hypotheses
    │  • Qdrant search: mitigation_workflows (top hypothesis)
    │  • sets: root_causes, mitigation_workflows
    │  phase="communicate", event="root_cause_found"
    ▼
incident_communicator  (event: root_cause_found)
    │  • SEV2 → send_slack + send_teams + send_email via commsmcpserv
    │  • update_ticket_comms(event="root_cause_found", channels=[...])
    │  • upserts partial summary → Qdrant incident_summaries
    │  • prints 🔍 ROOT CAUSE IDENTIFIED banner
    │  next_phase="mitigate"
    ▼
incident_mitigator
    │  • calls execute_mitigation_step for each step of top workflow
    │  • calls check_mitigation_status to verify run
    │  • computes confidence from top workflow score
    │  ┌─ if confidence < threshold AND iter < max ─────────────────────┐
    │  │  • store_mitigation_feedback(outcome="low_confidence")         │
    │  │  phase="feedback" → back to root_cause_finder                  │
    │  └────────────────────────────────────────────────────────────────┘
    │  • if resolved after feedback: store_mitigation_feedback("resolved")
    │  phase="communicate", event="mitigation_complete"
    ▼
incident_communicator  (event: mitigation_complete)
    │  • SEV2 → send_slack + send_teams + send_email via commsmcpserv
    │  • update_ticket_comms(event="mitigation_complete", channels=[...])
    │  • upserts partial summary → Qdrant incident_summaries
    │  • prints 🔧 MITIGATION EXECUTED banner
    │  next_phase="summarize"
    ▼
incident_summarizer
    │  • composes full incident summary from state
    │  • prints 📋 INCIDENT SUMMARY
    │  • upserts final summary → Qdrant incident_summaries (overwrites partials)
    │  • sets: incident_summary, phase="done"
    ▼
  END
```

### Feedback loop detail

```
incident_mitigator
    │  confidence=0.42 < threshold=0.75, iter=0 < max=3
    │  store_mitigation_feedback(outcome="low_confidence") → Qdrant + log
    │  phase="feedback"
    ▼
root_cause_finder  (second pass)
    │  reads feedback_request: "Widen analysis — check upstream..."
    │  searches feedback_history → avoids dead-end hypotheses
    │  produces new hypotheses, fetches new workflows
    │  phase="communicate", event="root_cause_found"
    ▼
incident_communicator → next_phase="mitigate"
    ▼
incident_mitigator
    │  confidence=0.83 >= threshold=0.75
    │  store_mitigation_feedback(outcome="resolved") → Qdrant + log
    │  phase="communicate", event="mitigation_complete"
    ▼
  (continues to summarizer)
```

---

## Data Flow

### Qdrant read/write per agent

```
                    ┌──────────────────────────────────────────────────┐
                    │              Qdrant Collections                  │
                    │                                                  │
                    │  mitigation_workflows   rca_documents            │
                    │  change_context         incident_summaries       │
                    │  feedback_history                                │
                    └──────────────────────────────────────────────────┘
                          ▲  read              ▲  write
                          │                   │
    root_cause_finder ────┤ reads all 5 cols  │
                          │                   │
    incident_communicator ┼───────────────────┤ writes incident_summaries
                          │                   │  (after each of 3 events)
    incident_mitigator ───┼───────────────────┤ writes feedback_history
      (via mitigationmcpserv)                 │  (via store_mitigation_feedback)
    incident_summarizer ──┼───────────────────┤ writes incident_summaries
                          │                   │  (final authoritative upsert)
```

### Neo4j read/write via graphmcpserv

```
  incident_detector  ──► list_anomalies
                     ──► create_incident_ticket

  root_cause_finder  ──► get_relationships      (Anomaly –DETECTED_ON→ Node)
                     ──► root_cause_analysis    (downstream graph traversal)
                     ──► blast_radius           (dependent service count)
                     ──► get_change_tickets     (recent change events)
                     ──► get_rca_tickets        (historical RCA records)
```

### Mitigation execution via mitigationmcpserv

```
  incident_mitigator ──► execute_mitigation_step  (one call per step)
                     ──► check_mitigation_status  (verify run after steps)
                     ──► store_mitigation_feedback (feedback + resolved)
                              │
                              ▼
                     logs/mitigation_runs.log   +   Qdrant feedback_history
```

### Communication dispatch via commsmcpserv

```
  incident_communicator ──► send_email        → logs/email_outbox.log
                        ──► send_slack        → logs/slack_outbox.log
                        ──► send_teams        → logs/teams_outbox.log
                        ──► send_sms          → logs/sms_outbox.log
                        ──► page_oncall       → logs/pagerduty_outbox.log
                        ──► update_ticket_comms → logs/ticket_comms.log
```

### MySQL (LangGraph checkpoints)

```
  trigger.py  ──► MySQLSaver / MemorySaver (fallback)
                  • checkpoints table    — LangGraph state snapshots
                  • sessions table       — one row per incident run
                  • agent_memory table   — reserved for future use
                  • feedback_log table   — reserved for future use
```

### Cross-run learning

Each completed run enriches Qdrant so subsequent runs are smarter:

```
Run N completes
    │
    ├── incident_summaries ◄── communicator (3 partial upserts, same point ID)
    │                      ◄── summarizer   (1 final upsert, overwrites partials)
    │
    └── feedback_history   ◄── mitigator via mitigationmcpserv
                                (low_confidence + resolved episodes, unique UUIDs)

Run N+1 starts
    │
    └── root_cause_finder searches:
          incident_summaries  → "last time: connection pool, fixed by restarting pgBouncer"
          feedback_history    → "avoid: CPU spike hypothesis (low_confidence)
                                 prefer: memory leak hypothesis (resolved)"
```

---

## Qdrant Vector Collections

| Collection | Written by | Read by | Purpose |
|---|---|---|---|
| `mitigation_workflows` | seed_data.py | root_cause_finder, mitigationmcpserv | Runbook steps per node type |
| `rca_documents` | seed_data.py | root_cause_finder | Past RCA write-ups |
| `change_context` | seed_data.py | root_cause_finder | Recent deploy / config changes |
| `incident_summaries` | communicator, summarizer | root_cause_finder | Progressive incident record |
| `feedback_history` | mitigationmcpserv | root_cause_finder | Low-confidence + resolved feedback episodes |

All collections use **768-dim cosine similarity** vectors via `nomic-embed-text` (Ollama).

Summary upserts use a **deterministic point ID** (`uuid5("incident-summary:<incident_id>")`) so the communicator's partial updates are overwritten by the summarizer's final record — one point per incident.

Feedback history uses **random UUIDs** — every episode is a unique point, building a growing log of what failed and what worked.

---

## Shared State

All agents read and write a single `AgentState` TypedDict:

| Field | Type | Set by |
|---|---|---|
| `phase` | `str` | Every agent — supervisor routes on this |
| `session_id` | `str` | trigger.py |
| `feedback_iteration` | `int` | Mitigator (feedback loop counter) |
| `incident_id` | `str` | Incident Detector |
| `severity` | `str` | Incident Detector |
| `anomaly_nodes` | `list[dict]` | Incident Detector |
| `root_causes` | `list[dict]` | Root Cause Finder |
| `mitigation_workflows` | `list[dict]` | Root Cause Finder |
| `mitigation_confidence` | `float` | Incident Mitigator |
| `communications_sent` | `list[dict]` | Incident Communicator (one record per event) |
| `incident_summary` | `str` | Incident Summarizer |
| `messages` | `Annotated[list, add_messages]` | All agents (accumulates) |
| `feedback_request` | `str` | Mitigator → Root Cause Finder |
| `communication_event` | `str` | Each agent before → communicator |
| `next_phase` | `str` | Each agent before → communicator |

### Communication routing fields

Before routing to `phase="communicate"`, every agent sets two extra fields:

| `communication_event` | Set by | `next_phase` |
|---|---|---|
| `"incident_detected"` | incident_detector | `"root_cause"` |
| `"root_cause_found"` | root_cause_finder | `"mitigate"` |
| `"mitigation_complete"` | incident_mitigator | `"summarize"` |

The communicator reads both, dispatches notifications, then returns `phase=next_phase`.

---

## Supervisor Routing

The supervisor is a **deterministic routing function** (no LLM calls). It reads the `phase` field and returns the next node name:

| `phase` | next node |
|---|---|
| `detect` | `incident_detector` |
| `root_cause` | `root_cause_finder` |
| `mitigate` | `incident_mitigator` |
| `feedback` | `root_cause_finder` |
| `communicate` | `incident_communicator` |
| `summarize` | `incident_summarizer` |
| anything else | `END` |

---

## Agents

### Phase 1 — Incident Detector (`agents/incident_detector.py`)

**Model:** Qwen 2.5 7B (`qwen2.5:7b`, `format=json`)

**MCP tools used (graphmcpserv):**
- `list_anomalies` — fetches active anomalies from Neo4j
- `create_incident_ticket` — writes an `IncidentTicket` node to Neo4j

**Logic:**
1. Calls `list_anomalies(status="active", limit=50)`
2. LLM classifies: `{"is_incident": bool, "severity": "SEV1-4", "reasoning": "..."}`
3. If `is_incident=true`, generates `INC-<uuid>` and calls `create_incident_ticket`
4. Sets `phase="communicate"`, `communication_event="incident_detected"`, `next_phase="root_cause"`

**Output state:** `phase`, `incident_id`, `severity`, `anomaly_nodes`, `communication_event`, `next_phase`, `messages`

---

### Communicator — Incident Communicator (`agents/incident_communicator.py`)

**Model:** none (deterministic, no LLM)

**MCP tools used (commsmcpserv):** `send_email`, `send_slack`, `send_teams`, `send_sms`, `page_oncall`, `update_ticket_comms`

**Called after:** incident_detected · root_cause_found · mitigation_complete

**Logic:**
1. Reads `communication_event` and `next_phase` from state
2. Selects channels based on `severity`:
   - SEV1 → `page_oncall` + `send_slack` + `send_email`
   - SEV2 → `send_slack` + `send_teams` + `send_email`
   - SEV3 → `send_email` + `send_teams`
   - SEV4 → `send_email`
3. Builds event-appropriate subject/title/body from state
4. Dispatches to each selected channel via commsmcpserv tools
5. Calls `update_ticket_comms` with event type + list of channels dispatched
6. Upserts rolling partial summary to Qdrant `incident_summaries` (best-effort)
7. Prints formatted banner to stdout (🚨 / 🔍 / 🔧)
8. Appends record to `communications_sent` in state
9. Returns `phase=next_phase`, clears `communication_event` and `next_phase`

**Graceful degradation:** if a channel tool fails, remaining channels still dispatch. Falls back to print-only mode if no comms tools are wired.

**Qdrant writes:** `incident_summaries` (deterministic point ID — same doc updated 3×)

---

### Phase 2 — Root Cause Finder (`agents/root_cause_finder.py`)

**Model:** Llama 3.1 8B (`llama3.1:8b`, `format=json`)

**MCP tools used (graphmcpserv):**
- `get_relationships` — resolves Anomaly → DETECTED_ON → Node
- `root_cause_analysis` — downstream graph traversal from affected node
- `blast_radius` — count of services depending on the affected node
- `get_change_tickets` — recent change events from Neo4j
- `get_rca_tickets` — historical RCA tickets from Neo4j

**Qdrant collections searched:**

| Collection | Purpose |
|---|---|
| `rca_documents` | Past RCA write-ups for similar anomaly patterns |
| `change_context` | Recent deploys / config changes |
| `incident_summaries` | Summaries from previous runs (spot recurring patterns) |
| `feedback_history` | Past low-confidence and resolved feedback episodes |
| `mitigation_workflows` | Pre-fetches top-3 runbooks for the leading hypothesis |

**Logic:**
1. For each anomaly → `get_relationships` → find the node it is `DETECTED_ON`
2. `root_cause_analysis` (maxDepth=5) and `blast_radius` from that node
3. `get_change_tickets(limit=20)` and `get_rca_tickets(limit=10)`
4. Qdrant semantic search across all 4 context collections
5. LLM synthesises all evidence → ranked hypotheses with confidence scores
6. Qdrant search `mitigation_workflows` using top hypothesis text
7. Sets `phase="communicate"`, `communication_event="root_cause_found"`, `next_phase="mitigate"`

**On feedback pass:** reads `feedback_request` from state and includes it in the LLM context alongside `feedback_history` from Qdrant, directing the model to avoid dead-end hypotheses.

**Output state:** `phase`, `root_causes`, `mitigation_workflows`, `feedback_request` (cleared), `communication_event`, `next_phase`, `messages`

---

### Phase 3 — Incident Mitigator (`agents/incident_mitigator.py`)

**Model:** none (deterministic)

**MCP tools used (mitigationmcpserv):** `execute_mitigation_step`, `check_mitigation_status`, `store_mitigation_feedback`

**Qdrant writes:** `feedback_history` (via `store_mitigation_feedback`)

**Logic:**
1. Prints root cause hypotheses and matched workflows
2. Calls `execute_mitigation_step` for each step of the top workflow, tracking `run_id`
3. Calls `check_mitigation_status(run_id)` to verify the run
4. Computes `confidence` = top workflow score (or max root-cause confidence if no workflows)
5. **If** `confidence < CONFIDENCE_THRESHOLD` **and** `feedback_iteration < MAX_FEEDBACK_ITERATIONS`:
   - Calls `store_mitigation_feedback(outcome="low_confidence")` → Qdrant + log
   - Returns `phase="feedback"` with `feedback_request` message
6. **Otherwise** (confident or max iterations reached):
   - If prior feedback was used: calls `store_mitigation_feedback(outcome="resolved")` → Qdrant + log
   - Returns `phase="communicate"`, `communication_event="mitigation_complete"`, `next_phase="summarize"`

**Fallback:** if mitigationmcpserv tools are unavailable, falls back to direct Qdrant writes and prints steps without executing them.

**Feedback history payload (stored via MCP):**

```json
{
  "incident_id": "INC-XXXXXXXX",
  "workflow_id": "WF-001",
  "run_id": "abc123",
  "outcome": "low_confidence" | "resolved",
  "notes": "...",
  "query_text": "Connection pool exhausted node db-001 type Storage",
  "recorded_at": "2026-04-10T09:41:00Z",
  "source": "mitigation-mcp"
}
```

---

### Phase 5 — Incident Summarizer (`agents/incident_summarizer.py`)

**Model:** none (deterministic — built from state, no LLM)

**Qdrant writes:** `incident_summaries` (final authoritative upsert, same point ID as communicator)

**Logic:**
1. Composes a complete plain-text summary from the full `AgentState`:
   - Anomalies detected (count + details)
   - Root cause hypotheses (ranked by confidence)
   - Mitigation workflows matched (with steps)
   - Confidence score and feedback loop count
   - Communications sent (channels + events)
2. Prints 📋 INCIDENT SUMMARY to stdout
3. Upserts to Qdrant `incident_summaries` with rich payload (point ID = `uuid5("incident-summary:<incident_id>")`)
4. Sets `phase="done"`, `incident_summary=<text>`

**Qdrant payload:**

```json
{
  "incident_id": "INC-XXXXXXXX",
  "severity": "SEV2",
  "session_id": "...",
  "summary": "INCIDENT SUMMARY — ...",
  "anomaly_count": 2,
  "root_cause_count": 3,
  "workflow_count": 2,
  "mitigation_confidence": 0.83,
  "feedback_iterations": 1,
  "completed_at": "2026-04-10T09:45:00Z"
}
```

---

## Project Structure

```
autoincrespagent/
├── Dockerfile                               # Build context: incident-response/
├── pyproject.toml                           # Package + dependency config
├── .env.example                             # Environment variable template
├── trigger.py                               # CLI entry point
│
├── autoincrespagent/                        # Python package
│   ├── config.py                            # Settings (reads .env via pydantic-settings)
│   ├── llm/
│   │   └── factory.py                       # get_llm(agent_name) → ChatOllama
│   ├── agents/
│   │   ├── state.py                         # AgentState TypedDict (shared)
│   │   ├── supervisor.py                    # Deterministic phase → node router
│   │   ├── incident_detector.py             # Phase 1 — anomaly classification
│   │   ├── incident_communicator.py         # Cross-phase — commsmcpserv dispatch
│   │   ├── root_cause_finder.py             # Phase 2 — graph + RAG + LLM synthesis
│   │   ├── incident_mitigator.py            # Phase 3 — mitigationmcpserv execution
│   │   └── incident_summarizer.py           # Phase 5 — final summary + Qdrant upsert
│   ├── graph/
│   │   ├── mcp_client.py                    # MultiServerMCPClient (3 servers)
│   │   └── workflow.py                      # StateGraph assembly + tool routing
│   ├── memory/
│   │   └── mysql_saver.py                   # LangGraph BaseCheckpointSaver → MySQL
│   └── vector/
│       └── qdrant_search.py                 # Shared async Qdrant search helper
│
├── sql/
│   └── schema.sql                           # MySQL DDL (sessions, checkpoints, etc.)
│
└── tests/
    ├── conftest.py
    └── unit/
        ├── test_supervisor.py               # 12 routing tests
        ├── test_detector.py                 # 18 incident detector tests
        ├── test_root_cause_finder.py        # 17 root cause finder tests
        ├── test_incident_mitigator.py       # 31 mitigator + MCP + feedback tests
        ├── test_incident_communicator.py    # 34 communicator + MCP + channel tests
        └── test_incident_summarizer.py      # 12 summarizer + Qdrant persistence tests
```

**Sibling MCP server packages (spawned as subprocesses):**

```
../graphmcpserv/        # 10 Neo4j tools — wraps graphserv REST API
../mitigationmcpserv/   # 4 mitigation tools — workflow search + stub execution
../commsmcpserv/        # 6 comms tools — email, Slack, Teams, SMS, PagerDuty, ticket
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `langgraph` | ≥1.0 | StateGraph orchestration + checkpointing |
| `langchain` | ≥1.0 | Tool calling, prompts, message types |
| `langchain-core` | ≥1.0 | Base types (BaseTool, messages, RunnableConfig) |
| `langchain-ollama` | ≥1.0 | ChatOllama + OllamaEmbeddings |
| `langchain-mcp-adapters` | ≥0.2 | Converts MCP tools → LangChain BaseTool |
| `mcp` | ≥1.0 | MCP protocol SDK (spawns MCP server subprocesses) |
| `aiomysql` | ≥0.2 | Async MySQL driver for checkpoint saver |
| `qdrant-client` | ≥1.7 | Async Qdrant vector search (`AsyncQdrantClient`) |
| `pydantic-settings` | ≥2.0 | Environment variable loading |
| `python-dotenv` | ≥1.0 | `.env` file support |

**MCP servers** run from their own source directories via `cwd` — no editable install into the agent venv required.

---

## Configuration

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|---|---|---|
| `GRAPHSERV_URL` | `http://localhost:8080` | graphserv REST API base URL |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API base URL |
| `QDRANT_HOST` | `localhost` | Qdrant host |
| `QDRANT_PORT` | `6333` | Qdrant port |
| `MYSQL_HOST` | `127.0.0.1` | MySQL host |
| `MYSQL_PORT` | `3306` | MySQL port |
| `MYSQL_USER` | `ir_user` | MySQL username |
| `MYSQL_PASSWORD` | *(empty)* | MySQL password |
| `MYSQL_DATABASE` | `incident_response` | MySQL database name |
| `GRAPHMCPSERV_PATH` | `../graphmcpserv` | Path to graphmcpserv source directory |
| `MITIGATIONMCPSERV_PATH` | `../mitigationmcpserv` | Path to mitigationmcpserv source directory |
| `COMMSMCPSERV_PATH` | `../commsmcpserv` | Path to commsmcpserv source directory |
| `CONFIDENCE_THRESHOLD` | `0.75` | Mitigation confidence below which feedback loop triggers |
| `MAX_FEEDBACK_ITERATIONS` | `3` | Max Mitigator ↔ Root Cause Finder cycles |
| `POLL_INTERVAL_SECONDS` | `60` | Seconds between anomaly polls |

---

## Setup

### Prerequisites

| Service | Purpose | Check |
|---|---|---|
| Neo4j 5.x | Topology graph | `curl http://localhost:7474` |
| graphserv | REST API for Neo4j | `curl http://localhost:8080/health` |
| Ollama | Local LLM runtime | `ollama list` |
| MySQL 8 | Session checkpoints | `mysql -u ir_user -p` |
| Qdrant | Vector database | `curl http://localhost:6333/healthz` |

### Install

```bash
cd autoincrespagent

python3 -m venv .venv
source .venv/bin/activate

# Install the agent package only — MCP servers run from their own directories
pip install -e .

cp .env.example .env
# edit .env with your credentials
```

> The three MCP servers (`graphmcpserv`, `mitigationmcpserv`, `commsmcpserv`) do **not** need to be installed into this venv. The agent spawns each one with `cwd` set to its source directory, so Python resolves the `mcp_servers.*` namespace directly from there.

### Apply MySQL schema

```bash
mysql -u root -p incident_response < sql/schema.sql
```

### Pull required Ollama models

```bash
ollama pull qwen2.5:7b          # Incident Detector
ollama pull llama3.1:8b         # Root Cause Finder
ollama pull nomic-embed-text    # Embeddings (768-dim) for all Qdrant writes
```

### Seed Qdrant vector database

```bash
cd ../qdrant_data

# Requires Qdrant and Ollama to be running
python seed_data.py
```

This creates and populates all 5 Qdrant collections:

```
mitigation_workflows  5 runbooks  (DB pool, memory leak, network, disk I/O, CPU)
rca_documents         5 RCAs      (PaymentService, OrderService, NetworkVPC, InventoryDB, AuthService)
incident_summaries    5 summaries (matching each RCA)
change_context        6 entries   (deploys and config changes)
feedback_history      0 points    (empty — populated at runtime by the mitigator)
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

The three MCP servers start automatically when `trigger.py` runs. You will see:

```
INFO:mcp.server.lowlevel.server:Processing request of type ListToolsRequest   (×3)
INFO trigger: loaded 20 MCP tools: ['list_anomalies', ..., 'send_email', ...]
```

### Inject a test anomaly

```bash
curl -X POST http://localhost:8080/anomalies \
  -H "Content-Type: application/json" \
  -d '{
    "id": "ANO-TEST-001",
    "type": "latency_spike",
    "severity": "high",
    "status": "active",
    "description": "Test latency spike on payment service"
  }'
```

### Example output (full run)

```
════════════════════════════════════════════════════════════════════════
  🚨  INCIDENT DECLARED
════════════════════════════════════════════════════════════════════════
  Incident ID  :  INC-17FEA9A4
  Severity     :  SEV2
  Anomalies    :  1 active
                  • latency_spike [high]
  Channels     :  send_slack, send_teams, send_email
  → Next       :  Root Cause Analysis
────────────────────────────────────────────────────────────────────────

[SLACK] 🟠 INC-17FEA9A4 → #incidents | Incident Detected — INC-17FEA9A4 [SEV2]
[TEAMS] INC-17FEA9A4 | Incident Detected — INC-17FEA9A4 [SEV2] [warning]
[EMAIL] INC-17FEA9A4 → oncall@example.com | [SEV2] Incident Detected: INC-17FEA9A4
[TICKET] INC-17FEA9A4 [incident_detected] channels=send_slack,send_teams,send_email

════════════════════════════════════════════════════════════════════════
  🔍  ROOT CAUSE IDENTIFIED
════════════════════════════════════════════════════════════════════════
  Incident ID  :  INC-17FEA9A4  [SEV2]
  Top Cause    :  Connection pool exhausted on payment-db
  Node         :  payment-db-001 (Storage)
  Confidence   :  88%
  Workflows    :  3 matching workflow(s)
  Channels     :  send_slack, send_teams, send_email
  → Next       :  Mitigation
────────────────────────────────────────────────────────────────────────

  ────────────────────────────────────────────────────────────────────────
  INCIDENT MITIGATOR  —  INC-17FEA9A4  [SEV2]
  ...
  [CONFIDENCE]  0.87  (threshold: 0.75)
  [EXECUTING — WF-001: Restart DB Connection Pool]
    step 1: Check active connection count  [completed]
    step 2: Increase max_connections       [completed]
    step 3: Restart application pods       [completed]
    Run abc-123: 3/3 steps completed

════════════════════════════════════════════════════════════════════════
  🔧  MITIGATION EXECUTED
════════════════════════════════════════════════════════════════════════
  Incident ID  :  INC-17FEA9A4  [SEV2]
  Workflow     :  WF-001 — Restart DB Connection Pool
  Confidence   :  0.87
  Channels     :  send_slack, send_teams, send_email
  → Next       :  Incident Summarization
────────────────────────────────────────────────────────────────────────
```

### Comms log files

After a run, check what was "sent":

```bash
cat logs/slack_outbox.log   | python3 -m json.tool
cat logs/email_outbox.log   | python3 -m json.tool
cat logs/ticket_comms.log   | python3 -m json.tool
cat logs/mitigation_runs.log | python3 -m json.tool
```

### Force feedback loop (for testing)

```bash
# .env — lower threshold to force loop
CONFIDENCE_THRESHOLD=0.99

python trigger.py
# mitigator will request feedback 1–3 times before advancing
```

### Fallback behaviour

| Service unavailable | Behaviour |
|---|---|
| MySQL | Falls back to in-memory checkpointer — no persistence |
| Qdrant | Vector search skipped gracefully — agents still run |
| graphserv | Detector logs error and exits cleanly with `phase=done` |
| mitigationmcpserv unavailable | Mitigator falls back to direct Qdrant writes + print-only execution |
| commsmcpserv unavailable | Communicator falls back to print-only mode |
| Individual comms tool fails | Remaining channels still dispatch (partial failure resilience) |
| LLM bad JSON | Treated as no-incident (detector) / empty hypotheses (RCA) |

---

## Testing

```bash
source .venv/bin/activate
pytest tests/unit/ -v
```

### Test coverage

| Test file | Tests | What's covered |
|---|---|---|
| `test_supervisor.py` | 12 | All phase → node routing, unknown/empty/missing phase → END |
| `test_detector.py` | 18 | No anomalies, no incident, incident declared, error handling |
| `test_root_cause_finder.py` | 17 | Factory, graph traversal, Qdrant search (5 collections), feedback |
| `test_incident_mitigator.py` | 31 | Confidence routing, MCP execution, feedback via MCP + direct Qdrant fallback |
| `test_incident_communicator.py` | 34 | Channel selection (SEV1-4), tool args, communications_sent, graceful degradation |
| `test_incident_summarizer.py` | 12 | Summary content, Qdrant persistence, deterministic point ID |
| **Total** | **124** | |

### Unit test approach

- MCP tools mocked with `unittest.mock.AsyncMock` — no running servers needed
- LLM injected via `make_<agent>(tools, llm=fake_llm)` — no Ollama needed
- Qdrant client mocked — `client.query_points`, `client.upsert`, `client.get_collections`
- All tests run offline in < 1 second

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
docker compose run --rm ollama ollama pull nomic-embed-text

# Seed Qdrant (after stack is up)
docker compose up -d qdrant ollama
docker compose run --rm agent python /app/qdrant_data/seed_data.py

# Start all services
docker compose up -d

# Watch agent logs
docker compose logs -f agent

# Stop everything
docker compose down
```

### Services

| Service | Image | Ports | Memory |
|---|---|---|---|
| `neo4j` | neo4j:5-community | 7474, 7687 | 1 GB |
| `graphserv` | rajeshkurup77/graphserv:latest | 8080 | 256 MB |
| `mysql` | mysql:8 | 3306 | 512 MB |
| `qdrant` | qdrant/qdrant | 6333 | 512 MB |
| `ollama` | ollama/ollama | 11434 | 8 GB |
| `agent` | local build | — | 2 GB |

> **Note:** Default password is `changeme` in `docker-compose.yml`. Change before any non-local use.
