# Build context must be the incident-response/ parent directory.
# docker-compose sets this automatically via the context: .. setting.
#
# Manual build (from incident-response/):
#   docker build -f autoincrespagent/Dockerfile -t autoincrespagent .

FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install graphmcpserv — the agent spawns it as an MCP stdio subprocess,
# so both packages must live in the same image.
COPY graphmcpserv/pyproject.toml /tmp/graphmcpserv/
COPY graphmcpserv/mcp_servers/   /tmp/graphmcpserv/mcp_servers/
RUN pip install --no-cache-dir /tmp/graphmcpserv && rm -rf /tmp/graphmcpserv

# Install autoincrespagent
COPY autoincrespagent/pyproject.toml .
COPY autoincrespagent/autoincrespagent/ autoincrespagent/
RUN pip install --no-cache-dir .

# Entry point
COPY autoincrespagent/trigger.py .

# Default env — override via docker-compose or -e flags
ENV GRAPHSERV_URL=http://graphserv:8080
ENV OLLAMA_BASE_URL=http://ollama:11434
ENV MYSQL_HOST=mysql
ENV MYSQL_PORT=3306
ENV MYSQL_USER=ir_user
ENV MYSQL_PASSWORD=changeme
ENV MYSQL_DATABASE=incident_response
ENV CONFIDENCE_THRESHOLD=0.75
ENV MAX_FEEDBACK_ITERATIONS=3
ENV POLL_INTERVAL_SECONDS=60

RUN useradd -m agent
USER agent

CMD ["python", "trigger.py", "--poll"]
