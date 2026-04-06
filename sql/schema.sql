-- Incident response database schema
-- Apply with: mysql -u root -p incident_response < sql/schema.sql

CREATE DATABASE IF NOT EXISTS incident_response
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE incident_response;

-- ── Sessions ─────────────────────────────────────────────────────────
-- One row per incident response run.
CREATE TABLE IF NOT EXISTS sessions (
    id           VARCHAR(36)  NOT NULL PRIMARY KEY,   -- UUID session_id
    created_at   DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at   DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    status       ENUM('running', 'completed', 'failed', 'escalated')
                              NOT NULL DEFAULT 'running',
    incident_id  VARCHAR(64),
    trigger_node VARCHAR(128),
    severity     VARCHAR(8),
    summary      TEXT
) ENGINE=InnoDB;

-- ── LangGraph Checkpoints ─────────────────────────────────────────────
-- Serialised state at each node boundary for replay and recovery.
CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id            VARCHAR(36)   NOT NULL,
    checkpoint_ns        VARCHAR(128)  NOT NULL DEFAULT '',
    checkpoint_id        VARCHAR(36)   NOT NULL,
    parent_checkpoint_id VARCHAR(36),
    type                 VARCHAR(16)   NOT NULL DEFAULT 'pickle',
    checkpoint           LONGBLOB      NOT NULL,
    metadata             LONGBLOB,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
) ENGINE=InnoDB;

-- ── Agent Memory ──────────────────────────────────────────────────────
-- Long-term facts extracted by agents for future runs.
CREATE TABLE IF NOT EXISTS agent_memory (
    id          BIGINT       NOT NULL AUTO_INCREMENT PRIMARY KEY,
    created_at  DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    session_id  VARCHAR(36),
    agent_name  VARCHAR(64)  NOT NULL,
    memory_type ENUM('fact', 'pattern', 'preference', 'feedback')
                             NOT NULL,
    content     TEXT         NOT NULL,
    context_key VARCHAR(256),
    confidence  FLOAT        NOT NULL DEFAULT 1.0,
    INDEX idx_agent_memory_session (session_id),
    INDEX idx_agent_memory_agent  (agent_name)
) ENGINE=InnoDB;

-- ── Feedback Log ─────────────────────────────────────────────────────
-- Audit trail for Mitigator ↔ Root Cause Finder feedback cycles.
CREATE TABLE IF NOT EXISTS feedback_log (
    id          BIGINT      NOT NULL AUTO_INCREMENT PRIMARY KEY,
    created_at  DATETIME    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    session_id  VARCHAR(36) NOT NULL,
    from_agent  VARCHAR(64) NOT NULL,
    to_agent    VARCHAR(64) NOT NULL,
    feedback_msg TEXT       NOT NULL,
    iteration   INT         NOT NULL DEFAULT 1,
    INDEX idx_feedback_session (session_id)
) ENGINE=InnoDB;
