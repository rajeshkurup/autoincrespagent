"""MySQL-backed LangGraph checkpoint saver.

Persists LangGraph state checkpoints to the `checkpoints` table so
incident runs can be replayed and recovered after failures.

Prerequisites:
  - MySQL table created via sql/schema.sql
  - aiomysql pool passed at construction time
"""

import json
import logging
import pickle
from typing import Any, AsyncIterator, Iterator, Optional, Sequence, Tuple

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)

logger = logging.getLogger(__name__)


class MySQLSaver(BaseCheckpointSaver):
    """Async checkpoint saver backed by MySQL.

    Args:
        pool: An aiomysql connection pool created by aiomysql.create_pool().
    """

    def __init__(self, pool) -> None:
        super().__init__()
        self._pool = pool

    # ── Async interface (used by graph.ainvoke) ───────────────────────

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Any,
    ) -> RunnableConfig:
        thread_id    = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]
        parent_id    = config["configurable"].get("checkpoint_id")

        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    INSERT INTO checkpoints
                        (thread_id, checkpoint_ns, checkpoint_id,
                         parent_checkpoint_id, type, checkpoint, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                        checkpoint = VALUES(checkpoint),
                        metadata   = VALUES(metadata)
                    """,
                    (
                        thread_id,
                        checkpoint_ns,
                        checkpoint_id,
                        parent_id,
                        "pickle",
                        pickle.dumps(checkpoint),
                        pickle.dumps(metadata),
                    ),
                )
                await conn.commit()

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        thread_id    = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"].get("checkpoint_id")

        async with self._pool.acquire() as conn:
            async with conn.cursor() as cur:
                if checkpoint_id:
                    await cur.execute(
                        """
                        SELECT checkpoint, metadata, parent_checkpoint_id
                        FROM checkpoints
                        WHERE thread_id=%s AND checkpoint_ns=%s AND checkpoint_id=%s
                        """,
                        (thread_id, checkpoint_ns, checkpoint_id),
                    )
                else:
                    await cur.execute(
                        """
                        SELECT checkpoint, metadata, parent_checkpoint_id
                        FROM checkpoints
                        WHERE thread_id=%s AND checkpoint_ns=%s
                        ORDER BY checkpoint_id DESC
                        LIMIT 1
                        """,
                        (thread_id, checkpoint_ns),
                    )
                row = await cur.fetchone()

        if row is None:
            return None

        checkpoint = pickle.loads(row[0])
        metadata   = pickle.loads(row[1])
        parent_id  = row[2]

        parent_config: Optional[RunnableConfig] = None
        if parent_id:
            parent_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": parent_id,
                }
            }

        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint["id"],
                }
            },
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
        )

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[dict] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        # Minimal implementation — returns nothing (sufficient for Phase 1).
        # Full history iteration will be added when replay is needed.
        return
        yield  # make this an async generator

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        # Pending writes are not persisted in this implementation.
        pass

    # ── Sync interface (required by BaseCheckpointSaver) ─────────────
    # These are not used by async graph execution but must be present.

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        raise NotImplementedError("Use async methods: aget_tuple")

    def list(self, config, **kwargs) -> Iterator[CheckpointTuple]:
        raise NotImplementedError("Use async methods: alist")

    def put(self, config, checkpoint, metadata, new_versions) -> RunnableConfig:
        raise NotImplementedError("Use async methods: aput")

    def put_writes(self, config, writes, task_id, task_path="") -> None:
        raise NotImplementedError("Use async methods: aput_writes")

    def delete_thread(self, config: RunnableConfig) -> None:
        raise NotImplementedError("Use async methods: adelete_thread")

    async def adelete_thread(self, config: RunnableConfig) -> None:
        pass
