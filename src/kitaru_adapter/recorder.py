# Kitaru session lifecycle for one recorded turn.
#
# Same order as Kitaru's released adapters: create the session and an
# in-progress root before the app runs, buffer child nodes while it runs
# (sync, no network), then flush them, replace the root, and update the
# session to the same terminal state. Each write is a separate request, so a
# failure part-way leaves a partial session that is marked failed if possible.
import asyncio
import uuid
from datetime import UTC, datetime

from kitaru.api_models.v1.session import (
    SessionCreateRequest,
    SessionOrigin,
    SessionStatus,
    SessionUpdateRequest,
)
from kitaru.api_models.v1.session_node import (
    NodeStatus,
    NodeType,
    SessionNodeBatchRequest,
    SessionNodeCreateRequest,
)

from src.kitaru_adapter.capture import capture, capture_error

FRAMEWORK = "openai-responses"
ADAPTER_VERSION = "0.2.0"    # 0.2.0: pipeline steps recorded as spans
ROOT_ID = "turn"
BATCH_SIZE = 100
# A failure write must not hang the worker after the turn already failed.
FAIL_TIMEOUT_SECONDS = 30


class KitaruRecordingError(RuntimeError):
    """
    Kitaru could not record the turn. phase is "start" (before the app ran;
    retrying is safe) or "finalize" (after it ran; result holds the answer,
    and retrying would repeat model and tool calls).
    """

    def __init__(self, *, phase, session_id, result=None):
        self.phase = phase
        self.session_id = session_id
        self.result = result
        self.retry_safe = phase == "start"
        self.side_effects_possible = phase != "start"
        super().__init__(f"Kitaru recording failed during {phase}"
                         + ("" if self.retry_safe else "; the turn already ran, do not retry blindly"))


def now():
    return datetime.now(UTC)


class Recorder:
    """One turn's session: its id, root, and the buffer of child nodes."""

    def __init__(self, client, secrets):
        self.client = client
        self.secrets = secrets
        self.session_id = None
        self.started_at = None
        self.root_inputs = None
        self.buffer = []
        self.confirmed = 0    # child nodes the server has stored

    def capture(self, value):
        return capture(value, self.secrets)

    async def start(self, *, inputs, input_text_selector, root_inputs, replay, name=None,
                    agent_id=None, agent_version_id=None, metadata=None):
        self.started_at = now()
        self.root_inputs = self.capture(root_inputs)
        session = await self.client.sessions.create(SessionCreateRequest(
            agent_id=agent_id,
            agent_version_id=agent_version_id,
            metadata=self.capture(metadata or {}),
            origin=SessionOrigin.REPLAY if replay else SessionOrigin.RECORDED,
            status=SessionStatus.IN_PROGRESS,
            name=name,
            inputs=self.capture(inputs),
            outputs=None,
            input_text_selector=input_text_selector,
            output_text_selector="/answer",
            started_at=self.started_at,
            framework=FRAMEWORK,
            adapter_version=ADAPTER_VERSION,
        ))
        self.session_id = session.id
        await self._write_root(NodeStatus.IN_PROGRESS)
        return session

    def add_node(self, *, node_type, name, parent, started_at, inputs, outputs,
                 status=NodeStatus.COMPLETED, error=None, external_id=None, **fields):
        """Buffer one child node; returns its external id for children of its own."""
        external_id = external_id or uuid.uuid4().hex
        self.buffer.append(SessionNodeCreateRequest(
            external_id=external_id,
            parent_external_id=parent,
            node_type=node_type,
            name=name,
            status=status,
            error=error,
            started_at=started_at,
            ended_at=now(),
            inputs=inputs,
            outputs=outputs,
            attributes=fields.pop("attributes", None) or {},
            **fields,
        ))
        return external_id

    async def flush(self):
        while self.buffer:
            batch = self.buffer[:BATCH_SIZE]
            await self.client.sessions.ingest_nodes(self.session_id, SessionNodeBatchRequest(nodes=batch))
            del self.buffer[:len(batch)]
            self.confirmed += len(batch)

    async def complete(self, answer):
        outputs = {"answer": self.capture(answer)}
        await self.flush()
        ended_at = now()
        await self._write_root(NodeStatus.COMPLETED, outputs=outputs, ended_at=ended_at)
        await self.client.sessions.update(self.session_id, SessionUpdateRequest(
            status=SessionStatus.COMPLETED, outputs=outputs, ended_at=ended_at))

    async def fail(self, error):
        """Best effort: keep what was buffered, mark root and session failed."""
        if self.session_id is None:
            return
        text = capture_error(error, self.secrets)
        first = None
        try:
            await self.flush()
            await self._write_root(NodeStatus.FAILED, error=text, ended_at=now())
        except Exception as failure:
            first = failure
        try:
            await self.client.sessions.update(self.session_id, SessionUpdateRequest(
                status=SessionStatus.FAILED, error=text, ended_at=now()))
        except Exception as failure:
            first = first or failure
        if first is not None:
            raise first

    async def fail_quietly(self, error):
        """Record a failure without masking it: problems become notes on error."""
        try:
            await asyncio.wait_for(self.fail(error), FAIL_TIMEOUT_SECONDS)
        except Exception as failure:
            error.add_note(f"Kitaru could not mark session {self.session_id} failed "
                           f"({type(failure).__name__}); {self.confirmed} child nodes were stored")

    async def _write_root(self, status, outputs=None, error=None, ended_at=None):
        await self.client.sessions.ingest_nodes(self.session_id, SessionNodeBatchRequest(nodes=[
            SessionNodeCreateRequest(
                external_id=ROOT_ID,
                parent_external_id=None,
                node_type=NodeType.SPAN,
                name="turn",
                status=status,
                error=error,
                started_at=self.started_at,
                ended_at=ended_at,
                input_text_selector="/question",
                output_text_selector="/answer" if outputs else None,
                inputs=self.root_inputs,
                outputs=outputs,
                attributes={},
            )
        ]))
