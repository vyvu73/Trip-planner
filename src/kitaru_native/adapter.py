# Native Kitaru recording and replay for one chat turn of the trip planner.
#
# Unlike the Braintrust importer-backed adapter (kitaru_replay.py), this one
# sits in the process: it wraps the OpenAI clients and the tool dispatcher the
# app reads as module globals, so a replay can apply model/prompt overrides
# and answer tool calls from a tool policy (static or recorded history).
#
# Boundary: one run per process, from a plain script with no running event
# loop (a worker replay task). The Kitaru client is async and the app is sync,
# so each write runs in its own asyncio.run(), the same pattern as Kitaru's
# released ImporterBackedAdapter.run(). Not for use inside the Gradio app.
import asyncio
import json
import os
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime

from kitaru.api_models.v1.replay import ToolLookupRequest
from kitaru.api_models.v1.replay_config import (
    HistoryConfig,
    HistoryScope,
    LLMConfig,
    PassthroughConfig,
    StaticConfig,
    StaticMatchMode,
    ToolPolicyOnMiss,
)
from kitaru.api_models.v1.session import (
    SessionCreateRequest,
    SessionOrigin,
    SessionStatus,
    SessionUpdateRequest,
    TokenUsage,
)
from kitaru.api_models.v1.session_node import (
    NodeStatus,
    NodeType,
    SessionNodeBatchRequest,
    SessionNodeCreateRequest,
)
from kitaru.cache_keys import compute_tool_cache_key
from kitaru.client.api_client import KitaruAPIClient

ADAPTER_VERSION = "0.1.0"
FRAMEWORK = "openai-responses"
MODEL_PROVIDER = "openai"

# Overridable model settings, and the request settings worth recording.
# Nothing else from a request is recorded (no clients, headers, or keys).
OVERRIDE_PARAM_KEYS = ("reasoning", "temperature", "max_output_tokens")
RECORDED_PARAM_KEYS = (*OVERRIDE_PARAM_KEYS, "tool_choice", "stream")

# Recorded strings are cut at this length; a cut tool payload is marked so the
# server never serves it as a history result.
MAX_TEXT = 50_000
NODE_BATCH_SIZE = 500
SECRET_SUFFIXES = ("_KEY", "_TOKEN", "_SECRET", "_PASSWORD")


class ToolPolicyError(RuntimeError):
    """A tool call the replay's tool policy refused to answer."""


def now():
    return datetime.now(UTC)


def new_id():
    return uuid.uuid4().hex


# --- projection ---------------------------------------------------------------

def secret_values():
    """Values of credential-like environment variables, never to be recorded."""
    return [value for name, value in os.environ.items()
            if name.endswith(SECRET_SUFFIXES) and len(value) >= 8]


def redact(text, secrets):
    for secret in secrets:
        text = text.replace(secret, "[redacted]")
    return text


def project(value, secrets, depth=0):
    """
    Bounded JSON copy of a recorded value: strings redacted and cut,
    containers capped, anything else recorded by type name only.
    Returns (projected value, whether anything was cut).
    """
    if value is None or isinstance(value, (bool, int, float)):
        return value, False
    if isinstance(value, str):
        text = redact(value, secrets)
        return (text[:MAX_TEXT], True) if len(text) > MAX_TEXT else (text, False)
    if depth >= 6:
        return f"<{type(value).__name__}>", True
    if isinstance(value, dict):
        items = list(value.items())
        out, cut = {}, len(items) > 200
        for key, item in items[:200]:
            out[str(key)], item_cut = project(item, secrets, depth + 1)
            cut = cut or item_cut
        return out, cut
    if isinstance(value, (list, tuple)):
        out, cut = [], len(value) > 200
        for item in value[:200]:
            projected, item_cut = project(item, secrets, depth + 1)
            out.append(projected)
            cut = cut or item_cut
        return out, cut
    if hasattr(value, "model_dump"):
        return project(value.model_dump(mode="json"), secrets, depth + 1)
    return f"<{type(value).__name__}>", True


def usage_tokens(usage):
    if usage is None:
        return None
    input_details = getattr(usage, "input_tokens_details", None)
    output_details = getattr(usage, "output_tokens_details", None)
    return TokenUsage(
        input_tokens=getattr(usage, "input_tokens", None),
        output_tokens=getattr(usage, "output_tokens", None),
        cached_input_tokens=getattr(input_details, "cached_tokens", None),
        reasoning_tokens=getattr(output_details, "reasoning_tokens", None),
    )


def response_outputs(response):
    """The answer text and requested tool calls of one model response."""
    calls = [
        {"name": item.name, "call_id": item.call_id, "arguments": item.arguments}
        for item in getattr(response, "output", None) or []
        if getattr(item, "type", None) == "function_call"
    ]
    parsed = getattr(response, "output_parsed", None)
    outputs = {"output_text": getattr(response, "output_text", None) or "", "function_calls": calls}
    if parsed is not None:
        outputs["output_parsed"] = parsed
    return outputs


# --- Kitaru writes --------------------------------------------------------------

class KitaruWriter:
    """Sync facade over the async Kitaru client: one fresh client per call."""

    def __init__(self, client_factory=KitaruAPIClient):
        self._client_factory = client_factory

    def _run(self, operation):
        async def call():
            async with self._client_factory() as client:
                return await operation(client)
        return asyncio.run(call())

    def get_replay(self, replay_id):
        return self._run(lambda client: client.replays.get(replay_id))

    def create_session(self, request):
        return self._run(lambda client: client.sessions.create(request))

    def ingest(self, session_id, nodes):
        for start in range(0, len(nodes), NODE_BATCH_SIZE):
            batch = SessionNodeBatchRequest(nodes=nodes[start:start + NODE_BATCH_SIZE])
            self._run(lambda client, batch=batch: client.sessions.ingest_nodes(session_id, batch))

    def update_session(self, session_id, request):
        return self._run(lambda client: client.sessions.update(session_id, request))

    def tool_lookup(self, replay_id, request):
        return self._run(lambda client: client.replays.tool_lookup(replay_id, request))


# --- overrides ------------------------------------------------------------------

def overridden_model(requested, override):
    """A string override replaces every model; a map replaces matching ones."""
    if override is None or override.model is None:
        return requested
    if isinstance(override.model, str):
        return override.model
    return override.model.get(requested, requested)


# --- one run --------------------------------------------------------------------

class NativeRun:
    """
    State of one recorded invocation: the session, its buffered child nodes,
    the replay config, and per-key history occurrence counters. Children are
    buffered and written with the terminal root, so a run that dies midway
    leaves an in-progress or failed session, never a falsely complete one.
    """

    def __init__(self, writer, replay_id=None, override=None, tool_policy=None):
        self.writer = writer
        self.replay_id = replay_id
        self.override = override
        self.tool_policy = tool_policy
        self.secrets = secret_values()
        self.session_id = None
        self.root_id = new_id()
        self.started_at = None
        self.nodes = []
        self.last_llm_id = None
        self.occurrences = {}
        self.policy_failure = None

    def project(self, value):
        return project(value, self.secrets)

    def add_node(self, node_type, name, inputs, outputs, started_at, status=NodeStatus.COMPLETED,
                 error=None, parent_id=None, node_id=None, **fields):
        inputs, inputs_cut = self.project(inputs)
        outputs, outputs_cut = self.project(outputs)
        node_id = node_id or new_id()
        self.nodes.append(SessionNodeCreateRequest(
            external_id=node_id,
            parent_external_id=parent_id or self.root_id,
            node_type=node_type,
            name=name,
            status=status,
            error=redact(error, self.secrets)[:MAX_TEXT] if error else None,
            started_at=started_at,
            ended_at=now(),
            inputs=inputs,
            outputs=outputs,
            attributes={"inputs_bounded": inputs_cut, "outputs_bounded": outputs_cut},
            **fields,
        ))
        return node_id

    # Model calls

    def model_call(self, call, kwargs, role):
        """Apply overrides to one Responses API call and record it."""
        kwargs = dict(kwargs)
        requested = kwargs.get("model")
        kwargs["model"] = overridden_model(requested, self.override)
        if role == "answer" and self.override is not None:
            if self.override.system_prompt is not None:
                kwargs["instructions"] = self.override.system_prompt
            for key, value in (self.override.model_params or {}).items():
                if key in OVERRIDE_PARAM_KEYS:
                    kwargs[key] = value

        name = f"openai.responses.{'parse' if role == 'router' else 'create'}"
        inputs = {"instructions": kwargs.get("instructions"), "input": kwargs.get("input"),
                  "previous_response_id": kwargs.get("previous_response_id")}
        fields = {
            "requested_model": requested,
            "model_provider": MODEL_PROVIDER,
            "model_params": self.project({k: kwargs[k] for k in RECORDED_PARAM_KEYS if k in kwargs})[0],
            "input_text_selector": "/input" if isinstance(kwargs.get("input"), str) else None,
            "system_prompt_selector": "/instructions" if kwargs.get("instructions") else None,
            "output_text_selector": "/output_text",
            "metadata": {"role": role},
        }
        node_id = new_id()
        self.last_llm_id = node_id if role == "answer" else self.last_llm_id
        started_at = now()

        def record(response=None, error=None):
            self.add_node(
                NodeType.LLM_CALL, name, inputs,
                response_outputs(response) if response is not None else None,
                started_at, status=NodeStatus.FAILED if error else NodeStatus.COMPLETED,
                error=error, node_id=node_id,
                model=getattr(response, "model", None) or kwargs["model"],
                tokens=usage_tokens(getattr(response, "usage", None)), **fields,
            )

        if kwargs.get("stream"):
            return self._stream(call, kwargs, record)
        try:
            response = call(**kwargs)
        except BaseException as error:
            record(error=f"{type(error).__name__}: {error}")
            raise
        record(response)
        return response

    def _stream(self, call, kwargs, record):
        """Pass events through; the node is recorded once the stream ends."""
        response = None
        try:
            for event in call(**kwargs):
                if getattr(event, "type", None) == "response.completed":
                    response = event.response
                yield event
        except GeneratorExit:
            record(response, error="Stream abandoned before completion")
            raise
        except BaseException as error:
            record(response, error=f"{type(error).__name__}: {error}")
            raise
        if response is None:
            record(error="Stream ended without a response.completed event")
        else:
            record(response)

    # Tool calls

    def tool_call(self, name, arguments_json, execute):
        """
        Answer one tool call per the tool policy and record it. The node's
        inputs are the raw arguments string, as the Braintrust traced_tool
        span records it, so history keys match imported and native baselines.
        """
        started_at = now()
        config = self._tool_config(name)
        source = "live"
        try:
            if isinstance(config, PassthroughConfig):
                result = execute()
            else:
                result, source = self._policy_result(name, arguments_json, config)
                if source == "miss":
                    result, source = self._on_miss(name, config.on_miss, execute)
        except ToolPolicyError as error:
            self.policy_failure = self.policy_failure or error
            self.add_node(NodeType.TOOL_CALL, name, arguments_json, None, started_at,
                          status=NodeStatus.FAILED, error=str(error), tool_name=name,
                          parent_id=self.last_llm_id, metadata={"policy": config.type})
            raise
        self.add_node(NodeType.TOOL_CALL, name, arguments_json, result, started_at,
                      tool_name=name, parent_id=self.last_llm_id,
                      metadata={"policy": config.type, "source": source})
        return result

    def _tool_config(self, name):
        if self.tool_policy is None:
            return PassthroughConfig()
        return self.tool_policy.tools.get(name, self.tool_policy.default)

    def _policy_result(self, name, arguments_json, config):
        if isinstance(config, StaticConfig):
            arguments = json.loads(arguments_json)
            for case in config.cases:
                if static_matches(case, arguments):
                    return as_tool_text(case.result), "static"
            return None, "miss"
        if isinstance(config, HistoryConfig):
            return self._history_result(name, arguments_json, config)
        raise ToolPolicyError(f"Tool policy '{config.type}' is not supported for {name}")

    def _history_result(self, name, arguments_json, config):
        key = compute_tool_cache_key(name, arguments_json)
        baseline = config.scope is HistoryScope.BASELINE
        occurrence = self.occurrences.get(key, 0) if baseline else None
        response = self.writer.tool_lookup(self.replay_id, ToolLookupRequest(
            tool_name=name, cache_key=key, **({"occurrence": occurrence} if baseline else {}),
        ))
        match = response.match
        # A null result is the zero-length node wrap_openai records for the
        # model's request, not a tool result: treat it as a miss.
        if response.rejected_candidate or match is None or match.result is None:
            return None, "miss"
        if baseline:
            self.occurrences[key] = occurrence + 1
        if match.status == NodeStatus.FAILED:
            return f"Tool {name} failed: {match.error}", "history"
        return as_tool_text(match.result), "history"

    def _on_miss(self, name, on_miss, execute):
        if on_miss == ToolPolicyOnMiss.PASSTHROUGH:
            return execute(), "live"
        if on_miss == ToolPolicyOnMiss.ERROR_RESULT:
            return f"Tool {name} failed: no recorded result matches this call", "error_result"
        raise ToolPolicyError(f"No recorded result for {name}; the tool policy says to fail")

    # Pipeline steps

    def span(self, name, inputs, run):
        started_at = now()
        try:
            output = run()
        except BaseException as error:
            self.add_node(NodeType.SPAN, name, inputs, None, started_at,
                          status=NodeStatus.FAILED, error=f"{type(error).__name__}: {error}")
            raise
        self.add_node(NodeType.SPAN, name, inputs, output, started_at)
        return output


def static_matches(case, arguments):
    if case.match is None:
        return True
    if case.match_mode == StaticMatchMode.EXACT:
        return arguments == case.match
    return isinstance(case.match, dict) and all(
        key in arguments and arguments[key] == value for key, value in case.match.items()
    )


def as_tool_text(result):
    """The app's tools return text to the model; keep that contract."""
    return result if isinstance(result, str) else json.dumps(result)


# --- wrapped clients --------------------------------------------------------------

class RecordingResponses:
    def __init__(self, responses, run, role):
        self._responses = responses
        self._run = run
        self._role = role

    def create(self, **kwargs):
        return self._run.model_call(self._responses.create, kwargs, self._role)

    def parse(self, **kwargs):
        return self._run.model_call(self._responses.parse, kwargs, self._role)


class RecordingClient:
    """Stands in for an OpenAI client; only .responses is used by the app."""

    def __init__(self, client, run, role):
        self.responses = RecordingResponses(client.responses, run, role)


@contextmanager
def patched_app(run):
    """
    Point the app's module globals at recording wrappers for one run and put
    the originals back afterwards. These are the names the app looks up at
    call time; tests/test_kitaru_native_adapter.py fails if they move.
    """
    import app
    from src.generation import generate_response, memory

    original_run_tool = generate_response.run_tool
    original_retrieve = app.retrieve_chunks
    original_alerts = app.alerts_context
    saved = [
        (generate_response, "client", generate_response.client),
        (memory, "client", memory.client),
        (generate_response, "run_tool", original_run_tool),
        (app, "retrieve_chunks", original_retrieve),
        (app, "alerts_context", original_alerts),
    ]
    generate_response.client = RecordingClient(generate_response.client, run, "answer")
    memory.client = RecordingClient(memory.client, run, "router")
    generate_response.run_tool = lambda name, arguments_json: run.tool_call(
        name, arguments_json, lambda: original_run_tool(name, arguments_json))
    app.retrieve_chunks = lambda question, k=7: run.span(
        "retrieve_chunks", {"question": question, "k": k}, lambda: original_retrieve(question, k=k))
    app.alerts_context = lambda park_codes: run.span(
        "alerts_context", {"park_codes": park_codes}, lambda: original_alerts(park_codes))
    try:
        yield
    finally:
        for module, name, value in saved:
            setattr(module, name, value)


# --- entry ----------------------------------------------------------------------

_active = False


def run_turn(answer, question, history, writer=None, replay_id=None):
    """
    Record one chat turn as a Kitaru session and return the answer text.

    answer(question, history) runs the app and returns its final text. In a
    worker replay task, KITARU_REPLAY_ID selects the override and tool policy.
    A Kitaru write failure fails the run (fail-fast); an app failure stays the
    primary error, with later recording errors attached as notes.
    """
    global _active
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError("The native Kitaru adapter cannot run inside an event loop")
    if _active:
        raise RuntimeError("Only one native Kitaru run per process is supported")

    writer = writer or KitaruWriter()
    replay_id = replay_id or os.environ.get("KITARU_REPLAY_ID")
    override = tool_policy = None
    if replay_id is not None:
        replay_id = uuid.UUID(str(replay_id))
        replay = writer.get_replay(replay_id)
        override, tool_policy = replay.override, replay.tool_policy
        if any(isinstance(config, LLMConfig)
               for config in [tool_policy.default, *tool_policy.tools.values()]):
            raise RuntimeError("The native adapter does not support the llm tool policy")
    if override is not None and override.prompt is not None:
        question = override.prompt

    run = NativeRun(writer, replay_id, override, tool_policy)
    turns = [{"inputs": history[i]["content"], "outputs": history[i + 1]["content"]}
             for i in range(0, len(history) - 1, 2)]
    turns.append({"inputs": question, "outputs": None})
    run.started_at = now()
    session = writer.create_session(SessionCreateRequest(
        origin=SessionOrigin.REPLAY if replay_id else SessionOrigin.RECORDED,
        status=SessionStatus.IN_PROGRESS,
        name="chat_turn",
        inputs=run.project({"turns": turns})[0],
        outputs=None,
        input_text_selector=f"/turns/{len(turns) - 1}/inputs",
        started_at=run.started_at,
        framework=FRAMEWORK,
        adapter_version=ADAPTER_VERSION,
    ))
    run.session_id = session.id
    root = lambda status, outputs=None, error=None: SessionNodeCreateRequest(
        external_id=run.root_id, node_type=NodeType.SPAN, name="kitaru-run", status=status,
        error=error, started_at=run.started_at,
        ended_at=None if status == NodeStatus.IN_PROGRESS else now(),
        inputs=run.project(question)[0], outputs=outputs, attributes=None,
    )
    try:
        writer.ingest(run.session_id, [root(NodeStatus.IN_PROGRESS)])
    except BaseException as error:
        mark_failed(run, f"Root node write failed: {error}", error)
        raise

    _active = True
    try:
        with patched_app(run):
            text = answer(question, history)
        if run.policy_failure is not None:
            raise run.policy_failure
    except BaseException as error:
        # The app turns a tool-policy refusal into its friendly error message,
        # so the entrypoint's generic failure would hide it: report the policy.
        if run.policy_failure is not None and error is not run.policy_failure:
            run.policy_failure.__cause__ = error
            error = run.policy_failure
        message = redact(f"{type(error).__name__}: {error}", run.secrets)[:MAX_TEXT]
        try:
            writer.ingest(run.session_id, [*run.nodes, root(NodeStatus.FAILED, error=message)])
        except BaseException as write_error:
            error.add_note(f"Kitaru: recording the failed nodes also failed: {write_error}")
        mark_failed(run, message, error)
        raise error
    finally:
        _active = False

    # The server rejects replacing a session's inputs, so the last turn keeps
    # outputs=None; replaying this session only needs earlier turns' answers.
    text_out = run.project(text)[0]
    try:
        writer.ingest(run.session_id, [*run.nodes, root(NodeStatus.COMPLETED, outputs=text_out)])
        writer.update_session(run.session_id, SessionUpdateRequest(
            status=SessionStatus.COMPLETED,
            outputs=text_out,
            output_text_selector="",
            ended_at=now(),
        ))
    except BaseException as error:
        mark_failed(run, f"Recording the completed run failed: {error}", error)
        raise
    return text


def mark_failed(run, message, error):
    """Best effort: mark the session failed without replacing the primary error."""
    try:
        run.writer.update_session(run.session_id, SessionUpdateRequest(
            status=SessionStatus.FAILED, error=message, ended_at=now()))
    except BaseException as write_error:
        error.add_note(f"Kitaru: marking the session failed also failed: {write_error}")
