# Record one trip-planner turn as a Kitaru session, applying replay settings.
#
# Hooks the places the app talks to the outside world: memory.client (the
# router), generate_response.client (each answer round),
# generate_response.run_tool (tool dispatch), and the pipeline steps the app
# runs on every question, retrieve_chunks and alerts_context, as app.py
# calls them. The app's code is unchanged. The hooks are installed once and
# send each call to the turn recording in the current context (current_run),
# or straight through when none is, so concurrent Gradio chats each record
# their own turn (see live.py).
#
# Pipeline steps are recorded as spans marked kind: pipeline, beside the
# router at the top of the turn: the app's code runs them, the model never
# asks for them. The server's history lookup only serves tool calls, so
# steps replay from static cases or run live (see policy.py).
#
# run_turn is the worker's entry: one turn per process, from a plain script.
# The app is sync and Kitaru's client is async, so run_turn owns one event
# loop, the way Kitaru's OpenAI Agents adapter runs run_sync. The loop is
# idle while the app runs, so a history lookup during the turn completes on
# it directly: no thread, no second loop.
import asyncio
import inspect
import json
import os
import sys
import uuid
import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from kitaru.api_models.v1.session import TokenUsage
from kitaru.api_models.v1.session_node import NodeStatus, NodeType
from kitaru.api_models.v1.task import AgentTaskDetails
from kitaru.client import KitaruAPIClient

from src.kitaru_adapter.capture import capture_error, has_capture_marker, secret_values
from src.kitaru_adapter.policy import APPROVED_PASSTHROUGH_TOOLS, lookup_match, prepare_policies
from src.kitaru_adapter.recorder import ROOT_ID, KitaruRecordingError, Recorder, now

ROUTER, ANSWER = "router", "answer"
MODEL_PROVIDER = "openai"
OVERRIDE_PARAMS = ("reasoning", "temperature", "max_output_tokens")
RECORDED_PARAMS = (*OVERRIDE_PARAMS, "tool_choice", "stream")

# What a Kitaru write failure does to a worker turn: fail it, since a replay
# with a broken recording is useless. See recording_failed. Live chats are
# best effort instead (live.py): the chat must keep working.
AVAILABILITY = "fail_fast"

# The turn being recorded in this context, if any.
current_run = ContextVar("kitaru_current_run", default=None)

# Fields of a retrieved chunk the prompt is built from, recorded and replayed.
CHUNK_FIELDS = ("source_type", "title", "park_name", "content", "metadata")


def chunks_record(chunks):
    """Retrieved rows as JSON: the prompt's fields, plus id and distance to inspect."""
    return [{**{key: chunk.get(key) for key in CHUNK_FIELDS},
             "id": None if chunk.get("id") is None else str(chunk["id"]),
             "distance": None if chunk.get("distance") is None else float(chunk["distance"])}
            for chunk in chunks]


# Pipeline steps app.py calls on every question, and how each result is recorded.
STEPS = {
    "retrieve_chunks": chunks_record,
    "alerts_context": lambda text: text,
}


class ReplayOverrideError(ValueError):
    """A replay override this adapter cannot apply."""


@dataclass
class TurnInput:
    recorded: object      # the task's inputs as stored, recorded as session inputs
    question: str
    history: list
    replay: object        # ReplayResponse, or None for a recorded run


@dataclass
class TurnResult:
    answer: str
    session_id: uuid.UUID | None
    recording: str = "complete"


def question_and_history(inputs):
    """
    A plain question, or {"turns": [{"inputs", "outputs"}, ...]} as imported
    Braintrust sessions store one chat: the last turn's input is the question
    and earlier complete turns become the history the app would have had.
    """
    if isinstance(inputs, str):
        return inputs, []
    turns = inputs.get("turns") if isinstance(inputs, dict) else None
    if not turns or not isinstance(turns[-1].get("inputs"), str):
        raise ValueError("Task inputs have no text question to replay")
    history = []
    for turn in turns[:-1]:
        if not (isinstance(turn.get("inputs"), str) and isinstance(turn.get("outputs"), str)):
            raise ValueError("An earlier turn has no text input or output; cannot rebuild history")
        history += [{"role": "user", "content": turn["inputs"]},
                    {"role": "assistant", "content": turn["outputs"]}]
    return turns[-1]["inputs"], history


def input_text_selector(inputs):
    if isinstance(inputs, dict) and inputs.get("turns"):
        return f"/turns/{len(inputs['turns']) - 1}/inputs"
    return None


async def resolve_input(client):
    """The task's inputs and replay, read once before the session exists."""
    task_value = os.environ.get("KITARU_TASK_ID")
    if not task_value:
        raise RuntimeError("run_turn records only inside a Kitaru worker task (no KITARU_TASK_ID)")
    replay_value = os.environ.get("KITARU_REPLAY_ID")
    raw_inputs = os.environ.get("KITARU_TASK_INPUTS")
    if raw_inputs is not None:
        recorded = json.loads(raw_inputs)
    else:
        spec = await client.tasks.get_spec(uuid.UUID(task_value))
        if not isinstance(spec.details, AgentTaskDetails):
            raise RuntimeError(f"Task {task_value} is not an agent task")
        recorded = spec.details.inputs
        replay_value = replay_value or (spec.details.replay_id and str(spec.details.replay_id))
    replay = await client.replays.get(uuid.UUID(replay_value)) if replay_value else None

    question, history = question_and_history(recorded)
    if replay is not None:
        check_override(replay.override)
        if replay.override is not None and replay.override.prompt is not None:
            question = replay.override.prompt
    return TurnInput(recorded, question, history, replay)


def check_override(override):
    unsupported = sorted(set((override and override.model_params) or {}) - set(OVERRIDE_PARAMS))
    if unsupported:
        raise ReplayOverrideError(f"Model params cannot be overridden: {', '.join(unsupported)}")


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


class TurnRun:
    """Hook state for one turn: overrides, tool policies, and the first failure."""

    def __init__(self, recorder, policies, override):
        self.recorder = recorder
        self.policies = policies
        self.override = override
        self.first_failure = None
        self.answer_node = None    # the latest answer round: parent of its tool calls

    def note(self, error):
        # The app turns errors into a friendly message; keep the real one.
        if self.first_failure is None:
            self.first_failure = error

    # --- model calls ---------------------------------------------------------

    def model_call(self, call, kwargs, role):
        kwargs, requested = self.apply_override(dict(kwargs), role)
        started = now()

        def record(response=None, error=None):
            self.record_model(role, kwargs, requested, started, response, error)

        try:
            response = call(**kwargs)
        except BaseException as error:
            record(error=error)
            self.note(error)
            raise
        if kwargs.get("stream"):
            return self.observe_stream(response, record)
        record(response=response)
        return response

    def apply_override(self, kwargs, role):
        requested = kwargs.get("model")
        override = self.override
        if override is None:
            return kwargs, requested
        if isinstance(override.model, str):
            kwargs["model"] = override.model
        elif isinstance(override.model, dict) and requested in override.model:
            kwargs["model"] = override.model[requested]
        if role == ANSWER:
            if override.system_prompt is not None:
                kwargs["instructions"] = override.system_prompt
            kwargs.update(override.model_params or {})
        return kwargs, requested

    def observe_stream(self, stream, record):
        """Pass events through; the round ends at response.completed, or fails."""
        response = None
        try:
            for event in stream:
                if event.type == "response.completed":
                    response = event.response
                yield event
        except GeneratorExit as error:
            record(error=RuntimeError("Stream abandoned before it finished"))
            self.note(RuntimeError("The app abandoned a model stream"))
            raise error
        except BaseException as error:
            record(error=error)
            self.note(error)
            raise
        if response is None:
            error = RuntimeError("Model stream ended without response.completed")
            record(error=error)
            self.note(error)
        else:
            record(response=response)

    def record_model(self, role, kwargs, requested, started, response, error):
        recorder = self.recorder
        inputs = {key: recorder.capture(kwargs.get(key))
                  for key in ("instructions", "input", "previous_response_id")}
        outputs = None
        if response is not None:
            outputs = {"output_text": recorder.capture(getattr(response, "output_text", None)),
                       "output": recorder.capture(getattr(response, "output", None))}
            if role == ROUTER:
                outputs["output_parsed"] = recorder.capture(getattr(response, "output_parsed", None))
        external_id = recorder.add_node(
            node_type=NodeType.LLM_CALL,
            name=f"openai.responses.{'parse' if role == ROUTER else 'create'}",
            parent=ROOT_ID,
            started_at=started,
            inputs=inputs,
            outputs=outputs,
            status=NodeStatus.FAILED if error is not None else NodeStatus.COMPLETED,
            error=capture_error(error, recorder.secrets) if error is not None else None,
            external_id=getattr(response, "id", None) if response is not None else None,
            requested_model=requested,
            model=getattr(response, "model", None) if response is not None else None,
            model_provider=MODEL_PROVIDER,
            tokens=usage_tokens(getattr(response, "usage", None)) if response is not None else None,
            model_params=recorder.capture({key: kwargs[key] for key in RECORDED_PARAMS if key in kwargs}),
            input_text_selector="/input" if isinstance(kwargs.get("input"), str) else None,
            system_prompt_selector="/instructions" if kwargs.get("instructions") else None,
            output_text_selector="/output_text" if outputs else None,
            attributes={"role": role},
        )
        if role == ANSWER:
            self.answer_node = external_id

    # --- tool calls ----------------------------------------------------------

    def tool_call(self, name, arguments_json, run_tool):
        started = now()
        decision = None
        try:
            decision = self.policies.decide(name, arguments_json)
            if decision.passthrough:
                output = run_tool(name, arguments_json)
                failed = isinstance(output, str) and output.startswith(f"Tool {name} failed:")
            else:
                output, failed = decision.value, decision.failed
        except BaseException as error:
            self.record_tool(name, arguments_json, started, decision, None,
                             capture_error(error, self.recorder.secrets))
            self.note(error)
            raise
        self.record_tool(name, arguments_json, started, decision, output, output if failed else None)
        return output

    def record_tool(self, name, arguments_json, started, decision, output, error):
        recorder = self.recorder
        attributes = {}
        if decision is not None and decision.policy:
            attributes.update(mocked=True, policy=decision.policy)
        try:
            inputs = recorder.capture(json.loads(arguments_json))
        except (TypeError, ValueError, RecursionError):
            inputs = None
            attributes["kitaru.tool_arguments"] = "invalid_json"
        if inputs is not None and has_capture_marker(inputs):
            # A cut copy would give a wrong cache key; record none instead.
            inputs = None
            attributes["kitaru.tool_arguments"] = "capture_loss"
        recorder.add_node(
            node_type=NodeType.TOOL_CALL,
            name=name,
            parent=self.answer_node or ROOT_ID,
            started_at=started,
            inputs=inputs,
            outputs=recorder.capture(output),
            status=NodeStatus.FAILED if error is not None else NodeStatus.COMPLETED,
            error=error,
            tool_name=name,
            attributes=attributes,
        )

    # --- pipeline steps ------------------------------------------------------

    def step_call(self, name, function, args, kwargs):
        started = now()
        bound = inspect.signature(function).bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)
        decision = None
        try:
            decision = self.policies.decide_arguments(name, arguments)
            if decision.passthrough:
                value = function(*args, **kwargs)
            elif decision.error is not None:
                # The step raised in the recorded run; the app saw an exception.
                raise RuntimeError(f"Recorded {name} failed: {decision.error}")
            else:
                value = decision.value
        except BaseException as error:
            self.record_step(name, arguments, started, decision, None,
                             capture_error(error, self.recorder.secrets))
            self.note(error)
            raise
        self.record_step(name, arguments, started, decision, STEPS[name](value), None)
        return value

    def record_step(self, name, arguments, started, decision, output, error):
        recorder = self.recorder
        attributes = {"kind": "pipeline"}
        if decision is not None and decision.policy:
            attributes.update(mocked=True, policy=decision.policy)
        inputs = recorder.capture(arguments)
        if has_capture_marker(inputs):
            inputs = None
            attributes["kitaru.tool_arguments"] = "capture_loss"
        recorder.add_node(
            node_type=NodeType.SPAN,
            name=name,
            parent=ROOT_ID,
            started_at=started,
            inputs=inputs,
            outputs=recorder.capture(output),
            status=NodeStatus.FAILED if error is not None else NodeStatus.COMPLETED,
            error=error,
            attributes=attributes,
        )


class RecordingResponses:
    def __init__(self, responses, role):
        self._responses, self._role = responses, role

    def create(self, **kwargs):
        run = current_run.get()
        if run is None:
            return self._responses.create(**kwargs)
        return run.model_call(self._responses.create, kwargs, self._role)

    def parse(self, **kwargs):
        run = current_run.get()
        if run is None:
            return self._responses.parse(**kwargs)
        return run.model_call(self._responses.parse, kwargs, self._role)


class RecordingClient:
    """An OpenAI client whose Responses calls go through the current run, if any."""

    def __init__(self, client, role):
        self._client = client
        self.responses = RecordingResponses(client.responses, role)

    def __getattr__(self, name):
        return getattr(self._client, name)


class RecordingTool:
    """The app's tool dispatcher, through the current run if any."""

    def __init__(self, run_tool):
        self.run_tool = run_tool

    def __call__(self, name, arguments_json):
        run = current_run.get()
        if run is None:
            return self.run_tool(name, arguments_json)
        return run.tool_call(name, arguments_json, self.run_tool)


class RecordingStep:
    """One pipeline step as app.py calls it, through the current run if any."""

    def __init__(self, name, function):
        self.name, self.function = name, function

    def __call__(self, *args, **kwargs):
        run = current_run.get()
        if run is None:
            return self.function(*args, **kwargs)
        return run.step_call(self.name, self.function, args, kwargs)


def app_namespace():
    """app.py's globals, where answer_turn looks up its pipeline steps."""
    module = sys.modules.get("app")
    if module is None:
        raise RuntimeError("app.py is not imported, so its pipeline steps cannot be recorded")
    return vars(module)


def install_hooks(app_globals):
    """
    Route the app's model clients, tool dispatcher, and pipeline steps through
    current_run. Idempotent. app_globals is app.py's namespace: it runs as
    __main__ under `python app.py`, so callers pass it rather than import app.
    """
    from src.generation import generate_response, memory

    seams = [(memory, "client"), (generate_response, "client"), (generate_response, "run_tool")]
    missing = [f"{module.__name__}.{name}" for module, name in seams if not hasattr(module, name)]
    missing += [f"app.{name}" for name in STEPS if name not in app_globals]
    if missing:
        raise RuntimeError(f"App seams moved, nothing would be recorded: {', '.join(missing)}")
    if not isinstance(memory.client, RecordingClient):
        memory.client = RecordingClient(memory.client, ROUTER)
    if not isinstance(generate_response.client, RecordingClient):
        generate_response.client = RecordingClient(generate_response.client, ANSWER)
    if not isinstance(generate_response.run_tool, RecordingTool):
        generate_response.run_tool = RecordingTool(generate_response.run_tool)
    for name in STEPS:
        if not isinstance(app_globals[name], RecordingStep):
            app_globals[name] = RecordingStep(name, app_globals[name])


@contextmanager
def recording(run, app_globals):
    """Record the app's calls in this context to run."""
    install_hooks(app_globals)
    if current_run.get() is not None:
        raise RuntimeError("A turn is already recording in this context")
    token = current_run.set(run)
    try:
        yield
    finally:
        current_run.reset(token)


def recording_failed(phase, recorder, error, result=None):
    """
    The one place AVAILABILITY decides what a Kitaru write failure does to the
    turn. Fail-fast raises. A best-effort mode would instead return
    TurnResult(result, recorder.session_id, recording="partial") for
    "finalize", and run the app unrecorded for "start".
    """
    raise KitaruRecordingError(phase=phase, session_id=recorder.session_id, result=result) from error


def run_turn(answer, *, client_factory=KitaruAPIClient, approved=APPROVED_PASSTHROUGH_TOOLS):
    """
    Run answer(question, history) -> text for the worker task's inputs and
    record it as one Kitaru session. Raises the first failure the turn hit,
    even when the app turned it into a friendly message.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError("run_turn cannot be called from a running event loop")

    loop = asyncio.new_event_loop()
    try:
        return _run_turn(loop.run_until_complete, answer, client_factory(), approved)
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()


def _run_turn(wait, answer, client, approved):
    from src.tools import TOOL_FUNCTIONS

    recorder = Recorder(client, secret_values())
    try:
        # Setup failures are replay-safety refusals, never downgraded.
        turn_input = wait(resolve_input(client))
        replay = turn_input.replay
        def lookup(tool_name, cache_key, occurrence):
            return wait(lookup_match(client, replay.id, tool_name, cache_key, occurrence))

        policies = prepare_policies(replay, list(TOOL_FUNCTIONS), lookup, approved, list(STEPS))

        try:
            wait(recorder.start(
                inputs=turn_input.recorded,
                input_text_selector=input_text_selector(turn_input.recorded),
                root_inputs={"question": turn_input.question, "history": turn_input.history},
                replay=replay is not None,
                name=os.environ.get("KITARU_SESSION_NAME"),
            ))
        except Exception as error:
            wait(recorder.fail_quietly(error))
            recording_failed("start", recorder, error)

        run = TurnRun(recorder, policies, replay.override if replay else None)
        try:
            with recording(run, app_namespace()):
                text = answer(turn_input.question, turn_input.history)
            if run.first_failure is not None:
                raise run.first_failure
        except BaseException as error:
            primary = run.first_failure or error
            if primary is not error:
                primary.add_note(f"The app then raised {type(error).__name__}")
            wait(recorder.fail_quietly(primary))
            raise primary

        try:
            wait(recorder.complete(text))
        except Exception as error:
            wait(recorder.fail_quietly(error))
            return recording_failed("finalize", recorder, error, result=text)
        return TurnResult(text, recorder.session_id)
    finally:
        try:
            wait(client.close())
        except Exception as error:
            warnings.warn(f"Kitaru client did not close cleanly: {type(error).__name__}")
