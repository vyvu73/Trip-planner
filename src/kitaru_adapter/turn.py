# Record one trip-planner turn as a Kitaru session, applying replay settings.
#
# Hooks the three places the app talks to the outside world, for one run:
# memory.client (the router), generate_response.client (each answer round),
# and generate_response.run_tool (tool dispatch). The app's code is unchanged.
#
# Boundary: one turn per process, from a plain script (a Kitaru worker task).
# The app is sync and Kitaru's client is async, so run_turn owns one event
# loop and runs Kitaru work on it before and after the app, never during it,
# the way Kitaru's OpenAI Agents adapter runs run_sync. Not for the Gradio app:
# the hooks are process-wide.
import asyncio
import json
import os
import uuid
import warnings
from contextlib import contextmanager
from dataclasses import dataclass

from kitaru.api_models.v1.session import TokenUsage
from kitaru.api_models.v1.session_node import NodeStatus, NodeType
from kitaru.api_models.v1.task import AgentTaskDetails
from kitaru.client import KitaruAPIClient

from src.kitaru_adapter.capture import capture_error, has_capture_marker, secret_values
from src.kitaru_adapter.policy import APPROVED_PASSTHROUGH_TOOLS, prepare_policies
from src.kitaru_adapter.recorder import ROOT_ID, KitaruRecordingError, Recorder, now

ROUTER, ANSWER = "router", "answer"
MODEL_PROVIDER = "openai"
OVERRIDE_PARAMS = ("reasoning", "temperature", "max_output_tokens")
RECORDED_PARAMS = (*OVERRIDE_PARAMS, "tool_choice", "stream")

# What a Kitaru write failure does to the turn. Only "fail_fast" exists: the
# app has no channel to report a degraded recording. See recording_failed.
AVAILABILITY = "fail_fast"


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
                output, failed = decision.result, decision.failed
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


class RecordingResponses:
    def __init__(self, responses, run, role):
        self._responses, self._run, self._role = responses, run, role

    def create(self, **kwargs):
        return self._run.model_call(self._responses.create, kwargs, self._role)

    def parse(self, **kwargs):
        return self._run.model_call(self._responses.parse, kwargs, self._role)


class RecordingClient:
    """An OpenAI client whose Responses calls go through the run's hooks."""

    def __init__(self, client, run, role):
        self._client = client
        self.responses = RecordingResponses(client.responses, run, role)

    def __getattr__(self, name):
        return getattr(self._client, name)


@contextmanager
def hooked(run):
    """Point the app's model clients and tool dispatcher at run, then restore them."""
    from src.generation import generate_response, memory

    seams = [(memory, "client"), (generate_response, "client"), (generate_response, "run_tool")]
    missing = [f"{module.__name__}.{name}" for module, name in seams if not hasattr(module, name)]
    if missing:
        raise RuntimeError(f"App seams moved, nothing would be recorded: {', '.join(missing)}")
    originals = [(module, name, getattr(module, name)) for module, name in seams]
    if any(isinstance(value, RecordingClient) for _, _, value in originals):
        raise RuntimeError("Another turn is already recording in this process")

    original_run_tool = generate_response.run_tool
    memory.client = RecordingClient(memory.client, run, ROUTER)
    generate_response.client = RecordingClient(generate_response.client, run, ANSWER)
    generate_response.run_tool = lambda name, arguments_json: run.tool_call(
        name, arguments_json, original_run_tool)
    try:
        yield
    finally:
        for module, name, value in originals:
            setattr(module, name, value)


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
        policies = wait(prepare_policies(client, replay, list(TOOL_FUNCTIONS), approved))

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
            with hooked(run):
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
