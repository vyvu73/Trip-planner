# Record each question asked in the Gradio app as a Kitaru session.
#
# On when the app starts with KITARU_RECORD=1: app.py wraps respond with
# record_turns. Each chat turn becomes its own session under the
# trip-planner agent, with the chat's conversation_id in its metadata and its
# inputs in the {"turns": [...]} shape the worker replays read back.
#
# Best effort: the chat must work even when Kitaru does not. If the session
# cannot be created the turn runs unrecorded; if a later write fails, the
# session is marked failed when possible. Every outcome is counted in
# recording_health(), and each failure is logged.
#
# Gradio runs an async handler on its event loop, so Kitaru's async client is
# awaited there, before and after the turn. The app's sync generator is
# stepped on a worker thread, as Gradio steps sync generators, inside a
# context whose current_run is this turn, so concurrent chats stay apart.
import contextvars
import functools
import logging
import os
import uuid
from collections import Counter

import anyio
from kitaru.client import KitaruAPIClient

from src.kitaru_adapter.capture import secret_values
from src.kitaru_adapter.policy import ToolPolicies
from src.kitaru_adapter.recorder import Recorder
from src.kitaru_adapter.turn import TurnRun, current_run, input_text_selector, install_hooks

AGENT_NAME = os.getenv("KITARU_AGENT") or "trip-planner"
# A recording write must not hold the chat open for long once the answer is out.
WRITE_TIMEOUT_SECONDS = 10

log = logging.getLogger(__name__)
health = Counter()    # outcome -> turns: complete, failed_turn, partial, unrecorded
_DONE = object()


def recording_health():
    """How many turns ended in each recording outcome since the app started."""
    return dict(health)


def chat_inputs(question, prior):
    """The turn as {"turns": [...]}, or the bare question if the history is not plain text pairs."""
    pairs = list(zip(prior[0::2], prior[1::2]))
    plain = len(prior) % 2 == 0 and isinstance(question, str) and all(
        user.get("role") == "user" and assistant.get("role") == "assistant"
        and isinstance(user.get("content"), str) and isinstance(assistant.get("content"), str)
        for user, assistant in pairs)
    if not plain:
        return question
    turns = [{"inputs": user["content"], "outputs": assistant["content"]} for user, assistant in pairs]
    return {"turns": turns + [{"inputs": question}]}


class Agent:
    """The Kitaru agent and latest version live sessions attach to, looked up once."""

    def __init__(self, name):
        self.name = name
        self.ids = None

    async def resolve(self, client):
        if self.ids is None:
            async for agent in client.agents.iter():
                if agent.name == self.name:
                    latest = None
                    async for version in client.agents.iter_versions(agent.id):
                        if latest is None or version.version > latest.version:
                            latest = version
                    self.ids = (agent.id, latest.id if latest else None)
                    break
            else:
                raise LookupError(f"No Kitaru agent named '{self.name}'; register it first")
        return self.ids


class LiveTurn:
    """One chat turn's recording. Never raises: a failure is counted and logged."""

    def __init__(self, client_factory, agent, error_message):
        self.client_factory = client_factory
        self.agent = agent
        self.error_message = error_message
        self.client = self.recorder = self.run = None

    async def start(self, question, prior, conversation_id):
        try:
            self.client = self.client_factory()
            self.recorder = Recorder(self.client, secret_values())
            with anyio.fail_after(WRITE_TIMEOUT_SECONDS):
                agent_id, version_id = await self.agent.resolve(self.client)
                inputs = chat_inputs(question, prior)
                await self.recorder.start(
                    inputs=inputs,
                    input_text_selector=input_text_selector(inputs),
                    root_inputs={"question": question, "history": prior},
                    replay=False,
                    agent_id=agent_id,
                    agent_version_id=version_id,
                    metadata={"conversation_id": conversation_id, "source": "gradio",
                              "turn": sum(1 for entry in prior if entry.get("role") == "user") + 1},
                )
        except Exception as error:
            await self._end("unrecorded", error, self.recorder and self.recorder.fail_quietly(error))
            return None
        self.run = TurnRun(self.recorder, ToolPolicies(configs={}), override=None)
        return self.run

    async def finish(self, answer):
        failure = self.run.first_failure
        if failure is None and (not answer or answer == self.error_message):
            failure = RuntimeError("The app could not answer")
        try:
            with anyio.fail_after(WRITE_TIMEOUT_SECONDS):
                if failure is None:
                    await self.recorder.complete(answer)
                else:
                    await self.recorder.fail(failure)
        except Exception as error:
            await self._end("partial", error, self.recorder.fail_quietly(error))
            return
        await self._end("complete" if failure is None else "failed_turn")

    async def abandon(self, error):
        """The chat stopped mid-answer: the tab closed, or the app raised."""
        failure = self.run.first_failure or error
        await self._end("failed_turn", None, self.recorder.fail_quietly(failure))

    async def _end(self, outcome, error=None, cleanup=None):
        health[outcome] += 1
        if error is not None:
            session = self.recorder.session_id if self.recorder else None
            log.warning("Kitaru recording %s for session %s: %s: %s",
                        outcome, session, type(error).__name__, error)
        try:
            if cleanup is not None:
                with anyio.fail_after(WRITE_TIMEOUT_SECONDS):
                    await cleanup
        except Exception:
            pass    # fail_quietly already notes its own problems on the error
        finally:
            if self.client is not None:
                try:
                    await self.client.close()
                except Exception as close_error:
                    log.warning("Kitaru client did not close cleanly: %s", type(close_error).__name__)


async def _stepped(steps, context):
    """Step a sync generator on a worker thread, inside context."""
    while True:
        item = await anyio.to_thread.run_sync(context.run, next, steps, _DONE)
        if item is _DONE:
            return
        yield item


def answer_of(update):
    """The assistant text in respond's last (chatbot, state, conversation_id) update."""
    if not update or not update[0] or update[0][-1].get("role") != "assistant":
        return None
    return update[0][-1].get("content")


def record_turns(respond, *, error_message, client_factory=KitaruAPIClient, agent_name=AGENT_NAME):
    """Wrap app.respond so each turn it answers is recorded as a Kitaru session."""
    install_hooks(respond.__globals__)    # app.py's namespace, even when run as __main__
    agent = Agent(agent_name)

    @functools.wraps(respond)
    async def recorded(history, conversation_id):
        # respond would mint this itself; minting it here lets the session carry it.
        conversation_id = conversation_id or str(uuid.uuid4())
        steps = respond(history, conversation_id)
        context = contextvars.copy_context()
        if not history or history[-1]["role"] != "user":
            async for update in _stepped(steps, context):
                yield update
            return

        turn = LiveTurn(client_factory, agent, error_message)
        run = await turn.start(history[-1]["content"], history[:-1], conversation_id)
        if run is not None:
            context.run(current_run.set, run)
        last = None
        try:
            async for update in _stepped(steps, context):
                last = update
                yield update
        except BaseException as error:
            if run is not None:
                with anyio.CancelScope(shield=True):
                    await turn.abandon(error)
            raise
        if run is not None:
            await turn.finish(answer_of(last))

    return recorded
