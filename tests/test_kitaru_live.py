import inspect
import json
import threading
import unittest
import uuid
from types import SimpleNamespace
from unittest import mock

import anyio

# Shares the adapter tests' environment setup and fakes.
from tests.test_kitaru_adapter import (
    SENTINEL,
    FakeKitaru,
    function_call,
    real_alerts_context,
    real_retrieve_chunks,
    router_response,
    stream_round,
)

import gradio as gr
from kitaru.api_models.v1.session import SessionOrigin, SessionStatus
from kitaru.api_models.v1.session_node import NodeStatus, NodeType

import app
from src.generation import generate_response, memory
from src.generation.memory import RouteDecision
from src.kitaru_adapter import live
from src.kitaru_adapter.recorder import ROOT_ID

AGENT_ID = uuid.uuid4()
VERSION_IDS = {1: uuid.uuid4(), 2: uuid.uuid4()}
PARKS = {"Yosemite": "yose", "Lassen": "lavo"}


class LiveKitaru(FakeKitaru):
    """A per-turn Kitaru client that also knows the trip-planner agent."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.agents = SimpleNamespace(iter=self._agents, iter_versions=self._versions)

    async def _agents(self, params=None):
        yield SimpleNamespace(id=uuid.uuid4(), name="other-agent")
        yield SimpleNamespace(id=AGENT_ID, name="trip-planner")

    async def _versions(self, agent_id, params=None):
        for number, version_id in VERSION_IDS.items():
            yield SimpleNamespace(id=version_id, version=number)


def park_in(text):
    return next(code for name, code in PARKS.items() if name in text)


class ParkResponses:
    """Stateless fake OpenAI: answers follow the park named in the request, so
    concurrent chats get their own answers whatever order they run in."""

    def parse(self, **kwargs):
        decision = router_response().output_parsed.model_copy(
            update={"park_codes": [park_in(kwargs["input"])]})
        return SimpleNamespace(**{**vars(router_response()), "output_parsed": decision})

    def create(self, **kwargs):
        previous = kwargs.get("previous_response_id")
        if previous is None:
            park = park_in(kwargs["input"])
            return stream_round(f"resp_{park}_1",
                                calls=[function_call("get_weather", {"park_code": park}, f"call_{park}")])
        park = previous.split("_")[1]
        return stream_round(f"resp_{park}_2", text=f"Answer for {park}.")


class FailingRouter(ParkResponses):
    def parse(self, **kwargs):
        raise TimeoutError("openai timed out")


def chat(question, prior=()):
    return [*prior, {"role": "user", "content": question}]


async def collect(updates):
    return [update async for update in updates]


class LiveTestCase(unittest.TestCase):
    def setUp(self):
        live.health.clear()
        self.clients = []
        self.weather = mock.Mock(side_effect=lambda park_code: f"forecast for {park_code}")
        self.retrieve = mock.create_autospec(real_retrieve_chunks, return_value=[])
        for patcher in [
            mock.patch.object(app, "retrieve_chunks", self.retrieve),
            mock.patch.object(app, "alerts_context", mock.create_autospec(real_alerts_context, return_value="")),
            mock.patch.dict("src.tools.TOOL_FUNCTIONS", {"get_weather": lambda **kw: self.weather(**kw)}),
            mock.patch.object(memory, "client", SimpleNamespace(responses=ParkResponses())),
            mock.patch.object(generate_response, "client", SimpleNamespace(responses=ParkResponses())),
        ]:
            patcher.start()
            self.addCleanup(patcher.stop)

    def factory(self, **kitaru_options):
        def make():
            client = LiveKitaru(**kitaru_options)
            self.clients.append(client)
            return client
        return make

    def recorded(self, **kitaru_options):
        return live.record_turns(app.respond, error_message=app.ERROR_MESSAGE,
                                 client_factory=self.factory(**kitaru_options))

    def run_chat(self, history, conversation_id="conv-1", **kitaru_options):
        return anyio.run(collect, self.recorded(**kitaru_options)(history, conversation_id))


class RecordTurnTests(LiveTestCase):
    def test_a_chat_turn_becomes_a_completed_session(self):
        prior = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]
        updates = self.run_chat(chat("Weather in Yosemite?", prior))

        self.assertEqual(updates[-1][0][-1], {"role": "assistant", "content": "Answer for yose."})
        self.assertEqual(updates[-1][2], "conv-1")
        [kitaru] = self.clients
        session = kitaru.session
        self.assertEqual((session.agent_id, session.agent_version_id), (AGENT_ID, VERSION_IDS[2]))
        self.assertEqual(session.origin, SessionOrigin.RECORDED)
        self.assertEqual(session.metadata, {"conversation_id": "conv-1", "source": "gradio", "turn": 2})
        self.assertEqual(session.inputs, {"turns": [{"inputs": "Hi", "outputs": "Hello!"},
                                                    {"inputs": "Weather in Yosemite?"}]})
        self.assertEqual(kitaru.updates[-1].status, SessionStatus.COMPLETED)
        [tool] = kitaru.tools()
        self.assertEqual(tool.outputs, "forecast for yose")
        steps = kitaru.steps()
        self.assertEqual(steps["retrieve_chunks"].inputs, {"question": "Weather in Yosemite?", "k": 7})
        self.assertEqual(steps["alerts_context"].inputs, {"park_codes": ["yose"]})
        self.assertEqual(kitaru.closed, 1)
        self.assertEqual(live.recording_health(), {"complete": 1})

    def test_recording_does_not_change_what_the_chat_shows(self):
        history = chat("Weather in Yosemite?")
        plain = list(app.respond(history, "conv-1"))
        self.assertEqual(self.run_chat(history), plain)

    def test_concurrent_chats_record_separately(self):
        both_in_tools = threading.Barrier(2, timeout=5)

        def weather(park_code):
            both_in_tools.wait()    # both turns are mid-answer at the same time
            return f"forecast for {park_code}"
        self.weather.side_effect = weather
        recorded = self.recorded()

        async def main():
            results = {}

            async def one(question, conversation_id):
                results[conversation_id] = await collect(recorded(chat(question), conversation_id))
            async with anyio.create_task_group() as group:
                group.start_soon(one, "Weather in Yosemite?", "conv-yose")
                group.start_soon(one, "Weather in Lassen?", "conv-lavo")
            return results

        results = anyio.run(main)

        self.assertEqual(results["conv-yose"][-1][0][-1]["content"], "Answer for yose.")
        self.assertEqual(results["conv-lavo"][-1][0][-1]["content"], "Answer for lavo.")
        self.assertEqual(len(self.clients), 2)
        for kitaru in self.clients:
            park = kitaru.session.metadata["conversation_id"].split("-")[1]
            [tool] = kitaru.tools()
            self.assertEqual((tool.inputs, tool.outputs), ({"park_code": park}, f"forecast for {park}"))
            rounds = [node.external_id for node in kitaru.children(NodeType.LLM_CALL)][1:]
            self.assertEqual(rounds, [f"resp_{park}_1", f"resp_{park}_2"])
            self.assertEqual(tool.parent_external_id, f"resp_{park}_1")
            self.assertEqual(kitaru.updates[-1].outputs, {"answer": f"Answer for {park}."})

    def test_messages_without_a_question_are_not_recorded(self):
        updates = self.run_chat([])
        self.assertEqual(updates, [([], [], "conv-1")])
        self.assertEqual(self.clients, [])

    def test_secrets_never_reach_kitaru(self):
        self.run_chat(chat(f"Weather in Yosemite? {SENTINEL}"))
        self.assertNotIn(SENTINEL, self.clients[0].everything_sent())


class BestEffortTests(LiveTestCase):
    def test_kitaru_down_leaves_the_chat_working(self):
        history = chat("Weather in Yosemite?")
        plain = list(app.respond(history, "conv-1"))
        self.weather.reset_mock()

        with self.assertLogs("src.kitaru_adapter.live", "WARNING"):
            updates = self.run_chat(history, fail={"create": 0})

        self.assertEqual(updates, plain)
        self.assertIsNone(self.clients[0].session)
        self.assertEqual(live.recording_health(), {"unrecorded": 1})
        self.weather.assert_called_once_with(park_code="yose")

    def test_unregistered_agent_leaves_the_chat_working(self):
        recorded = live.record_turns(app.respond, error_message=app.ERROR_MESSAGE,
                                     client_factory=self.factory(), agent_name="missing-agent")
        with self.assertLogs("src.kitaru_adapter.live", "WARNING") as logs:
            updates = anyio.run(collect, recorded(chat("Weather in Yosemite?"), "conv-1"))

        self.assertEqual(updates[-1][0][-1]["content"], "Answer for yose.")
        self.assertIn("register it first", logs.output[0])
        self.assertEqual(live.recording_health(), {"unrecorded": 1})

    def test_failed_final_write_marks_the_session_partial(self):
        with self.assertLogs("src.kitaru_adapter.live", "WARNING"):
            updates = self.run_chat(chat("Weather in Yosemite?"), fail={"update": 0})

        self.assertEqual(updates[-1][0][-1]["content"], "Answer for yose.")
        self.assertEqual([u.status for u in self.clients[0].updates],
                         [SessionStatus.COMPLETED, SessionStatus.FAILED])
        self.assertEqual(live.recording_health(), {"partial": 1})

    def test_app_error_shows_the_message_and_records_the_real_cause(self):
        with mock.patch.object(memory, "client", SimpleNamespace(responses=FailingRouter())):
            updates = self.run_chat(chat("Weather in Yosemite?"))

        self.assertEqual(updates[-1][0][-1]["content"], app.ERROR_MESSAGE)
        kitaru = self.clients[0]
        self.assertEqual(kitaru.updates[-1].status, SessionStatus.FAILED)
        self.assertIn("openai timed out", kitaru.updates[-1].error)
        self.assertEqual(kitaru.nodes[ROOT_ID].status, NodeStatus.FAILED)
        self.assertEqual(live.recording_health(), {"failed_turn": 1})

    def test_database_failure_shows_the_message_and_records_the_cause(self):
        self.retrieve.side_effect = ConnectionError("database down")
        updates = self.run_chat(chat("Weather in Yosemite?"))

        self.assertEqual(updates[-1][0][-1]["content"], app.ERROR_MESSAGE)
        kitaru = self.clients[0]
        self.assertIn("database down", kitaru.updates[-1].error)
        self.assertEqual(kitaru.steps()["retrieve_chunks"].status, NodeStatus.FAILED)
        self.assertEqual(live.recording_health(), {"failed_turn": 1})

    def test_a_closed_tab_marks_the_session_failed(self):
        async def first_update_then_close():
            updates = self.recorded()(chat("Weather in Yosemite?"), "conv-1")
            await updates.__anext__()
            await updates.aclose()

        anyio.run(first_update_then_close)

        kitaru = self.clients[0]
        self.assertEqual(kitaru.updates[-1].status, SessionStatus.FAILED)
        self.assertEqual(live.recording_health(), {"failed_turn": 1})


class GradioTests(LiveTestCase):
    def test_gradio_streams_the_recorded_handler(self):
        recorded = self.recorded()
        self.assertTrue(inspect.isasyncgenfunction(recorded))
        self.assertEqual(list(inspect.signature(recorded).parameters), ["history", "conversation_id"])

        with gr.Blocks() as demo:
            state, conversation, chatbot = gr.State(), gr.State(), gr.Chatbot()
            gr.Button().click(recorded, inputs=[state, conversation], outputs=[chatbot, state, conversation])
        [block_fn] = demo.fns.values()

        async def stream():
            outputs, iterator = [], None
            while True:
                result = await demo.call_function(block_fn, [chat("Weather in Yosemite?"), "conv-1"],
                                                  iterator=iterator)
                if not result["is_generating"]:
                    return outputs
                outputs.append(result["prediction"])
                iterator = result["iterator"]

        outputs = anyio.run(stream)

        self.assertEqual(outputs[-1][0][-1]["content"], "Answer for yose.")
        self.assertEqual(self.clients[0].updates[-1].status, SessionStatus.COMPLETED)

    def test_app_without_the_switch_keeps_the_plain_handler(self):
        self.assertTrue(inspect.isgeneratorfunction(app.respond))


if __name__ == "__main__":
    unittest.main()
