import asyncio
import json
import os
import unittest
import uuid
from types import SimpleNamespace
from unittest import mock

# The app builds OpenAI clients at import time; tracing must stay off in tests.
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ["BRAINTRUST_API_KEY"] = ""
SENTINEL = "sentinel-secret-7f3a9c"
os.environ["ADAPTER_TEST_API_KEY"] = SENTINEL

from kitaru.api_models.v1.replay_config import (
    HistoryConfig,
    HistoryScope,
    LLMConfig,
    PassthroughConfig,
    ReplayOverride,
    StaticCase,
    StaticConfig,
    StaticMatchMode,
    ToolPolicy,
    ToolPolicyOnMiss,
)
from kitaru.api_models.v1.session import (
    SessionCreateRequest,
    SessionOrigin,
    SessionStatus,
    SessionUpdateRequest,
)
from kitaru.api_models.v1.session_node import NodeStatus, NodeType, SessionNodeBatchRequest
from kitaru.cache_keys import compute_tool_cache_key

import app
import kitaru_replay
from src.generation import generate_response, memory
from src.generation.memory import RouteDecision
from src.kitaru_adapter.policy import ToolPolicyError, ToolPolicyMissError
from src.kitaru_adapter.recorder import ROOT_ID, KitaruRecordingError
from src.kitaru_adapter.turn import ReplayOverrideError, question_and_history, run_turn

WEATHER_ARGS = {"park_code": "yose"}


# --- fakes --------------------------------------------------------------------

class Obj(SimpleNamespace):
    """A stand-in for an OpenAI response object."""

    def model_dump(self, mode="json"):
        return {key: value for key, value in vars(self).items()
                if isinstance(value, (str, int, float, bool, type(None)))}


def usage(input_tokens=10, output_tokens=5):
    return Obj(input_tokens=input_tokens, output_tokens=output_tokens,
               input_tokens_details=Obj(cached_tokens=0),
               output_tokens_details=Obj(reasoning_tokens=1))


def router_response(route="in_scope"):
    decision = RouteDecision(route=route, standalone_question="Weather in Yosemite?",
                             park_codes=["yose"], clarifying_question=None, redirect_message=None)
    return Obj(id="resp_router", model="gpt-5-mini-served", output_text="", output=[],
               output_parsed=decision, usage=usage())


def function_call(name, arguments, call_id):
    return Obj(type="function_call", name=name, arguments=json.dumps(arguments), call_id=call_id)


def stream_round(response_id, text="", calls=(), complete=True):
    events = [Obj(type="response.output_text.delta", delta=text)] if text else []
    if complete:
        response = Obj(id=response_id, model="gpt-5-mini-served", output_text=text,
                       output=list(calls), usage=usage())
        events.append(Obj(type="response.completed", response=response))
    return iter(events)


class FakeResponses:
    def __init__(self, parse=(), create=()):
        self.parse_results = list(parse)
        self.create_results = list(create)
        self.calls = []

    def _next(self, results, kind, kwargs):
        self.calls.append((kind, kwargs))
        result = results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    def parse(self, **kwargs):
        return self._next(self.parse_results, "parse", kwargs)

    def create(self, **kwargs):
        return self._next(self.create_results, "create", kwargs)


class FakeSessions:
    def __init__(self, kitaru):
        self.kitaru = kitaru

    async def create(self, request, idempotency_key=None):
        assert isinstance(request, SessionCreateRequest)
        self.kitaru.maybe_fail("create")
        self.kitaru.session = request
        self.kitaru.session_id = uuid.uuid4()
        return SimpleNamespace(id=self.kitaru.session_id)

    async def ingest_nodes(self, session_id, batch):
        assert isinstance(batch, SessionNodeBatchRequest) and session_id == self.kitaru.session_id
        self.kitaru.maybe_fail("ingest")
        for node in batch.nodes:
            self.kitaru.order.append(node.external_id)
            self.kitaru.nodes[node.external_id] = node    # an external id is replaced whole

    async def update(self, session_id, request):
        assert isinstance(request, SessionUpdateRequest)
        self.kitaru.updates.append(request)
        self.kitaru.maybe_fail("update")

    async def get(self, session_id):
        return SimpleNamespace(status=self.kitaru.baseline_status)

    async def iter_nodes(self, session_id, params):
        assert params.include_payloads
        for node in self.kitaru.baseline:
            yield node


class FakeKitaru:
    def __init__(self, replay=None, baseline=(), baseline_status=SessionStatus.COMPLETED, fail=None):
        self.sessions = FakeSessions(self)
        self.replays = SimpleNamespace(get=self._get_replay)
        self.replay = replay
        self.baseline = list(baseline)
        self.baseline_status = baseline_status
        self.fail = dict(fail or {})    # operation -> calls to let through before failing
        self.session = self.session_id = None
        self.nodes, self.order, self.updates = {}, [], []
        self.closed = 0

    async def _get_replay(self, replay_id):
        return self.replay

    def maybe_fail(self, operation):
        if operation in self.fail:
            if self.fail[operation] == 0:
                raise ConnectionError(f"kitaru {operation} down")
            self.fail[operation] -= 1

    async def close(self):
        self.closed += 1

    # inspection
    def children(self, node_type):
        return [node for node in self.nodes.values() if node.node_type is node_type]

    def everything_sent(self):
        dumped = [self.session.model_dump(mode="json")] if self.session else []
        dumped += [node.model_dump(mode="json") for node in self.nodes.values()]
        dumped += [update.model_dump(mode="json") for update in self.updates]
        return json.dumps(dumped)


def replay(tool_policy=None, override=None):
    return SimpleNamespace(id=uuid.uuid4(), baseline_session_id=uuid.uuid4(), override=override,
                           tool_policy=tool_policy or ToolPolicy(default=PassthroughConfig()))


def baseline_tool(name, arguments, outputs, status=NodeStatus.COMPLETED, error=None, cache_key="auto"):
    return SimpleNamespace(
        node_type=NodeType.TOOL_CALL, tool_name=name, status=status, inputs=arguments,
        outputs=outputs, error=error,
        cache_key=compute_tool_cache_key(name, arguments) if cache_key == "auto" else cache_key)


def history(on_miss=ToolPolicyOnMiss.FAIL, scope=HistoryScope.BASELINE):
    return HistoryConfig(scope=scope, on_miss=on_miss)


def static(*cases, on_miss=ToolPolicyOnMiss.FAIL):
    return StaticConfig(cases=list(cases), on_miss=on_miss)


def all_tools(config, **tools):
    """A policy that sets every tool, so nothing defaults to passthrough."""
    names = ("get_weather", "get_route", "get_transportation_cost")
    return ToolPolicy(default=config, tools={name: tools.get(name, config) for name in names})


# --- harness ------------------------------------------------------------------

class AdapterTestCase(unittest.TestCase):
    def setUp(self):
        self.weather = mock.Mock(return_value="LIVE forecast: sunny")
        patchers = [
            mock.patch.object(app, "retrieve_chunks", mock.Mock(return_value=[])),
            mock.patch.object(app, "alerts_context", mock.Mock(return_value="")),
            mock.patch.dict("src.tools.TOOL_FUNCTIONS", {"get_weather": self.weather}),
        ]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_turn(self, responses, kitaru, inputs="Weather in Yosemite?", **kwargs):
        env = {"KITARU_TASK_ID": str(uuid.uuid4()), "KITARU_TASK_INPUTS": json.dumps(inputs)}
        if kitaru.replay is not None:
            env["KITARU_REPLAY_ID"] = str(kitaru.replay.id)
        with mock.patch.dict(os.environ, env), \
                mock.patch.object(memory, "client", SimpleNamespace(responses=responses)), \
                mock.patch.object(generate_response, "client", SimpleNamespace(responses=responses)):
            return run_turn(kitaru_replay.answer, client_factory=lambda: kitaru, **kwargs)


def weather_turn(rounds=None):
    """Router, a round that calls get_weather, then a round that answers."""
    return FakeResponses(parse=[router_response()], create=rounds or [
        stream_round("resp_1", calls=[function_call("get_weather", WEATHER_ARGS, "call_1")]),
        stream_round("resp_2", text="Pack layers."),
    ])


# --- recording ----------------------------------------------------------------

class RecordingTests(AdapterTestCase):
    def test_records_one_turn_as_a_completed_session(self):
        kitaru = FakeKitaru()
        result = self.run_turn(weather_turn(), kitaru)

        self.assertEqual(result.answer, "Pack layers.")
        self.assertEqual(result.recording, "complete")
        self.assertEqual(result.session_id, kitaru.session_id)
        self.assertEqual(kitaru.session.origin, SessionOrigin.RECORDED)
        self.assertEqual(kitaru.session.framework, "openai-responses")
        self.assertEqual(kitaru.updates[-1].status, SessionStatus.COMPLETED)
        self.assertEqual(kitaru.updates[-1].outputs, {"answer": "Pack layers."})
        self.assertEqual(kitaru.closed, 1)

        root = kitaru.nodes[ROOT_ID]
        self.assertEqual(root.status, NodeStatus.COMPLETED)
        self.assertEqual(root.inputs, {"question": "Weather in Yosemite?", "history": []})
        self.assertEqual(kitaru.order[0], ROOT_ID)    # in-progress root before any child

        router, *rounds = kitaru.children(NodeType.LLM_CALL)
        self.assertEqual(router.name, "openai.responses.parse")
        self.assertEqual(router.outputs["output_parsed"]["route"], "in_scope")
        self.assertEqual([node.external_id for node in rounds], ["resp_1", "resp_2"])
        self.assertEqual(rounds[0].requested_model, "gpt-5-mini")
        self.assertEqual(rounds[0].model, "gpt-5-mini-served")
        self.assertEqual(rounds[0].tokens.input_tokens, 10)
        self.assertTrue(all(node.parent_external_id == ROOT_ID for node in [router, *rounds]))

        [tool] = kitaru.children(NodeType.TOOL_CALL)
        self.assertEqual((tool.tool_name, tool.inputs, tool.outputs),
                         ("get_weather", WEATHER_ARGS, "LIVE forecast: sunny"))
        self.assertEqual(tool.parent_external_id, "resp_1")
        self.assertEqual(tool.attributes, {})
        self.weather.assert_called_once_with(park_code="yose")    # recording runs tools live

    def test_hooks_are_removed_after_the_turn(self):
        original_run_tool = generate_response.run_tool
        self.run_turn(weather_turn(), FakeKitaru())
        self.assertIs(generate_response.run_tool, original_run_tool)

    def test_turns_inputs_become_question_and_history(self):
        inputs = {"turns": [{"inputs": "Hi", "outputs": "Hello!"}, {"inputs": "Weather in Yosemite?"}]}
        kitaru = FakeKitaru()
        self.run_turn(weather_turn(), kitaru, inputs=inputs)

        self.assertEqual(kitaru.session.inputs, inputs)
        self.assertEqual(kitaru.session.input_text_selector, "/turns/1/inputs")
        self.assertEqual(kitaru.nodes[ROOT_ID].inputs["history"],
                         [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}])

    def test_question_and_history_rejects_incomplete_turns(self):
        with self.assertRaises(ValueError):
            question_and_history({"turns": [{"inputs": "Hi"}, {"inputs": "Then?"}]})

    def test_secrets_never_reach_kitaru(self):
        self.weather.return_value = f"forecast {SENTINEL}"
        kitaru = FakeKitaru()
        self.run_turn(weather_turn(), kitaru, inputs=f"Weather in Yosemite? {SENTINEL}")

        sent = kitaru.everything_sent()
        self.assertNotIn(SENTINEL, sent)
        self.assertIn("[redacted]", sent)

    def test_oversized_tool_arguments_are_not_recorded_as_matchable(self):
        rounds = [stream_round("resp_1", calls=[function_call("get_weather", {"park_code": "y" * 60_000}, "c")]),
                  stream_round("resp_2", text="Done.")]
        kitaru = FakeKitaru()
        self.run_turn(weather_turn(rounds), kitaru)

        [tool] = kitaru.children(NodeType.TOOL_CALL)
        self.assertIsNone(tool.inputs)
        self.assertEqual(tool.attributes["kitaru.tool_arguments"], "capture_loss")


# --- failures -----------------------------------------------------------------

class FailureTests(AdapterTestCase):
    def test_model_failure_is_raised_even_though_the_app_hides_it(self):
        responses = FakeResponses(parse=[TimeoutError("openai timed out")])
        kitaru = FakeKitaru()

        with self.assertRaises(TimeoutError):
            self.run_turn(responses, kitaru)

        [router] = kitaru.children(NodeType.LLM_CALL)
        self.assertEqual(router.status, NodeStatus.FAILED)
        self.assertIn("openai timed out", router.error)
        self.assertEqual(kitaru.nodes[ROOT_ID].status, NodeStatus.FAILED)
        self.assertEqual(kitaru.updates[-1].status, SessionStatus.FAILED)
        self.assertEqual(kitaru.closed, 1)

    def test_stream_without_completion_fails_the_turn(self):
        rounds = [stream_round("resp_1", text="Partial", complete=False)]
        kitaru = FakeKitaru()

        with self.assertRaisesRegex(RuntimeError, "without response.completed"):
            self.run_turn(weather_turn(rounds), kitaru)
        self.assertEqual(kitaru.updates[-1].status, SessionStatus.FAILED)

    def test_session_create_failure_stops_before_the_app_runs(self):
        responses = weather_turn()
        kitaru = FakeKitaru(fail={"create": 0})

        with self.assertRaises(KitaruRecordingError) as caught:
            self.run_turn(responses, kitaru)

        self.assertEqual(caught.exception.phase, "start")
        self.assertTrue(caught.exception.retry_safe)
        self.assertEqual(responses.calls, [])
        self.assertEqual(kitaru.closed, 1)

    def test_root_failure_after_create_marks_the_session_failed(self):
        responses = weather_turn()
        kitaru = FakeKitaru(fail={"ingest": 0})

        with self.assertRaises(KitaruRecordingError) as caught:
            self.run_turn(responses, kitaru)

        self.assertEqual(caught.exception.session_id, kitaru.session_id)
        self.assertEqual(kitaru.updates[-1].status, SessionStatus.FAILED)
        self.assertEqual(responses.calls, [])

    def test_finalize_failure_keeps_the_answer_and_says_retry_is_unsafe(self):
        kitaru = FakeKitaru(fail={"update": 0})

        with self.assertRaises(KitaruRecordingError) as caught:
            self.run_turn(weather_turn(), kitaru)

        error = caught.exception
        self.assertEqual((error.phase, error.result, error.retry_safe), ("finalize", "Pack layers.", False))
        self.assertEqual(error.session_id, kitaru.session_id)
        self.assertEqual([update.status for update in kitaru.updates],
                         [SessionStatus.COMPLETED, SessionStatus.FAILED])
        self.assertTrue(any("could not mark" in note for note in error.__cause__.__notes__))

    def test_app_failure_stays_primary_when_recording_also_fails(self):
        responses = FakeResponses(parse=[TimeoutError("openai timed out")])
        kitaru = FakeKitaru(fail={"update": 0})

        with self.assertRaises(TimeoutError) as caught:
            self.run_turn(responses, kitaru)
        self.assertTrue(any("could not mark" in note for note in caught.exception.__notes__))

    def test_partial_child_writes_are_kept(self):
        # Root (1) and first child batch (2) succeed; the root's terminal write fails.
        kitaru = FakeKitaru(fail={"ingest": 2})

        with self.assertRaises(KitaruRecordingError):
            self.run_turn(weather_turn(), kitaru)
        self.assertEqual(len(kitaru.children(NodeType.LLM_CALL)), 3)
        self.assertEqual(kitaru.updates[-1].status, SessionStatus.FAILED)


# --- replay: overrides ----------------------------------------------------------

class OverrideTests(AdapterTestCase):
    def test_overrides_apply_to_model_calls(self):
        override = ReplayOverride(model="gpt-5", system_prompt="Be brief.", prompt="Weather in Lassen?",
                                  model_params={"reasoning": {"effort": "high"}})
        policy = all_tools(static(StaticCase(match_mode=StaticMatchMode.SUBSET, result="Static sun")))
        responses = weather_turn()
        kitaru = FakeKitaru(replay=replay(policy, override))
        self.run_turn(responses, kitaru)

        (_, router_kwargs), (_, first_round), *_ = responses.calls
        self.assertEqual(router_kwargs["model"], "gpt-5")
        self.assertIn("Weather in Lassen?", router_kwargs["input"])
        self.assertNotIn("instructions", router_kwargs)       # system prompt is the answer model's
        self.assertEqual(first_round["model"], "gpt-5")
        self.assertEqual(first_round["instructions"], "Be brief.")
        self.assertEqual(first_round["reasoning"], {"effort": "high"})

        self.assertEqual(kitaru.session.origin, SessionOrigin.REPLAY)
        self.assertEqual(kitaru.nodes[ROOT_ID].inputs["question"], "Weather in Lassen?")
        self.assertEqual(kitaru.session.inputs, "Weather in Yosemite?")    # stored task inputs
        self.assertEqual(kitaru.nodes["resp_1"].requested_model, "gpt-5-mini")

    def test_model_map_replaces_only_listed_models(self):
        override = ReplayOverride(model={"gpt-4o": "gpt-5"})
        responses = weather_turn()
        policy = all_tools(static(StaticCase(match_mode=StaticMatchMode.SUBSET, result="sun")))
        self.run_turn(responses, FakeKitaru(replay=replay(policy, override)))
        self.assertTrue(all(kwargs["model"] == "gpt-5-mini" for _, kwargs in responses.calls))

    def test_unsupported_model_param_is_refused_before_recording(self):
        override = ReplayOverride(model_params={"top_p": 0.5})
        kitaru = FakeKitaru(replay=replay(all_tools(history()), override))

        with self.assertRaises(ReplayOverrideError):
            self.run_turn(weather_turn(), kitaru)
        self.assertIsNone(kitaru.session)


# --- replay: tool policies ------------------------------------------------------

class ToolPolicyTests(AdapterTestCase):
    def test_history_replays_recorded_results_in_order(self):
        calls = [function_call("get_weather", WEATHER_ARGS, "c1"), function_call("get_weather", WEATHER_ARGS, "c2")]
        rounds = [stream_round("resp_1", calls=calls), stream_round("resp_2", text="Done.")]
        baseline = [baseline_tool("get_weather", WEATHER_ARGS, "Recorded: rain"),
                    baseline_tool("get_weather", WEATHER_ARGS, {"periods": ["snow"]})]
        responses = weather_turn(rounds)
        kitaru = FakeKitaru(replay=replay(all_tools(history())), baseline=baseline)
        self.run_turn(responses, kitaru)

        self.weather.assert_not_called()
        first, second = kitaru.children(NodeType.TOOL_CALL)
        self.assertEqual(first.outputs, "Recorded: rain")
        self.assertEqual(second.outputs, '{"periods": ["snow"]}')
        self.assertEqual(first.attributes, {"mocked": True, "policy": "history"})
        sent = responses.calls[2][1]["input"]
        self.assertEqual([item["output"] for item in sent], ["Recorded: rain", '{"periods": ["snow"]}'])

    def test_history_miss_fails_the_replay(self):
        kitaru = FakeKitaru(replay=replay(all_tools(history())), baseline=[])

        with self.assertRaises(ToolPolicyMissError):
            self.run_turn(weather_turn(), kitaru)
        self.weather.assert_not_called()
        [tool] = kitaru.children(NodeType.TOOL_CALL)
        self.assertEqual(tool.status, NodeStatus.FAILED)
        self.assertEqual(kitaru.updates[-1].status, SessionStatus.FAILED)

    def test_history_miss_can_return_an_error_result(self):
        kitaru = FakeKitaru(replay=replay(all_tools(history(on_miss=ToolPolicyOnMiss.ERROR_RESULT))))
        result = self.run_turn(weather_turn(), kitaru)

        self.assertEqual(result.answer, "Pack layers.")
        [tool] = kitaru.children(NodeType.TOOL_CALL)
        self.assertEqual(tool.status, NodeStatus.FAILED)
        self.assertEqual(json.loads(tool.outputs), {"error": "No history result for tool 'get_weather'"})

    def test_recorded_tool_failure_replays_as_the_same_text(self):
        baseline = [baseline_tool("get_weather", WEATHER_ARGS, None, NodeStatus.FAILED, "NWS down")]
        kitaru = FakeKitaru(replay=replay(all_tools(history())), baseline=baseline)
        self.run_turn(weather_turn(), kitaru)

        [tool] = kitaru.children(NodeType.TOOL_CALL)
        self.assertEqual(tool.outputs, "Tool get_weather failed: NWS down")
        self.assertEqual(tool.status, NodeStatus.FAILED)

    def test_lossy_history_result_is_refused(self):
        lossy = {"value": "cut", "_kitaru_truncated": {"reason": "max_characters", "original_characters": 9}}
        kitaru = FakeKitaru(replay=replay(all_tools(history())),
                            baseline=[baseline_tool("get_weather", WEATHER_ARGS, lossy)])

        with self.assertRaisesRegex(ToolPolicyError, "cannot be replayed safely"):
            self.run_turn(weather_turn(), kitaru)

    def test_in_progress_baseline_is_refused_before_recording(self):
        kitaru = FakeKitaru(replay=replay(all_tools(history())), baseline_status=SessionStatus.IN_PROGRESS)

        with self.assertRaisesRegex(ToolPolicyError, "still in progress"):
            self.run_turn(weather_turn(), kitaru)
        self.assertIsNone(kitaru.session)

    def test_static_cases_match_exactly_or_by_subset(self):
        cases = [StaticCase(match={"park_code": "jotr"}, match_mode=StaticMatchMode.EXACT, result="Desert heat"),
                 StaticCase(match={"park_code": "yose"}, match_mode=StaticMatchMode.SUBSET, result={"sky": "clear"})]
        kitaru = FakeKitaru(replay=replay(all_tools(static(*cases))))
        self.run_turn(weather_turn(), kitaru)

        [tool] = kitaru.children(NodeType.TOOL_CALL)
        self.assertEqual(tool.outputs, '{"sky": "clear"}')
        self.assertEqual(tool.attributes["policy"], "static")
        self.weather.assert_not_called()

    def test_unapproved_live_calls_are_refused_before_recording(self):
        live = [
            ToolPolicy(default=PassthroughConfig()),
            all_tools(history(), get_weather=PassthroughConfig()),
            all_tools(history(), get_weather=history(on_miss=ToolPolicyOnMiss.PASSTHROUGH)),
        ]
        for policy in live:
            with self.subTest(policy=policy):
                kitaru = FakeKitaru(replay=replay(policy))
                with self.assertRaisesRegex(ToolPolicyError, "not approved"):
                    self.run_turn(weather_turn(), kitaru)
                self.assertIsNone(kitaru.session)
        self.weather.assert_not_called()

    def test_approved_tool_runs_live(self):
        policy = all_tools(history(), get_weather=PassthroughConfig())
        kitaru = FakeKitaru(replay=replay(policy))
        self.run_turn(weather_turn(), kitaru, approved=frozenset({"get_weather"}))

        self.weather.assert_called_once_with(park_code="yose")
        [tool] = kitaru.children(NodeType.TOOL_CALL)
        self.assertEqual(tool.attributes, {})

    def test_unsupported_policies_are_refused_before_recording(self):
        refused = {
            "llm": all_tools(history(), get_weather=LLMConfig(model="gpt-5")),
            "scope": all_tools(history(scope=HistoryScope.AGENT)),
            "unknown tools": ToolPolicy(default=history(), tools={"book_hotel": history()}),
        }
        for reason, policy in refused.items():
            with self.subTest(reason):
                kitaru = FakeKitaru(replay=replay(policy))
                with self.assertRaises(ToolPolicyError):
                    self.run_turn(weather_turn(), kitaru)
                self.assertIsNone(kitaru.session)


class EventLoopTests(unittest.TestCase):
    def test_run_turn_refuses_a_running_event_loop(self):
        async def inside_loop():
            run_turn(lambda question, history: "", client_factory=FakeKitaru)

        with self.assertRaisesRegex(RuntimeError, "running event loop"):
            asyncio.run(inside_loop())


if __name__ == "__main__":
    unittest.main()
