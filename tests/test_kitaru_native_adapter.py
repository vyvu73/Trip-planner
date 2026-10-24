import asyncio
import json
import os
import unittest
import uuid
from types import SimpleNamespace
from unittest import mock

# memory.py and retrieve_chunks.py build OpenAI clients at import time. An
# empty Braintrust key keeps app spans local (load_dotenv does not override).
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ["BRAINTRUST_API_KEY"] = ""
SECRET = "sentinel-secret-value-1234"
os.environ["TEST_SENTINEL_API_KEY"] = SECRET

from kitaru.api_models.v1.replay import ToolLookupMatch, ToolLookupResponse
from kitaru.api_models.v1.replay_config import ReplayOverride, ToolPolicy
from kitaru.cache_keys import compute_tool_cache_key

import app
from src.generation import generate_response, memory
from src.generation.memory import RouteDecision
from src.kitaru_native import adapter
from src.kitaru_native.adapter import KitaruWriter, run_turn

ROUTE_ARGS = '{"origin":"San Jose, CA","destination":"yose"}'
COST_ARGS = '{"miles":182,"origin_state":"CA","mpg":null}'


# --- fakes ---------------------------------------------------------------------

class FakeKitaru:
    """Stores what the adapter writes; fail_on names an operation to break."""

    def __init__(self, replay=None, lookups=None, fail_on=None):
        self.sessions = {}
        self.nodes = {}
        self.updates = []
        self.replay = replay
        self.lookups = lookups or {}
        self.lookup_calls = []
        self.fail_on = fail_on
        self.ingest_count = 0

    def check(self, operation):
        if self.fail_on == operation:
            raise ConnectionError(f"{operation} failed")

    def factory(self):
        return FakeAsyncClient(self)

    def tree(self, session_id):
        return self.nodes[session_id]

    def all_text(self):
        dumped = [request.model_dump_json() for request in self.sessions.values()]
        dumped += [node.model_dump_json() for nodes in self.nodes.values() for node in nodes.values()]
        dumped += [update.model_dump_json() for update in self.updates]
        return "\n".join(dumped)


class FakeAsyncClient:
    def __init__(self, fake):
        self.sessions = FakeSessions(fake)
        self.replays = FakeReplays(fake)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeSessions:
    def __init__(self, fake):
        self.fake = fake

    async def create(self, request):
        self.fake.check("create")
        session_id = uuid.uuid4()
        self.fake.sessions[session_id] = request
        self.fake.nodes[session_id] = {}
        return SimpleNamespace(id=session_id)

    async def ingest_nodes(self, session_id, batch):
        self.fake.ingest_count += 1
        self.fake.check(f"ingest{self.fake.ingest_count}")
        for node in batch.nodes:
            self.fake.nodes[session_id][node.external_id] = node   # replaced whole
        return []

    async def update(self, session_id, request):
        # Like the server: inputs are fixed once a (non-Mastra) session exists.
        if "inputs" in request.model_fields_set:
            raise ValueError("422: invalid Mastra replay finalization")
        self.fake.check("update")
        if self.fake.fail_on == "first_update" and not self.fake.updates:
            self.fake.updates.append(None)
            raise ConnectionError("update failed")
        self.fake.updates.append(request)
        return SimpleNamespace(id=session_id)


class FakeReplays:
    def __init__(self, fake):
        self.fake = fake

    async def get(self, replay_id):
        return self.fake.replay

    async def tool_lookup(self, replay_id, request):
        self.fake.lookup_calls.append(request)
        result = self.fake.lookups.get((request.cache_key, request.occurrence))
        if result is None:
            return ToolLookupResponse()
        return ToolLookupResponse(match=ToolLookupMatch(result=result, status="completed"))


def replay_config(override=None, tool_policy=None):
    return SimpleNamespace(
        override=ReplayOverride.model_validate(override) if override else None,
        tool_policy=ToolPolicy.model_validate(tool_policy or {"default": {"type": "passthrough"}}),
    )


def function_call(name, arguments, call_id):
    return SimpleNamespace(type="function_call", name=name, arguments=arguments, call_id=call_id)


def model_response(text="", calls=(), model="gpt-5-mini-2025-08-07"):
    return SimpleNamespace(
        id=f"resp-{uuid.uuid4().hex[:6]}", model=model, output=list(calls), output_text=text,
        usage=SimpleNamespace(input_tokens=100, output_tokens=20,
                              input_tokens_details=SimpleNamespace(cached_tokens=0),
                              output_tokens_details=SimpleNamespace(reasoning_tokens=5)),
    )


class FakeAnswerModel:
    """Streams scripted rounds; each round is (text, function calls)."""

    def __init__(self, rounds, fail_mid_stream=False):
        self.rounds = list(rounds)
        self.requests = []
        self.fail_mid_stream = fail_mid_stream
        self.responses = SimpleNamespace(create=self.create)

    def create(self, **kwargs):
        self.requests.append(kwargs)
        text, calls = self.rounds.pop(0)

        def events():
            for start in range(0, len(text), 8):
                yield SimpleNamespace(type="response.output_text.delta", delta=text[start:start + 8])
                if self.fail_mid_stream:
                    raise TimeoutError(f"stream dropped, key {SECRET}")
            yield SimpleNamespace(type="response.completed", response=model_response(text, calls))
        return events()


class FakeRouter:
    def __init__(self):
        self.requests = []
        self.responses = SimpleNamespace(parse=self.parse)

    def parse(self, **kwargs):
        self.requests.append(kwargs)
        response = model_response()
        response.output_parsed = RouteDecision(
            route="in_scope", standalone_question="Drive San Jose to Yosemite and gas cost?",
            park_codes=["yose"], clarifying_question=None, redirect_message=None)
        return response


TOOL_RESULTS = {"get_route": "182 miles, about 4 h 13 min. Origin state: CA",
                "get_transportation_cost": "One way: $43.91. Round trip: $87.81."}

TOOL_ROUNDS = [
    ("", [function_call("get_route", ROUTE_ARGS, "c1")]),
    ("", [function_call("get_transportation_cost", COST_ARGS, "c2")]),
    ("About 4 h 13 min, and gas is about $43.91 one way.", []),
]


def answer(question, history):
    text = ""
    for text in app.stream_answer(question, history):
        pass
    if text == app.ERROR_MESSAGE or not text:
        raise RuntimeError("The app could not answer")
    return text


class NativeAdapterTest(unittest.TestCase):
    def setUp(self):
        self.router = FakeRouter()
        self.tool_calls = []

        def fake_run_tool(name, arguments_json):
            self.tool_calls.append((name, arguments_json))
            return TOOL_RESULTS[name]

        patches = [
            mock.patch.object(memory, "client", self.router),
            mock.patch.object(generate_response, "run_tool", fake_run_tool),
            mock.patch.object(app, "retrieve_chunks", lambda question, k=7: [
                {"park_name": "Yosemite", "title": "Fees", "content": "c", "source_type": "nps"}]),
            mock.patch.object(app, "alerts_context", lambda park_codes: ""),
            mock.patch.object(generate_response, "build_input", lambda *a, **k: "PROMPT"),
            mock.patch("builtins.print"),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def run_with(self, fake, rounds=TOOL_ROUNDS, history=(), question="Drive and gas?",
                 replay_id=None, **model_kwargs):
        self.model = FakeAnswerModel(rounds, **model_kwargs)
        with mock.patch.object(generate_response, "client", self.model):
            return run_turn(answer, question, list(history),
                            writer=KitaruWriter(fake.factory), replay_id=replay_id)

    def only_session(self, fake):
        (session_id, request), = fake.sessions.items()
        return session_id, request, fake.tree(session_id)

    # Lifecycle and node tree

    def test_records_one_completed_session_with_tool_nodes_under_their_model_call(self):
        fake = FakeKitaru()
        text = self.run_with(fake)

        self.assertEqual(text, "About 4 h 13 min, and gas is about $43.91 one way.")
        session_id, created, nodes = self.only_session(fake)
        self.assertEqual(created.origin, "recorded")
        self.assertEqual(created.status, "in_progress")
        self.assertEqual(fake.updates[-1].status, "completed")
        self.assertEqual(fake.updates[-1].outputs, text)

        root = next(n for n in nodes.values() if n.name == "kitaru-run")
        self.assertIsNone(root.parent_external_id)
        self.assertEqual(root.status, "completed")
        by_name = {}
        for node in nodes.values():
            by_name.setdefault(node.name, []).append(node)
        self.assertEqual(len(by_name["openai.responses.create"]), 3)
        self.assertEqual(len(by_name["openai.responses.parse"]), 1)
        self.assertEqual([n.name for n in nodes.values() if n.node_type == "tool_call"],
                         ["get_route", "get_transportation_cost"])   # once each

        route = by_name["get_route"][0]
        requesting = nodes[route.parent_external_id]
        self.assertEqual(requesting.name, "openai.responses.create")
        self.assertEqual(requesting.outputs["function_calls"][0]["name"], "get_route")
        self.assertEqual(route.inputs, ROUTE_ARGS)
        self.assertEqual(route.outputs, TOOL_RESULTS["get_route"])
        self.assertEqual(requesting.tokens.reasoning_tokens, 5)
        self.assertEqual(self.tool_calls, [("get_route", ROUTE_ARGS),
                                           ("get_transportation_cost", COST_ARGS)])

    def test_session_inputs_keep_every_turn_so_the_replay_can_be_replayed(self):
        fake = FakeKitaru()
        history = [{"role": "user", "content": "I'm visiting Lassen with my kids"},
                   {"role": "assistant", "content": "When are you going?"}]
        self.run_with(fake, rounds=[("Usually closed in June.", [])], history=history,
                      question="Is Bumpass Hell worth it in June?")

        _, created, _ = self.only_session(fake)
        self.assertEqual(created.input_text_selector, "/turns/1/inputs")
        self.assertEqual(created.inputs["turns"], [
            {"inputs": "I'm visiting Lassen with my kids", "outputs": "When are you going?"},
            {"inputs": "Is Bumpass Hell worth it in June?", "outputs": None},
        ])
        self.assertEqual(fake.updates[-1].outputs, "Usually closed in June.")
        self.assertEqual(self.router.requests[0]["input"].count("I'm visiting Lassen"), 1)

    def test_replay_session_origin_and_module_globals_restored(self):
        fake = FakeKitaru(replay=replay_config())
        originals = (generate_response.run_tool, app.retrieve_chunks, app.alerts_context, memory.client)
        self.run_with(fake, replay_id=uuid.uuid4())

        _, created, _ = self.only_session(fake)
        self.assertEqual(created.origin, "replay")
        self.assertEqual(originals, (generate_response.run_tool, app.retrieve_chunks,
                                     app.alerts_context, memory.client))

    # Failures

    def test_session_create_failure_starts_no_app_work(self):
        fake = FakeKitaru(fail_on="create")
        with self.assertRaises(ConnectionError):
            self.run_with(fake)
        self.assertEqual(self.router.requests, [])

    def test_root_write_failure_marks_session_failed_before_app_work(self):
        fake = FakeKitaru(fail_on="ingest1")
        with self.assertRaises(ConnectionError):
            self.run_with(fake)
        self.assertEqual(fake.updates[-1].status, "failed")
        self.assertEqual(self.router.requests, [])

    def test_final_node_write_failure_fails_the_run_and_never_completes_the_session(self):
        fake = FakeKitaru(fail_on="ingest2")
        with self.assertRaises(ConnectionError):
            self.run_with(fake)
        self.assertNotIn("completed", [u.status for u in fake.updates])

    def test_failed_completion_update_marks_the_session_failed(self):
        fake = FakeKitaru(fail_on="first_update")
        with self.assertRaises(ConnectionError):
            self.run_with(fake)
        self.assertEqual(fake.updates[-1].status, "failed")

    def test_app_failure_stays_primary_when_recording_also_fails(self):
        fake = FakeKitaru(fail_on="ingest2")
        with self.assertRaises(RuntimeError) as raised:
            self.run_with(fake, rounds=[("Partial answer that drops", [])], fail_mid_stream=True)
        self.assertIn("could not answer", str(raised.exception))
        self.assertTrue(any("recording the failed nodes" in note for note in raised.exception.__notes__))

    def test_mid_stream_failure_records_a_failed_model_call_and_failed_session(self):
        fake = FakeKitaru()
        with self.assertRaises(RuntimeError):
            self.run_with(fake, rounds=[("Partial answer that drops", [])], fail_mid_stream=True)
        _, _, nodes = self.only_session(fake)
        failed = [n for n in nodes.values() if n.name == "openai.responses.create"]
        self.assertEqual(failed[0].status, "failed")
        self.assertIn("TimeoutError", failed[0].error)
        self.assertEqual(fake.updates[-1].status, "failed")

    def test_refuses_to_run_inside_an_event_loop(self):
        async def inside_loop():
            return run_turn(answer, "q", [], writer=KitaruWriter(FakeKitaru().factory))
        with self.assertRaisesRegex(RuntimeError, "event loop"):
            asyncio.run(inside_loop())

    # Overrides

    def test_overrides_apply_to_model_system_prompt_params_and_prompt(self):
        fake = FakeKitaru(replay=replay_config(override={
            "model": "gpt-5", "system_prompt": "Be brief.", "prompt": "Replacement question?",
            "model_params": {"reasoning": {"effort": "high"}, "unknown_param": 1},
        }))
        self.run_with(fake, replay_id=uuid.uuid4())

        first = self.model.requests[0]
        self.assertEqual(first["model"], "gpt-5")
        self.assertEqual(first["instructions"], "Be brief.")
        self.assertEqual(first["reasoning"], {"effort": "high"})
        self.assertNotIn("unknown_param", first)
        self.assertEqual(self.router.requests[0]["model"], "gpt-5")
        self.assertNotIn("instructions", self.router.requests[0])   # router keeps its prompt
        self.assertIn("Replacement question?", self.router.requests[0]["input"])

        _, created, nodes = self.only_session(fake)
        self.assertEqual(created.inputs["turns"][-1]["inputs"], "Replacement question?")
        answer_node = next(n for n in nodes.values() if n.name == "openai.responses.create")
        self.assertEqual(answer_node.requested_model, generate_response.MODEL)
        self.assertEqual(answer_node.inputs["instructions"], "Be brief.")

    def test_model_map_override_only_replaces_matching_models(self):
        fake = FakeKitaru(replay=replay_config(override={"model": {"some-other-model": "gpt-5"}}))
        self.run_with(fake, replay_id=uuid.uuid4())
        self.assertEqual(self.model.requests[0]["model"], generate_response.MODEL)

    # Tool policies

    def test_static_policy_answers_without_running_the_tool(self):
        fake = FakeKitaru(replay=replay_config(tool_policy={
            "default": {"type": "passthrough"},
            "tools": {"get_route": {"type": "static", "on_miss": "fail", "cases": [
                {"match": {"destination": "yose"}, "match_mode": "subset", "result": "STATIC ROUTE"}]}},
        }))
        self.run_with(fake, replay_id=uuid.uuid4())

        self.assertEqual(self.tool_calls, [("get_transportation_cost", COST_ARGS)])
        _, _, nodes = self.only_session(fake)
        route = next(n for n in nodes.values() if n.name == "get_route")
        self.assertEqual((route.outputs, route.metadata["source"]), ("STATIC ROUTE", "static"))

    def test_history_policy_replays_recorded_results_by_occurrence(self):
        route_key = compute_tool_cache_key("get_route", ROUTE_ARGS)
        cost_key = compute_tool_cache_key("get_transportation_cost", COST_ARGS)
        fake = FakeKitaru(
            replay=replay_config(tool_policy={
                "default": {"type": "history", "scope": "baseline", "on_miss": "fail"}}),
            lookups={(route_key, 0): "RECORDED ROUTE", (cost_key, 0): "RECORDED COST"},
        )
        self.run_with(fake, replay_id=uuid.uuid4())

        self.assertEqual(self.tool_calls, [])
        self.assertEqual([c.occurrence for c in fake.lookup_calls], [0, 0])
        tool_outputs = [o["output"] for o in self.model.requests[1]["input"]]
        self.assertEqual(tool_outputs, ["RECORDED ROUTE"])

    def test_repeated_identical_calls_use_successive_occurrences(self):
        key = compute_tool_cache_key("get_route", ROUTE_ARGS)
        fake = FakeKitaru(
            replay=replay_config(tool_policy={
                "default": {"type": "history", "scope": "baseline", "on_miss": "error_result"}}),
            lookups={(key, 0): "FIRST", (key, 1): "SECOND"},
        )
        rounds = [("", [function_call("get_route", ROUTE_ARGS, "c1"),
                        function_call("get_route", ROUTE_ARGS, "c2")]), ("Done here, both routes.", [])]
        self.run_with(fake, rounds=rounds, replay_id=uuid.uuid4())
        self.assertEqual([o["output"] for o in self.model.requests[1]["input"]], ["FIRST", "SECOND"])

    def test_history_miss_with_fail_fails_the_run_with_the_policy_error(self):
        fake = FakeKitaru(replay=replay_config(tool_policy={
            "default": {"type": "history", "scope": "baseline", "on_miss": "fail"}}))
        with self.assertRaisesRegex(adapter.ToolPolicyError, "No recorded result for get_route"):
            self.run_with(fake, replay_id=uuid.uuid4())
        self.assertEqual(self.tool_calls, [])
        self.assertEqual(fake.updates[-1].status, "failed")

    def test_history_miss_with_error_result_tells_the_model(self):
        fake = FakeKitaru(replay=replay_config(tool_policy={
            "default": {"type": "history", "scope": "baseline", "on_miss": "error_result"}}))
        self.run_with(fake, replay_id=uuid.uuid4())
        self.assertEqual(self.tool_calls, [])
        self.assertIn("no recorded result", self.model.requests[1]["input"][0]["output"])

    def test_history_miss_with_passthrough_runs_the_live_tool(self):
        fake = FakeKitaru(replay=replay_config(tool_policy={
            "default": {"type": "history", "scope": "baseline", "on_miss": "passthrough"}}))
        self.run_with(fake, replay_id=uuid.uuid4())
        self.assertEqual(len(self.tool_calls), 2)

    def test_null_recorded_result_is_a_miss_not_an_empty_answer(self):
        # Imported baselines also hold wrap_openai's zero-length node per call,
        # whose output is null.
        key = compute_tool_cache_key("get_route", ROUTE_ARGS)
        fake = FakeKitaru(replay=replay_config(tool_policy={
            "default": {"type": "history", "scope": "baseline", "on_miss": "error_result"}}))
        fake.lookups[(key, 0)] = None
        self.run_with(fake, replay_id=uuid.uuid4())
        self.assertIn("no recorded result", self.model.requests[1]["input"][0]["output"])

    def test_wider_history_scope_sends_no_occurrence(self):
        fake = FakeKitaru(replay=replay_config(tool_policy={
            "default": {"type": "history", "scope": "agent", "on_miss": "error_result"}}))
        self.run_with(fake, replay_id=uuid.uuid4())
        self.assertEqual([c.occurrence for c in fake.lookup_calls], [None, None])

    def test_llm_tool_policy_is_rejected_before_any_write(self):
        fake = FakeKitaru(replay=replay_config(tool_policy={
            "default": {"type": "llm", "model": "gpt-5-mini"}}))
        with self.assertRaisesRegex(RuntimeError, "llm tool policy"):
            self.run_with(fake, replay_id=uuid.uuid4())
        self.assertEqual(fake.sessions, {})

    # Privacy

    def test_credential_values_never_reach_recorded_data(self):
        fake = FakeKitaru()
        rounds = [("", [function_call("get_route", json.dumps({"origin": SECRET, "destination": "yose"}), "c1")]),
                  ("Answer quoting " + SECRET, [])]
        with mock.patch.object(generate_response, "build_input", lambda *a, **k: f"PROMPT {SECRET}"):
            self.run_with(fake, rounds=rounds, question=f"Question with {SECRET}")

        self.assertNotIn(SECRET, fake.all_text())
        self.assertIn("[redacted]", fake.all_text())

    def test_failure_path_redacts_credentials_in_errors(self):
        fake = FakeKitaru()
        with self.assertRaises(RuntimeError):
            self.run_with(fake, rounds=[("Partial answer that drops", [])], fail_mid_stream=True)
        self.assertNotIn(SECRET, fake.all_text())


if __name__ == "__main__":
    unittest.main()
