import os
import unittest
from unittest import mock

from pydantic import ValidationError

# memory.py and retrieve_chunks.py build OpenAI clients at import time.
os.environ.setdefault("OPENAI_API_KEY", "test-key")

from src.generation import memory
from src.generation.memory import RouteDecision, route_message
import app


def parsed(decision):
    return mock.Mock(output_parsed=decision)


def history_of(n):
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": f"msg-{i}"} for i in range(n)]


class RouteMessageTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(memory.client.responses, "parse")
        self.parse = patcher.start()
        self.addCleanup(patcher.stop)

    def test_router_called_on_first_message(self):
        decision = RouteDecision(route="in_scope", standalone_question="Best hikes in Yosemite?", park_codes=["yose"], clarifying_question=None, redirect_message=None)
        self.parse.return_value = parsed(decision)

        result = route_message([], "Best hikes in Yosemite?")

        self.parse.assert_called_once()
        self.assertEqual(self.parse.call_args.kwargs["text_format"], RouteDecision)
        self.assertEqual(result, decision)

    def test_only_last_six_history_messages_sent(self):
        self.parse.return_value = parsed(
            RouteDecision(route="in_scope", standalone_question="q", park_codes=[], clarifying_question=None, redirect_message=None)
        )

        route_message(history_of(10), "what about day 2?")

        prompt = self.parse.call_args.kwargs["input"]
        for i in range(4):
            self.assertNotIn(f"msg-{i}\n", prompt)
        for i in range(4, 10):
            self.assertIn(f"msg-{i}", prompt)

    def test_api_exception_propagates(self):
        self.parse.side_effect = RuntimeError("api down")

        with self.assertRaises(RuntimeError):
            route_message(history_of(2), "hello")

    def test_ambiguous_without_clarifying_question_raises(self):
        self.parse.return_value = parsed(
            RouteDecision(route="ambiguous", standalone_question="weather this weekend?", park_codes=[], clarifying_question=None, redirect_message=None)
        )

        with self.assertRaises(ValueError):
            route_message([], "weather this weekend?")

    def test_out_of_scope_without_redirect_message_raises(self):
        self.parse.return_value = parsed(
            RouteDecision(route="out_of_scope", standalone_question="Write a poem", park_codes=[], clarifying_question=None, redirect_message=None)
        )

        with self.assertRaises(ValueError):
            route_message([], "Write a poem")

    def test_prompt_lists_park_codes(self):
        self.parse.return_value = parsed(
            RouteDecision(route="in_scope", standalone_question="q", park_codes=[], clarifying_question=None, redirect_message=None)
        )

        route_message([], "hello")

        self.assertIn("Yosemite National Park (yose)", self.parse.call_args.kwargs["input"])

    def test_park_codes_reject_unsupported_code(self):
        with self.assertRaises(ValidationError):
            RouteDecision(route="in_scope", standalone_question="q", park_codes=["kica"], clarifying_question=None, redirect_message=None)


class HandleMessageTests(unittest.TestCase):
    def setUp(self):
        self.route = self.patch("app.route_message")
        self.retrieve = self.patch("app.retrieve_chunks")
        self.generate = self.patch("app.stream_response")
        self.alerts = self.patch("app.alerts_context")

    def patch(self, target):
        patcher = mock.patch(target)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def answer(self, message, history=None):
        chat, textbox, state = app.handle_message(message, history or [])
        self.assertEqual(textbox, "")
        self.assertEqual(chat, state)
        return chat[-1]["content"]

    def test_out_of_scope_returns_redirect_message(self):
        redirect = "I'm more of a California parks AI than a Yellowstone one!"
        self.route.return_value = RouteDecision(
            route="out_of_scope",
            standalone_question="Plan a trip to Yellowstone",
            park_codes=[],
            clarifying_question=None,
            redirect_message=redirect,
        )

        self.assertEqual(self.answer("Plan a trip to Yellowstone"), redirect)
        self.retrieve.assert_not_called()
        self.generate.assert_not_called()
        self.alerts.assert_not_called()

    def test_ambiguous_returns_clarifying_question(self):
        self.route.return_value = RouteDecision(
            route="ambiguous",
            standalone_question="What's the weather this weekend?",
            park_codes=[],
            clarifying_question="Which park are you visiting?",
            redirect_message=None,
        )

        self.assertEqual(self.answer("What's the weather this weekend?"), "Which park are you visiting?")
        self.retrieve.assert_not_called()
        self.generate.assert_not_called()
        self.alerts.assert_not_called()

    def test_in_scope_retrieves_with_standalone_and_generates_with_raw(self):
        history = history_of(2)
        self.route.return_value = RouteDecision(
            route="in_scope", standalone_question="Day 2 of a Yosemite trip?", park_codes=["yose"], clarifying_question=None, redirect_message=None
        )
        self.retrieve.return_value = ["chunk"]
        self.alerts.return_value = "ALERTS"
        self.generate.return_value = iter(["Day", "Day 2 plan"])

        self.assertEqual(self.answer("what about day 2?", history), "Day 2 plan")
        self.retrieve.assert_called_once_with("Day 2 of a Yosemite trip?", k=7)
        self.alerts.assert_called_once_with(["yose"])
        self.generate.assert_called_once_with("what about day 2?", ["chunk"], recent=history, alerts="ALERTS")

    def test_missing_redirect_message_shows_error_message(self):
        self.route.side_effect = ValueError("Router returned out_of_scope without a redirect message")

        self.assertTrue(self.answer("Write a poem").startswith("Sorry, I ran into a problem"))
        self.retrieve.assert_not_called()
        self.generate.assert_not_called()

    def test_router_error_shows_error_message(self):
        self.route.side_effect = RuntimeError("api down")

        self.assertTrue(self.answer("hello").startswith("Sorry, I ran into a problem"))
        self.retrieve.assert_not_called()
        self.generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
