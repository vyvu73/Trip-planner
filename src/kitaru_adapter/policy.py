# Replay tool policies for the trip planner's local tools and pipeline steps
# (retrieval and alerts, which the app runs on every question; see turn.py).
#
# Everything that can refuse a replay runs before the app does: unknown tool
# names, unsupported policies, and live calls nobody approved. During the
# run, policies are sync calls.
#
# History asks the Kitaru server (replays.tool_lookup), like Kitaru's released
# adapters. A replay task may not read its baseline session directly, and the
# server owns matching for every scope. With baseline scope, the nth
# identical call replays the nth recorded one. The server keys tool calls
# only, so pipeline steps (spans) cannot replay history: static or approved
# passthrough only.
import json
from dataclasses import dataclass, field

from kitaru.api_models.v1.replay import ToolLookupRequest
from kitaru.api_models.v1.replay_config import (
    HistoryConfig,
    HistoryScope,
    PassthroughConfig,
    StaticConfig,
    StaticMatchMode,
    ToolPolicyOnMiss,
)
from kitaru.api_models.v1.session_node import NodeStatus
from kitaru.cache_keys import compute_tool_cache_key

from src.kitaru_adapter.capture import has_capture_marker

# Tools and pipeline steps a replay may run for real, against live APIs and
# the database. Anything not listed must be answered from a static case or
# the baseline's history. Add a name to approve live calls, e.g. "get_weather".
# Approved 2026-10-24: the two read-only pipeline steps (Postgres search, one
# embedding call, public NPS alerts), because a replayed router rewrites the
# question and the search query then misses the recorded history.
APPROVED_PASSTHROUGH_TOOLS = frozenset({"retrieve_chunks", "alerts_context"})


class ToolPolicyError(RuntimeError):
    """A replay tool policy that cannot be applied safely."""


class ToolPolicyMissError(ToolPolicyError):
    """A tool call with no static or history result, under on_miss: fail."""


@dataclass
class Decision:
    """What a policy did with one call: run it for real, or return value instead."""
    passthrough: bool = False
    value: object = None
    policy: str | None = None    # set when the value did not come from the call
    failed: bool = False         # the value stands for a failed call
    error: str | None = None     # a recorded failure's error, when replaying one


@dataclass
class ToolPolicies:
    """
    Per-run policy state: configs by tool, and for history the server lookup,
    lookup(tool_name, cache_key, occurrence) -> ToolLookupMatch or None.
    """
    configs: dict
    lookup: object = None
    occurrences: dict = field(default_factory=dict)   # (tool, cache_key) -> next baseline occurrence

    def decide(self, name, arguments_json):
        """A model tool call: arguments and any replayed result are text."""
        config = self.configs.get(name)
        if config is None or isinstance(config, PassthroughConfig):
            return Decision(passthrough=True)
        try:
            arguments = json.loads(arguments_json)
        except (TypeError, ValueError) as error:
            raise ToolPolicyError(f"Invalid JSON arguments for tool '{name}'") from error
        decision = self.decide_arguments(name, arguments)
        if decision.passthrough:
            return decision
        if decision.error is not None:
            # The app's tools never raise; a failed call reached the model as this text.
            decision.value = f"Tool {name} failed: {decision.error}"
        else:
            decision.value = as_tool_text(decision.value)
        return decision

    def decide_arguments(self, name, arguments):
        """Any call by its parsed arguments; a replayed value is returned as recorded."""
        config = self.configs.get(name)
        if config is None or isinstance(config, PassthroughConfig):
            return Decision(passthrough=True)
        if isinstance(config, StaticConfig):
            case = next((case for case in config.cases if case_matches(case, arguments)), None)
            if case is not None:
                return Decision(value=case.result, policy="static")
        else:
            decision = self._history(name, arguments, config.scope)
            if decision is not None:
                return decision
        return miss(config, name)

    def _history(self, name, arguments, scope):
        cache_key = compute_tool_cache_key(name, arguments)
        if cache_key is None:
            raise ToolPolicyError(f"Arguments for tool '{name}' cannot be replayed safely")
        if scope is HistoryScope.BASELINE:
            # Count per key and advance only on a hit, so repeated identical
            # calls replay their own recorded results in order.
            key = (name, cache_key)
            occurrence = self.occurrences.get(key, 0)
            match = self.lookup(name, cache_key, occurrence)
            if match is not None:
                self.occurrences[key] = occurrence + 1
        else:
            match = self.lookup(name, cache_key, None)    # wider scopes: the latest match
        if match is None:
            return None
        if match.status is NodeStatus.FAILED:
            return Decision(policy="history", failed=True, error=match.error or "unknown error")
        if match.status is not NodeStatus.COMPLETED:
            raise ToolPolicyError(f"History lookup for tool '{name}' returned status '{match.status.value}'")
        if has_capture_marker(match.result):
            raise ToolPolicyError(f"History result for tool '{name}' cannot be replayed safely")
        return Decision(value=match.result, policy="history")


async def lookup_match(client, replay_id, tool_name, cache_key, occurrence):
    """One server history lookup, as Kitaru's released adapters make it."""
    response = await client.replays.tool_lookup(replay_id, ToolLookupRequest(
        tool_name=tool_name, cache_key=cache_key, occurrence=occurrence))
    if "match" not in response.model_fields_set:
        raise ToolPolicyError("The Kitaru server's tool lookup has no 'match'; upgrade the server for history replay")
    return response.match


def case_matches(case, arguments):
    """Exact or shallow-subset matching against parsed call arguments."""
    if case.match is None:
        return True
    if case.match_mode is StaticMatchMode.EXACT:
        return arguments == case.match
    if not isinstance(case.match, dict) or not isinstance(arguments, dict):
        return False
    return all(name in arguments and arguments[name] == value for name, value in case.match.items())


def miss(config, name):
    if config.on_miss is ToolPolicyOnMiss.PASSTHROUGH:
        return Decision(passthrough=True)
    message = f"No {config.type} result for tool '{name}'"
    if config.on_miss is ToolPolicyOnMiss.ERROR_RESULT:
        return Decision(value={"error": message}, policy=config.type, failed=True)
    raise ToolPolicyMissError(message)


def as_tool_text(result):
    """Tools hand the model text; a structured result is sent as JSON."""
    return result if isinstance(result, str) else json.dumps(result)


def runs_live(config):
    return isinstance(config, PassthroughConfig) or (
        isinstance(config, (StaticConfig, HistoryConfig))
        and config.on_miss is ToolPolicyOnMiss.PASSTHROUGH)


def check_policy(tool_policy, tool_names, approved=APPROVED_PASSTHROUGH_TOOLS, step_names=()):
    """
    Each tool's and pipeline step's effective config, or ToolPolicyError if
    the replay is unsafe. Steps are spans, which the server's history lookup
    does not serve, so history is refused for them. They feed the app data,
    not the model text, so an {"error": ...} stand-in would break the app:
    error_result is refused too.
    """
    step_names = list(step_names)
    tool_names = [*tool_names, *step_names]
    unknown = sorted(set(tool_policy.tools) - set(tool_names))
    if unknown:
        raise ToolPolicyError(f"Tool policy names unknown tools: {', '.join(unknown)}")

    configs = {}
    for name in tool_names:
        config = tool_policy.tools.get(name, tool_policy.default)
        if not isinstance(config, (PassthroughConfig, StaticConfig, HistoryConfig)):
            raise ToolPolicyError(f"Tool policy '{config.type}' is not supported for '{name}'")
        if name in step_names and isinstance(config, HistoryConfig):
            raise ToolPolicyError(
                f"History is not supported for pipeline step '{name}'; give it a static "
                f"policy, or passthrough if it is in APPROVED_PASSTHROUGH_TOOLS")
        if (name in step_names and isinstance(config, StaticConfig)
                and config.on_miss is ToolPolicyOnMiss.ERROR_RESULT):
            raise ToolPolicyError(f"on_miss: error_result is not supported for pipeline step '{name}'")
        if runs_live(config) and name not in approved:
            raise ToolPolicyError(
                f"Replay could call '{name}' live, which is not approved; give it a static "
                f"or history policy without on_miss: passthrough, or add it to "
                f"APPROVED_PASSTHROUGH_TOOLS")
        configs[name] = config
    return configs


def prepare_policies(replay, tool_names, lookup, approved=APPROVED_PASSTHROUGH_TOOLS, step_names=()):
    """Tool policies for one run: all passthrough when recording, checked when replaying."""
    if replay is None:
        return ToolPolicies(configs={})
    return ToolPolicies(configs=check_policy(replay.tool_policy, tool_names, approved, step_names),
                        lookup=lookup)
