# Replay tool policies for the trip planner's local tools.
#
# Everything that can refuse a replay runs before the app does: unknown tool
# names, unsupported policies, live tool calls nobody approved, and a baseline
# that cannot serve history. During the run, policies are plain sync lookups.
#
# History is preloaded from the baseline session (the app is sync, so it
# cannot ask the Kitaru server mid-run). That limits history to the baseline
# scope; the nth identical call replays the nth recorded one, in baseline
# order, like the server's occurrence lookup.
import json
from dataclasses import dataclass, field

from kitaru.api_models.v1.replay_config import (
    HistoryConfig,
    HistoryScope,
    PassthroughConfig,
    StaticConfig,
    StaticMatchMode,
    ToolPolicyOnMiss,
)
from kitaru.api_models.v1.session import SessionStatus
from kitaru.api_models.v1.session_node import NodeStatus, NodeType, SessionNodeListParams
from kitaru.cache_keys import compute_tool_cache_key

from src.kitaru_adapter.capture import has_capture_marker

# Tools a replay may run for real, against live APIs. Empty: every replay
# must answer every tool call from a static case or the baseline's history.
# Add a tool name here to approve live calls to it, e.g. "get_weather".
APPROVED_PASSTHROUGH_TOOLS = frozenset()


class ToolPolicyError(RuntimeError):
    """A replay tool policy that cannot be applied safely."""


class ToolPolicyMissError(ToolPolicyError):
    """A tool call with no static or history result, under on_miss: fail."""


@dataclass
class Recorded:
    """One baseline tool call: its status, result, and whether it replays safely."""
    status: NodeStatus
    result: object
    error: str | None
    replayable: bool


@dataclass
class Decision:
    """What a policy did with one call: a result to return, or run the tool."""
    result: str | None = None
    policy: str | None = None    # set when the result did not come from the tool
    failed: bool = False

    @property
    def passthrough(self):
        return self.result is None


@dataclass
class ToolPolicies:
    """Per-run policy state: configs by tool, the baseline index, occurrence counters."""
    configs: dict
    history: dict = field(default_factory=dict)       # (tool, cache_key) -> [Recorded]
    occurrences: dict = field(default_factory=dict)   # (tool, cache_key) -> next index

    def decide(self, name, arguments_json):
        config = self.configs.get(name)
        if config is None or isinstance(config, PassthroughConfig):
            return Decision()
        try:
            arguments = json.loads(arguments_json)
        except (TypeError, ValueError) as error:
            raise ToolPolicyError(f"Invalid JSON arguments for tool '{name}'") from error
        if isinstance(config, StaticConfig):
            case = next((case for case in config.cases if case_matches(case, arguments)), None)
            if case is not None:
                return Decision(as_tool_text(case.result), "static")
        else:
            decision = self._history(name, arguments)
            if decision is not None:
                return decision
        return miss(config, name)

    def _history(self, name, arguments):
        cache_key = compute_tool_cache_key(name, arguments)
        if cache_key is None:
            return None
        key = (name, cache_key)
        index = self.occurrences.get(key, 0)
        matches = self.history.get(key, [])
        if index >= len(matches):
            return None
        self.occurrences[key] = index + 1
        match = matches[index]
        if not match.replayable:
            raise ToolPolicyError(f"History result for tool '{name}' cannot be replayed safely")
        if match.status is NodeStatus.FAILED:
            # The app's tools never raise; a failed call reached the model as this text.
            return Decision(f"Tool {name} failed: {match.error}", "history", failed=True)
        return Decision(as_tool_text(match.result), "history")


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
        return Decision()
    message = f"No {config.type} result for tool '{name}'"
    if config.on_miss is ToolPolicyOnMiss.ERROR_RESULT:
        return Decision(json.dumps({"error": message}), config.type, failed=True)
    raise ToolPolicyMissError(message)


def as_tool_text(result):
    """Tools hand the model text; a structured result is sent as JSON."""
    return result if isinstance(result, str) else json.dumps(result)


def runs_live(config):
    return isinstance(config, PassthroughConfig) or (
        isinstance(config, (StaticConfig, HistoryConfig))
        and config.on_miss is ToolPolicyOnMiss.PASSTHROUGH)


def check_policy(tool_policy, tool_names, approved=APPROVED_PASSTHROUGH_TOOLS):
    """Each tool's effective config, or ToolPolicyError if the replay is unsafe."""
    unknown = sorted(set(tool_policy.tools) - set(tool_names))
    if unknown:
        raise ToolPolicyError(f"Tool policy names unknown tools: {', '.join(unknown)}")

    configs = {}
    for name in tool_names:
        config = tool_policy.tools.get(name, tool_policy.default)
        if not isinstance(config, (PassthroughConfig, StaticConfig, HistoryConfig)):
            raise ToolPolicyError(f"Tool policy '{config.type}' is not supported for '{name}'")
        if isinstance(config, HistoryConfig) and config.scope is not HistoryScope.BASELINE:
            raise ToolPolicyError(
                f"History scope '{config.scope.value}' is not supported for '{name}'; use baseline")
        if runs_live(config) and name not in approved:
            raise ToolPolicyError(
                f"Replay could call '{name}' live, which is not approved; give it a static "
                f"or history policy without on_miss: passthrough, or add it to "
                f"APPROVED_PASSTHROUGH_TOOLS")
        configs[name] = config
    return configs


async def load_history(client, session_id, tool_names):
    """Index a terminal baseline's tool calls by (tool, cache key), in baseline order."""
    session = await client.sessions.get(session_id)
    if session.status is SessionStatus.IN_PROGRESS:
        raise ToolPolicyError("Baseline session is still in progress; its history is incomplete")

    history = {}
    params = SessionNodeListParams(include_payloads=True)
    async for node in client.sessions.iter_nodes(session_id, params):
        if node.node_type is not NodeType.TOOL_CALL or node.tool_name not in tool_names:
            continue
        if node.status is NodeStatus.IN_PROGRESS:
            raise ToolPolicyError(f"Baseline call to '{node.tool_name}' never finished")
        cache_key = node.cache_key
        if cache_key is None and not has_capture_marker(node.inputs):
            cache_key = compute_tool_cache_key(node.tool_name, node.inputs)
        if cache_key is None:
            continue    # arguments were not recorded whole: this call can never match
        history.setdefault((node.tool_name, cache_key), []).append(Recorded(
            status=node.status,
            result=node.outputs,
            error=node.error,
            replayable=(node.status is NodeStatus.FAILED
                        or (node.outputs is not None and not has_capture_marker(node.outputs))),
        ))
    return history


async def prepare_policies(client, replay, tool_names, approved=APPROVED_PASSTHROUGH_TOOLS):
    """Tool policies for one run: all passthrough when recording, checked when replaying."""
    if replay is None:
        return ToolPolicies(configs={})
    policies = ToolPolicies(configs=check_policy(replay.tool_policy, tool_names, approved))
    history_tools = {name for name, config in policies.configs.items()
                     if isinstance(config, HistoryConfig)}
    if history_tools:
        policies.history = await load_history(client, replay.baseline_session_id, history_tools)
    return policies
