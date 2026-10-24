# Bounded, redacted copies of the values a Kitaru session records.
#
# Lossy values carry the same markers as Kitaru's released adapters
# (_kitaru_truncated, _kitaru_unsupported_type), so a cut payload is never
# mistaken for a complete one, in particular as a history result.
import math
import os

MAX_STRING_CHARS = 50_000
MAX_COLLECTION_ITEMS = 200
MAX_DEPTH = 8
SECRET_SUFFIXES = ("_KEY", "_TOKEN", "_SECRET", "_PASSWORD")
MIN_SECRET_CHARS = 8


def secret_values():
    """Values of credential-like environment variables, never to be recorded."""
    return [value for name, value in os.environ.items()
            if name.upper().endswith(SECRET_SUFFIXES) and len(value) >= MIN_SECRET_CHARS]


def redact(text, secrets):
    for secret in secrets:
        text = text.replace(secret, "[redacted]")
    return text


def capture(value, secrets, depth=0):
    """JSON-safe copy of value: strings redacted and cut, containers capped."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"_kitaru_unsupported_type": "non_finite_float"}
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        text = redact(value, secrets)
        if len(text) <= MAX_STRING_CHARS:
            return text
        return {"value": text[:MAX_STRING_CHARS],
                "_kitaru_truncated": {"reason": "max_characters", "original_characters": len(text)}}
    if depth >= MAX_DEPTH:
        return {"_kitaru_truncated": {"reason": "max_depth"}}
    if isinstance(value, dict):
        captured, omitted = {}, max(0, len(value) - MAX_COLLECTION_ITEMS)
        for key, item in list(value.items())[:MAX_COLLECTION_ITEMS]:
            if not isinstance(key, str):
                omitted += 1
                continue
            captured[key] = capture(item, secrets, depth + 1)
        if omitted:
            captured["_kitaru_truncated"] = {"reason": "max_items_or_non_string_keys", "omitted": omitted}
        return captured
    if isinstance(value, (list, tuple)):
        captured = [capture(item, secrets, depth + 1) for item in value[:MAX_COLLECTION_ITEMS]]
        if len(value) > MAX_COLLECTION_ITEMS:
            captured.append({"_kitaru_truncated": {"reason": "max_items",
                                                   "omitted": len(value) - MAX_COLLECTION_ITEMS}})
        return captured
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump(mode="json")
        except Exception:
            dumped = None
        if isinstance(dumped, (dict, list)):
            return capture(dumped, secrets, depth + 1)
    kind = type(value)
    return {"_kitaru_unsupported_type": f"{kind.__module__}.{kind.__qualname__}"}


def capture_error(error, secrets):
    text = capture(f"{type(error).__name__}: {error}", secrets)
    return text if isinstance(text, str) else text["value"]


def has_capture_marker(value):
    """Whether a recorded value lost data when it was captured."""
    if isinstance(value, dict):
        if "_kitaru_truncated" in value or "_kitaru_unsupported_type" in value:
            return True
        return any(has_capture_marker(item) for item in value.values())
    if isinstance(value, list):
        return any(has_capture_marker(item) for item in value)
    return False
