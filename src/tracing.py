# Send traces to Braintrust: one trace per chat turn, with spans for each step
import contextvars
import os
import subprocess

from braintrust import init_logger, start_span
from dotenv import load_dotenv

load_dotenv()

BRAINTRUST_PROJECT_ID = "2b49b5fa-c0cc-4b9a-9e18-57c9808835b4"


def git_branch():
    """Tag traces with the branch so runs from different branches can be compared."""
    try:
        return subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


GIT_BRANCH = git_branch()

# Without a key, spans are no-ops and the app runs untraced.
if os.getenv("BRAINTRUST_API_KEY"):
    init_logger(project_id=BRAINTRUST_PROJECT_ID)


def run_in_one_context(generator):
    """
    Gradio advances a sync generator on worker threads, each with a fresh
    copy of the context, so the current span set before one yield would be
    gone after it. Running every step in one saved context keeps the whole
    turn (and tool calls made mid-stream) under one trace.
    """
    context = contextvars.copy_context()
    while True:
        try:
            value = context.run(next, generator)
        except StopIteration:
            return
        yield value


def traced_tool(name, arguments_json, run):
    """One span per tool call, named after the tool."""
    with start_span(name=name, type="tool") as span:
        span.log(input=arguments_json)
        output = run()
        span.log(output=output)
        return output
