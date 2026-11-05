# Ask the deployed app the test questions in docs/test-questions.md, one chat
# per question, through its Gradio /respond endpoint (what the chat box calls).
# With KITARU_RECORD=1 on the deployment, each turn is recorded as a Kitaru
# session. Each chat gets its own id (q001 ... q112), used as the Kitaru
# session's conversation_id.
# Usage, from the project root:
#   uv run python run_questions.py <app url> [answers.json] [group prefix]
# A group prefix (e.g. "limits") asks only the groups whose name starts with it.
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from gradio_client import Client

QUESTIONS = "docs/test-questions.md"
ERROR_MESSAGE = "Sorry, I ran into a problem answering that. Please try again."  # app.py
RUN = time.strftime("%Y%m%d-%H%M")
WORKERS = 3  # OpenRouteService's free tier allows about 40 requests a minute


def read_questions(path):
    """Each "N. question" line starts a chat; "- Follow-up: ..." lines below it continue it."""
    items, group = [], None
    with open(path) as f:
        for line in f:
            if line.startswith("## "):
                group = line[3:].strip()
            elif match := re.match(r"(\d+)\. (.+)", line):
                items.append({"n": int(match[1]), "group": group, "turns": [match[2].strip()]})
            elif match := re.match(r"\s+- Follow-up: (.+)", line):
                items[-1]["turns"].append(match[1].strip())
    return items


def run_chat(url, item):
    client = Client(url, verbose=False)
    session = f"q{item['n']:03d}-{RUN}"
    history, answers = [], []
    for message in item["turns"]:
        started = time.time()
        history = history + [{"role": "user", "content": message}]
        try:
            _, history, _ = client.predict(history, session, api_name="/respond")
            answer = history[-1]["content"] if history[-1]["role"] == "assistant" else ""
        except Exception as error:
            answer = f"{ERROR_MESSAGE} ({type(error).__name__}: {error})"
            history = history + [{"role": "assistant", "content": answer}]
        answers.append({"message": message, "answer": answer,
                        "seconds": round(time.time() - started, 1)})
    failed = any(not turn["answer"] or turn["answer"].startswith(ERROR_MESSAGE) for turn in answers)
    print(f"q{item['n']:03d} {'FAILED' if failed else 'ok'}", flush=True)
    return {**item, "session_id": session, "answers": answers, "failed": failed}


def main(url, answers_path, group=""):
    items = [item for item in read_questions(QUESTIONS) if item["group"].startswith(group)]
    with ThreadPoolExecutor(WORKERS) as pool:
        results = list(pool.map(lambda item: run_chat(url, item), items))
    with open(answers_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    failed = [r["n"] for r in results if r["failed"]]
    print(f"done: {len(results)} chats, {len(failed)} failed {failed}, run {RUN}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else f"answers-{RUN}.json",
         sys.argv[3] if len(sys.argv) > 3 else "")
