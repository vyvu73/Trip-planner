# Kitaru worker entrypoint: answer one recorded question through the app and
# record the turn as a Kitaru session, applying the replay's overrides and
# tool policies (src/kitaru_adapter). A worker runs this for each task.
# Locally, with no task, it only answers and records nothing:
#   python kitaru_replay.py "question"
import sys

from kitaru.task import get_task_id

from app import ERROR_MESSAGE, stream_answer
from src.kitaru_adapter.turn import run_turn


def answer(question, history):
    text = ""
    for text in stream_answer(question, history):
        pass
    # The app turns failures into a friendly message; a recorded turn must fail instead.
    if text == ERROR_MESSAGE or not text:
        raise RuntimeError("The app could not answer; see the session's failed nodes")
    return text


def main():
    if get_task_id() is None:
        if len(sys.argv) < 2:
            sys.exit('usage: python kitaru_replay.py "question"  (or run as a Kitaru worker task)')
        print(answer(" ".join(sys.argv[1:]), []))
        return
    result = run_turn(answer)
    print(f"session {result.session_id}: {result.recording}")
    print(result.answer)


if __name__ == "__main__":
    main()
