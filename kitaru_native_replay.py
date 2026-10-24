# Kitaru replay entrypoint through the native adapter (src/kitaru_native):
# re-run one recorded question through the app, recording model calls, tool
# calls, and pipeline steps as a Kitaru session while applying the replay's
# override and tool policy. A worker runs this for a replay; locally:
#   python kitaru_native_replay.py "question"   (answers without recording)
from kitaru.task import get_task_id

from app import ERROR_MESSAGE, stream_answer
from kitaru_replay import load_task_inputs, question_and_history
from src.kitaru_native.adapter import run_turn


def answer(question, history):
    text = ""
    for text in stream_answer(question, history):
        pass
    # The app turns failures into a friendly message; a replay must fail instead.
    if text == ERROR_MESSAGE or not text:
        raise RuntimeError("The app could not answer; see the failed nodes for the error")
    return text


def main():
    question, history = question_and_history(load_task_inputs())
    if get_task_id() is None:
        # Outside a worker task there is no agent to record under: just answer.
        print(answer(question, history))
        return
    print(run_turn(answer, question, history))


if __name__ == "__main__":
    main()
