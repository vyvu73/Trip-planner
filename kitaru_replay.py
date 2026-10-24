# Kitaru replay entrypoint: re-run one recorded question through the app and
# record the run as a Kitaru session through the Braintrust importer-backed
# adapter. A worker runs this for a replay; locally: python kitaru_replay.py "question"
import sys

from braintrust import current_span
from kitaru.task import get_task_id, get_task_inputs
from kitaru_braintrust_importer.adapter import BraintrustAdapter

from app import ERROR_MESSAGE, stream_answer


def load_task_inputs():
    """The baseline session's recorded inputs, or a question from the command line."""
    if get_task_id() is not None:
        return get_task_inputs()
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:])
    sys.exit('usage: python kitaru_replay.py "question"  (or run as a Kitaru replay task)')


def question_and_history(inputs):
    """
    Imported Braintrust sessions store {"turns": [{"inputs", "outputs"}, ...]},
    one turn per chat message. The last turn's input is the question; earlier
    complete turns become the chat history the app would have had.
    """
    if isinstance(inputs, str):
        return inputs, []

    turns = inputs.get("turns") if isinstance(inputs, dict) else None
    if not turns or not isinstance(turns[-1].get("inputs"), str):
        raise ValueError("Task inputs have no text question to replay")

    history = []
    for turn in turns[:-1]:
        if not (isinstance(turn.get("inputs"), str) and isinstance(turn.get("outputs"), str)):
            raise ValueError("An earlier turn has no text input or output; cannot rebuild history")
        history += [
            {"role": "user", "content": turn["inputs"]},
            {"role": "assistant", "content": turn["outputs"]},
        ]
    return turns[-1]["inputs"], history


def answer(question, history):
    """
    Runs inside the adapter's "kitaru-run" Braintrust span. The importer takes a
    session's inputs and outputs from that outermost span, so log them there;
    without them a recorded replay could not itself be replayed.
    """
    run_span = current_span()
    run_span.log(input=question)

    text = ""
    for text in stream_answer(question, history):
        pass
    run_span.log(output=text)

    # The app turns failures into a friendly message; a replay must fail instead.
    if text == ERROR_MESSAGE or not text:
        raise RuntimeError("The app could not answer; see the chat_turn span for the error")
    return text


def main():
    question, history = question_and_history(load_task_inputs())
    if get_task_id() is None:
        # Outside a worker task there is no agent to record under: just answer.
        print(answer(question, history))
        return
    print(BraintrustAdapter().run(answer, question, history))


if __name__ == "__main__":
    main()
