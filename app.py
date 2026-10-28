import os
import uuid

import gradio as gr
from dotenv import load_dotenv

from braintrust import start_span

from src.tracing import GIT_BRANCH, run_in_one_context
from src.retrieval.retrieve_chunks import retrieve_chunks
from src.retrieval.fetch_alerts import alerts_context, get_alerts
from src.generation.generate_response import MODEL, stream_response
from src.generation.memory import route_message
from src.tools.transportation_cost import warm_gas_price
from src.ui.theme import CSS, HEAD, THEME

STORAGE_KEY = "trip_planner_chat_v1"
CONVERSATION_STORAGE_KEY = "trip_planner_conversation_v1"
STORAGE_SECRET = "12345"

EMPTY_HISTORY = []
ERROR_MESSAGE = "Sorry, I ran into a problem answering that. Please try again."

WELCOME = """### Where to next?
Ask about trails, camping, fees, or a day-by-day itinerary
for any of California's national parks."""

EXAMPLES = [
    {"text": "Is Bumpass Hell worth it in June?"},
    {"text": "Suggest a park for me to visit"},
    {"text": "What's the entrance fee at Joshua Tree?"},
]


def load_history(history):
    """demo.load handler: populate the Chatbot from BrowserState on page load."""
    return history


def stream_answer(message, history, conversation_id=None):
    """
    Yield the assistant's answer as it grows (full text so far each time).
    Router replies arrive whole; generated answers stream token by token.
    Always yields at least once. Each turn is one Braintrust trace; turns of
    one chat share conversation_id so they can be filtered together.
    """
    yield from run_in_one_context(answer_turn(message, history, conversation_id))


def answer_turn(message, history, conversation_id):
    turn = sum(1 for entry in history if entry["role"] == "user") + 1
    with start_span(name="chat_turn", type="task") as span:
        span.log(input=message, metadata={
            "conversation_id": conversation_id,
            "turn": turn,
            "git_branch": GIT_BRANCH,
            "model": MODEL,
        })
        answer = ""
        try:
            decision = route_message(history, message)
            span.log(metadata={"route": decision.route, "park_codes": decision.park_codes})
            if decision.route == "out_of_scope":
                answer = decision.redirect_message
                yield answer
                return
            if decision.route == "ambiguous":
                answer = decision.clarifying_question
                yield answer
                return

            chunks = retrieve_chunks(decision.standalone_question, k=7)
            alerts = alerts_context(decision.park_codes)
            for answer in stream_response(message, chunks, recent=history, alerts=alerts):
                yield answer
            if not answer:
                raise ValueError("Model returned an empty answer")

        except Exception as error:
            print(f"stream_answer failed: {error}")
            span.log(error=str(error))
            answer = ERROR_MESSAGE
            yield answer
        finally:
            span.log(output=answer)


def handle_message(message, history):
    """
    Answer one message without streaming (used by tests and scripts).
    returns: (chatbot history, cleared textbox, state history)
    """
    answer = ERROR_MESSAGE
    for answer in stream_answer(message, history):
        pass

    history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": answer}]
    return history, "", history


def add_user_message(message, history):
    """
    Step 1 of a submit: show the user's message and clear the box right away,
    before the (slow) router + retrieval + generation run.
    """
    if not message.strip():
        return history, message, history

    history = history + [{"role": "user", "content": message}]
    return history, "", history


def respond(history, conversation_id):
    """
    Step 2 of a submit: stream the answer to the last user message into the
    chat. Each yield replaces the assistant message with the text so far.
    The first message of a chat creates its conversation_id.
    """
    conversation_id = conversation_id or str(uuid.uuid4())
    if not history or history[-1]["role"] != "user":
        yield history, history, conversation_id
        return

    message, prior = history[-1]["content"], history[:-1]
    for answer in stream_answer(message, prior, conversation_id):
        updated = history + [{"role": "assistant", "content": answer}]
        yield updated, updated, conversation_id


def pick_example(history, event: gr.SelectData):
    """chatbot.example_select handler: send the clicked example as a message."""
    return add_user_message(event.value["text"], history)


def handle_clear():
    """clear_btn.click handler -- resets the display, the stored history, and
    the conversation_id (the next message starts a new one)."""
    return EMPTY_HISTORY, EMPTY_HISTORY, ""

# KITARU_RECORD=1 records each turn as a Kitaru session (src/kitaru_adapter/live.py).
if os.getenv("KITARU_RECORD") == "1":
    from src.kitaru_adapter.live import record_turns
    respond = record_turns(respond, error_message=ERROR_MESSAGE)

load_dotenv()
with gr.Blocks(title="Lupine", fill_height=True) as demo:
    state = gr.BrowserState(EMPTY_HISTORY, storage_key=STORAGE_KEY, secret=STORAGE_SECRET)
    conversation = gr.BrowserState("", storage_key=CONVERSATION_STORAGE_KEY, secret=STORAGE_SECRET)

    with gr.Row(elem_id="app-header", equal_height=True):
        gr.HTML("""
            <div class="brand">
              <span class="brand-eyebrow">California national parks</span>
              <h1>Lupine</h1>
              <p>Trails, camping, fees, and day-by-day itineraries.</p>
            </div>
        """)
        clear_btn = gr.Button("New conversation", elem_id="new-chat", scale=0)

    chatbot = gr.Chatbot(
        elem_id="chat",
        show_label=False,
        height="65vh",    # fixed size; long conversations scroll inside
        min_height="65vh",
        max_height="65vh",
        placeholder=WELCOME,
        examples=EXAMPLES,
    )

    msg = gr.Textbox(
        elem_id="composer",
        placeholder="Ask about a park, trail, or trip…",
        show_label=False,
        container=False,
        lines=2,          # multi-line: Enter adds a new line, the button sends
        max_lines=6,
        autofocus=True,
        submit_btn=True,
    )

    demo.load(load_history, inputs=[state], outputs=[chatbot])

    msg.submit(
        fn=add_user_message,
        inputs=[msg, state],
        outputs=[chatbot, msg, state],
    ).then(
        fn=respond,
        inputs=[state, conversation],
        outputs=[chatbot, state, conversation],
    )

    chatbot.example_select(
        fn=pick_example,
        inputs=[state],
        outputs=[chatbot, msg, state],
    ).then(
        fn=respond,
        inputs=[state, conversation],
        outputs=[chatbot, state, conversation],
    )

    clear_btn.click(handle_clear, outputs=[chatbot, state, conversation])

if __name__ == "__main__":
    get_alerts([])   # warm the alerts cache; a failure starts the 1-hour backoff
    warm_gas_price() # EIA can take 20s; fetch California's price before the first question
    demo.launch(theme=THEME, css=CSS, head=HEAD)
