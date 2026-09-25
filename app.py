import os
import gradio as gr
from dotenv import load_dotenv

from src.retrieval.retrieve_chunks import retrieve_chunks
from src.retrieval.fetch_alerts import alerts_context, get_alerts
from src.generation.generate_response import stream_response
from src.generation.memory import route_message
from src.ui.theme import CSS, HEAD, THEME

STORAGE_KEY = "trip_planner_chat_v1"
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


def stream_answer(message, history):
    """
    Yield the assistant's answer as it grows (full text so far each time).
    Router replies arrive whole; generated answers stream token by token.
    Always yields at least once.
    """
    try:
        decision = route_message(history, message)
        if decision.route == "out_of_scope":
            yield decision.redirect_message
            return
        if decision.route == "ambiguous":
            yield decision.clarifying_question
            return

        chunks = retrieve_chunks(decision.standalone_question, k=7)
        alerts = alerts_context(decision.park_codes)
        answer = ""
        for answer in stream_response(message, chunks, recent=history, alerts=alerts):
            yield answer
        if not answer:
            raise ValueError("Model returned an empty answer")

    except Exception as error:
        print(f"stream_answer failed: {error}")
        yield ERROR_MESSAGE


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


def respond(history):
    """
    Step 2 of a submit: stream the answer to the last user message into the
    chat. Each yield replaces the assistant message with the text so far.
    """
    if not history or history[-1]["role"] != "user":
        yield history, history
        return

    message, prior = history[-1]["content"], history[:-1]
    for answer in stream_answer(message, prior):
        updated = history + [{"role": "assistant", "content": answer}]
        yield updated, updated


def pick_example(history, event: gr.SelectData):
    """chatbot.example_select handler: send the clicked example as a message."""
    return add_user_message(event.value["text"], history)


def handle_clear():
    """clear_btn.click handler -- resets both the display and the stored state."""
    return EMPTY_HISTORY, EMPTY_HISTORY

load_dotenv()
with gr.Blocks(title="Lupine", fill_height=True) as demo:
    state = gr.BrowserState(EMPTY_HISTORY, storage_key=STORAGE_KEY, secret=STORAGE_SECRET)

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
        inputs=[state],
        outputs=[chatbot, state],
    )

    chatbot.example_select(
        fn=pick_example,
        inputs=[state],
        outputs=[chatbot, msg, state],
    ).then(
        fn=respond,
        inputs=[state],
        outputs=[chatbot, state],
    )

    clear_btn.click(handle_clear, outputs=[chatbot, state])

if __name__ == "__main__":
    get_alerts([])   # warm the alerts cache; a failure starts the 1-hour backoff
    demo.launch(theme=THEME, css=CSS, head=HEAD)
