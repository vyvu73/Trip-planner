import os
import gradio as gr
from dotenv import load_dotenv

from src.retrieval.retrieve_chunks import retrieve_chunks
from src.generation.generate_response import generate_response
from src.generation.memory import condense_question 

STORAGE_KEY = "trip_planner_chat_v1"
STORAGE_SECRET = "12345"

EMPTY_HISTORY = [] 
def load_history(history):
    """demo.load handler: populate the Chatbot from BrowserState on page load."""
    return history


def handle_message(message, history):
    """
    msg.submit handler.
    inputs:  [msg textbox, state]      -> (message, history)
    outputs: [chatbot, msg textbox, state]
    """
    try:
        question = condense_question(history, message)
        chunks = retrieve_chunks(question, k=10)
        answer = generate_response(message, chunks, recent=history)
    except Exception as error:
        print(f"handle_message failed: {error}")
        answer = "Sorry, I ran into a problem answering that. Please try again."

    history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": answer}]
    return history, "", history


def handle_clear():
    """clear_btn.click handler -- resets both the display and the stored state."""
    return EMPTY_HISTORY, EMPTY_HISTORY

load_dotenv()
with gr.Blocks(title="Plan your trip to Calinifornia National Parks") as demo:
    gr.Markdown("Plan your trip to California National Parks")

    msg = gr.Textbox(placeholder="ask about a park...", show_label=False)
    state = gr.BrowserState(EMPTY_HISTORY, storage_key=STORAGE_KEY, secret=STORAGE_SECRET)
    clear_btn = gr.Button("Clear conversation")
    chatbot = gr.Chatbot(height=500)

    demo.load(load_history, inputs=[state], outputs=[chatbot])

    msg.submit(
        fn=handle_message,
        inputs=[msg, state],
        outputs=[chatbot, msg, state]
    )
    
    clear_btn.click(handle_clear, outputs=[chatbot, state])

demo.launch()