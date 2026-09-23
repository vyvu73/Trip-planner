# Rewrite follow up into standalone queries
import os
import openai
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=API_KEY)
MODEL = os.getenv("OPENAI_MODEL")

def format_turns(history):
    """history: list of {"role": "user"|"assistant", "content": str}"""
    return "\n".join(f"{turn['role']}: {turn['content']}" for turn in history)

def condense_question(history, question):
    if not history:
        return question

    prompt = f"""
    {format_turns(history)}
     
    latest user message: {question}

    Rewrite the latest user message as one standalone question that includes
    any park name, dates, or preferences it implicitly refers to from the
    conversation above. If it is already standalone, return it unchanged.
    Reply with only the rewritten question, nothing else
    """

    try:
        response = client.responses.create(model=MODEL, input=prompt)
        return response.output_text.strip()
    except Exception:
        return question