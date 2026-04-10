"""
llm.py — Swappable LLM backend.
Currently: Groq API with Llama 3.3. To switch to self-hosted: change call() to point at Ollama/vLLM endpoint.
"""

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "llama-3.1-8b-instant"


def call(system_prompt: str, user_message: str, max_tokens: int = 1500) -> str:
    """Single LLM call. Swap this function to change backend."""
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content


def call_with_history(system_prompt: str, history: list, max_tokens: int = 1500) -> str:
    """Multi-turn call with conversation history."""
    messages = [{"role": "system", "content": system_prompt}] + history
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=messages,
    )
    return response.choices[0].message.content