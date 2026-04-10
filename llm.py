import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "llama-3.1-8b-instant"


def call(system_prompt: str, user_message: str, max_tokens: int = 1500) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=max_tokens,
        temperature=0,  # 🔥 critical
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return response.choices[0].message.content