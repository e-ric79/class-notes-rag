"""Answer a question from retrieved chunks using Groq.

Concept: the prompt does three jobs. It hands the model the evidence, forces it
to cite which evidence it used, and gives it permission to say "not in the notes"
so it doesn't fall back on made-up answers.
"""
from groq import Groq

from .config import GROQ_API_KEY, GROQ_MODEL

SYSTEM = """You are a study assistant answering questions from the student's own class notes.
Rules:
- Use ONLY the numbered notes excerpts provided. Do not use outside knowledge.
- Cite every claim with the excerpt number in square brackets, like [1] or [2][3].
- If the excerpts don't contain the answer, say "I couldn't find this in your notes." and stop.
- Explain clearly and concisely, as if tutoring."""


def format_context(chunks: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] ({c['course']} | {c['source']} | {c['location']})\n{c['text']}"
        for i, c in enumerate(chunks, start=1)
    )


def answer(question: str, chunks: list[dict]) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")
    client = Groq(api_key=GROQ_API_KEY)
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Notes excerpts:\n{format_context(chunks)}\n\nQuestion: {question}"},
        ],
    )
    return resp.choices[0].message.content
