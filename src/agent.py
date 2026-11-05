"""A hand-written agent loop (no framework).

Concept: an agent is a while-loop around a chat model.

    messages = [system, user question]
    repeat:
        reply = model(messages, tools)
        if reply has no tool calls -> that's the final answer, stop
        otherwise run each requested tool, append the results, go round again

The model decides *what to do next* (search, search again, ask you, or answer).
Our code only executes tools and enforces limits. When you later port this to
LangGraph, this loop becomes the graph: the model call is one node, tool execution
is another, and "has tool calls?" is the conditional edge between them.
"""
import json
import re

from groq import Groq

from .config import GROQ_API_KEY, GROQ_MODEL
from .tools import TOOL_SCHEMAS, Toolbox

SYSTEM = """You are a study assistant for one student's own class notes. You can search the notes with tools.

How to work:
- For anything about course content, search the notes first. Do not answer course content from your own knowledge.
- Write focused search queries (key terms, not the whole question). If a question has several parts, search for each part separately.
- If the first results look off-topic, rephrase and search again, or try another course.
- If you are unsure which course is meant, call list_courses. Only call ask_user if the question is truly ambiguous.
- Cite every claim with the excerpt number in square brackets, like [1] or [2][3].
- If the notes don't contain the answer after a reasonable search, say "I couldn't find this in your notes." Do not guess.
- Greetings or questions about what you can do can be answered directly without tools.
- Keep answers clear and concise, as if tutoring."""


class Agent:
    def __init__(self, index, ask_user_fn, max_steps: int = 6, verbose: bool = True):
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")
        self.client = Groq(api_key=GROQ_API_KEY)
        self.tools = Toolbox(index, ask_user_fn)
        self.max_steps = max_steps
        self.verbose = verbose
        self.reset()

    def reset(self):
        """Forget the conversation (history and citation numbers)."""
        self.messages = [{"role": "system", "content": SYSTEM}]
        self.tools.sources.clear()

    def _log(self, text: str):
        if self.verbose:
            print(text)

    def _model_call(self, use_tools: bool):
        kwargs = dict(model=GROQ_MODEL, messages=self.messages, temperature=0.2)
        if use_tools:
            kwargs.update(tools=TOOL_SCHEMAS, tool_choice="auto")
        return self.client.chat.completions.create(**kwargs).choices[0].message

    @staticmethod
    def _as_dict(msg) -> dict:
        """Convert the SDK's reply object back into the plain dict form the API expects in history."""
        out = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            out["tool_calls"] = [
                {"id": c.id, "type": "function",
                 "function": {"name": c.function.name, "arguments": c.function.arguments or "{}"}}
                for c in msg.tool_calls
            ]
        return out

    def run(self, question: str) -> str:
        self.messages.append({"role": "user", "content": question})

        for step in range(1, self.max_steps + 1):
            msg = self._model_call(use_tools=True)
            self.messages.append(self._as_dict(msg))

            if not msg.tool_calls:                 # no tool requested -> this is the answer
                return msg.content or ""

            for call in msg.tool_calls:
                name = call.function.name
                try:
                    args = json.loads(call.function.arguments or "{}")
                    result = self.tools.call(name, args)
                except json.JSONDecodeError:
                    args, result = {}, "Error: tool arguments were not valid JSON. Try again."
                self._log(f"  step {step}: {name}({json.dumps(args)})")
                # every tool call MUST be answered with a tool message carrying its id
                self.messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

        # Step budget used up: force a final answer from what we have instead of looping forever.
        self._log("  step limit reached, asking for a final answer")
        self.messages.append({"role": "user", "content": "Stop searching. Answer now using only what you found, with citations."})
        msg = self._model_call(use_tools=False)
        self.messages.append(self._as_dict(msg))
        return msg.content or ""

    def cited_sources(self, answer: str) -> list[tuple[int, dict]]:
        """Which numbered excerpts did the answer actually cite?"""
        numbers = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})
        return [(n, self.tools.sources[n - 1]) for n in numbers if 1 <= n <= len(self.tools.sources)]