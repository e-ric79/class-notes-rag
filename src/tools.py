"""The tools the agent can call.

Concept: a "tool" is just a normal Python function plus a JSON schema that tells the
model its name, what it does and what arguments it takes. The model never runs
anything. It only *asks* for a call by emitting JSON, and our code runs it.

Two design rules worth noticing:
  1. Tools return plain strings, including for errors. An error the model can read
     ("unknown course, valid ones are ...") lets it fix its own mistake and retry.
  2. Every excerpt shown to the model gets a number that stays unique for the whole
     conversation, so [3] always means the same chunk even after several searches.
"""
import json

from .config import TOP_K
from .retrieve import retrieve


class Toolbox:
    def __init__(self, index, ask_user_fn):
        self.index = index
        self.ask_user_fn = ask_user_fn   # how to put a question to the human (CLI input, Streamlit, ...)
        self.sources: list[dict] = []    # every chunk the model has seen; position + 1 = citation number

    # ---- the tools ---------------------------------------------------------------

    def list_courses(self) -> str:
        return json.dumps(self.index.courses)

    def search_notes(self, query: str, course: str | None = None, k: int = TOP_K) -> str:
        if course and course not in self.index.courses:
            return f"Error: unknown course '{course}'. Valid courses: {self.index.courses}"
        chunks = retrieve(self.index, query, mode="hybrid", k=min(int(k), 10), course=course)
        if not chunks:
            return "No matching excerpts found."
        return "\n\n".join(self._register(c) for c in chunks)

    def ask_user(self, question: str) -> str:
        return self.ask_user_fn(question) or "(the user gave no answer)"

    # ---- plumbing ----------------------------------------------------------------

    def _register(self, chunk: dict) -> str:
        """Give the chunk a conversation-wide number (reuse it if we've shown this chunk before)."""
        key = (chunk["source"], chunk["location"], chunk["text"])
        for i, seen in enumerate(self.sources):
            if (seen["source"], seen["location"], seen["text"]) == key:
                n = i + 1
                break
        else:
            self.sources.append(chunk)
            n = len(self.sources)
        return f"[{n}] ({chunk['course']} | {chunk['source']} | {chunk['location']})\n{chunk['text']}"

    def call(self, name: str, args: dict) -> str:
        """Run a tool by name. Never raises: any failure becomes text the model can react to."""
        fn = getattr(self, name, None)
        if name not in TOOL_NAMES or fn is None:
            return f"Error: no tool named '{name}'. Available: {sorted(TOOL_NAMES)}"
        try:
            return fn(**args)
        except TypeError as e:    # wrong / missing arguments
            return f"Error: bad arguments for {name}: {e}"
        except Exception as e:
            return f"Error while running {name}: {e}"


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_notes",
            "description": (
                "Search the student's class notes (hybrid keyword + semantic search). Returns numbered "
                "excerpts with course, file and page/slide. Use short, specific queries. For a question "
                "with several parts, call this once per part."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for, e.g. 'backpropagation chain rule'"},
                    "course": {"type": "string", "description": "Optional course name to restrict the search. Use list_courses to see valid names."},
                    "k": {"type": "integer", "description": "How many excerpts to return (default 5, max 10)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_courses",
            "description": "List the course names that have notes. Call this when unsure which course a question belongs to.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": (
                "Ask the student a short clarifying question. Only use when the question is genuinely "
                "ambiguous (for example it could refer to two different courses) and searching can't resolve it."
            ),
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            },
        },
    },
]

TOOL_NAMES = {s["function"]["name"] for s in TOOL_SCHEMAS}