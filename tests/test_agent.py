"""Offline test of the agent loop (no API key, no model downloads):  python tests/test_agent.py"""
import os
import sys
import types
from types import SimpleNamespace as NS

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)
os.environ["GROQ_API_KEY"] = "test"

# Stub heavy / unavailable third-party modules so we only test our own logic.
for name in ["groq", "sentence_transformers", "rank_bm25", "dotenv", "numpy"]:
    try:
        __import__(name)
    except ImportError:
        m = types.ModuleType(name)
        m.Groq = lambda **kw: None
        m.SentenceTransformer = object
        m.BM25Okapi = object
        m.load_dotenv = lambda *a, **k: None
        sys.modules[name] = m

import src.tools as tools_mod
from src.agent import Agent


def call(id_, name, args):
    return NS(id=id_, function=NS(name=name, arguments=args))


def reply(content=None, calls=None):
    return NS(content=content, tool_calls=calls)


CHUNKS = {
    "backprop": [{"course": "ml", "source": "w4.pptx", "location": "slide 3", "text": "Backprop applies the chain rule."}],
    "sgd": [{"course": "ml", "source": "w5.pdf", "location": "p. 2", "text": "SGD updates weights per batch."}],
}


def fake_retrieve(index, query, mode="hybrid", k=5, course=None):
    return CHUNKS["backprop"] if "backprop" in query else CHUNKS["sgd"] if "sgd" in query else []


tools_mod.retrieve = fake_retrieve
INDEX = NS(courses=["ml", "networks"])


class FakeClient:
    def __init__(self, script):
        self.script, self.calls = list(script), []
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **kw):
        self.calls.append(kw)
        return NS(choices=[NS(message=self.script.pop(0))])


def make_agent(script, ask=lambda q: "ml"):
    a = Agent(INDEX, ask_user_fn=ask, max_steps=3, verbose=False)
    a.client = FakeClient(script)
    return a


# 1. two searches (different parts), shared numbering, final cited answer
a = make_agent([
    reply(calls=[call("c1", "search_notes", '{"query": "backprop"}'), call("c2", "search_notes", '{"query": "sgd"}')]),
    reply("Backprop uses the chain rule [1]; SGD updates per batch [2]."),
])
ans = a.run("explain backprop and sgd")
assert ans.startswith("Backprop"), ans
assert [n for n, _ in a.cited_sources(ans)] == [1, 2]
roles = [m["role"] for m in a.messages]
assert roles == ["system", "user", "assistant", "tool", "tool", "assistant"], roles
assert a.messages[3]["tool_call_id"] == "c1" and a.messages[4]["tool_call_id"] == "c2"
assert "[1] (ml | w4.pptx | slide 3)" in a.messages[3]["content"]
print("1 OK  parallel tool calls, ids, citations")

# 2. same chunk searched again keeps its number (no duplicate registration)
a = make_agent([
    reply(calls=[call("c1", "search_notes", '{"query": "backprop"}')]),
    reply(calls=[call("c2", "search_notes", '{"query": "backprop again"}')]),
    reply("done [1]"),
])
a.run("q")
assert len(a.tools.sources) == 1, a.tools.sources
print("2 OK  stable citation numbers")

# 3. bad course -> readable error; unknown tool; bad args; invalid JSON; none crash the loop
a = make_agent([
    reply(calls=[
        call("c1", "search_notes", '{"query": "x", "course": "chemistry"}'),
        call("c2", "no_such_tool", "{}"),
        call("c3", "search_notes", '{"wrong": 1}'),
        call("c4", "list_courses", "not json"),
    ]),
    reply("recovered"),
])
a.run("q")
results = [m["content"] for m in a.messages if m["role"] == "tool"]
assert "unknown course 'chemistry'" in results[0]
assert "no tool named" in results[1]
assert "bad arguments" in results[2]
assert "not valid JSON" in results[3]
print("3 OK  errors become text, loop survives")

# 4. ask_user routes to the human callback
asked = []
a = make_agent([
    reply(calls=[call("c1", "ask_user", '{"question": "which course?"}')]),
    reply("ok"),
], ask=lambda q: asked.append(q) or "networks")
a.run("q")
assert asked == ["which course?"]
assert [m for m in a.messages if m["role"] == "tool"][0]["content"] == "networks"
print("4 OK  ask_user")

# 5. step limit: model never stops calling tools -> forced final answer without tools
loop = reply(calls=[call("c", "list_courses", "{}")])
a = make_agent([loop, loop, loop, reply("forced answer")])
ans = a.run("q")
assert ans == "forced answer"
assert "tools" not in a.client.calls[-1], "final forced call must not offer tools"
assert len(a.client.calls) == 4
print("5 OK  step limit forces an answer")

# 6. reset clears history and citations; no-tool reply returns directly
a = make_agent([reply(calls=[call("c1", "search_notes", '{"query": "backprop"}')]), reply("a [1]"), reply("hi")])
a.run("q")
a.reset()
assert len(a.messages) == 1 and a.tools.sources == []
assert a.run("hello") == "hi"
print("6 OK  reset + direct answer")