"""Chat with the notes agent in the terminal:  python agent_cli.py

Commands: /reset clears the conversation, /quit exits.
You'll see each tool call the agent makes, which is the best way to learn how it behaves.
"""
from src.agent import Agent
from src.index import Index, IndexMissing


def ask_user_in_terminal(question: str) -> str:
    return input(f"\n  agent asks: {question}\n  you> ").strip()


def main():
    try:
        index = Index()
    except IndexMissing as e:
        raise SystemExit(str(e))

    agent = Agent(index, ask_user_fn=ask_user_in_terminal)
    print("Notes agent ready. Ask a question (/reset, /quit).\n")

    while True:
        question = input("you> ").strip()
        if not question:
            continue
        if question == "/quit":
            break
        if question == "/reset":
            agent.reset()
            print("conversation cleared\n")
            continue

        answer = agent.run(question)
        print(f"\n{answer}\n")
        for n, c in agent.cited_sources(answer):
            print(f"  [{n}] {c['course']} · {c['source']} · {c['location']}")
        print()


if __name__ == "__main__":
    main()