"""Compare BM25 vs dense vs hybrid retrieval on your own questions.

Usage:  python -m eval.run_eval eval/questions.json

Each question lists the file that should contain the answer. A "hit" means that
file shows up in the top-k retrieved chunks. We report hit@k and MRR
(mean reciprocal rank: 1 if the right file is first, 1/2 if second, ...).
No LLM involved, so this measures retrieval alone and is free to run repeatedly.
"""
import json
import sys

from src.index import Index
from src.retrieve import retrieve

K = 5


def evaluate(index: Index, questions: list[dict], mode: str) -> tuple[float, float]:
    hits, rr = 0, 0.0
    for q in questions:
        results = retrieve(index, q["question"], mode=mode, k=K)   # no course filter: harder, fairer test
        sources = [r["source"] for r in results]
        if q["source"] in sources:
            hits += 1
            rr += 1.0 / (sources.index(q["source"]) + 1)
    n = len(questions)
    return hits / n, rr / n


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "eval/questions.json"
    questions = json.load(open(path, encoding="utf-8"))
    index = Index()
    print(f"{len(questions)} questions, k={K}\n")
    print(f"{'mode':<8} {'hit@k':>7} {'MRR':>7}")
    for mode in ("bm25", "dense", "hybrid"):
        hit, mrr = evaluate(index, questions, mode)
        print(f"{mode:<8} {hit:>7.2f} {mrr:>7.2f}")
