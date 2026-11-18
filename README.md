# Class Notes RAG

Ask questions about your own class notes (PDF, DOCX, PPTX) and get answers with citations
back to the exact file and page/slide/section.

**Stack:** pypdf / python-docx / python-pptx · sentence-transformers · Qdrant (vector database) ·
BM25 · reciprocal rank fusion · Groq (gpt-oss-120b; change with `GROQ_MODEL` in `.env`) · Streamlit

## How it works

```
notes -> ingest (chunk by page/slide/heading) -> embeddings -> Qdrant (vectors + chunk metadata)
question -> hybrid retrieval (BM25 + Qdrant dense search, fused by rank) -> Groq answers with [n] citations
```

## Vector database

Embeddings and chunk metadata are stored in [Qdrant](https://qdrant.tech/), running in embedded
local mode (files in `data/index/qdrant/`, no server needed).

- **Incremental indexing:** each chunk's id is a hash of its content. Re-running `python -m src.index`
  embeds only new or edited chunks and removes chunks that no longer exist. Use
  `python -m src.index --rebuild` to wipe and re-embed everything.
- **Filtering in the database:** the course filter is applied by Qdrant, not in Python.
- **Switching embedding models** (`EMBED_MODEL` in `.env`) creates a new collection automatically.
- **One process at a time:** local mode locks the folder, so stop the Streamlit app / agent CLI
  before re-indexing. To avoid this, run a Qdrant server and set `QDRANT_URL` in `.env`:
  `docker run -p 6333:6333 qdrant/qdrant` then `QDRANT_URL=http://localhost:6333`.
- BM25 stays in memory and is rebuilt from the stored chunk text at startup.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add your GROQ_API_KEY
```

`requirements.txt` installs the CPU-only build of PyTorch, so no GPU or CUDA is needed.

## Use

1. Put notes in `data/notes/<course_name>/` (the folder name becomes the course filter).
2. Build the index: `python -m src.index`
3. Run the app: `streamlit run app.py`

Re-run step 2 whenever you add or change notes (it only embeds what changed).

## Agent mode

`python agent_cli.py` starts an agent that decides for itself how to answer: it can search the notes
(several times, with rewritten queries), list courses, or ask you a clarifying question. Each tool
call is printed so you can see its reasoning path. The loop is hand-written in `src/agent.py`
(no framework) so every step is visible.

## Tests

Both run offline (no API key, no model download):

```bash
python tests/test_agent.py   # agent loop, with a scripted fake model
python tests/test_store.py   # vector store + incremental indexing, against real Qdrant
```

## Evaluate retrieval

Copy `eval/questions.example.json` to `eval/questions.json`, write 30-40 real questions with the
file that answers each, then:

```bash
python -m eval.run_eval eval/questions.json
```

This compares BM25, dense and hybrid on hit@5 and MRR.

## Known limits

- Scanned or handwritten notes need OCR first (not included).
- Images and diagrams inside slides are ignored; only text is indexed.
- Chunking is structure-based, so a very large unstructured page is split by size.

## Ideas to extend

- Reranker (cross-encoder) on the top 20 before generation
- Quiz generation from a lecture
- OCR for handwritten notes
- Conversation memory for follow-up questions