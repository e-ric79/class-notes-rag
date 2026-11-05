# Class Notes RAG

Ask questions about your own class notes (PDF, DOCX, PPTX) and get answers with citations
back to the exact file and page/slide/section.

**Stack:** pypdf / python-docx / python-pptx · sentence-transformers · BM25 · reciprocal rank
fusion · Groq (Llama 3.3 70B) · Streamlit

## How it works

```
notes -> ingest (chunk by page/slide/heading) -> embeddings + BM25 index
question -> hybrid retrieval (BM25 + dense, fused by rank) -> Groq answers with [n] citations
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add your GROQ_API_KEY
```

## Use

1. Put notes in `data/notes/<course_name>/` (the folder name becomes the course filter).
2. Build the index: `python -m src.index`
3. Run the app: `streamlit run app.py`

Re-run step 2 whenever you add or change notes.

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
