"""Central settings. Everything tunable lives here so experiments are one-line changes."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
NOTES_DIR = ROOT / "data" / "notes"    # put notes in data/notes/<course_name>/...
INDEX_DIR = ROOT / "data" / "index"

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Vector database (Qdrant). Leave QDRANT_URL empty to use the embedded local store in QDRANT_DIR.
QDRANT_DIR = INDEX_DIR / "qdrant"
QDRANT_URL = os.getenv("QDRANT_URL") or None          # e.g. http://localhost:6333
# The collection name includes the embedding model, so switching EMBED_MODEL never mixes
# incompatible vectors: it just creates a fresh collection and re-embeds.
COLLECTION = "class_notes__" + EMBED_MODEL.replace("/", "_").replace("-", "_")

CHUNK_CHARS = 900      # target max size of a chunk
CHUNK_OVERLAP = 150    # overlap when a long page/section has to be split
TOP_K = 5              # chunks passed to the LLM
RRF_K = 60             # reciprocal rank fusion constant (60 is the standard default)