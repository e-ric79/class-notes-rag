"""Turn PDFs, DOCX and PPTX files into chunks with metadata.

Concept: a chunk is the unit of retrieval. We want one chunk to be "one idea",
so we split on the file's natural structure first (page / slide / heading) and
only fall back to a character window when a unit is too long.

Folder layout decides the course tag:  data/notes/<course>/<file>
"""
from dataclasses import dataclass, asdict
from pathlib import Path

from docx import Document
from pptx import Presentation
from pypdf import PdfReader

from .config import CHUNK_CHARS, CHUNK_OVERLAP, NOTES_DIR


@dataclass
class Chunk:
    text: str
    course: str
    source: str      # file name
    location: str    # "p. 4", "slide 7", "section: Backpropagation"

    def to_dict(self):
        return asdict(self)


def split_long(text: str) -> list[str]:
    """Split on paragraph boundaries into windows of ~CHUNK_CHARS, with a small overlap."""
    text = text.strip()
    if len(text) <= CHUNK_CHARS:
        return [text] if text else []
    pieces, current = [], ""
    for para in text.split("\n"):
        if len(current) + len(para) + 1 > CHUNK_CHARS and current:
            pieces.append(current.strip())
            current = current[-CHUNK_OVERLAP:]   # carry a tail so context isn't cut mid-thought
        current += para + "\n"
    if current.strip():
        pieces.append(current.strip())
    return pieces


def load_pdf(path: Path):
    """One unit per page."""
    for i, page in enumerate(PdfReader(str(path)).pages, start=1):
        yield f"p. {i}", page.extract_text() or ""


def load_pptx(path: Path):
    """One unit per slide: all text frames plus speaker notes."""
    for i, slide in enumerate(Presentation(str(path)).slides, start=1):
        parts = [s.text_frame.text for s in slide.shapes if s.has_text_frame and s.text_frame.text.strip()]
        if slide.has_notes_slide:
            parts.append(slide.notes_slide.notes_text_frame.text)
        yield f"slide {i}", "\n".join(parts)


def load_docx(path: Path):
    """One unit per heading section (paragraphs under the same Heading style)."""
    heading, buf = "intro", []
    for para in Document(str(path)).paragraphs:
        if para.style.name.lower().startswith("heading") and para.text.strip():
            if buf:
                yield f"section: {heading}", "\n".join(buf)
            heading, buf = para.text.strip(), []
        elif para.text.strip():
            buf.append(para.text)
    if buf:
        yield f"section: {heading}", "\n".join(buf)


LOADERS = {".pdf": load_pdf, ".pptx": load_pptx, ".docx": load_docx}


def ingest(notes_dir: Path = NOTES_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(notes_dir.rglob("*")):
        loader = LOADERS.get(path.suffix.lower())
        if loader is None or path.name.startswith("~$"):   # skip Word temp files
            continue
        rel = path.relative_to(notes_dir)
        course = rel.parts[0] if len(rel.parts) > 1 else "general"
        try:
            for location, text in loader(path):
                for piece in split_long(text):
                    chunks.append(Chunk(piece, course, path.name, location))
        except Exception as e:   # one bad file shouldn't kill the whole ingest
            print(f"[skip] {path.name}: {e}")
    return chunks


if __name__ == "__main__":
    cs = ingest()
    print(f"{len(cs)} chunks from {len({c.source for c in cs})} files")
