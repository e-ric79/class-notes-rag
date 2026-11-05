"""Streamlit UI:  streamlit run app.py"""
import streamlit as st

from src.config import INDEX_DIR, TOP_K
from src.generate import answer
from src.index import Index
from src.retrieve import retrieve

st.set_page_config(page_title="Class Notes Assistant", page_icon="📚")
st.title("📚 Class Notes Assistant")


@st.cache_resource
def load_index() -> Index:
    return Index()


if not (INDEX_DIR / "chunks.json").exists():
    st.error("No index yet. Add notes to data/notes/<course>/ and run: python -m src.index")
    st.stop()

index = load_index()

with st.sidebar:
    course = st.selectbox("Course", ["All courses"] + index.courses)
    mode = st.radio("Retrieval", ["hybrid", "dense", "bm25"])
    k = st.slider("Excerpts to use", 2, 10, TOP_K)

question = st.text_input("Ask something about your notes")

if question:
    chunks = retrieve(index, question, mode=mode, k=k, course=None if course == "All courses" else course)
    if not chunks:
        st.warning("Nothing retrieved for that course.")
        st.stop()
    with st.spinner("Thinking..."):
        st.markdown(answer(question, chunks))
    st.subheader("Sources")
    for i, c in enumerate(chunks, start=1):
        with st.expander(f"[{i}] {c['course']} · {c['source']} · {c['location']}"):
            st.write(c["text"])
