import os
import glob
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

_EMBED_DIM = 384  # all-MiniLM-L6-v2 output dimension

_index: faiss.IndexFlatL2 | None = None
_chunks: list[str] = []
_model: SentenceTransformer | None = None


def _read_file(path: str) -> str:
    if path.lower().endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    with open(path, encoding="utf-8") as f:
        return f.read()


def build_vector_store(docs_dir: str) -> None:
    global _index, _chunks, _model

    if not os.path.isdir(docs_dir):
        raise FileNotFoundError(f"docs_dir not found: {docs_dir!r}")

    paths = glob.glob(os.path.join(docs_dir, "*.txt")) + \
            glob.glob(os.path.join(docs_dir, "*.pdf"))
    if not paths:
        raise ValueError(f"No .txt or .pdf files found in {docs_dir!r}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    all_chunks: list[str] = []
    for path in paths:
        raw = _read_file(path)
        if raw.strip():
            all_chunks.extend(splitter.split_text(raw))

    if not all_chunks:
        raise ValueError("Documents were empty after reading.")

    _model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = _model.encode(all_chunks, convert_to_numpy=True).astype(np.float32)
    assert embeddings.shape[1] == _EMBED_DIM

    _index = faiss.IndexFlatL2(_EMBED_DIM)
    _index.add(embeddings)
    _chunks = all_chunks


def search(query: str, k: int = 4) -> list[str]:
    if _index is None or _model is None:
        raise RuntimeError("Call build_vector_store() before search().")

    q_vec = _model.encode([query], convert_to_numpy=True).astype(np.float32)
    _, indices = _index.search(q_vec, k)
    return [_chunks[i] for i in indices[0] if 0 <= i < len(_chunks)]
