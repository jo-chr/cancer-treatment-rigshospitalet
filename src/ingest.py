"""Ingest hospital PDFs into a local Chroma vector store.

Usage:
    python -m src.ingest
"""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import pypdf
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings

# Maps top-level folder under hospital_documents/ to a category label
# used both for retrieval filtering and for routing in the graph.
CATEGORY_LABELS = {
    "thyroid": "thyroid_cancer",
    "patient": "patient_general",
    "paaroerende": "relatives",
}


def _read_pdf(path: Path) -> str:
    reader = pypdf.PdfReader(str(path))
    return "\n\n".join((p.extract_text() or "") for p in reader.pages)


def _doc_metadata(path: Path) -> dict:
    rel = path.relative_to(settings.docs_dir)
    category_folder = rel.parts[0]
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return {
        "source_file": path.name,
        "source_path": str(rel).replace("\\", "/"),
        "category": CATEGORY_LABELS.get(category_folder, category_folder),
        "last_modified": mtime.strftime("%Y-%m-%d"),
        "last_modified_ts": int(mtime.timestamp()),
    }


def load_documents() -> list[Document]:
    docs: list[Document] = []
    for pdf_path in sorted(settings.docs_dir.rglob("*.pdf")):
        text = _read_pdf(pdf_path).strip()
        if not text:
            print(f"  ! empty: {pdf_path.name}")
            continue
        docs.append(Document(page_content=text, metadata=_doc_metadata(pdf_path)))
        print(f"  + {pdf_path.relative_to(settings.docs_dir)}  ({len(text)} chars)")
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def build_vector_store(reset: bool = True) -> Chroma:
    if reset and settings.chroma_dir.exists():
        shutil.rmtree(settings.chroma_dir)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading PDFs from {settings.docs_dir} ...")
    raw_docs = load_documents()
    print(f"Loaded {len(raw_docs)} documents.")

    chunks = split_documents(raw_docs)
    print(f"Split into {len(chunks)} chunks.")

    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )

    print("Embedding + persisting to Chroma ...")
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(settings.chroma_dir),
        collection_name="rigshospitalet",
    )
    print(f"Done. Vector store at {settings.chroma_dir}")
    return store


if __name__ == "__main__":
    build_vector_store(reset=True)
