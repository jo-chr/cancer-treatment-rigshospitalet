"""Ingest hospital PDFs into a local Chroma vector store.

Usage:
    python -m src.ingest
"""
from __future__ import annotations

import re
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

# Danish month names -> month number, for parsing "Sidst opdateret: 24. juli 2024"
_DA_MONTHS = {
    "januar": 1, "februar": 2, "marts": 3, "april": 4, "maj": 5, "juni": 6,
    "juli": 7, "august": 8, "september": 9, "oktober": 10, "november": 11,
    "december": 12,
}
_LAST_UPDATED_RE = re.compile(
    r"Sidst opdateret:\s*(\d{1,2})\.\s*([a-zæøå]+)\s*(\d{4})",
    re.IGNORECASE,
)


def _read_pdf(path: Path) -> str:
    reader = pypdf.PdfReader(str(path))
    return "\n\n".join((p.extract_text() or "") for p in reader.pages)


def _parse_last_updated(text: str) -> datetime | None:
    """Parse the 'Sidst opdateret: 24. juli 2024' footer used on Rigshospitalet PDFs."""
    m = _LAST_UPDATED_RE.search(text)
    if not m:
        return None
    day, month_name, year = m.group(1), m.group(2).lower(), m.group(3)
    month = _DA_MONTHS.get(month_name)
    if not month:
        return None
    try:
        return datetime(int(year), month, int(day))
    except ValueError:
        return None


def _doc_metadata(path: Path, text: str) -> dict:
    rel = path.relative_to(settings.docs_dir)
    category_folder = rel.parts[0]

    last_updated = _parse_last_updated(text)
    if last_updated is None:
        # Fall back to file mtime if the PDF has no 'Sidst opdateret' line.
        last_updated = datetime.fromtimestamp(path.stat().st_mtime)
        date_source = "file_mtime"
    else:
        date_source = "pdf_footer"

    return {
        "source_file": path.name,
        "source_path": str(rel).replace("\\", "/"),
        "category": CATEGORY_LABELS.get(category_folder, category_folder),
        "last_modified": last_updated.strftime("%Y-%m-%d"),
        "last_modified_ts": int(last_updated.timestamp()),
        "date_source": date_source,
    }


def load_documents() -> list[Document]:
    docs: list[Document] = []
    for pdf_path in sorted(settings.docs_dir.rglob("*.pdf")):
        text = _read_pdf(pdf_path).strip()
        if not text:
            print(f"  ! empty: {pdf_path.name}")
            continue
        meta = _doc_metadata(pdf_path, text)
        docs.append(Document(page_content=text, metadata=meta))
        print(
            f"  + {pdf_path.relative_to(settings.docs_dir)}  "
            f"({len(text)} chars, updated {meta['last_modified']} "
            f"[{meta['date_source']}])"
        )
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
