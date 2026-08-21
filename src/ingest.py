"""Belge yükleme, parçalama, vektörleştirme (embedding) ve Chroma'ya kaydetme.

Desteklenen dosya türleri: .pdf, .txt, .md

Bu modül tamamen yerelde çalışır: PDF metni pypdf ile çıkarılır, embedding
Ollama üzerinden (varsayılan: nomic-embed-text) üretilir ve sonuçlar diske
kalıcı bir Chroma koleksiyonu olarak yazılır. Hiçbir veri dışarı gönderilmez.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from .config import settings

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def _read_plain_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_file_text(path: Path) -> str:
    """Bir dosyanın düz metnini çıkarır. Desteklenmeyen türlerde hata fırlatır."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix in {".txt", ".md"}:
        return _read_plain_text(path)
    raise ValueError(f"Desteklenmeyen dosya türü: {suffix}")


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=settings.embedding_model, base_url=settings.ollama_host)


def get_vectorstore() -> Chroma:
    """Diskteki kalıcı Chroma koleksiyonuna bağlanır (yoksa oluşturur)."""
    Path(settings.chroma_dir).mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=settings.chroma_dir,
    )


def _split_into_documents(path: Path, text: str) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_text(text)
    return [
        Document(
            page_content=chunk,
            metadata={"source": path.name, "chunk_index": i},
        )
        for i, chunk in enumerate(chunks)
    ]


def ingest_files(file_paths: Iterable[str | Path]) -> int:
    """Verilen dosyaları okur, parçalar, vektörleştirir ve Chroma'ya ekler.

    Returns:
        Veritabanına eklenen toplam parça (chunk) sayısı.
    """
    all_documents: list[Document] = []
    for raw_path in file_paths:
        path = Path(raw_path)
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        text = load_file_text(path)
        if not text.strip():
            continue
        all_documents.extend(_split_into_documents(path, text))

    if not all_documents:
        return 0

    store = get_vectorstore()
    store.add_documents(all_documents)
    return len(all_documents)


def list_indexed_sources() -> list[str]:
    """Şu anda vektör veritabanında bulunan benzersiz kaynak dosya adlarını döndürür."""
    store = get_vectorstore()
    raw = store.get(include=["metadatas"])
    sources = {meta.get("source") for meta in raw.get("metadatas", []) if meta}
    return sorted(s for s in sources if s)


def collection_size() -> int:
    """Vektör veritabanındaki toplam parça (chunk) sayısını döndürür."""
    store = get_vectorstore()
    raw = store.get(include=[])
    return len(raw.get("ids", []))


def reset_vectorstore() -> None:
    """Tüm indekslenmiş veriyi siler (koleksiyonu yeniden oluşturur)."""
    store = get_vectorstore()
    store.delete_collection()
