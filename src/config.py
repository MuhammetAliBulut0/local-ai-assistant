"""Uygulama genelinde kullanılan ayarlar.

Tüm değerler .env dosyasından (veya ortam değişkenlerinden) okunur.
Bulut servislerine hiçbir bağımlılık yoktur; her şey yerel Ollama
sunucusuna ve yerel diske yazılan Chroma veritabanına dayanır.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    chat_model: str = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:7b")
    embedding_model: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

    chroma_dir: str = os.getenv("CHROMA_DIR", "data/chroma_db")
    collection_name: str = os.getenv("CHROMA_COLLECTION", "local_documents")
    upload_dir: str = os.getenv("UPLOAD_DIR", "data/uploads")

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "4"))

    # Ollama'nın varsayılan bağlam penceresi (num_ctx) çoğu modelde sadece
    # 2048 token'dır. Belge parçaları + geçmiş + talimatlar bunu kolayca
    # aşabildiğinden, model fark ettirmeden içeriği baştan kırpabilir. Bunu
    # önlemek için daha büyük bir varsayılan kullanıyoruz.
    num_ctx: int = int(os.getenv("OLLAMA_NUM_CTX", "8192"))


settings = Settings()
