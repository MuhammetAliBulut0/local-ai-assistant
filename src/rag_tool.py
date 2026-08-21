"""Agent'ın çağırabileceği belge-arama aracı (RAG retrieval tool).

Microsoft Agent Framework, Python fonksiyonlarını `@tool` dekoratörüyle
işaretleyip modele "araç" (function calling) olarak sunmaya izin verir.
Model, kullanıcının sorusunu yanıtlamak için belgelere bakması gerektiğine
karar verirse bu fonksiyonu otomatik olarak çağırır.
"""

from __future__ import annotations

from agent_framework import tool

from .config import settings
from .ingest import collection_size, get_vectorstore

# Küçük/orta boy belge koleksiyonlarında (ör. tek bir birkaç sayfalık PDF),
# anlamsal arama sırasında alakalı ama kelimesi kelimesine örtüşmeyen bir
# parçanın (ör. "Yıllık Etkinlik Takvimi" tablosu) sabit küçük bir k (ör. 4)
# yüzünden dışarıda kalması mümkün. Bunu önlemek için, koleksiyon küçükse
# pratikte TÜM parçaları getiriyoruz; büyük koleksiyonlarda ise bu üst sınırla
# (MAX_CONTEXT_CHUNKS) makul bir bağlam boyutunda kalıyoruz.
MAX_CONTEXT_CHUNKS = 20


def retrieve_context(query: str) -> str:
    """Vektör veritabanında `query` ile en alakalı parçaları arar.

    Bu, hem `search_documents` aracı hem de app.py'deki "her zaman önce ara"
    mekanizması tarafından kullanılan paylaşılan mantıktır.
    """
    store = get_vectorstore()
    total_chunks = collection_size()
    k = min(max(settings.retrieval_k, total_chunks), MAX_CONTEXT_CHUNKS)
    results = store.similarity_search(query, k=k)

    if not results:
        return (
            "Bu sorguyla ilgili hiçbir belge bulunamadı. Henüz belge "
            "yüklenmemiş olabilir ya da soru yüklenen belgelerin kapsamı "
            "dışında olabilir."
        )

    parts = []
    for i, doc in enumerate(results, start=1):
        source = doc.metadata.get("source", "bilinmeyen kaynak")
        parts.append(f"[Kaynak {i}: {source}]\n{doc.page_content}")

    return "\n\n---\n\n".join(parts)


@tool(
    name="search_documents",
    description=(
        "Kullanıcının bu asistana yüklediği PDF/metin belgeleri içinde anlam "
        "tabanlı (semantic) arama yapar ve en alakalı metin parçalarını, "
        "kaynak dosya adlarıyla birlikte döndürür. İlk sorudaki belge "
        "parçaları yetersiz kalırsa veya farklı bir konuyu aramak "
        "gerekirse bu aracı BAŞKA bir sorguyla tekrar çağırabilirsin."
    ),
)
def search_documents(query: str) -> str:
    """Vektör veritabanında `query` ile en alakalı parçaları arar.

    Args:
        query: Belgeler içinde aranacak doğal dil sorgusu.
    """
    return retrieve_context(query)
