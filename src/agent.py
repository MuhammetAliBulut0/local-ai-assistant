"""Yerel RAG asistanının Microsoft Agent Framework ile kurulması.

OllamaChatClient, buluta hiç bağlanmadan yerel Ollama sunucusuyla konuşur.
Ajana, belgelerde arama yapabilmesi için `search_documents` aracı verilir.
"""

from __future__ import annotations

from agent_framework import Agent
from agent_framework_ollama import OllamaChatClient

from .config import settings
from .rag_tool import search_documents

SYSTEM_INSTRUCTIONS = """\
Sen, kullanıcının kendi bilgisayarında tamamen yerel olarak çalışan, \
gizliliği önceleyen bir belge asistanısın. Hiçbir veri internete veya \
buluta gönderilmez; tüm işlemler bu cihazda gerçekleşir.

Kurallar:
1. Kullanıcı belgelere dayalı bir soru sorduğunda, cevap vermeden ÖNCE \
mutlaka `search_documents` aracını çağırarak ilgili belge parçalarını getir.
2. Yanıtını YALNIZCA bulduğun belge parçalarına dayandır. Belgede yer \
almayan bilgiyi uydurma.
3. Belgelerde yeterli bilgi yoksa bunu açıkça belirt.
4. Yanıtının sonunda kullandığın kaynak dosya adlarını kısaca listele.
5. Türkçe ve anlaşılır bir dille yanıt ver.
"""


def build_agent() -> Agent:
    client = OllamaChatClient(model=settings.chat_model, host=settings.ollama_host)
    return client.as_agent(
        name="LocalRAGAssistant",
        instructions=SYSTEM_INSTRUCTIONS,
        tools=[search_documents],
        # num_ctx: Ollama'nın varsayılan (genelde 2048 token'lık) bağlam
        # penceresini büyütüyoruz ki belge parçaları + geçmiş + talimatlar
        # sessizce kırpılmasın.
        default_options={"num_ctx": settings.num_ctx},
    )
