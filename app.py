"""Yerel, gizlilik odaklı RAG asistanı — Streamlit arayüzü.

Çalıştırmak için:
    streamlit run app.py

Ön koşullar:
    - Ollama yerelde kurulu ve çalışıyor olmalı (https://ollama.com)
    - `ollama pull <OLLAMA_CHAT_MODEL>` ve `ollama pull <OLLAMA_EMBEDDING_MODEL>`
      ile gerekli modeller indirilmiş olmalı (bkz. .env.example)
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import streamlit as st
from agent_framework import Message

from src.agent import build_agent
from src.config import settings
from src.ingest import ingest_files, list_indexed_sources, reset_vectorstore
from src.rag_tool import retrieve_context

st.set_page_config(page_title="Yerel Belge Asistanı", page_icon="🔒", layout="wide")


def run_agent(history: list[dict]):
    """Sohbet geçmişini ajana gönderir ve yanıtı senkron şekilde döndürür.

    Küçük yerel modeller, talimatta yazsa bile her zaman güvenilir şekilde
    "önce search_documents aracını çağır" davranışını sergilemeyebiliyor
    (ki bu, modelin belgede aslında var olan bilgiyi "bulamadım" demesine
    yol açabilir). Bu yüzden en son kullanıcı sorusuyla ilgili belge
    parçalarını BİZ burada önceden getirip mesaja ekliyoruz; model gerekirse
    yine de `search_documents` aracını farklı bir sorguyla tekrar çağırabilir.

    Ajan (ve içindeki Ollama async istemcisi) BİLEREK önbelleğe alınmıyor:
    `asyncio.run()` her çağrıda yeni bir event loop açıp kapatıyor, önbelleğe
    alınmış bir istemci ise ilk oluşturulduğu (artık kapanmış) loop'a bağlı
    kalıp "Event loop is closed" hatasına yol açıyordu. Her seferinde taze
    bir ajan oluşturmak bu sorunu ortadan kaldırıyor; maliyeti göz ardı
    edilebilir düzeydedir (gerçek ağ çağrısı sadece agent.run() sırasında
    yapılır).
    """
    agent = build_agent()

    last_question = history[-1]["content"]
    context = retrieve_context(last_question)

    messages = [Message(role=m["role"], contents=[m["content"]]) for m in history[:-1]]
    messages.append(
        Message(
            role="user",
            contents=[
                "İlgili belge parçaları:\n\n"
                f"{context}\n\n"
                "Yukarıdaki belge parçalarını kullanarak şu soruyu yanıtla "
                "(gerekirse ek arama yapmak için search_documents aracını "
                f"tekrar kullanabilirsin): {last_question}"
            ],
        )
    )
    return asyncio.run(agent.run(messages))


def save_uploaded_files(uploaded_files) -> list[str]:
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for uploaded in uploaded_files:
        dest = upload_dir / uploaded.name
        dest.write_bytes(uploaded.getbuffer())
        saved_paths.append(str(dest))
    return saved_paths


# ---------------------------------------------------------------- Sidebar --
with st.sidebar:
    st.header("🔒 Yerel Belge Asistanı")
    st.caption(
        "Tüm işlemler bu cihazda çalışır. Hiçbir belge veya soru internete "
        "gönderilmez."
    )

    st.subheader("Ayarlar")
    st.text(f"Sohbet modeli: {settings.chat_model}")
    st.text(f"Embedding modeli: {settings.embedding_model}")
    st.text(f"Ollama adresi: {settings.ollama_host}")

    st.divider()
    st.subheader("Belge Yükle")
    uploaded_files = st.file_uploader(
        "PDF, TXT veya MD dosyaları seçin",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if st.button("Belgeleri İşle", type="primary", disabled=not uploaded_files):
        with st.spinner("Belgeler okunuyor, parçalanıyor ve vektörleştiriliyor..."):
            try:
                paths = save_uploaded_files(uploaded_files)
                chunk_count = ingest_files(paths)
                st.success(f"{len(paths)} dosya işlendi, {chunk_count} parça indekslendi.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Belge işlenirken hata oluştu: {exc}")

    st.divider()
    st.subheader("İndekslenmiş Belgeler")
    try:
        sources = list_indexed_sources()
    except Exception as exc:  # noqa: BLE001
        sources = []
        st.warning(f"Belge listesi alınamadı: {exc}")

    if sources:
        for source in sources:
            st.text(f"• {source}")
    else:
        st.caption("Henüz belge indekslenmedi.")

    if st.button("Tüm indeksi sıfırla"):
        reset_vectorstore()
        st.rerun()

# ------------------------------------------------------------------ Chat --
st.title("Belgelerinizle Sohbet Edin")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Belgeleriniz hakkında bir soru sorun...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Belgeler taranıyor ve yanıt hazırlanıyor..."):
            try:
                response = run_agent(st.session_state.chat_history)
                answer = response.text
            except Exception as exc:  # noqa: BLE001
                answer = (
                    "Bir hata oluştu. Ollama'nın çalıştığından ve gerekli "
                    f"modellerin indirildiğinden emin olun.\n\nHata: {exc}"
                )
        st.markdown(answer)

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
