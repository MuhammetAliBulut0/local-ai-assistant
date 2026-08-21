# Yerel Belge Asistanı (Local Privacy-First RAG Assistant)

Buluttan tamamen bağımsız, kullanıcının kendi bilgisayarında çalışan, PDF ve
metin belgelerini analiz edip bu belgelere dayalı yanıtlar üreten bir yapay
zeka asistanı. Hiçbir belge veya soru internete gönderilmez; tüm çıkarım
(inference) yerel bir [Ollama](https://ollama.com) sunucusu üzerinden yapılır.

## Mimari

```
┌──────────────┐      ┌───────────────────┐      ┌──────────────────┐
│  Streamlit   │─────▶│  Microsoft Agent   │─────▶│  Ollama (yerel)   │
│  Arayüzü     │      │  Framework Agent   │      │  sohbet modeli    │
│  (app.py)    │◀─────│  + search_documents│◀─────│  (örn. qwen2.5)   │
└──────────────┘      │  aracı             │      └──────────────────┘
        │              └─────────┬─────────┘
        │ PDF / TXT / MD                   │
        ▼                                  ▼
┌──────────────┐               ┌──────────────────────┐
│  İçe Aktarma │──────────────▶│  Chroma Vektör Veri   │
│  (ingest.py) │  embedding    │  Tabanı (yerel disk)  │
└──────────────┘  (Ollama)     └──────────────────────┘
```

- **RAG (Retrieval-Augmented Generation):** Belgeler parçalanır, Ollama ile
  vektörleştirilir (embedding) ve yerel bir Chroma veritabanında saklanır.
- **Agent katmanı:** [Microsoft Agent Framework](https://learn.microsoft.com/en-us/agent-framework/user-guide/overview)
  kullanılarak, modelin gerektiğinde `search_documents` aracını çağırıp
  belgelerden bilgi getirmesi sağlanır (function calling / tool use).
  Framework tamamen açık kaynaklıdır ve `agent-framework-ollama` paketi
  sayesinde buluta hiç çıkmadan, doğrudan yerel Ollama sunucusuyla çalışır —
  bu nedenle Copilot Studio veya Azure AI Foundry gibi bulut tabanlı
  Microsoft servisleri **kullanılmamıştır** (projenin "tamamen yerel"
  hedefiyle çelişirler).
- **Arayüz:** Streamlit ile basit bir sohbet ve belge yükleme ekranı.

## Kurulum

### 1. Ollama'yı kurun ve modelleri indirin

[ollama.com](https://ollama.com) üzerinden işletim sisteminize uygun Ollama'yı
kurun, ardından bir sohbet ve bir embedding modeli indirin:

```bash
ollama pull qwen2.5:7b          # sohbet modeli (araç çağırmayı destekler)
ollama pull nomic-embed-text    # embedding modeli
```

> Daha küçük/hızlı bir model isterseniz `qwen2.5:3b` veya `llama3.1:8b` gibi
> araç çağırmayı (tool calling) destekleyen başka modeller de kullanabilirsiniz.

### 2. Python ortamını kurun

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Ortam değişkenlerini ayarlayın

```bash
cp .env.example .env
# gerekiyorsa .env içindeki model adlarını / adresleri düzenleyin
```

### 4. Uygulamayı çalıştırın

```bash
streamlit run app.py
```

Tarayıcıda açılan arayüzden PDF/TXT/MD dosyalarınızı yükleyip "Belgeleri
İşle" butonuna basın, ardından sohbet kutusundan belgelerinizle ilgili
sorular sorun.

### Alternatif: Arayüzsüz (CLI) indeksleme

```bash
python scripts/ingest_cli.py /belgelerinizin/oldugu/klasor
```

## Proje Yapısı

```
.
├── app.py                # Streamlit arayüzü
├── src/
│   ├── config.py          # .env tabanlı ayarlar
│   ├── ingest.py           # PDF/metin okuma, parçalama, embedding, Chroma
│   ├── rag_tool.py         # Agent'ın çağırdığı search_documents aracı
│   └── agent.py            # Microsoft Agent Framework ile ajan kurulumu
├── scripts/
│   └── ingest_cli.py       # Terminalden toplu belge indeksleme
├── data/
│   ├── uploads/            # Yüklenen ham dosyalar (git'e girmez)
│   └── chroma_db/          # Kalıcı vektör veritabanı (git'e girmez)
├── requirements.txt
└── .env.example
```

## Sorun Giderme

- **"Ollama'nın çalıştığından emin olun" hatası:** `ollama serve` komutunun
  çalıştığından ve `.env` içindeki `OLLAMA_HOST` adresinin doğru olduğundan
  emin olun.
- **Model bulunamadı hatası:** `ollama pull <model_adı>` ile modeli
  indirdiğinizden ve `.env` dosyasındaki model adlarıyla eşleştiğinden emin
  olun.
- **`pip install` sırasında `ResolutionImpossible` / `ollama` paketiyle
  ilgili çakışma hatası:** `agent-framework-ollama`, `ollama` paketinin
  `<0.5.4` sürümünü, `langchain-ollama`'nın yeni sürümleri (1.x) ise
  `>=0.6.1` sürümünü istiyor — bu ikisi asla aynı anda karşılanamaz.
  `requirements.txt` bu yüzden `ollama==0.5.3` ve `langchain-ollama==0.3.10`
  olarak kasıtlı şekilde sabitlenmiştir. Hâlâ bu hatayı alıyorsanız
  `requirements.txt` dosyasının güncel (bu sabitlemeleri içeren) sürümünü
  kullandığınızdan emin olun; gerekirse `pip install -r requirements.txt
  --force-reinstall` deneyin.
- **Yanıtlar yavaş:** Daha küçük bir model deneyin (örn. `qwen2.5:3b`) veya
  GPU hızlandırmasının aktif olduğundan emin olun.

## Kullanılan Kaynaklar

Bu proje aşağıdaki Microsoft kaynakları referans alınarak tasarlanmıştır:

- [Microsoft Agent Framework – Genel Bakış](https://learn.microsoft.com/en-us/agent-framework/user-guide/overview)
- [GitHub Copilot Fundamentals (Microsoft Learn)](https://learn.microsoft.com/en-us/training/paths/copilot/)
- [AI For Beginners (Microsoft)](https://microsoft.github.io/AI-For-Beginners/)
- [Microsoft Copilot dokümantasyonu](https://learn.microsoft.com/en-us/copilot/)

Not: Microsoft Copilot Studio ve Azure AI Foundry Agent Service bulut tabanlı
servislerdir; bu projenin "tamamen yerel/gizlilik odaklı" hedefiyle doğrudan
örtüşmedikleri için mimaride kullanılmamış, yalnızca öğrenme/karşılaştırma
kaynağı olarak referans listesinde bırakılmıştır. İleride isteğe bağlı bir
bulut entegrasyon modu eklenmek istenirse bu servisler üzerinden
genişletilebilir.

## Lisans

Bu proje MIT lisansı ile paylaşılabilir; `LICENSE` dosyasını kendi
tercihinize göre ekleyebilirsiniz.
