# Zammad RAG Assistant WebUI

![Hero](Antwort%20von%20KI%20basierend%20auf%20vorherige%20Tickets.png)

Die einfache KI‑Erweiterung für Zammad: Diese Weboberfläche verbindet Ihr Zammad‑Helpdesk mit moderner AI/LLM‑Technologie. Dank Retrieval‑Augmented Generation (RAG) nutzt die App Ihre vorhandenen Tickets als Wissensquelle, um schneller bessere Antworten zu finden – ohne Ihre Daten an Dritte zu senden, **LOKAL UND SICHER**.

Kurz gesagt: Der Zammad RAG Assistant liest neue oder markierte Tickets, sucht automatisch nach ähnlichen Fällen in Ihrer Ticket‑Historie und erstellt eine hilfreiche interne Notiz mit Lösungsvorschlägen. Sie behalten die Kontrolle, sparen Recherchezeit und erhöhen die Erstlösungsquote.

Repository: https://github.com/DominikWoh/zammad-rag-assistant

---

## Was ist RAG – in 30 Sekunden (für Anwender)

RAG steht für “Retrieval‑Augmented Generation”. Bevor ein KI‑Modell (LLM) antwortet, holt es sich passendes Wissen aus Ihren eigenen Daten (z. B. gelöste Tickets in Zammad). Die KI formuliert dann auf Basis dieser Fakten eine Antwort. Ergebnis: weniger Halluzinationen, mehr Praxisbezug.

Typische Vorteile:
- Schnellere Antworten durch Wiederverwendung von Lösungen aus ähnlichen Fällen
- Konsistente Qualität – die KI schreibt Hinweise für Techniker, kein Marketingtext
- Datenschutz: Ihre Daten bleiben bei Ihnen (Ollama + Qdrant können on‑prem laufen)

## Badges

- Image: `ghcr.io/DominikWoh/zammad-rag-assistant`
- Lizenz: MIT
- Stack: FastAPI, Vanilla JS, Qdrant (Vektor‑DB), Ollama (lokale LLMs), SentenceTransformers

---

## Inhalt

- ✨ Features
- 🧩 Systemanforderungen
- 🏗️ Architektur
- 🗃️ Datenmodell in Qdrant
- 🖥️ UI / Screens
- 🚀 Quickstart
- ⚙️ Konfiguration (.env)
- 🔌 API‑Überblick
- 🔁 Interna: Poller, Batch‑Import, Scheduler
- 📝 Logging & Aktivitäten
- 🔐 Sicherheit & Auth
- 🗺️ Roadmap
- 📄 Lizenz
- 📦 Datei‑Hinweise

---

## So funktioniert es (einfach erklärt)

1) Sie verbinden die App mit Ihrem Zammad, Qdrant (Vektor‑Suche) und Ollama (LLM/AI).
2) Die App liest passende Tickets (neu/offen oder mit “askai” markiert).
3) Zu jedem Ticket sucht sie ähnliche Fälle in Ihrer Ticket‑Historie (RAG).
4) Die KI erstellt eine interne Notiz mit konkreten Lösungsschritten – kein Marketing, keine Floskeln.
5) Ihr Team prüft/ergänzt und antwortet schneller.

Hinweis: Der Assistent schreibt standardmäßig interne Notizen – Ihre Kunden sehen diese nicht.

## Features

- 🤖 Einfache KI‑Erweiterung für Zammad mit RAG und AskAI
- 🔒 **Datenschutz: lokale KI/AI & LLMs** mit Ollama – Daten bleiben on‑prem
- ⚡ **Effizienzsteigerung bei vielen Tickets** durch Wiederverwendung gelöster Fälle
- 🧱 Single‑Container Web‑UI, keinerlei systemd im Container nötig
- 🔗 Externe Services: Qdrant (Vektordatenbank) und Ollama (LLM/AI)
- 🎫 Zammad‑Integration per Token; Poller verarbeitet Tickets und postet Notizen
- 🔀 Zwei Modi:
  - 📎 RAG: bei “neu/offen” und 1 Artikel
  - ✍️ AskAI: wenn im letzten Artikel “askai” vorkommt
- 🔍 Multi‑Vektor‑Suche in Qdrant: `kurzbeschreibung`, `beschreibung`, `lösung`, `all`
- 📊 Dashboard: Service‑Status, Poller‑Steuerung, Ingest‑Status, letzte Aktivitäten
- 📦 Batch‑Import (Zammad → Qdrant) mit einfachem Zeitplaner
- 🧰 Konfiguration via `.env` (persistente Volumes), Flags mit Hot‑Reload
- 🧹 Sicherheitsnetz: Entfernt automatisch `<think>`/Reasoning‑Teile aus LLM‑Antworten

---

## Installation: One‑Liner (Download → Build → ENV → Up)

Führt alles in einem Rutsch aus (Ubuntu). Danach ist die WebUI unter http://localhost:5000 erreichbar und Qdrant/Ollama laufen automatisch via docker compose.

```bash
sudo apt update && sudo apt install -y ca-certificates curl && \
sudo install -m 0755 -d /etc/apt/keyrings && \
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo tee /etc/apt/keyrings/docker.asc > /dev/null && \
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null && \
sudo apt update && sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin && \
sudo usermod -aG docker $USER && \
newgrp docker <<'NG' && \
set -e && \
git clone https://github.com/DominikWoh/zammad-rag-assistant.git || true && \
cd zammad-rag-assistant && \
git pull --rebase || true && \
mkdir -p ./config ./cache ./logs ./qdrant_storage ./ollama && \
docker compose build && \
# Optional: Qdrant API-Key setzen (leer lassen = kein API-Key)
echo "QDRANT_API_KEY=" > .env && \
cat > ./config/ticket_ingest.env << 'EOF'
ZAMMAD_URL=http://localhost:8080
ZAMMAD_TOKEN=
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
COLLECTION_NAME=zammad-collection
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
EMBED_MODEL=intfloat/multilingual-e5-base

ENABLE_ASKKI=true
ENABLE_RAG_NOTE=true

TOP_K_RESULTS=5
MAX_TOKENS=800
TEMPERATURE=0.1
TIMEOUT_SECONDS=220

MIN_CLOSED_DAYS=14
MIN_TICKET_DATE=2025-01-01

PROMPTS_DIR=/data/config/prompts
INGEST_SCHEDULE=@daily 23:00
EOF
docker compose up -d && \
docker ps && \
docker compose logs -f --tail=50 & sleep 3 && \
echo "Fertig: WebUI auf http://localhost:5000"
NG
```

Hinweise
- Falls newgrp docker in deiner Shell nicht wirkt, melde dich einmal ab/an oder starte ein neues Terminal.
- Ports belegt? In docker-compose.yml Host‑Ports anpassen (z. B. "5001:5000", "6334:6333", "11435:11434").
- Standard‑Volumes: ./config → /data/config, ./cache → /data/cache, ./qdrant_storage → /qdrant/storage, ./ollama → /root/.ollama.

## Systemanforderungen

Für einen reibungslosen Betrieb mit lokaler Qdrant‑Datenbank und Ollama‑LLMs empfehlen wir:

- 🐧 **Betriebssystem:** Ubuntu 24.04 LTS
- 🧮 **CPU:** mind. 12 Threads
- 🧠 **RAM:** 12–16 GB
- 💾 **Speicher:** mind. 50 GB (Modelle + Vektordaten + Logs)
- 🌐 **Netzwerk:** stabile Verbindung zwischen UI, Qdrant und Ollama (lokal oder im LAN)

Hinweise:
- ✅ Die App skaliert mit der Ticketmenge.
- 🚀 Besonders bei vielen Tickets spürbare **Effizienzsteigerung** durch RAG‑Suche und Wiederverwendung von Lösungen.

## Architektur

- Single Container
  - FastAPI Web‑UI (Backend + statische Frontend‑Dateien)
  - Interne Threads:
    - RAG‑Poller: holt Tickets aus Zammad, entscheidet RAG/AskAI, fragt LLM, postet Notiz
    - Batch‑Import: einmaliger Lauf zum Befüllen von Qdrant (Zammad‑Historie)
    - Scheduler: triggert Batch‑Import gemäß ENV‑Plan
- Externe Services
  - Qdrant (z. B. `http://host:6333`)
  - Ollama (z. B. `http://host:11434`, Modelle via `/api/pull`)

---

## Datenmodell in Qdrant

- Collection: frei wählbar (default `zammad-collection`)
- Vektorfelder (je 768, Cosine):
  - `kurzbeschreibung`
  - `beschreibung`
  - `lösung`
  - `all`
- Payload‑Felder (Beispiele):
  - `ticket_id`, `kurzbeschreibung`, `beschreibung`, `lösung`, `title`, `erstelldatum`, …

---

## UI / Screens

- Dashboard
  - Service‑Status (Qdrant, Ollama, Zammad), System‑Metriken
  - Poller‑Steuerung (Start/Stop/Restart)
  - Ingest‑Status und Start
  - Letzte Aktivitäten (Top‑5; neueste oben; Uhrzeit rechts)

  ![Dashboard](Zammad%20RAG%20Assistant%20Dashboard.png)

- Config
  - Form zur Eingabe von URLs, Tokens, Flags (`ENABLE_ASKKI`, `ENABLE_RAG_NOTE`)
  - Qdrant‑Collection‑Check/Erstellung (Multi‑Vektor)
  - Ollama‑Modelle anzeigen/pullen/löschen
  - Ingest‑Schedule setzen und Ingest starten

  ![Konfiguration](Zammad%20RAG%20Assistant%20Konfiguration.png)

---

## Installationsskripte (Compose‑basiert)

Empfohlene Reihenfolge: 1) Update + Docker, 2) RAG‑UI + ENV, 3) Start via docker compose (inkl. Qdrant + Ollama).

### Teil 1: Update + Docker installieren (Ubuntu)
```bash
sudo apt update && sudo apt install -y ca-certificates curl && \
sudo install -m 0755 -d /etc/apt/keyrings && \
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo tee /etc/apt/keyrings/docker.asc > /dev/null && \
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
sudo tee /etc/apt/sources.list.d/docker.list > /dev/null && \
sudo apt update && sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin && \
sudo usermod -aG docker $USER && \
echo "Docker installiert. Bitte ggf. neu anmelden oder 'newgrp docker' ausführen."
```

### Teil 2: RAG‑UI herunterladen, bauen und ENV erstellen
```bash
set -e
# ggf. neue Shell mit Docker-Gruppe: newgrp docker
git clone https://github.com/DominikWoh/zammad-rag-assistant.git || true
cd zammad-rag-assistant
git pull --rebase || true
mkdir -p ./config ./cache ./logs ./qdrant_storage ./ollama
docker compose build
# Optional: Qdrant API-Key setzen (leer lassen = kein API-Key)
echo "QDRANT_API_KEY=" > .env
cat > ./config/ticket_ingest.env << 'EOF'
ZAMMAD_URL=http://localhost:8080
ZAMMAD_TOKEN=
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
COLLECTION_NAME=zammad-collection
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
EMBED_MODEL=intfloat/multilingual-e5-base

ENABLE_ASKKI=true
ENABLE_RAG_NOTE=true

TOP_K_RESULTS=5
MAX_TOKENS=800
TEMPERATURE=0.1
TIMEOUT_SECONDS=220

MIN_CLOSED_DAYS=14
MIN_TICKET_DATE=2025-01-01

PROMPTS_DIR=/data/config/prompts
INGEST_SCHEDULE=@daily 23:00
EOF
```

### Teil 3: Starten (alle Services)
```bash
docker compose up -d
docker ps
docker compose logs -f --tail=50
echo "WebUI erreichbar unter: http://localhost:5000"
```

---

## Konfiguration (.env) – lokale Defaults schnell aktivieren

Pfad: `/data/config/ticket_ingest.env` (über `ENV_FILE` steuerbar)

Schnellstart‑Defaults für lokale Nutzung (Qdrant/Ollama via docker compose):
```
ZAMMAD_URL=http://localhost:8080
ZAMMAD_TOKEN=
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
COLLECTION_NAME=zammad-collection
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
EMBED_MODEL=intfloat/multilingual-e5-base

ENABLE_ASKKI=true
ENABLE_RAG_NOTE=true

TOP_K_RESULTS=5
MAX_TOKENS=800
TEMPERATURE=0.1
TIMEOUT_SECONDS=220

MIN_CLOSED_DAYS=14
MIN_TICKET_DATE=2025-01-01

PROMPTS_DIR=/data/config/prompts
INGEST_SCHEDULE=@daily 23:00
```

Zusätzliches Compose‑.env im Projektwurzelverzeichnis:
```
# Optional: Qdrant API Key aktivieren
QDRANT_API_KEY=
```

Hinweis: Die App verwendet intern die Service-Namen als Host (QDRANT_URL=http://qdrant:6333, OLLAMA_URL=http://ollama:11434), extern weiterhin localhost-Ports.

Hinweise
- `ZAMMAD_URL` und `ZAMMAD_TOKEN` bitte mit eurem System befüllen.
- `OLLAMA_MODEL` beliebig wählen; im UI unter “Modelle” per Pull laden.
- Flags `ENABLE_ASKKI`/`ENABLE_RAG_NOTE` werden alle 10 s im Poller nachgeladen (Hot‑Reload).
- Prompts: `askai.env` und `rag_prompt.env` in `PROMPTS_DIR` überschreiben Defaults.

Hinweise
- Die Flags `ENABLE_ASKKI`/`ENABLE_RAG_NOTE` werden alle 10 s im Poller nachgeladen (Hot‑Reload).
- Prompts: `askai.env` und `rag_prompt.env` in `PROMPTS_DIR` überschreiben Default‑Prompts.

---

## API‑Überblick (Auszug)

- Auth: Cookie‑basiert (Setup/Login über WebUI)

Allgemein:
- `GET /api/status` – Service‑ und Systemstatus
- `GET /api/config` / `POST /api/config` – Konfig laden/speichern

Qdrant:
- `POST /api/qdrant/test` – Healthcheck; optional Collection anlegen

Ollama:
- `GET /api/ollama/models` – Modelle aus `/api/tags`
- `POST /api/ollama/pull` – Modell herunterladen
- `DELETE /api/ollama/models/{model}` – Modell löschen

Poller:
- `GET /api/services/zammad_rag_poller/status`
- `POST /api/services/zammad_rag_poller/control?action=start|stop|restart|status`

Ingest:
- `GET /api/ingest/status`
- `POST /api/ingest/start`
- `GET /api/ingest/schedule`
- `POST /api/ingest/schedule`

---

## Interna: Poller, Batch‑Import, Scheduler

### Poller (RAG/AskAI)

- Datei: `web-ui/Services/zammad_rag_poller.py`
- Flags: `ENABLE_ASKKI`, `ENABLE_RAG_NOTE` (robust geparst, Hot‑Reload)
- Logik:
  - Tickets “neu/offen” mit nur einem Artikel → RAG
  - Mehrere Artikel und letzter Artikel enthält “askai” → AskAI
- Volltext‑Kontext:
  - Lädt alle Artikel des Tickets, bereinigt HTML, kombiniert Titel + Verlauf
- Query‑Expansion:
  - LLM generiert 3 alternative Suchsätze → Embeddings je Variante
- Multi‑Vector‑Suche:
  - Query gegen `kurzbeschreibung`, `beschreibung`, `lösung`, `all`
  - Top‑5 Treffer, dedupliziert, best score pro Punkt
- LLM‑Call:
  - Strikte System‑Prompts, Entfernen von `<think>` und Reasoning‑Präfixen vor Posting
- Posting:
  - Notiz via `/api/v1/ticket_articles` (internal=true)

### Batch‑Import (Zammad ‑> Qdrant)

- Datei: `web-ui/Services/ZammadToQdrant.py`
- Lädt Zammad‑Tickets, baut Multi‑Vektor‑Embeddings und schreibt Punkte nach Qdrant
- Verhindert Duplikate (`existing_ticket_ids`, `max_existing_tid`)
- Loggt Aktivitäten: `ingest`, `ingest_skip`, `ingest_summary`

### Scheduler

- Läuft beim App‑Start, prüft alle 10 s `INGEST_SCHEDULE`
- Triggert `ingest_start()` höchstens einmal pro Minute

---

## Logging & Aktivitäten

- Aktivitäten‑Log: `/data/log/activities.jsonl` (JSON Lines)
- `GET /api/activities`:
  - liest Log‑Datei, sortiert nach `processed_at` desc
  - Fallback: Qdrant scroll, weist `processed_at` aus Payload/`erstelldatum` zu
- Dashboard zeigt Top‑5, Uhrzeit rechts, Primärsortierung nach Ticket‑ID desc

---

## Sicherheit & Auth

- Setup/Anmeldung via WebUI, Cookie‑basierte Session
- `/api/*` erfordern gültige Session (401 bei Fehlen)
- Zammad‑Token in ENV, keine Speicherung außerhalb ENV

---

## Roadmap

- Live‑Streaming der Ingest‑Logs (WebSocket/SSE)
- Erweiterte Scheduler‑Syntax (Sekunden/vollständige Crontab)
- Poller‑Aktivitäten ins Aktivitäten‑Panel integrieren
- UI‑Verbesserungen (Dark Mode, Such-/Filterfunktionen)

---

## Lizenz

MIT

---

## Datei‑Hinweise

- Backend: [`backend.main.py`](web-ui/backend/main.py)
- Konfiguration: [`backend.config.Settings`](web-ui/backend/config.py:20)
- RAG‑Poller: [`Services.zammad_rag_poller`](web-ui/Services/zammad_rag_poller.py:1)
- Batch‑Import: [`Services.ZammadToQdrant`](web-ui/Services/ZammadToQdrant.py:1)
- Frontend: [`frontend/index.html`](web-ui/frontend/index.html), [`frontend/app.js`](web-ui/frontend/app.js), [`frontend/config.html`](web-ui/frontend/config.html), [`frontend/config.js`](web-ui/frontend/config.js)

---

## Hinweis zum Image

- Öffentliches Image: `ghcr.io/DominikWoh/zammad-rag-assistant`
- Alternativ: lokalen Build nutzen
  - `docker build -t zammad-rag-ui .`
  - `docker run … zammad-rag-ui`
