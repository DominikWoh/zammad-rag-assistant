<div align="center">

# 🔍 Zammad-LightRAG

### Verwandle dein Zammad-Helpdesk in eine KI-gestützte Wissensdatenbank

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![LightRAG](https://img.shields.io/badge/LightRAG-1.5+-purple.svg)](https://github.com/HKUDS/LightRAG)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Stars](https://img.shields.io/github/stars/duftertyp/zammad-lightrag?style=social)](https://github.com/duftertyp/zammad-lightrag/stargazers)

[🇬🇧 English](README.md) · [🇩🇪 Deutsch](README.de.md)

</div>

---

## ✨ Was es macht

```
  📥 Zammad              ⚙️  Sync              🧠 LightRAG            💬 KI-Suche
┌──────────┐    Tickets   ┌────────┐   Doku.   ┌──────────────┐    Anfrage
│ Tickets  │ ───────────→ │ Engine │ ────────→ │ Knowledge    │ ←────────
└──────────┘              └────────┘           │ Graph + RAG  │
                                              └──────────────┘
```

| Feature | Beschreibung |
|---------|--------------|
| 📥 **Automatischer Sync** | Geschlossene Tickets nach Zeitplan ingesten (stündlich/täglich/wöchentlich) |
| 🔄 **Delta-Sync** | Nach erstem Durchlauf nur neue/geänderte Tickets holen |
| 🚫 **Deduplizierung** | SQLite-Fortschrittsdatenbank verhindert doppelte Tickets |
| 🔁 **Auto-Retry** | Fehlgeschlagene Tickets werden im nächsten Durchlauf erneut versucht |
| 🤖 **Beliebiges LLM** | Funktioniert mit Ollama, LM Studio oder jeder OpenAI-kompatiblen API |
| 🪶 **Schlank** | Nur 2 Container — keine zusätzliche Datenbank nötig |
| 🇩🇪 **Deutsch / 🇬🇧 Englisch** | Erstklassige Unterstützung für deutsche Tickets |

---

## 🚀 Schnellstart

### Voraussetzungen

Stelle sicher, dass du folgendes installiert hast:

- 🐳 [Docker](https://docs.docker.com/get-docker/) + Docker Compose
- 📋 Eine laufende [Zammad](https://zammad.org/)-Instanz mit API-Zugriff
- 🤖 Eines davon:
  - 🦙 [Ollama](https://ollama.ai/) (empfohlen, kostenlos, lokal)
  - 🎨 [LM Studio](https://lmstudio.ai/) (OpenAI-kompatibel, lokal)
  - ☁️ Jede OpenAI-kompatible API

### Installation

```bash
# 1. Klonen
git clone https://github.com/duftertyp/zammad-lightrag.git
cd zammad-lightrag

# 2. Config kopieren und bearbeiten
cp .env.example .env
nano .env   # oder deinen Editor nutzen
```

**Minimale Config** (`.env`):

```ini
# Zammad
ZAMMAD_URL=http://dein-zammad:8080
ZAMMAD_TOKEN=dein_token_hier

# LLM (Ollama)
LLM_BINDING=ollama
LLM_BINDING_HOST=http://host.docker.internal:11434
LLM_MODEL=qwen3:8b
EMBEDDING_MODEL=bge-m3
```

```bash
# 3. Starten
docker compose up -d

# 4. Sync beobachten
docker logs -f zammad-sync
```

### 🎉 Tickets durchsuchen

Öffne **http://localhost:9621/webui** und frage:

- 🗨️ *"Wie wurde das VPN-Problem von Kunde Müller gelöst?"*
- 🗨️ *"Häufige Probleme beim Passwort-Reset"*
- 🗨️ *"Druckerfehler an Maschine 307"*

### 🔌 Open WebUI Integration

Willst du aus Open WebUI heraus suchen? LightRAG hat eine **eingebaute Ollama-kompatible Schnittstelle**.

**Schnellweg:** In Open WebUI → Settings → Connections → Ollama API → URL `http://localhost:9621` → Model `lightrag:latest` erscheint.

Für die volle Anleitung inkl. Custom-🔍-Tool siehe [`openwebui-functions/README.md`](openwebui-functions/README.md).

---

## ⚙️ Konfiguration

Alle Einstellungen in `.env`. Am wichtigsten:

| Variable | Standard | Was es tut |
|----------|----------|------------|
| `TICKET_MIN_AGE_DAYS` | `7` | Tickets überspringen, die < N Tage geschlossen sind |
| `START_DATE` | `2020-01-01` | Nur Tickets nach diesem Datum syncen |
| `SYNC_INTERVAL` | `daily` | `hourly`, `daily` oder `weekly` |
| `SYNC_TIME` | `02:00` | Uhrzeit für täglichen/wöchentlichen Sync (`HH:MM`, 24h) |
| `SYNC_LIMIT` | `0` | Max. Tickets pro Durchlauf (0 = unbegrenzt, 50 = Test) |
| `SUMMARY_LANGUAGE` | `German` | Sprache für Antworten (z.B. `English`, `German`) |

### LLM-Backend Beispiele

<details>
<summary>🦙 Ollama (empfohlen)</summary>

```ini
LLM_BINDING=ollama
LLM_BINDING_HOST=http://host.docker.internal:11434
LLM_MODEL=qwen3:8b
EMBEDDING_MODEL=bge-m3
```
</details>

<details>
<summary>🎨 LM Studio / OpenAI-kompatibel</summary>

```ini
LLM_BINDING=openai
LLM_BINDING_HOST=http://host.docker.internal:1234/v1
LLM_MODEL=google/gemma-3-27b-it
OPENAI_API_KEY=lm-studio
EMBEDDING_MODEL=text-embedding-bge-m3
```
> 💡 Ein Kompatibilitäts-Patch wird automatisch für LM Studio angewendet.
</details>

<details>
<summary>☁️ OpenAI / Cloud APIs</summary>

```ini
LLM_BINDING=openai
LLM_BINDING_HOST=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIM=1536
```
</details>

---

## 🛠️ Sync verwalten

```bash
# ▶️ Starten
docker start zammad-sync

# ⏹️ Stoppen (LightRAG läuft weiter)
docker stop zammad-sync

# 🔄 Neustart
docker restart zammad-sync

# 🧪 Einmaligen Test-Sync (50 Tickets)
docker exec zammad-sync python sync.py --once --limit=50

# 📊 Fortschritt prüfen
docker exec zammad-sync python -c "
import sqlite3; c = sqlite3.connect('data/sync_progress.db')
print('✅ Synced:', c.execute('SELECT COUNT(*) FROM sync_progress WHERE status=\"done\"').fetchone()[0])
print('❌ Failed:', c.execute('SELECT COUNT(*) FROM sync_progress WHERE status=\"failed\"').fetchone()[0])
"

# 📜 Logs anzeigen
docker logs zammad-sync --tail 50
```

### 🧪 Testmodus

Um nur 50 Tickets zu testen, in `.env` setzen:
```ini
SYNC_LIMIT=50
```
Dann `docker restart zammad-sync`. Auf `0` setzen (oder löschen) für vollen Sync.

---

## 🏗️ Architektur

```
┌─────────────────┐         ┌──────────────────┐
│   sync          │  POST   │   lightrag       │
│   (Cron-Loop)   │────────→│   (RAG + KG)     │──→ :9621/webui
└────┬────────────┘         │   Datei-Storage   │
     │ GET /tickets         └──────────────────┘
     ▼
┌─────────────────┐
│   Zammad        │
│   (REST API)    │
└─────────────────┘
```

**2 Docker-Container** — kein Qdrant, keine Web-UI, kein Docker-Socket:

| Container | Image | Zweck |
|-----------|-------|-------|
| 🧠 `zammad-lightrag` | `ghcr.io/hkuds/lightrag:latest` | RAG-Engine + Knowledge Graph + WebUI |
| ⚙️ `zammad-sync` | Custom (python:3.12-slim) | Zammad-Sync mit Cron-Scheduler |

---

## 🔐 Sicherheit

- 🔒 Dein Zammad-Token und LLM-Keys sind in `.env` (gitignored). Privat halten.
- 🌐 Die LightRAG WebUI hat keine Authentifizierung. Port `9621` nicht ohne Reverse-Proxy + Auth ins Internet freigeben.
- 🛡️ Alle Daten sind lokal in Dateien — deine Tickets verlassen nie deine Maschine.

---

## 🤝 Mitwirken

Beiträge sind willkommen! 🎉

1. Fork dieses Repository
2. Erstelle einen Feature-Branch (`git checkout -b feature/tolle-funktion`)
3. Commit deine Änderungen (`git commit -m 'Tolle Funktion hinzugefügt'`)
4. Push auf deinen Branch (`git push origin feature/tolle-funktion`)
5. Öffne einen Pull Request

Für größere Änderungen bitte zuerst ein Issue öffnen, um den Ansatz zu besprechen.

---

## 📜 Lizenz

MIT — siehe [LICENSE](LICENSE).

---

## 🙏 Danksagungen

Basiert auf diesen großartigen Projekten:

- 🧠 [LightRAG](https://github.com/HKUDS/LightRAG) — Einfache und schnelle RAG mit Knowledge Graph
- 📋 [Zammad](https://zammad.org/) — Open Source Helpdesk
- 🦙 [Ollama](https://ollama.ai/) — LLMs lokal ausführen
- 🎨 [LM Studio](https://lmstudio.ai/) — Lokale LLM-GUI

---

<div align="center">

Mit ❤️ gemacht für die DACH-Helpdesk-Community

⭐ **Repo mit Stern markieren** wenn es deinem Support-Team hilft!

</div>
