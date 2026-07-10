<div align="center">

# 🔍 Zammad-LightRAG

### Turn your Zammad helpdesk into an AI-powered knowledge base

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![LightRAG](https://img.shields.io/badge/LightRAG-1.5+-purple.svg)](https://github.com/HKUDS/LightRAG)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Stars](https://img.shields.io/github/stars/duftertyp/zammad-lightrag?style=social)](https://github.com/duftertyp/zammad-lightrag/stargazers)

[🇬🇧 English](README.md) · [🇩🇪 Deutsch](README.de.md)

</div>

---

## ✨ What it does

```
  📥 Zammad              ⚙️  Sync              🧠 LightRAG            💬 AI Search
┌──────────┐    tickets   ┌────────┐   docs    ┌──────────────┐    query
│ Tickets  │ ───────────→ │ Engine │ ────────→ │ Knowledge    │ ←────────
└──────────┘              └────────┘           │ Graph + RAG  │
                                              └──────────────┘
```

| Feature | Description |
|---------|-------------|
| 📥 **Automatic sync** | Closed tickets are ingested on a schedule (hourly/daily/weekly) |
| 🔄 **Delta sync** | Only new/changed tickets are fetched after initial run |
| 🚫 **Deduplication** | SQLite-based progress tracking prevents duplicates |
| 🔁 **Auto-retry** | Failed tickets are retried on the next cycle |
| 🤖 **Any LLM** | Works with Ollama, LM Studio, or any OpenAI-compatible API |
| 🪶 **Lightweight** | Just 2 containers — no extra database needed |
| 🇩🇪 **German / 🇬🇧 English** | First-class support for German tickets |

---

## 🚀 Quick Start

### Prerequisites

Make sure you have these installed:

- 🐳 [Docker](https://docs.docker.com/get-docker/) + Docker Compose
- 📋 A running [Zammad](https://zammad.org/) instance with API access
- 🤖 One of:
  - 🦙 [Ollama](https://ollama.ai/) (recommended, free, local)
  - 🎨 [LM Studio](https://lmstudio.ai/) (OpenAI-compatible, local)
  - ☁️ Any OpenAI-compatible API

### Installation

```bash
# 1. Clone
git clone https://github.com/duftertyp/zammad-lightrag.git
cd zammad-lightrag

# 2. Copy and edit config
cp .env.example .env
nano .env   # or use your editor
```

**Minimal config** (`.env`):

```ini
# Zammad
ZAMMAD_URL=http://your-zammad:8080
ZAMMAD_TOKEN=your_token_here

# LLM (Ollama)
LLM_BINDING=ollama
LLM_BINDING_HOST=http://host.docker.internal:11434
LLM_MODEL=qwen3:8b
EMBEDDING_MODEL=bge-m3
```

```bash
# 3. Start
docker compose up -d

# 4. Watch the sync
docker logs -f zammad-sync
```

### 🎉 Query Your Tickets

Open **http://localhost:9621/webui** and ask:

- 🗨️ *"How was the VPN issue for customer Müller resolved?"*
- 🗨️ *"Common password reset problems"*
- 🗨️ *"Druckerfehler an Maschine 307"*

### 🔌 Open WebUI Integration

Want to search tickets from Open WebUI? LightRAG has a **built-in Ollama-compatible interface**.

**Quick way:** In Open WebUI → Settings → Connections → Ollama API → URL `http://localhost:9621` → model `lightrag:latest` appears.

For the full guide including a custom 🔍 tool, see [`openwebui-functions/README.md`](openwebui-functions/README.md).

---

## ⚙️ Configuration

All settings are in `.env`. Most important:

| Variable | Default | What it does |
|----------|---------|--------------|
| `TICKET_MIN_AGE_DAYS` | `7` | Skip tickets closed less than N days ago |
| `START_DATE` | `2020-01-01` | Only sync tickets after this date |
| `SYNC_INTERVAL` | `daily` | `hourly`, `daily`, or `weekly` |
| `SYNC_TIME` | `02:00` | Time for daily/weekly sync (`HH:MM`, 24h) |
| `SYNC_LIMIT` | `0` | Max tickets per cycle (0 = unlimited, 50 = test) |
| `SUMMARY_LANGUAGE` | `German` | Language for responses (e.g. `English`, `German`) |

### LLM Backend Examples

<details>
<summary>🦙 Ollama (recommended)</summary>

```ini
LLM_BINDING=ollama
LLM_BINDING_HOST=http://host.docker.internal:11434
LLM_MODEL=qwen3:8b
EMBEDDING_MODEL=bge-m3
```
</details>

<details>
<summary>🎨 LM Studio / OpenAI-compatible</summary>

```ini
LLM_BINDING=openai
LLM_BINDING_HOST=http://host.docker.internal:1234/v1
LLM_MODEL=google/gemma-3-27b-it
OPENAI_API_KEY=lm-studio
EMBEDDING_MODEL=text-embedding-bge-m3
```
> 💡 A compatibility patch is auto-applied for LM Studio.
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

## 🛠️ Managing the Sync

```bash
# ▶️ Start
docker start zammad-sync

# ⏹️ Stop (LightRAG keeps running)
docker stop zammad-sync

# 🔄 Restart
docker restart zammad-sync

# 🧪 Run a one-off test sync (50 tickets)
docker exec zammad-sync python sync.py --once --limit=50

# 📊 Check progress
docker exec zammad-sync python -c "
import sqlite3; c = sqlite3.connect('data/sync_progress.db')
print('✅ Synced:', c.execute('SELECT COUNT(*) FROM sync_progress WHERE status=\"done\"').fetchone()[0])
print('❌ Failed:', c.execute('SELECT COUNT(*) FROM sync_progress WHERE status=\"failed\"').fetchone()[0])
"

# 📜 View logs
docker logs zammad-sync --tail 50
```

### 🧪 Test Mode

To try it out with just 50 tickets, set in `.env`:
```ini
SYNC_LIMIT=50
```
Then `docker restart zammad-sync`. Set to `0` (or remove) for full sync.

---

## 🏗️ Architecture

```
┌─────────────────┐         ┌──────────────────┐
│   sync          │  POST   │   lightrag       │
│   (cron loop)   │────────→│   (RAG + KG)     │──→ :9621/webui
└────┬────────────┘         │   File Storage   │
     │ GET /tickets         └──────────────────┘
     ▼
┌─────────────────┐
│   Zammad        │
│   (REST API)    │
└─────────────────┘
```

**2 Docker containers** — no Qdrant, no web UI, no Docker socket:

| Container | Image | Purpose |
|-----------|-------|---------|
| 🧠 `zammad-lightrag` | `ghcr.io/hkuds/lightrag:latest` | RAG engine + knowledge graph + WebUI |
| ⚙️ `zammad-sync` | Custom (python:3.12-slim) | Zammad ticket sync with cron scheduler |

---

## 🔐 Security

- 🔒 Your Zammad token and LLM keys are stored in `.env` (gitignored). Keep them private.
- 🌐 LightRAG's WebUI has no authentication. Don't expose port `9621` to the internet without a reverse proxy + auth.
- 🛡️ All storage is local file-based — your tickets never leave your machine.

---

## 🤝 Contributing

Contributions are welcome! 🎉

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to your branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please open an issue first for major changes to discuss the approach.

---

## 📜 License

MIT — see [LICENSE](LICENSE).

---

## 🙏 Acknowledgments

Built on top of these amazing projects:

- 🧠 [LightRAG](https://github.com/HKUDS/LightRAG) — Simple and Fast RAG with Knowledge Graph
- 📋 [Zammad](https://zammad.org/) — Open Source Helpdesk
- 🦙 [Ollama](https://ollama.ai/) — Run LLMs locally
- 🎨 [LM Studio](https://lmstudio.ai/) — Local LLM GUI

---

<div align="center">

Made with ❤️ for the DACH helpdesk community

⭐ **Star this repo** if it helps your support team!

</div>
