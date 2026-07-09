# Zammad-LightRAG

> Sync Zammad helpdesk tickets into a LightRAG knowledge graph for AI-powered search.

[Deutsch](README.de.md) | English

## What It Does

```
Zammad Tickets → Sync Engine → LightRAG (Knowledge Graph) → AI Search
```

- **Automatic sync**: Closed tickets are synced on a schedule (daily, hourly, weekly)
- **Delta sync**: Only new/changed tickets are fetched after the initial run
- **Dedup**: SQLite progress tracking prevents duplicate ingestion
- **Auto-retry**: Failed tickets are retried on the next cycle
- **Any LLM**: Works with Ollama, LM Studio, or any OpenAI-compatible API
- **2 containers**: Just LightRAG + sync — no extra databases

## Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) + Docker Compose
- A Zammad instance with API access
- An LLM backend: [Ollama](https://ollama.ai/), [LM Studio](https://lmstudio.ai/), or OpenAI-compatible API

### Setup

```bash
git clone https://github.com/duftertyp/zammad-lightrag.git
cd zammad-lightrag
cp .env.example .env
```

Edit `.env`:

```ini
# Zammad
ZAMMAD_URL=http://your-zammad:8080
ZAMMAD_TOKEN=your_api_token

# LLM Backend (pick one)

# Option A: Ollama
LLM_BINDING=ollama
LLM_BINDING_HOST=http://host.docker.internal:11434
LLM_MODEL=qwen3:8b
EMBEDDING_MODEL=bge-m3

# Option B: LM Studio
LLM_BINDING=openai
LLM_BINDING_HOST=http://host.docker.internal:1234/v1
LLM_MODEL=google/gemma-3-27b-it
OPENAI_API_KEY=lm-studio
EMBEDDING_MODEL=text-embedding-bge-m3
```

Start:

```bash
docker compose up -d
```

### Search

Open LightRAG's WebUI: **http://localhost:9621/webui**

Ask questions like:
- "How was the VPN issue for customer Müller resolved?"
- "Common password reset problems"
- "Printer errors on machine 307"

## Configuration

All settings are in `.env`. Key options:

| Setting | Default | Description |
|---------|---------|-------------|
| `TICKET_MIN_AGE_DAYS` | `7` | Only sync tickets closed ≥ N days ago |
| `START_DATE` | `2020-01-01` | Only sync tickets created after this date |
| `SYNC_INTERVAL` | `daily` | `hourly`, `daily`, or `weekly` |
| `SYNC_TIME` | `02:00` | Time for daily/weekly sync (24h `HH:MM`) |

### LLM Backend Examples

**Ollama:**
```ini
LLM_BINDING=ollama
LLM_BINDING_HOST=http://host.docker.internal:11434
LLM_MODEL=qwen3:8b
EMBEDDING_MODEL=bge-m3
```

**LM Studio / OpenAI-compatible:**
```ini
LLM_BINDING=openai
LLM_BINDING_HOST=http://host.docker.internal:1234/v1
LLM_MODEL=google/gemma-3-27b-it
OPENAI_API_KEY=lm-studio
EMBEDDING_MODEL=text-embedding-bge-m3
```

> When using LM Studio, a compatibility patch is automatically applied to handle `response_format` differences and disable reasoning mode.

## Architecture

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

| Container | Purpose |
|-----------|---------|
| `lightrag` | RAG engine + knowledge graph + WebUI (:9621) |
| `sync` | Zammad ticket sync with scheduled cron loop |

## Development

```bash
# Run one sync cycle manually
docker exec zammad-sync python sync.py --once --limit=10

# Check sync progress
docker exec zammad-sync python -c "
import sqlite3
c = sqlite3.connect('data/sync_progress.db')
print('Synced:', c.execute('SELECT COUNT(*) FROM sync_progress WHERE status=\"done\"').fetchone()[0])
print('Failed:', c.execute('SELECT COUNT(*) FROM sync_progress WHERE status=\"failed\"').fetchone()[0])
"

# View sync logs
docker logs zammad-sync --tail 20

# View LightRAG logs
docker logs zammad-lightrag --tail 20
```

## Security

- **API Tokens**: Your Zammad token and LLM keys are in `.env` (gitignored). Keep them private.
- **No Auth**: LightRAG's WebUI has no authentication. Don't expose port 9621 to the internet without a reverse proxy.

## License

MIT — see [LICENSE](LICENSE).
