# Zammad-LightRAG

> Synchronisiere Zammad-Tickets in einen LightRAG Knowledge Graph für KI-gestützte Suche.

[English](README.md) | Deutsch

## Was es macht

```
Zammad-Tickets → Sync-Engine → LightRAG (Knowledge Graph) → KI-Suche
```

- **Automatischer Sync**: Geschlossene Tickets werden nach Zeitplan synchronisiert (täglich, stündlich, wöchentlich)
- **Delta-Sync**: Nach dem ersten Durchlauf werden nur neue/geänderte Tickets geholt
- **Dedup**: SQLite-Fortschrittsdatenbank verhindert doppelte Erfassung
- **Auto-Retry**: Fehlgeschlagene Tickets werden im nächsten Durchlauf erneut versucht
- **Beliebiges LLM**: Funktioniert mit Ollama, LM Studio oder jeder OpenAI-kompatiblen API
- **2 Container**: Nur LightRAG + Sync — keine zusätzliche Datenbank

## Schnellstart

### Voraussetzungen

- [Docker](https://docs.docker.com/get-docker/) + Docker Compose
- Eine Zammad-Instanz mit API-Zugriff
- Ein LLM-Backend: [Ollama](https://ollama.ai/), [LM Studio](https://lmstudio.ai/) oder OpenAI-kompatible API

### Einrichtung

```bash
git clone https://github.com/duftertyp/zammad-lightrag.git
cd zammad-lightrag
cp .env.example .env
```

`.env` bearbeiten:

```ini
# Zammad
ZAMMAD_URL=http://your-zammad:8080
ZAMMAD_TOKEN=dein_api_token

# LLM Backend (eines wählen)

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

Starten:

```bash
docker compose up -d
```

### Suche

LightRAG WebUI öffnen: **http://localhost:9621/webui**

Beispiel-Fragen:
- "Wie wurde das VPN-Problem von Kunde Müller gelöst?"
- "Häufige Probleme beim Passwort-Reset"
- "Druckerfehler an Maschine 307"

## Konfiguration

Alle Einstellungen in `.env`. Wichtige Optionen:

| Einstellung | Standard | Beschreibung |
|-------------|----------|--------------|
| `TICKET_MIN_AGE_DAYS` | `7` | Nur Tickets syncen, die ≥ N Tage geschlossen sind |
| `START_DATE` | `2020-01-01` | Nur Tickets nach diesem Datum (YYYY-MM-DD) |
| `SYNC_INTERVAL` | `daily` | `hourly`, `daily` oder `weekly` |
| `SYNC_TIME` | `02:00` | Uhrzeit für täglichen/wöchentlichen Sync (`HH:MM`, 24h) |
| `SYNC_LIMIT` | `0` | Max. Tickets pro Sync-Durchlauf (0 = unbegrenzt, 50 = Testmodus) |

### Sync verwalten

```bash
# Sync stoppen (LightRAG läuft weiter)
docker stop zammad-sync

# Sync starten
docker start zammad-sync

# Einen manuellen Sync-Durchlauf starten (ignoriert Zeitplan)
docker exec zammad-sync python sync.py --once --limit=50

# Sync-Logs anzeigen
docker logs zammad-sync --tail 20

# Fortschritt prüfen
docker exec zammad-sync python -c "
import sqlite3; c=sqlite3.connect('data/sync_progress.db')
print('Synced:', c.execute('SELECT COUNT(*) FROM sync_progress WHERE status=\"done\"').fetchone()[0])
print('Failed:', c.execute('SELECT COUNT(*) FROM sync_progress WHERE status=\"failed\"').fetchone()[0])
"
```

### Testmodus

Um nur 50 Tickets zu testen, in `.env` setzen:
```ini
SYNC_LIMIT=50
```
Dann `docker restart zammad-sync`. Auf `0` setzen oder Zeile löschen für vollen Sync.

**Ollama:**
```ini
LLM_BINDING=ollama
LLM_BINDING_HOST=http://host.docker.internal:11434
LLM_MODEL=qwen3:8b
EMBEDDING_MODEL=bge-m3
```

**LM Studio / OpenAI-kompatibel:**
```ini
LLM_BINDING=openai
LLM_BINDING_HOST=http://host.docker.internal:1234/v1
LLM_MODEL=google/gemma-3-27b-it
OPENAI_API_KEY=lm-studio
EMBEDDING_MODEL=text-embedding-bge-m3
```

> Bei LM Studio wird automatisch ein Kompatibilitäts-Patch angewendet, der `response_format`-Unterschiede behandelt und den Reasoning-Modus deaktiviert.

## Architektur

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

| Container | Zweck |
|-----------|-------|
| `lightrag` | RAG-Engine + Knowledge Graph + WebUI (:9621) |
| `sync` | Zammad-Ticket-Sync mit geplantem Cron-Loop |

## Entwicklung

```bash
# Einen Sync-Durchlauf manuell starten
docker exec zammad-sync python sync.py --once --limit=10

# Sync-Fortschritt prüfen
docker exec zammad-sync python -c "
import sqlite3
c = sqlite3.connect('data/sync_progress.db')
print('Synced:', c.execute('SELECT COUNT(*) FROM sync_progress WHERE status=\"done\"').fetchone()[0])
print('Failed:', c.execute('SELECT COUNT(*) FROM sync_progress WHERE status=\"failed\"').fetchone()[0])
"

# Sync-Logs anzeigen
docker logs zammad-sync --tail 20

# LightRAG-Logs anzeigen
docker logs zammad-lightrag --tail 20
```

## Sicherheit

- **API-Token**: Dein Zammad-Token und LLM-Keys sind in `.env` (gitignored). Privat halten.
- **Keine Auth**: Die LightRAG WebUI hat keine Authentifizierung. Port 9621 nicht ohne Reverse-Proxy ins Internet freigeben.

## Lizenz

MIT — siehe [LICENSE](LICENSE).
