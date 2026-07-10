# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-07-10

### ✨ Added
- Initial release as `zammad-lightrag` (forked from `zammad-rag-assistant` v1)
- Reduced from 6 templates + Qdrant + MCP server to **2 minimal containers** (lightrag + sync)
- File-based storage only (NanoVectorDB + NetworkX + JsonKV) — **no separate vector DB**
- Scheduled sync with `SYNC_INTERVAL` + `SYNC_TIME` (e.g. daily at 02:00)
- Delta sync via Zammad search API (`updated_after`)
- `SYNC_LIMIT` env var for test mode
- Auto-detection of Ollama vs LM Studio backends
- LM Studio compatibility patch (response_format, enable_thinking, embedding URL)
- Graceful shutdown on SIGTERM/SIGINT
- Configurable summary language (`SUMMARY_LANGUAGE`)
- Ticket title filter (skip phishing/spam/auto-reports)

### 🗑️ Removed (vs v1)
- Qdrant container (file-based vector storage)
- MCP server (`lightrag_search_mcp.py`)
- Web dashboard / setup wizard (`demo_app.py` → 1.900 → 0 lines)
- Docker socket mount (no container restart from UI needed)
- OpenWebUI integration files
- Multiple templates (dashboard, settings, ai_settings → single README)

## [1.x] - zammad-rag-assistant (legacy)

See [DominikWoh/zammad-rag-assistant](https://github.com/DominikWoh/zammad-rag-assistant) for the legacy version with full web UI and Qdrant backend.

[2.0.0]: https://github.com/duftertyp/zammad-lightrag/releases/tag/v2.0.0
