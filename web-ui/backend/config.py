import os
from datetime import datetime
from dotenv import load_dotenv

# Einheitliches ENV-Handling für Docker:
# Standard auf /data/config/ticket_ingest.env, override via ENV_FILE
ENV_FILE = os.getenv("ENV_FILE", "/data/config/ticket_ingest.env")
# Laden ohne harte Betriebssystempfade
if os.path.exists(ENV_FILE):
    load_dotenv(ENV_FILE)

# Optional: zusätzliche Prompt-ENVs aus /data/config/prompts laden
PROMPTS_DIR = os.getenv("PROMPTS_DIR", "/data/config/prompts")
if os.path.isdir(PROMPTS_DIR):
    for fname in ("askai.env", "rag_prompt.env", "is_ticket_helpfull.env", "summarize_ticket.env"):
        fpath = os.path.join(PROMPTS_DIR, fname)
        if os.path.isfile(fpath):
            load_dotenv(fpath, override=True)

class Settings:
    # === ZAMMAD EINSTELLUNGEN ===
    ZAMMAD_URL = os.getenv("ZAMMAD_URL", "http://localhost:8080")
    ZAMMAD_TOKEN = os.getenv("ZAMMAD_TOKEN", "")

    # === QDRANT EINSTELLUNGEN ===
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "zammad-collection")

    # === OLLAMA EINSTELLUNGEN ===
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3n:latest")

    # === EMBEDDING MODEL ===
    EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")

    # === WEB-UI EINSTELLUNGEN ===
    WEBUI_HOST = os.getenv("WEBUI_HOST", "0.0.0.0")
    WEBUI_PORT = int(os.getenv("WEBUI_PORT", "5000"))

    # === RAG PARAMETER ===
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "800"))
    TIMEOUT_SECONDS = int(os.getenv("TIMEOUT_SECONDS", "220"))

    # === TICKET FILTER ===
    MIN_CLOSED_DAYS = int(os.getenv("MIN_CLOSED_DAYS", "14"))
    min_ticket_date_str = os.getenv("MIN_TICKET_DATE", "2025-01-01")
    MIN_TICKET_DATE = datetime.strptime(min_ticket_date_str, "%Y-%m-%d")

    # === SYSTEM PFADE ===
    INSTALL_DIR = os.getenv("INSTALL_DIR", "/app")
    LOG_FILE = os.getenv("LOG_FILE", "/data/log/zammad_to_qdrant.log")
    HUGGINGFACE_CACHE_DIR = os.getenv("HUGGINGFACE_CACHE_DIR", "/data/cache")

    # === URLS FÜR DASHBOARD ===
    OPENWEBUI_URL = os.getenv("OPENWEBUI_URL", "http://localhost:3000")
    QDRANT_DASHBOARD_URL = f"{QDRANT_URL}/dashboard"

settings = Settings()
