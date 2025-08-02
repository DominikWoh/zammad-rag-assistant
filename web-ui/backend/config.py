import os
from datetime import datetime
from dotenv import load_dotenv

# Für Windows-Entwicklung - falls .env lokal liegt
if os.path.exists(".env"):
    load_dotenv(".env")
else:
    # Für Production auf Linux
    load_dotenv("/opt/ai-suite/ticket_ingest.env")

print(f"DEBUG: ZAMMAD_TOKEN={os.getenv('MIN_TICKET_DATE')}")

class Settings:
    # === ZAMMAD EINSTELLUNGEN ===
    ZAMMAD_URL = os.getenv("ZAMMAD_URL", "http://192.168.0.120:8080")
    ZAMMAD_TOKEN = os.getenv("ZAMMAD_TOKEN")
    
    # === QDRANT EINSTELLUNGEN ===
    QDRANT_URL = os.getenv("QDRANT_URL", "http://192.168.0.120:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "99f87e9c0f6b3f1d37376ff69edd4298a6c23a20e78519567867b7ff82146ce3")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "zammad-collection")
    
    # === OLLAMA EINSTELLUNGEN ===
    OLLAMA_URL = os.getenv("OLLAMA_URL", "http://192.168.0.120:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3n:latest")
    
    # === EMBEDDING MODEL ===
    EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")
    
    # === WEB-UI EINSTELLUNGEN ===
    WEBUI_HOST = os.getenv("WEBUI_HOST", "localhost")
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
    INSTALL_DIR = os.getenv("INSTALL_DIR", "/opt/ai-suite")
    LOG_FILE = os.getenv("LOG_FILE", "/var/log/zammad_to_qdrant.log")
    
    # === URLS FÜR DASHBOARD ===
    OPENWEBUI_URL = os.getenv("OPENWEBUI_URL", "http://192.168.0.120:3000")
    QDRANT_DASHBOARD_URL = f"{QDRANT_URL}/dashboard"

settings = Settings()
