import os
import hashlib
import secrets
import requests
import logging
import json
from typing import Optional
from dotenv import dotenv_values
from fastapi import FastAPI, HTTPException, Request, Depends, status, Response, Cookie
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
# Im Container wird mit --app-dir web-ui gestartet; services liegt als web-ui/services.py vor.
# Daher nutzen wir den einfachen Import aus services.
try:
    from services import service_manager  # erwartet web-ui/services.py mit Objekt service_manager
except Exception as e:
    # Fallback: minimaler Stub, damit die WebUI startet, auch wenn services fehlt
    import logging as _logging
    _logging.getLogger("webui").warning(f"Service-Manager Import fehlgeschlagen: {e}")
    class _FallbackServiceManager:
        @staticmethod
        def check_qdrant_status():
            return {"status": "unknown", "error": "service manager unavailable"}
        @staticmethod
        def check_ollama_status():
            return {"status": "unknown", "error": "service manager unavailable"}
        @staticmethod
        def check_zammad_status():
            return {"status": "unknown", "error": "service manager unavailable"}
        @staticmethod
        def get_system_stats():
            return {"cpu": None, "mem": None}
        @staticmethod
        def get_recent_activities():
            return []
        @staticmethod
        def control_service(name, action):
            return {"service": name, "action": action, "success": False, "detail": "no systemd in container"}
        @staticmethod
        def check_systemd_service_status(name):
            return {"service": name, "status": "unknown", "detail": "no systemd in container"}
    service_manager = _FallbackServiceManager()

# config.py liegt unter web-ui/backend/config.py und wird bei --app-dir=web-ui unter dem Modulpfad "backend.config" gefunden.
# Daher Import auf backend.config.settings umstellen.
from backend.config import settings

# Logger initialisieren
logger = logging.getLogger("webui")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.setLevel(logging.INFO)

# --- Konfiguration und Pfaderkennung (Docker-freundlich) ---
COOKIE_NAME = "zammad_rag_session"

# Einheitlicher ENV-Pfad (wird auch zum Schreiben genutzt)
ENV_PATH = os.getenv("ENV_FILE", "/data/config/ticket_ingest.env")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

app = FastAPI(title="Zammad RAG Assistant Web-UI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Hilfsfunktionen ---
def read_env_config():
    # Fallback: leeres Dict, wenn Datei fehlt
    try:
        cfg = dotenv_values(ENV_PATH)
        # Korrektur: Zahl-Felder robust casten, inkl. 0 (kein Falsy-Verlust)
        normalized = dict(cfg)
        # Zahl- und Datumsfelder (sofern vorhanden)
        for key in ["MIN_CLOSED_DAYS", "MAX_TOKENS", "TOP_K_RESULTS"]:
            if key in normalized and normalized[key] is not None:
                try:
                    normalized[key] = str(int(str(normalized[key]).strip()))
                except Exception:
                    pass
        return normalized
    except Exception:
        return {}

def write_env_config(updates: dict):
    """KORRIGIERT: Schreibt Werte ohne Anführungszeichen"""
    try:
        with open(ENV_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        lines = []
    
    keys_updated = set()
    new_lines = []
    
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            new_lines.append(line)
            continue
        
        key, val = stripped.split("=", 1)
        if key in updates:
            # Werte immer als reine Strings schreiben (inkl. "0")
            new_val = updates[key]
            if new_val is None:
                new_val = ""
            new_lines.append(f"{key}={new_val}\n")
            keys_updated.add(key)
        else:
            new_lines.append(line)
    
    for k, v in updates.items():
        if k not in keys_updated:
            new_lines.append(f"{k}={v}\n")
    
    with open(ENV_PATH, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def check_login_required():
    config = read_env_config()
    return not (config.get("WEBUI_USERNAME") and config.get("WEBUI_PASSWORD"))

# --- Cookie-basierte Authentifizierung ---
async def api_auth(session: Optional[str] = Cookie(None, alias=COOKIE_NAME)):
    """Dependency für API-Endpunkte - wirft 401-Fehler für JavaScript"""
    if session is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Nicht angemeldet")
    
    config = read_env_config()
    valid_token = config.get("WEBUI_SESSION_TOKEN")
    
    if not valid_token or session != valid_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Ungültige Sitzung")
    
    return session

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Middleware für Seiten-Authentifizierung - leitet bei Bedarf um"""
    public_paths = ["/login", "/setup", "/api/login", "/api/setup", "/static", "/favicon.ico"]
    
    if any(request.url.path.startswith(p) for p in public_paths):
        return await call_next(request)
    
    if check_login_required():
        return RedirectResponse(url="/setup")
    
    session_token = request.cookies.get(COOKIE_NAME)
    if not session_token:
        return RedirectResponse(url="/login")
    
    config = read_env_config()
    valid_token = config.get("WEBUI_SESSION_TOKEN")
    
    if session_token != valid_token:
        response = RedirectResponse(url="/login")
        response.delete_cookie(COOKIE_NAME)
        return response
    
    return await call_next(request)

# --- Seiten-Routen ---
@app.get("/", response_class=HTMLResponse)
async def root():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/config", response_class=HTMLResponse)
async def config_page():
    config_path = os.path.join(FRONTEND_DIR, "config.html")
    with open(config_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/setup", response_class=HTMLResponse)
async def setup_page():
    if not check_login_required():
        return RedirectResponse(url="/login")
    
    setup_html = """<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Setup - Zammad RAG Assistant</title>
    <style>
        body { font-family: Arial, sans-serif; background: #f5f5f5; display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; }
        .setup-container { background: white; padding: 2rem; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); width: 100%; max-width: 400px; }
        .setup-container h1 { text-align: center; color: #333; margin-bottom: 1.5rem; }
        .setup-container input { width: 100%; padding: 0.75rem; margin-bottom: 1rem; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; }
        .setup-container button { width: 100%; padding: 0.75rem; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 1rem; }
        .setup-container button:hover { background: #0056b3; }
        #result { margin-top: 1rem; text-align: center; font-weight: bold; }
    </style>
</head>
<body>
    <div class="setup-container">
        <h1>Setup</h1>
        <p>Bitte erstellen Sie Ihre Anmeldedaten für das Web-Interface:</p>
        <form id="setup-form">
            <input type="text" id="username" placeholder="Benutzername" required>
            <input type="password" id="password" placeholder="Passwort" required>
            <button type="submit">Setup abschließen</button>
        </form>
        <div id="result"></div>
    </div>
    
    <script>
        document.getElementById('setup-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const username = document.getElementById('username').value;
            const password = document.getElementById('password').value;
            const result = document.getElementById('result');
            
            try {
                const response = await fetch('/api/setup', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username, password })
                });
                
                if (response.ok) {
                    result.style.color = 'green';
                    result.textContent = 'Setup erfolgreich! Weiterleitung...';
                    setTimeout(() => window.location.href = '/login', 2000);
                } else {
                    result.style.color = 'red';
                    result.textContent = 'Fehler beim Setup.';
                }
            } catch (error) {
                result.style.color = 'red';
                result.textContent = 'Verbindungsfehler.';
            }
        });
    </script>
</body>
</html>"""
    return HTMLResponse(content=setup_html)

@app.get("/login", response_class=HTMLResponse)
async def login_page():
    if check_login_required():
        return RedirectResponse(url="/setup")
    
    login_html = """<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login - Zammad RAG Assistant</title>
    <style>
        body { font-family: Arial, sans-serif; background: #f5f5f5; display: flex; justify-content: center; align-items: center; min-height: 100vh; margin: 0; }
        .login-container { background: white; padding: 2rem; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); width: 100%; max-width: 400px; }
        .login-container h1 { text-align: center; color: #333; margin-bottom: 1.5rem; }
        .login-container input { width: 100%; padding: 0.75rem; margin-bottom: 1rem; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; }
        .login-container button { width: 100%; padding: 0.75rem; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 1rem; }
        .login-container button:hover { background: #0056b3; }
        #result { margin-top: 1rem; text-align: center; font-weight: bold; }
    </style>
</head>
<body>
    <div class="login-container">
        <h1>Anmeldung</h1>
        <form id="login-form">
            <input type="text" id="username" placeholder="Benutzername" required>
            <input type="password" id="password" placeholder="Passwort" required>
            <button type="submit">Anmelden</button>
        </form>
        <div id="result"></div>
    </div>
    
    <script>
        document.getElementById('login-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const username = document.getElementById('username').value;
            const password = document.getElementById('password').value;
            const result = document.getElementById('result');
            
            try {
                const response = await fetch('/api/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username, password })
                });
                
                if (response.ok) {
                    result.style.color = 'green';
                    result.textContent = 'Anmeldung erfolgreich! Weiterleitung...';
                    setTimeout(() => window.location.href = '/', 1000);
                } else {
                    result.style.color = 'red';
                    result.textContent = 'Ungültige Anmeldedaten.';
                }
            } catch (error) {
                result.style.color = 'red';
                result.textContent = 'Verbindungsfehler.';
            }
        });
    </script>
</body>
</html>"""
    return HTMLResponse(content=login_html)

# --- API-Endpunkte ---
@app.post("/api/setup")
async def api_setup(request: Request):
    if not check_login_required():
        raise HTTPException(status_code=400, detail="Setup bereits abgeschlossen")
    
    data = await request.json()
    username = data.get("username")
    password = data.get("password")
    
    if not username or not password:
        raise HTTPException(status_code=400, detail="Benutzername und Passwort erforderlich")
    
    session_token = secrets.token_urlsafe(32)
    password_hash = hash_password(password)
    
    write_env_config({
        "WEBUI_USERNAME": username,
        "WEBUI_PASSWORD": password_hash,
        "WEBUI_SESSION_TOKEN": session_token
    })
    
    return {"success": True}

@app.post("/api/login")
async def api_login(request: Request, response: Response):
    config = read_env_config()
    data = await request.json()
    
    username = data.get("username")
    password = data.get("password")
    
    stored_username = config.get("WEBUI_USERNAME")
    stored_password = config.get("WEBUI_PASSWORD")
    
    if not stored_username or not stored_password:
        raise HTTPException(status_code=400, detail="Setup erforderlich")
    
    if username != stored_username or hash_password(password) != stored_password:
        raise HTTPException(status_code=401, detail="Ungültige Anmeldedaten")
    
    session_token = config.get("WEBUI_SESSION_TOKEN")
    if not session_token:
        session_token = secrets.token_urlsafe(32)
        write_env_config({"WEBUI_SESSION_TOKEN": session_token})
    
    response.set_cookie(
        key=COOKIE_NAME,
        value=session_token,
        httponly=True,
        secure=False,
        samesite="lax"
    )
    
    return {"success": True}

@app.post("/api/logout")
async def api_logout(response: Response):
    response.delete_cookie(COOKIE_NAME)
    return {"success": True}

@app.get("/api/config")
async def get_config(session: str = Depends(api_auth)):
    """Lädt die aktuelle Konfiguration (inkl. aktueller QDRANT_URL)"""
    config = read_env_config()
    # Defaults anwenden: ENABLE_ASKKI=false, ENABLE_RAG_NOTE=true
    normalized = dict(config)
    normalized["ENABLE_ASKKI"] = (config.get("ENABLE_ASKKI") or "false")
    normalized["ENABLE_RAG_NOTE"] = (config.get("ENABLE_RAG_NOTE") or "true")
    # QDRANT_URL aus Settings als Fallback, falls noch nicht in ENV-Datei vorhanden
    try:
        from backend.config import settings as _settings
        if not normalized.get("QDRANT_URL"):
            normalized["QDRANT_URL"] = _settings.QDRANT_URL
    except Exception:
        pass
    return normalized

@app.post("/api/config")
async def save_config(request: Request, session: str = Depends(api_auth)):
    """Speichert die Konfiguration (inkl. QDRANT_URL)"""
    data = await request.json()

    # Validierung/Normalisierung der neuen Flags
    def to_bool_str(val, default):
        if isinstance(val, bool):
            return "true" if val else "false"
        if isinstance(val, str):
            low = val.strip().lower()
            if low in ["true", "1", "yes", "on"]:
                return "true"
            if low in ["false", "0", "no", "off"]:
                return "false"
        return "true" if default else "false"

    updates = dict(data)

    # Normalisiere Flags
    updates["ENABLE_ASKKI"] = to_bool_str(data.get("ENABLE_ASKKI", "false"), default=False)
    updates["ENABLE_RAG_NOTE"] = to_bool_str(data.get("ENABLE_RAG_NOTE", "true"), default=True)

    # Numerische Felder robust übernehmen (auch "0" zulassen)
    def norm_int(name, fallback=None):
        if name in data and data[name] is not None and str(data[name]).strip() != "":
            try:
                updates[name] = str(int(str(data[name]).strip()))
            except Exception:
                if fallback is not None:
                    updates[name] = str(fallback)

    norm_int("MIN_CLOSED_DAYS")
    norm_int("TOP_K_RESULTS")
    norm_int("MAX_TOKENS")

    # QDRANT_URL optional trimmen
    qurl = updates.get("QDRANT_URL")
    if isinstance(qurl, str):
        updates["QDRANT_URL"] = qurl.strip()

    write_env_config(updates)
    return {"success": True}

@app.get("/api/status")
async def get_status(session: str = Depends(api_auth)):
    """
    Läd Service-Status (direkte Live-Checks, nicht systemd) und Systemstatistiken.
    - Qdrant: /healthz gegen QDRANT_URL aus ENV
    - Ollama: /api/tags gegen OLLAMA_URL aus ENV
    - Zammad: /api/v1/users/me (nur wenn Token vorhanden)
    - System: CPU/RAM/Disk lokal via psutil (falls verfügbar)
    """
    status_payload = {"services": {}, "system": {}}

    # --- Konfiguration aus Settings/ENV holen
    from backend.config import settings as _settings
    qdrant_url = (_settings.QDRANT_URL or "").rstrip("/")
    ollama_url = (_settings.OLLAMA_URL or "").rstrip("/")
    zammad_url = (_settings.ZAMMAD_URL or "").rstrip("/")
    zammad_token = (_settings.ZAMMAD_TOKEN or "").strip()

    # --- Qdrant Check
    q_status = {"status": "unknown"}
    if qdrant_url:
        try:
            r = requests.get(f"{qdrant_url}/healthz", timeout=3)
            if r.status_code == 200:
                q_status = {"status": "ok"}
            else:
                q_status = {"status": "error", "detail": f"HTTP {r.status_code}"}
        except Exception as e:
            q_status = {"status": "error", "detail": str(e)}
    else:
        q_status = {"status": "unknown", "detail": "QDRANT_URL not set"}
    status_payload["services"]["qdrant"] = q_status

    # --- Ollama Check
    o_status = {"status": "unknown"}
    if ollama_url:
        try:
            r = requests.get(f"{ollama_url}/api/tags", timeout=3)
            if r.status_code == 200:
                o_status = {"status": "ok"}
            else:
                o_status = {"status": "error", "detail": f"HTTP {r.status_code}"}
        except Exception as e:
            o_status = {"status": "error", "detail": str(e)}
    else:
        o_status = {"status": "unknown", "detail": "OLLAMA_URL not set"}
    status_payload["services"]["ollama"] = o_status

    # --- Zammad Check
    z_status = {"status": "unknown"}
    if zammad_url and zammad_token:
        try:
            headers = {"Authorization": f"Token token={zammad_token}"}
            r = requests.get(f"{zammad_url}/api/v1/users/me", headers=headers, timeout=5)
            if r.status_code == 200:
                z_status = {"status": "ok"}
            else:
                z_status = {"status": "error", "detail": f"HTTP {r.status_code}"}
        except Exception as e:
            z_status = {"status": "error", "detail": str(e)}
    else:
        miss = []
        if not zammad_url: miss.append("ZAMMAD_URL")
        if not zammad_token: miss.append("ZAMMAD_TOKEN")
        z_status = {"status": "unknown", "detail": f"missing {', '.join(miss)}"}
    status_payload["services"]["zammad"] = z_status

    # --- System Stats (psutil optional)
    try:
        import psutil, shutil
        cpu = psutil.cpu_percent(interval=0.2)
        mem = psutil.virtual_memory().percent
        total, used, free = shutil.disk_usage("/")
        disk = round(used / total * 100, 1) if total else None
        status_payload["system"] = {"cpu": cpu, "mem": mem, "disk": disk}
    except Exception:
        status_payload["system"] = {"cpu": None, "mem": None, "disk": None}

    return status_payload

@app.post("/api/zammad/test")
async def test_zammad_connection(request: Request, session: str = Depends(api_auth)):
    """
    Serverseitiger Verbindungs-Test zu Zammad, um CORS/Mixed-Content im Browser zu vermeiden.
    Body: { "url": "...", "token": "..." }
    """
    try:
        data = await request.json()
        url = data.get("url", "").rstrip("/")
        token = data.get("token", "")
        if not url or not token:
            return JSONResponse(status_code=400, content={"success": False, "error": "URL und Token erforderlich"})
        
        headers = {"Authorization": f"Token token={token}"}
        resp = requests.get(f"{url}/api/v1/users/me", headers=headers, timeout=10)
        if resp.status_code == 200:
            user = resp.json().get("login", "Unbekannt")
            return {"success": True, "user": user}
        else:
            return JSONResponse(status_code=resp.status_code, content={"success": False, "error": f"HTTP {resp.status_code}"})
    except requests.exceptions.RequestException as e:
        return JSONResponse(status_code=502, content={"success": False, "error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

@app.get("/api/activities")
async def get_activities(session: str = Depends(api_auth)):
    """
    Liefert die zuletzt verarbeiteten Aktivitäten als Liste.
    Quelle: JSONL /data/log/activities.jsonl und Fallback auf Qdrant‑Top‑Items,
    wenn keine Logdatei vorhanden ist.
    """
    log_path = "/data/log/activities.jsonl"
    max_items = 50
    items = []

    # 1) Versuche aus JSONL‑Log zu lesen
    try:
        if os.path.isfile(log_path):
            with open(log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        items.append(obj)
                    except Exception:
                        continue
            def ts(x):
                v = x.get("processed_at")
                try:
                    return float(v)
                except Exception:
                    return 0.0
            items.sort(key=ts, reverse=True)
            items = items[:max_items]
            if items:
                return items
    except Exception as e:
        # Logging lesen fehlgeschlagen – wir fallen auf Qdrant zurück
        logger.warning(f"Aktivitätslog konnte nicht gelesen werden: {e}")

    # 2) Fallback: Zeige letzte Punkte aus Qdrant‑Collection als „Aktivität“
    try:
        from backend.config import settings as _settings
        coll = _settings.COLLECTION_NAME
        if not coll:
            return []
        from qdrant_client import QdrantClient
        client = QdrantClient(url=_settings.QDRANT_URL, api_key=_settings.QDRANT_API_KEY or None)
        # Scroll einige Punkte mit Payload; limitieren und als Aktivitäten rendern
        limit = 20
        result = client.scroll(collection_name=coll, with_payload=True, limit=limit)
        points = result[0] if isinstance(result, (list, tuple)) else []
        out = []
        for p in points or []:
            try:
                payload = getattr(p, "payload", {}) or {}
                tid = payload.get("ticket_id") or payload.get("id")
                # Titel-Priorität: kurzbeschreibung > beschreibung > title > empty
                title = (
                    payload.get("kurzbeschreibung")
                    or payload.get("beschreibung")
                    or payload.get("title")
                    or ""
                )
                # processed_at ableiten: bevorzugt erstelldatum (YYYY-MM-DD) -> epoch, sonst jetzt
                ts = None
                date_str = payload.get("erstelldatum")
                if isinstance(date_str, str) and len(date_str) >= 10:
                    try:
                        import datetime as _dt, time as _time
                        dt = _dt.datetime.strptime(date_str[:10], "%Y-%m-%d")
                        ts = _time.mktime(dt.timetuple())
                    except Exception:
                        ts = None
                if ts is None:
                    from time import time as _now
                    ts = _now()
                out.append({
                    "kind": "qdrant_point",
                    "ticket_id": tid,
                    "title": title if isinstance(title, str) else "",
                    "processed_at": float(ts)
                })
            except Exception:
                continue
        # Neueste zuerst sortieren und auf 50 begrenzen (Konsistenz zum Log-Pfad)
        out.sort(key=lambda x: (x.get("processed_at") or 0.0), reverse=True)
        return out[:50]
    except Exception as e:
        logger.warning(f"Fallback Qdrant‑Aktivitäten fehlgeschlagen: {e}")
        return []

@app.get("/api/ollama/models")
async def get_ollama_models(session: str = Depends(api_auth)):
    """Lade verfügbare Ollama-Modelle (robuster Healthcheck + klare Fehler)."""
    try:
        # 1) Settings laden
        from backend.config import settings as _settings
        base = (_settings.OLLAMA_URL or "").rstrip("/")

        # 2) Healthcheck gegen /api/tags (liefert Liste der Modelle)
        tags_url = f"{base}/api/tags" if base else ""
        if not tags_url:
            return {"success": False, "error": "OLLAMA_URL nicht gesetzt", "models": []}

        try:
            resp = requests.get(tags_url, timeout=5)
        except requests.exceptions.RequestException as rexc:
            return {"success": False, "error": f"Ollama Healthcheck fehlgeschlagen: {rexc}", "models": []}

        if resp.status_code != 200:
            # Detaillierte Fehlermeldung versuchen
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            return {"success": False, "error": f"Ollama Healthcheck HTTP {resp.status_code}: {detail}", "models": []}

        # 3) Modelle aus Healthcheck extrahieren
        try:
            data = resp.json()
            models = [m.get("name") for m in data.get("models", []) if isinstance(m, dict) and m.get("name")]
        except Exception as jex:
            return {"success": False, "error": f"Fehler beim Parsen von /api/tags: {jex}", "models": []}

        # 4) Optional: aktuelles Modell aus ENV anzeigen
        current = getattr(_settings, "OLLAMA_MODEL", None) or os.environ.get("OLLAMA_MODEL", None)

        return {"success": True, "models": models or [], "current": current}
    except Exception as e:
        return {"success": False, "error": str(e), "models": []}

@app.post("/api/ollama/pull")
async def pull_ollama_model(request: Request, session: str = Depends(api_auth)):
    """
    Lädt ein Ollama-Modell herunter (robuster: Healthcheck + klare Fehlertexte).
    - Body {"model": "..."} oder {"name": "..."} (Alias); Fallback auf ENV OLLAMA_MODEL.
    - Prüft zunächst /api/tags, bevor /api/pull aufgerufen wird.
    """
    try:
        from backend.config import settings as _settings
        base = (_settings.OLLAMA_URL or "").rstrip("/")
        if not base:
            return JSONResponse(status_code=400, content={"success": False, "error": "OLLAMA_URL nicht gesetzt"})

        # Body lesen (robust gegen leeren Body)
        model_name = None
        try:
            data = await request.json()
            if isinstance(data, dict):
                model_name = (data.get("model") or data.get("name") or "").strip()
        except Exception:
            data = {}

        if not model_name:
            env_model = (getattr(_settings, "OLLAMA_MODEL", "") or os.environ.get("OLLAMA_MODEL", "")).strip()
            model_name = env_model

        if not model_name:
            return JSONResponse(status_code=400, content={"success": False, "error": "Modellname erforderlich"})

        # Healthcheck /api/tags
        tags_url = f"{base}/api/tags"
        try:
            health = requests.get(tags_url, timeout=5)
        except requests.exceptions.RequestException as rexc:
            return JSONResponse(status_code=502, content={"success": False, "error": f"Ollama nicht erreichbar: {rexc}"})
        if health.status_code != 200:
            try:
                detail = health.json()
            except Exception:
                detail = health.text
            return JSONResponse(status_code=health.status_code, content={
                "success": False,
                "error": f"Ollama Healthcheck HTTP {health.status_code}: {detail}"
            })

        # Pull aufrufen
        api_url = f"{base}/api/pull"
        payload = {"model": model_name}
        logger.info(f"Ollama Pull: url={api_url}, model='{model_name}'")

        response = requests.post(api_url, json=payload, timeout=600)

        if response.status_code == 200:
            msg = f"Modell {model_name} erfolgreich heruntergeladen"
            try:
                body = response.json()
                return {"success": True, "message": msg, "ollama": body}
            except Exception:
                return {"success": True, "message": msg}
        else:
            try:
                err_json = response.json()
                err_text = err_json.get("error") or err_json
            except Exception:
                err_text = response.text
            return JSONResponse(
                status_code=response.status_code,
                content={"success": False, "error": f"Ollama Fehler: {err_text or ('HTTP ' + str(response.status_code))}"}
            )

    except requests.exceptions.RequestException as e:
        return JSONResponse(status_code=502, content={"success": False, "error": f"Netzwerkfehler zu Ollama: {str(e)}"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

@app.delete("/api/ollama/models/{model_name}")
async def delete_ollama_model(model_name: str, session: str = Depends(api_auth)):
    """Löscht ein Ollama-Modell"""
    try:
        from backend.config import settings
        api_url = settings.OLLAMA_URL.rstrip('/') + '/api/delete'
        
        response = requests.delete(api_url, json={"name": model_name}, timeout=60)
        
        if response.status_code == 200:
            return {"success": True, "message": f"Modell {model_name} erfolgreich gelöscht"}
        else:
            return {"success": False, "error": f"HTTP {response.status_code}"}
            
    except Exception as e:
        return {"success": False, "error": str(e)}

# --- Interner Prozessmanager für den RAG-Poller (Single-Container) + Ingest-Job ---
from threading import Thread, Event, Lock
from time import time
import datetime

# Poller
_poller_thread = None
_poller_stop_event = None
_poller_status = {"running": False, "started_at": None, "last_tick": None}

def _run_poller():
    # Laufzeitimport, um Importzeiten zu reduzieren und zirkuläre Imports zu vermeiden
    try:
        # Bevorzugter Importpfad entspricht der tatsächlichen Projektstruktur
        from Services.zammad_rag_poller import process_tickets  # type: ignore
    except Exception as e1:
        try:
            # Fallback: absoluter Pfad relativ zum Arbeitsverzeichnis via importlib
            import importlib.util, sys, os as _os
            base_dir = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # web-ui/
            poller_path = _os.path.join(base_dir, "Services", "zammad_rag_poller.py")
            spec = importlib.util.spec_from_file_location("zammad_rag_poller_dyn", poller_path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                sys.modules["zammad_rag_poller_dyn"] = mod
                spec.loader.exec_module(mod)  # type: ignore
                process_tickets = getattr(mod, "process_tickets")
            else:
                raise ImportError("Konnte Spec für zammad_rag_poller nicht erstellen")
        except Exception as e2:
            logger.error(f"Poller-Import fehlgeschlagen: {e1} | Fallback: {e2}")
            return

    global _poller_stop_event, _poller_status
    _poller_status["running"] = True
    _poller_status["started_at"] = time()
    try:
        process_tickets(stop_event=_poller_stop_event, status_holder=_poller_status)
    finally:
        _poller_status["running"] = False

def poller_status():
    return {
        "running": _poller_status["running"],
        "started_at": _poller_status["started_at"],
        "last_tick": _poller_status["last_tick"]
    }

def poller_start():
    global _poller_thread, _poller_stop_event
    if _poller_thread and _poller_thread.is_alive():
        return {"success": True, "status": "already_running"}
    _poller_stop_event = Event()
    _poller_thread = Thread(target=_run_poller, name="rag_poller", daemon=True)
    _poller_thread.start()
    return {"success": True, "status": "started"}

def poller_stop(timeout=10.0):
    global _poller_thread, _poller_stop_event
    if not _poller_thread or not _poller_thread.is_alive():
        return {"success": True, "status": "already_stopped"}
    _poller_stop_event.set()
    _poller_thread.join(timeout=timeout)
    if _poller_thread.is_alive():
        return {"success": False, "status": "stop_timeout"}
    return {"success": True, "status": "stopped"}

def poller_restart():
    stop_res = poller_stop()
    start_res = poller_start()
    return {"success": stop_res.get("success") and start_res.get("success"), "stop": stop_res, "start": start_res}

# Zentrale Liste der Poller-Service-Namen zur Wiederverwendung
_POLLER_SERVICE_ALIASES = {"rag-poller", "poller", "zammad_rag_poller"}

def _is_poller(service_name: str) -> bool:
    return service_name in _POLLER_SERVICE_ALIASES

# --- Ingest-Job (Batch: Zammad -> Qdrant) ---
_ingest_thread = None
_ingest_stop_event = None  # für spätere Erweiterung, aktuell einmaliger Job
_ingest_lock = Lock()
_ingest_status = {
    "running": False,
    "started_at": None,
    "finished_at": None,
    "last_message": None,
    "last_success": None,
    "error": None
}
# Zeitplanung (persistiert in ENV): einfache CRON/Preset-Strings
# ENV Keys: INGEST_SCHEDULE (z.B. "@hourly", "@daily 23:00", "0 * * * *", "0 23 * * *")
def _read_env_value(key: str) -> str:
    cfg = read_env_config()
    return (cfg.get(key) or "").strip()

def _write_env_value(key: str, value: str):
    write_env_config({key: value})

def _should_trigger_now(schedule: str, now: datetime.datetime) -> bool:
    """
    Sehr einfache Scheduler-Logik:
    - "@hourly": jede volle Stunde (Minute==0)
    - "@daily HH:MM": täglich zur Uhrzeit
    - "0 * * * *" / "M H * * *" (nur Minute und Stunde ausgewertet)
    """
    schedule = (schedule or "").strip()
    if not schedule:
        return False
    try:
        if schedule == "@hourly":
            return now.minute == 0
        if schedule.startswith("@daily"):
            # @daily HH:MM (default 23:00 wenn keine Zeit)
            parts = schedule.split()
            hhmm = parts[1] if len(parts) > 1 else "23:00"
            hh, mm = hhmm.split(":")
            return now.hour == int(hh) and now.minute == int(mm)
        # primitive crontab: "M H * * *"
        if schedule.count(" ") == 4:
            m_str, h_str, _, _, _ = schedule.split()
            def match_field(field, val):
                if field == "*":
                    return True
                try:
                    return int(field) == val
                except:
                    return False
            return match_field(m_str, now.minute) and match_field(h_str, now.hour)
    except Exception as e:
        logger.warning(f"Schedule-Parsing Fehler: {e}")
    return False

def _append_activity(entry: dict):
    """Schreibt eine Aktivität als JSONL nach /data/log/activities.jsonl"""
    try:
        # Sichere Log‑Pfad‑Erzeugung und kompaktes JSON
        os.makedirs("/data/log", exist_ok=True)
        with open("/data/log/activities.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")
    except Exception as e:
        logger.warning(f"Aktivitäts-Log konnte nicht geschrieben werden: {e}")

def _run_ingest_once():
    global _ingest_status
    with _ingest_lock:
        if _ingest_status["running"]:
            return False, "Ingest läuft bereits"
        _ingest_status.update({
            "running": True,
            "started_at": time(),
            "finished_at": None,
            "last_message": "Gestartet",
            "error": None
        })
    try:
        # Laufzeitimport des Batch-Importers
        try:
            from Services.ZammadToQdrant import fetch_and_process_zammad_tickets  # type: ignore
        except Exception:
            from web_ui.Services.ZammadToQdrant import fetch_and_process_zammad_tickets  # type: ignore
        # Wrapper: führe Import aus und erzeuge eine Sammel-Aktivität
        start_ts = time()
        fetch_and_process_zammad_tickets()
        end_ts = time()
        _append_activity({
            "kind": "ingest_batch",
            "message": "Batch-Import abgeschlossen",
            "processed_at": end_ts,
            "duration_s": round(end_ts - start_ts, 2)
        })
        _ingest_status["last_success"] = end_ts
        _ingest_status["last_message"] = "Erfolgreich abgeschlossen"
        return True, "OK"
    except Exception as e:
        _ingest_status["error"] = str(e)
        _ingest_status["last_message"] = f"Fehler: {e}"
        logger.exception("Ingest-Fehler")
        _append_activity({
            "kind": "ingest_batch",
            "message": f"Batch-Import Fehler: {e}",
            "processed_at": time(),
            "error": True
        })
        return False, str(e)
    finally:
        _ingest_status["running"] = False
        _ingest_status["finished_at"] = time()

def ingest_start():
    global _ingest_thread
    if _ingest_thread and _ingest_thread.is_alive():
        return {"success": True, "status": "already_running"}
    _ingest_thread = Thread(target=_run_ingest_once, name="ingest_job", daemon=True)
    _ingest_thread.start()
    return {"success": True, "status": "started"}

def ingest_status():
    # simple snapshot
    return dict(_ingest_status)

# Hintergrund-Ticker für Scheduling
_scheduler_thread = None
_scheduler_stop_event = Event()

def _scheduler_loop():
    last_trigger_minute = None
    while not _scheduler_stop_event.is_set():
        try:
            schedule = _read_env_value("INGEST_SCHEDULE")
            now = datetime.datetime.now()
            minute_key = now.strftime("%Y-%m-%d %H:%M")
            if schedule and minute_key != last_trigger_minute and _should_trigger_now(schedule, now):
                last_trigger_minute = minute_key
                # nur starten, wenn nicht schon läuft
                if not _ingest_status.get("running"):
                    logger.info(f"Scheduler: starte Ingest (Plan: '{schedule}')")
                    ingest_start()
        except Exception as e:
            logger.warning(f"Scheduler-Loop Fehler: {e}")
        # alle 10s prüfen
        if _scheduler_stop_event.wait(10.0):
            break

def scheduler_start_once():
    global _scheduler_thread
    if _scheduler_thread and _scheduler_thread.is_alive():
        return
    _scheduler_thread = Thread(target=_scheduler_loop, name="ingest_scheduler", daemon=True)
    _scheduler_thread.start()

# Scheduler beim App-Start aktivieren
scheduler_start_once()

def _status_payload(service_name: str):
    res = poller_status()
    return {"service": service_name, "status": "running" if res["running"] else "stopped", "detail": res}

# HARMONISIERTE SERVICE-API (kompatibel zu /api/services/... aus dem Frontend-Log)
@app.get("/api/services/{service_name}/status")
async def get_service_status_compat(service_name: str, session: str = Depends(api_auth)):
    """Kompatibler Alias: Status eines 'Service' – hier: interner rag-poller"""
    if _is_poller(service_name):
        return _status_payload(service_name)
    # Fallback: alte service_manager-Checks optional
    try:
        return service_manager.check_systemd_service_status(service_name)
    except Exception:
        return {"service": service_name, "status": "unknown", "detail": "no systemd in container"}

@app.get("/api/service/{service_name}/status")
async def get_service_status(service_name: str, session: str = Depends(api_auth)):
    """Status eines 'Service' – hier: interner rag-poller"""
    if _is_poller(service_name):
        return _status_payload(service_name)
    try:
        return service_manager.check_systemd_service_status(service_name)
    except Exception:
        return {"service": service_name, "status": "unknown", "detail": "no systemd in container"}

@app.post("/api/services/{service_name}/control")
async def control_service_compat(service_name: str, request: Request, session: str = Depends(api_auth)):
    """Kompatibel: ?action=start|stop|restart|status"""
    action = request.query_params.get("action")
    if action not in ["start", "stop", "restart", "status"]:
        raise HTTPException(status_code=400, detail="Ungültige Aktion")
    return await control_service(service_name, action, session)

@app.post("/api/service/{service_name}/{action}")
async def control_service(service_name: str, action: str, session: str = Depends(api_auth)):
    """Steuert internen rag-poller: start/stop/restart/status"""
    if action not in ["start", "stop", "restart", "status"]:
        raise HTTPException(status_code=400, detail="Ungültige Aktion")
    if _is_poller(service_name):
        if action == "start":
            res = poller_start()
        elif action == "stop":
            res = poller_stop()
        elif action == "restart":
            res = poller_restart()
        else:
            res = _status_payload(service_name)
        try:
            logger.info(f"Poller control: service={service_name}, action={action}, result={res}")
        except Exception:
            pass
        return res
    # Fallback für andere 'Services'
    try:
        return service_manager.control_service(service_name, action)
    except Exception:
        return {"service": service_name, "action": action, "success": False, "detail": "no systemd in container"}

# --- Ingest-API ---
@app.get("/api/ingest/status")
async def api_ingest_status(session: str = Depends(api_auth)):
    return ingest_status()

@app.post("/api/ingest/start")
async def api_ingest_start(session: str = Depends(api_auth)):
    return ingest_start()

@app.post("/api/ingest/stop")
async def api_ingest_stop(session: str = Depends(api_auth)):
    # einmaliger Job – kein Live-Stop vorgesehen; reserviert für Zukunft
    st = ingest_status()
    return {"success": not st.get("running", False), "detail": "Stop nicht unterstützt für Batch; wartet auf Abschluss"}

@app.get("/api/ingest/schedule")
async def api_ingest_get_schedule(session: str = Depends(api_auth)):
    return {"schedule": _read_env_value("INGEST_SCHEDULE")}

@app.post("/api/ingest/schedule")
async def api_ingest_set_schedule(request: Request, session: str = Depends(api_auth)):
    body = await request.json()
    schedule = (body.get("schedule") or "").strip()
    _write_env_value("INGEST_SCHEDULE", schedule)
    return {"success": True, "schedule": schedule}

@app.post("/api/qdrant/test")
async def test_qdrant_connection(request: Request, session: str = Depends(api_auth)):
    """
    Serverseitiger Test gegen Qdrant.
    Body: { "url": "...", "collection": "...", "api_key": "...", "create_if_missing": true|false }
    - Prüft Health.
    - Prüft optional Collection und liefert points_count.
    - Wenn create_if_missing=true und die Collection fehlt, wird sie angelegt (mit Multi-Vektor-Layout).
    """
    try:
        from backend.config import settings as _settings
        data = await request.json()
        url_override = (data.get("url") or "").strip()
        collection = (data.get("collection") or "").strip()
        api_key_override = (data.get("api_key") or "").strip()
        create_if_missing = bool(data.get("create_if_missing", False))

        base_url = (url_override or _settings.QDRANT_URL or "").rstrip("/")
        if not base_url:
            return JSONResponse(status_code=400, content={"success": False, "error": "QDRANT URL fehlt"})

        headers = {}
        api_key = api_key_override if api_key_override else _settings.QDRANT_API_KEY
        if api_key:
            headers["api-key"] = api_key

        # 1) Healthcheck
        health_resp = requests.get(f"{base_url}/healthz", headers=headers, timeout=5)
        if health_resp.status_code != 200:
            try:
                detail = health_resp.json()
            except Exception:
                detail = health_resp.text
            return JSONResponse(status_code=health_resp.status_code, content={
                "success": False,
                "error": f"Healthcheck fehlgeschlagen (HTTP {health_resp.status_code}): {detail}"
            })

        result = {"success": True, "url": base_url}

        # 2) Optional: Collection prüfen
        if collection:
            coll_url = f"{base_url}/collections/{collection}"
            coll_resp = requests.get(coll_url, headers=headers, timeout=10)
            if coll_resp.status_code == 200:
                info = coll_resp.json()
                points_count = info.get("result", {}).get("points_count")
                result.update({"collection": collection, "points_count": points_count})
            elif coll_resp.status_code == 404 and create_if_missing:
                # Collection anlegen im Multi-Vektor-Layout (wie im Ingest-Skript)
                create_url = f"{base_url}/collections/{collection}"
                create_payload = {
                    "vectors": {
                        "kurzbeschreibung": {"size": 768, "distance": "Cosine"},
                        "beschreibung": {"size": 768, "distance": "Cosine"},
                        "lösung": {"size": 768, "distance": "Cosine"},
                        "all": {"size": 768, "distance": "Cosine"}
                    }
                }
                c_resp = requests.put(create_url, headers={**headers, "Content-Type": "application/json"}, json=create_payload, timeout=15)
                if c_resp.status_code in (200, 201):
                    # Nach Anlage erneut abfragen
                    verify = requests.get(coll_url, headers=headers, timeout=10)
                    if verify.status_code == 200:
                        vinfo = verify.json()
                        points_count = vinfo.get("result", {}).get("points_count")
                        result.update({"collection": collection, "points_count": points_count, "created": True})
                    else:
                        return JSONResponse(status_code=verify.status_code, content={
                            "success": False,
                            "error": f'Collection "{collection}" nach Anlage nicht lesbar (HTTP {verify.status_code})'
                        })
                else:
                    try:
                        cdetail = c_resp.json()
                    except Exception:
                        cdetail = c_resp.text
                    return JSONResponse(status_code=c_resp.status_code, content={
                        "success": False,
                        "error": f'Collection "{collection}" konnte nicht angelegt werden (HTTP {c_resp.status_code}): {cdetail}'
                    })
            else:
                try:
                    cdetail = coll_resp.json()
                except Exception:
                    cdetail = coll_resp.text
                return JSONResponse(status_code=coll_resp.status_code, content={
                    "success": False,
                    "error": f'Collection "{collection}" nicht erreichbar (HTTP {coll_resp.status_code}): {cdetail}'
                })

        return result
    except requests.exceptions.RequestException as e:
        return JSONResponse(status_code=502, content={"success": False, "error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})

# Statische Dateien
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    from backend.config import settings
    uvicorn.run(app, host=settings.WEBUI_HOST, port=settings.WEBUI_PORT)
