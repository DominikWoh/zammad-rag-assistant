import os
import hashlib
import secrets
import requests
import logging
from typing import Optional
from dotenv import dotenv_values
from fastapi import FastAPI, HTTPException, Request, Depends, status, Response, Cookie
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from services import service_manager

# Logger initialisieren
logger = logging.getLogger("webui")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.setLevel(logging.INFO)

# --- Konfiguration und Pfaderkennung ---
WINDOWS_ENV_PATH = os.path.join(os.getcwd(), ".env")
UBUNTU_ENV_PATH = "/opt/ai-suite/ticket_ingest.env"
COOKIE_NAME = "zammad_rag_session"

def detect_env_path():
    if os.path.isfile(WINDOWS_ENV_PATH):
        return WINDOWS_ENV_PATH
    elif os.path.isfile(UBUNTU_ENV_PATH):
        return UBUNTU_ENV_PATH
    else:
        open(WINDOWS_ENV_PATH, 'a').close()
        return WINDOWS_ENV_PATH

ENV_PATH = detect_env_path()
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
    return dotenv_values(ENV_PATH)

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
            new_lines.append(f"{key}={updates[key]}\n")
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
    """Lädt die aktuelle Konfiguration"""
    config = read_env_config()
    # Defaults anwenden: ENABLE_ASKKI=false, ENABLE_RAG_NOTE=true
    normalized = dict(config)
    normalized["ENABLE_ASKKI"] = (config.get("ENABLE_ASKKI") or "false")
    normalized["ENABLE_RAG_NOTE"] = (config.get("ENABLE_RAG_NOTE") or "true")
    return normalized

@app.post("/api/config")
async def save_config(request: Request, session: str = Depends(api_auth)):
    """Speichert die Konfiguration"""
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
    updates["ENABLE_ASKKI"] = to_bool_str(data.get("ENABLE_ASKKI", "false"), default=False)
    updates["ENABLE_RAG_NOTE"] = to_bool_str(data.get("ENABLE_RAG_NOTE", "true"), default=True)

    write_env_config(updates)
    return {"success": True}

@app.get("/api/status")
async def get_status(session: str = Depends(api_auth)):
    """Lädt Service-Status und Systemstatistiken"""
    return {
        "services": {
            "qdrant": service_manager.check_qdrant_status(),
            "ollama": service_manager.check_ollama_status(),
            "zammad": service_manager.check_zammad_status()
        },
        "system": service_manager.get_system_stats()
    }

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
    """Lädt letzte Aktivitäten"""
    return service_manager.get_recent_activities()

@app.get("/api/ollama/models")
async def get_ollama_models(session: str = Depends(api_auth)):
    """Lade verfügbare Ollama-Modelle"""
    try:
        ollama_status = service_manager.check_ollama_status()
        if ollama_status["status"] == "online":
            return {
                "success": True,
                "models": ollama_status["available_models"],
                "current": ollama_status["current_model"]
            }
        else:
            return {
                "success": False,
                "error": ollama_status.get("error", "Ollama nicht verfügbar"),
                "models": []
            }
    except Exception as e:
        return {"success": False, "error": str(e), "models": []}

@app.post("/api/ollama/pull")
async def pull_ollama_model(request: Request, session: str = Depends(api_auth)):
    """Lädt ein Ollama-Modell herunter"""
    try:
        data = await request.json()
        model_name = data.get("model")
        
        if not model_name:
            return {"success": False, "error": "Modellname erforderlich"}
        
        from config import settings
        api_url = settings.OLLAMA_URL.rstrip('/') + '/api/pull'
        
        response = requests.post(api_url, json={"name": model_name}, timeout=300)
        
        if response.status_code == 200:
            return {"success": True, "message": f"Modell {model_name} erfolgreich heruntergeladen"}
        else:
            return {"success": False, "error": f"HTTP {response.status_code}"}
            
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.delete("/api/ollama/models/{model_name}")
async def delete_ollama_model(model_name: str, session: str = Depends(api_auth)):
    """Löscht ein Ollama-Modell"""
    try:
        from config import settings
        api_url = settings.OLLAMA_URL.rstrip('/') + '/api/delete'
        
        response = requests.delete(api_url, json={"name": model_name}, timeout=60)
        
        if response.status_code == 200:
            return {"success": True, "message": f"Modell {model_name} erfolgreich gelöscht"}
        else:
            return {"success": False, "error": f"HTTP {response.status_code}"}
            
    except Exception as e:
        return {"success": False, "error": str(e)}

# HARMONISIERTE SERVICE-API (kompatibel zu /api/services/... aus dem Frontend-Log)
@app.get("/api/services/{service_name}/status")
async def get_service_status_compat(service_name: str, session: str = Depends(api_auth)):
    """Alias: Prüft den Status eines systemd-Services (kompatible Route)"""
    result = service_manager.check_systemd_service_status(service_name)
    try:
        logger.info(f"Service status (compat): unit={service_name}, status={result.get('status')}, detail={result.get('detail')}")
    except Exception:
        pass
    return result

@app.get("/api/service/{service_name}/status")
async def get_service_status(service_name: str, session: str = Depends(api_auth)):
    """Prüft den Status eines systemd-Services"""
    result = service_manager.check_systemd_service_status(service_name)
    try:
        logger.info(f"Service status: unit={service_name}, status={result.get('status')}, detail={result.get('detail')}")
    except Exception:
        pass
    return result

@app.post("/api/services/{service_name}/control")
async def control_service_compat(service_name: str, request: Request, session: str = Depends(api_auth)):
    """
    Alias: Steuert einen systemd-Service (kompatible Route)
    Erwartet Query-Parameter ?action=start|stop|restart|status
    """
    action = request.query_params.get("action")
    if action not in ["start", "stop", "restart", "status"]:
        raise HTTPException(status_code=400, detail="Ungültige Aktion")
    result = service_manager.control_service(service_name, action)
    try:
        logger.info(f"Service control (compat): unit={service_name}, action={action}, cmd={result.get('cmd')}, code={result.get('code')}, success={result.get('success')}")
    except Exception:
        pass
    return result

@app.post("/api/service/{service_name}/{action}")
async def control_service(service_name: str, action: str, session: str = Depends(api_auth)):
    """Steuert einen systemd-Service (start/stop/restart/status)"""
    if action not in ["start", "stop", "restart", "status"]:
        raise HTTPException(status_code=400, detail="Ungültige Aktion")
    result = service_manager.control_service(service_name, action)
    try:
        logger.info(f"Service control: unit={service_name}, action={action}, cmd={result.get('cmd')}, code={result.get('code')}, success={result.get('success')}")
    except Exception:
        pass
    return result

@app.post("/api/qdrant/test")
async def test_qdrant_connection(request: Request, session: str = Depends(api_auth)):
    """
    Serverseitiger Test gegen Qdrant.
    Optionaler Body: { "collection": "...", "api_key": "..." }
    - URL wird aus der gespeicherten Server-Konfiguration (settings.QDRANT_URL) gelesen.
    - Wenn api_key übergeben wird, wird er genutzt, sonst settings.QDRANT_API_KEY.
    - Prüft Health und (falls collection angegeben) ruft Collection-Info ab.
    """
    try:
        from config import settings
        data = await request.json()
        collection = (data.get("collection") or "").strip()
        api_key_override = (data.get("api_key") or "").strip()

        base_url = settings.QDRANT_URL.rstrip("/")
        headers = {}
        api_key = api_key_override if api_key_override else settings.QDRANT_API_KEY
        if api_key:
            headers["api-key"] = api_key

        # 1) Healthcheck
        health_resp = requests.get(f"{base_url}/healthz", headers=headers, timeout=5)
        if health_resp.status_code != 200:
            return JSONResponse(status_code=health_resp.status_code, content={
                "success": False,
                "error": f"Healthcheck fehlgeschlagen (HTTP {health_resp.status_code})"
            })

        result = {"success": True, "url": base_url}

        # 2) Optional: Collection prüfen
        if collection:
            coll_resp = requests.get(f"{base_url}/collections/{collection}", headers=headers, timeout=10)
            if coll_resp.status_code == 200:
                info = coll_resp.json()
                points_count = info.get("result", {}).get("points_count")
                result.update({"collection": collection, "points_count": points_count})
            else:
                return JSONResponse(status_code=coll_resp.status_code, content={
                    "success": False,
                    "error": f'Collection "{collection}" nicht erreichbar (HTTP {coll_resp.status_code})'
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
    from config import settings
    uvicorn.run(app, host=settings.WEBUI_HOST, port=settings.WEBUI_PORT)
