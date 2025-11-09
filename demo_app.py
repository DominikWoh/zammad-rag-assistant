#!/usr/bin/env python3
"""
Zammad Qdrant Web Interface - Connected Version
Integrated with zammad_to_qdrant.py
"""

import logging
import os
import json
import subprocess
import time
import glob
import socket
import platform
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("web_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Helper to update single env key (e.g., UI language)
def set_env_key(key: str, value: str) -> None:
    try:
        existing_lines = []
        if ENV_FILE.exists():
            with open(ENV_FILE, 'r', encoding='utf-8') as f:
                existing_lines = f.readlines()

        found = False
        new_lines = []
        for line in existing_lines:
            line_stripped = line.strip()
            if '=' in line_stripped and not line_stripped.startswith('#'):
                k = line_stripped.split('=')[0]
                if k == key:
                    new_lines.append(f'{key}="{value}"\n')
                    found = True
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)

        if not found:
            new_lines.append(f'{key}="{value}"\n')

        with open(ENV_FILE, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)

        # Reload env vars for running process
        load_dotenv(override=True)
        logger.info(f"Updated {key} in .env to: {value}")
    except Exception as e:
        logger.error(f"Error updating {key} in .env: {str(e)}")
        raise

# Data Models
@dataclass
class TransferStatus:
    is_running: bool = False
    progress: int = 0
    current_ticket: Optional[str] = None
    total_tickets: int = 0
    processed_tickets: int = 0
    start_time: Optional[datetime] = None
    error_message: Optional[str] = None

@dataclass
class ConfigData:
    zammad_url: str
    zammad_token: str
    qdrant_url: str
    qdrant_api_key: str
    bm25_cache: bool = False
    min_age_days: int = 14
    start_date: str = "2018-01-01"
    # KI-Einstellungen
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama2"
    ai_check_interval: int = 300  # 5 Minuten in Sekunden
    ai_ticket_max_age_days: int = 1  # Maximales Ticket-Alter in Tagen
    top_k: int = 5  # Anzahl der Vektor-Suchergebnisse
    top_tickets: int = 5  # Anzahl der relevantesten Tickets
    rag_search_prompt: str = "Erstelle einen prägnanten Suchbegriff für die RAG-Suche basierend auf folgendem Ticket: {ticket_content}"
    zammad_note_prompt: str = "Erstelle eine hilfreiche und professionelle Antwort für folgendes Zammad-Ticket basierend auf den verfügbaren Informationen: {ticket_content}\n\nRelevante Informationen:\n{search_results}"
    ai_enabled: bool = False

# Global state
transfer_status = TransferStatus()
current_process: Optional[subprocess.Popen] = None
mcp_server_process: Optional[subprocess.Popen] = None

# Schedule state
import threading
import atexit
from datetime import timedelta
import json

schedule_lock = threading.Lock()
schedule_thread: Optional[threading.Thread] = None
schedule_running = False
scheduled_transfers = {}  # Dict[str, ScheduleConfig]

# Environment file path
ENV_FILE = Path(".env")

@dataclass
class ScheduleConfig:
    """Zeitplan-Konfiguration für automatische Transfers"""
    id: str
    name: str
    interval: str  # "hourly", "daily", "weekly"
    time: Optional[str] = None  # "HH:MM" für täglich/wöchentlich, None für stündlich
    days: Optional[str] = None  # "0,1,2,3,4,5,6" für wöchentlich (Mo-So), None für stündlich/täglich
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    config: Optional[Dict[str, Any]] = None  # Transfer-Konfiguration

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'interval': self.interval,
            'time': self.time,
            'days': self.days,
            'enabled': self.enabled,
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'next_run': self.next_run.isoformat() if self.next_run else None,
            'config': self.config or {}
        }

def load_env_config() -> ConfigData:
    """Load configuration from .env file"""
    return ConfigData(
        zammad_url=os.getenv("ZAMMAD_URL", ""),
        zammad_token=os.getenv("ZAMMAD_TOKEN", ""),
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY", ""),
        bm25_cache=os.getenv("USE_CACHED_BM25", "false").lower() == "true",
        min_age_days=int(os.getenv("TICKET_MIN_AGE_DAYS", "14")),
        start_date=os.getenv("START_DATE", "2018-01-01"),
        # KI-Einstellungen
        ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama2"),
        ai_check_interval=int(os.getenv("AI_CHECK_INTERVAL", "300")),
        ai_ticket_max_age_days=int(os.getenv("AI_TICKET_MAX_AGE_DAYS", "1")),
        top_k=int(os.getenv("TOP_K", "5")),
        top_tickets=int(os.getenv("TOP_TICKETS", "5")),
        rag_search_prompt=os.getenv("RAG_SEARCH_PROMPT", "Erstelle einen prägnanten Suchbegriff für die RAG-Suche basierend auf folgendem Ticket: {ticket_content}"),
        zammad_note_prompt=os.getenv("ZAMMAD_NOTE_PROMPT", "Erstelle eine hilfreiche und professionelle Antwort für folgendes Zammad-Ticket basierend auf den verfügbaren Informationen: {ticket_content}\n\nRelevante Informationen:\n{search_results}"),
        ai_enabled=os.getenv("AI_ENABLED", "false").lower() == "true"
    )

def save_env_config(config: ConfigData) -> None:
    """Save configuration to .env file"""
    env_vars = {
        "ZAMMAD_URL": config.zammad_url,
        "ZAMMAD_TOKEN": config.zammad_token,
        "QDRANT_URL": config.qdrant_url,
        "QDRANT_API_KEY": config.qdrant_api_key,
        "USE_CACHED_BM25": str(config.bm25_cache).lower(),
        "TICKET_MIN_AGE_DAYS": str(config.min_age_days),
        "START_DATE": config.start_date,
        # KI-Einstellungen
        "OLLAMA_URL": config.ollama_url,
        "OLLAMA_MODEL": config.ollama_model,
        "AI_CHECK_INTERVAL": str(config.ai_check_interval),
        "AI_TICKET_MAX_AGE_DAYS": str(config.ai_ticket_max_age_days),
        "TOP_K": str(config.top_k),
        "TOP_TICKETS": str(config.top_tickets),
        "RAG_SEARCH_PROMPT": config.rag_search_prompt.replace('\n', '\\n').replace('\r', ''),
        "ZAMMAD_NOTE_PROMPT": config.zammad_note_prompt.replace('\n', '\\n').replace('\r', ''),
        "AI_ENABLED": str(config.ai_enabled).lower()
    }
    
    # Read existing .env content
    existing_lines = []
    if ENV_FILE.exists():
        with open(ENV_FILE, 'r', encoding='utf-8') as f:
            existing_lines = f.readlines()
    
    # Process lines
    processed_keys = set()
    new_lines = []
    
    for line in existing_lines:
        line_stripped = line.strip()
        if '=' in line_stripped and not line_stripped.startswith('#'):
            key = line_stripped.split('=')[0]
            if key in env_vars:
                # Replace existing key
                new_lines.append(f'{key}="{env_vars[key]}"\n')
                processed_keys.add(key)
            else:
                # Keep existing key
                new_lines.append(line)
        else:
            # Keep comments and empty lines
            new_lines.append(line)
    
    # Add new keys that weren't in existing file
    for key, value in env_vars.items():
        if key not in processed_keys:
            new_lines.append(f'{key}="{value}"\n')
    
    # Write back to file
    with open(ENV_FILE, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    logger.info("Configuration saved to .env file")

def check_zammad_connection() -> Dict[str, Any]:
    """Check Zammad API connection"""
    try:
        config = load_env_config()
        if not config.zammad_url or not config.zammad_token:
            return {"status": "disconnected", "message": "Configuration missing"}
        
        headers = {"Authorization": f"Token token={config.zammad_token}", "Content-Type": "application/json"}
        response = requests.get(f"{config.zammad_url.rstrip('/')}/api/v1/tickets?page=1&per_page=1", headers=headers, timeout=10)
        
        if response.status_code == 200:
            return {"status": "connected", "message": "Zammad API reachable"}
        else:
            return {"status": "disconnected", "message": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"status": "disconnected", "message": str(e)}

def get_qdrant_collections() -> Dict[str, Any]:
    """Get Qdrant collections information"""
    try:
        config = load_env_config()
        if not config.qdrant_url:
            return {"status": "error", "message": "Qdrant URL missing", "collections_count": 0}
        
        # Try to get collections
        url = f"{config.qdrant_url.rstrip('/')}/collections"
        if config.qdrant_api_key:
            headers = {"api-key": config.qdrant_api_key}
            response = requests.get(url, headers=headers, timeout=10)
        else:
            response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            collections_count = len(data.get("result", {}).get("collections", []))
            return {
                "status": "connected",
                "collections_count": collections_count,
                "collections": data.get("result", {}).get("collections", [])
            }
        else:
            return {"status": "error", "message": f"HTTP {response.status_code}", "collections_count": 0}
    except Exception as e:
        return {"status": "error", "message": str(e), "collections_count": 0}

def check_qdrant_connection() -> Dict[str, Any]:
    """Check Qdrant connection"""
    try:
        config = load_env_config()
        if not config.qdrant_url:
            return {"status": "disconnected", "message": "Qdrant URL missing"}
        
        # Try to get collections
        url = f"{config.qdrant_url.rstrip('/')}/collections"
        if config.qdrant_api_key:
            headers = {"api-key": config.qdrant_api_key}
            response = requests.get(url, headers=headers, timeout=10)
        else:
            response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            return {"status": "connected", "message": "Qdrant API reachable"}
        else:
            return {"status": "disconnected", "message": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"status": "disconnected", "message": str(e)}

def get_local_ip() -> str:
    """Get the local IP address of this machine"""
    try:
        # Try to get local IP by connecting to a remote address
        # This is a common method to get the local IP without external dependencies
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.1)  # Short timeout to prevent hanging
        try:
            # This doesn't actually connect, it just determines the local IP
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
        except Exception:
            # Fallback: try to get hostname and resolve
            try:
                hostname = socket.gethostname()
                local_ip = socket.gethostbyname(hostname)
            except Exception:
                # Final fallback: return localhost
                local_ip = "127.0.0.1"
        finally:
            s.close()
        
        # If we got a local IP that's not localhost, use it
        if local_ip and not local_ip.startswith("127."):
            return local_ip
        else:
            # Try to get IP from network interfaces as fallback
            try:
                import netifaces
                interfaces = netifaces.interfaces()
                for interface in interfaces:
                    addresses = netifaces.ifaddresses(interface)
                    if netifaces.AF_INET in addresses:
                        for addr_info in addresses[netifaces.AF_INET]:
                            ip = addr_info.get('addr')
                            if ip and not ip.startswith("127."):
                                return ip
            except ImportError:
                # netifaces not available, continue with fallback
                pass
            return "127.0.0.1"
    except Exception as e:
        logger.warning(f"Could not determine local IP: {str(e)}")
        return "127.0.0.1"

def read_live_log_entries() -> List[Dict[str, Any]]:
    """Get the last 5 log entries from current log file"""
    try:
        # Find the most recent log file
        log_files = glob.glob("zammad_ingest_*.log")
        if not log_files:
            return [{"timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "level": "INFO", "message": "Keine Log-Dateien gefunden"}]
        
        latest_log = max(log_files, key=os.path.getctime)
        
        # Read last 5 lines
        with open(latest_log, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Parse log entries
        entries = []
        for line in lines[-5:]:
            if line.strip():
                # Simple log parsing - assume format: [timestamp] LEVEL: message
                if '] ' in line and ': ' in line:
                    timestamp_part, rest = line.split('] ', 1)
                    timestamp = timestamp_part.strip('[')
                    if ': ' in rest:
                        level, message = rest.split(': ', 1)
                        entries.append({
                            "timestamp": timestamp,
                            "level": level.strip(),
                            "message": message.strip()
                        })
                    else:
                        entries.append({
                            "timestamp": timestamp,
                            "level": "INFO",
                            "message": rest.strip()
                        })
                else:
                    entries.append({
                        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        "level": "INFO",
                        "message": line.strip()
                    })
        
        return entries if entries else [{"timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "level": "INFO", "message": "Log-Datei ist leer"}]
    except Exception as e:
        return [{"timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "level": "ERROR", "message": f"Fehler beim Lesen der Logs: {str(e)}"}]

def load_schedules() -> Dict[str, ScheduleConfig]:
    """Lade Zeitpläne aus .env-Datei"""
    try:
        # Get schedules from .env
        schedules_json = os.getenv("SCHEDULES_JSON", "")
        if not schedules_json:
            return {}
        
        # Parse JSON
        data = json.loads(schedules_json)
        
        schedules = {}
        for schedule_id, schedule_data in data.items():
            # Convert datetime strings back to datetime objects
            if schedule_data.get('last_run'):
                schedule_data['last_run'] = datetime.fromisoformat(schedule_data['last_run'])
            if schedule_data.get('next_run'):
                schedule_data['next_run'] = datetime.fromisoformat(schedule_data['next_run'])
            
            schedules[schedule_id] = ScheduleConfig(**schedule_data)
        
        logger.info(f"Loaded {len(schedules)} schedules from .env")
        return schedules
    except Exception as e:
        logger.error(f"Error loading schedules from .env: {str(e)}")
        return {}

def save_schedules(schedules: Dict[str, ScheduleConfig]) -> None:
    """Speichere Zeitpläne in .env-Datei"""
    try:
        # Convert schedules to dict
        data = {}
        for schedule_id, schedule in schedules.items():
            schedule_dict = schedule.to_dict()
            data[schedule_id] = schedule_dict
        
        # Convert to JSON string
        schedules_json = json.dumps(data, indent=2, ensure_ascii=False)
        
        # Load current .env content
        existing_lines = []
        if ENV_FILE.exists():
            with open(ENV_FILE, 'r', encoding='utf-8') as f:
                existing_lines = f.readlines()
        
        # Process lines to update/remove SCHEDULES_JSON
        processed_keys = set()
        new_lines = []
        
        for line in existing_lines:
            line_stripped = line.strip()
            if '=' in line_stripped and not line_stripped.startswith('#'):
                key = line_stripped.split('=')[0]
                if key == "SCHEDULES_JSON":
                    # Replace with new schedules
                    new_lines.append(f'SCHEDULES_JSON=\'{schedules_json}\'\n')
                    processed_keys.add(key)
                else:
                    # Keep existing key
                    new_lines.append(line)
            else:
                # Keep comments and empty lines
                new_lines.append(line)
        
        # Add schedules if not present
        if "SCHEDULES_JSON" not in processed_keys:
            new_lines.append(f'\n# Transfer Schedules - Generated automatically\n')
            new_lines.append(f'SCHEDULES_JSON=\'{schedules_json}\'\n')
        
        # Write back to .env file
        with open(ENV_FILE, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        
        logger.info(f"Schedules saved to .env: {len(data)} schedules")
    except Exception as e:
        logger.error(f"Error saving schedules to .env: {str(e)}")
        raise

def run_scheduled_transfer(schedule_config: ScheduleConfig) -> None:
    """Führe geplanten Transfer aus"""
    global current_process, transfer_status
    
    try:
        logger.info(f"Starting scheduled transfer: {schedule_config.name}")
        
        # Update transfer status BEFORE starting
        transfer_status.is_running = True
        transfer_status.start_time = datetime.now()
        transfer_status.progress = 0
        transfer_status.error_message = None
        
        # Build command arguments
        cmd = ["python", "zammad_to_qdrant.py"]
        
        if schedule_config.config and schedule_config.config.get("bm25_cache"):
            cmd.append("--use-cached-bm25")
        
        # Start process
        current_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=os.getcwd()
        )
        
        logger.info(f"Scheduled transfer started with PID: {current_process.pid}")
        
        # Update last_run
        schedule_config.last_run = datetime.now()
        
    except Exception as e:
        logger.error(f"Failed to start scheduled transfer: {str(e)}")
        transfer_status.is_running = False
        transfer_status.error_message = str(e)

def scheduler_worker():
    """Worker-Thread für Scheduler"""
    global schedule_running, scheduled_transfers
    
    while schedule_running:
        try:
            # Reload schedules from file
            with schedule_lock:
                current_schedules = load_schedules()
                scheduled_transfers.update(current_schedules)
            
            # Check all enabled schedules
            for schedule_id, schedule_config in scheduled_transfers.items():
                if not schedule_config.enabled:
                    continue
                
                # Check if it's time to run
                now = datetime.now()
                
                if schedule_config.interval == "hourly":
                    # Run every hour
                    if not schedule_config.next_run or now >= schedule_config.next_run:
                        run_scheduled_transfer(schedule_config)
                        schedule_config.next_run = now + timedelta(hours=1)
                        save_schedules(scheduled_transfers)
                
                elif schedule_config.interval == "daily" and schedule_config.time:
                    # Run daily at specific time
                    try:
                        hour, minute = map(int, schedule_config.time.split(':'))
                        target_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                        
                        # If time has passed today, schedule for tomorrow
                        if now.time() > target_time.time():
                            target_time += timedelta(days=1)
                        
                        if not schedule_config.next_run or now >= schedule_config.next_run:
                            run_scheduled_transfer(schedule_config)
                            schedule_config.next_run = target_time
                            save_schedules(scheduled_transfers)
                    except ValueError:
                        logger.error(f"Invalid time format for schedule {schedule_id}: {schedule_config.time}")
                
                elif schedule_config.interval == "weekly" and schedule_config.time and schedule_config.days:
                    # Run weekly on specific days at specific time
                    try:
                        hour, minute = map(int, schedule_config.time.split(':'))
                        today = now.weekday()  # 0=Monday, 6=Sunday
                        
                        days = [int(d) for d in schedule_config.days.split(',')]
                        if today in days:
                            target_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                            
                            # If time has passed today, schedule for next occurrence
                            if now.time() > target_time.time():
                                # Find next scheduled day
                                days_ahead = 1
                                while (today + days_ahead) % 7 not in days:
                                    days_ahead += 1
                                target_time += timedelta(days=days_ahead)
                            else:
                                target_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                            
                            if not schedule_config.next_run or now >= schedule_config.next_run:
                                run_scheduled_transfer(schedule_config)
                                schedule_config.next_run = target_time
                                save_schedules(scheduled_transfers)
                    except (ValueError, IndexError) as e:
                        logger.error(f"Invalid weekly schedule format for {schedule_id}: {e}")
            
            # Wait 60 seconds before next check
            time.sleep(60)
            
        except Exception as e:
            logger.error(f"Error in scheduler worker: {str(e)}")
            time.sleep(60)

# FastAPI App Setup
app = FastAPI(
    title="Zammad Qdrant Web Interface",
    description="Clean UI für Ticket-Transfer und Hybrid Search",
    version="1.0.0"
)

# Statische Dateien und Templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.on_event("startup")
async def init_scheduler_on_startup():
    """Load schedules from .env and start scheduler at app startup."""
    global scheduled_transfers, schedule_running, schedule_thread
    try:
        with schedule_lock:
            loaded = load_schedules()
            # Replace in-memory schedules with loaded ones
            scheduled_transfers = loaded
            count = len(scheduled_transfers)
        logger.info(f"Startup: loaded {count} schedules from .env")

        # Auto-start scheduler if schedules exist
        if count > 0 and not schedule_running:
            schedule_running = True
            schedule_thread = threading.Thread(target=scheduler_worker, daemon=True)
            schedule_thread.start()
            logger.info(f"Startup: scheduler auto-started with {count} schedules")
    except Exception as e:
        logger.error(f"Startup: failed to initialize scheduler: {str(e)}")

# Routes
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Hauptdashboard mit Ticket-Transfer und Such-Interface"""
    ui_lang = os.getenv("UI_LANGUAGE", "DE").upper()
    return templates.TemplateResponse("dashboard.html", {"request": request, "ui_language": ui_lang})

@app.get("/settings", response_class=HTMLResponse)
async def settings(request: Request):
    """Einstellungsseite"""
    ui_lang = os.getenv("UI_LANGUAGE", "DE").upper()
    return templates.TemplateResponse("settings.html", {"request": request, "ui_language": ui_lang})

@app.get("/ai-settings", response_class=HTMLResponse)
async def ai_settings(request: Request):
    """KI-Einstellungsseite"""
    ui_lang = os.getenv("UI_LANGUAGE", "DE").upper()
    return templates.TemplateResponse("ai_settings.html", {"request": request, "ui_language": ui_lang})

# API Routes for Dashboard
@app.get("/api/transfer-status")
async def get_transfer_status():
    """Get current transfer status"""
    global transfer_status, current_process
    
    # Check if process is still running
    if current_process:
        poll_result = current_process.poll()
        if poll_result is None:
            transfer_status.is_running = True
            logger.debug(f"Process {current_process.pid} is still running")
        else:
            # Process has finished
            transfer_status.is_running = False
            return_code = current_process.returncode
            logger.info(f"Transfer process finished with return code: {return_code}")
            current_process = None
    else:
        logger.debug("No current process")
        transfer_status.is_running = False
    
    status_data = {
        "is_running": transfer_status.is_running,
        "progress": transfer_status.progress,
        "current_ticket": transfer_status.current_ticket,
        "total_tickets": transfer_status.total_tickets,
        "processed_tickets": transfer_status.processed_tickets,
        "start_time": transfer_status.start_time.isoformat() if transfer_status.start_time else None,
        "error_message": transfer_status.error_message,
        "process_running": current_process is not None and current_process.poll() is None,
        "process_pid": current_process.pid if current_process else None
    }
    
    logger.info(f"Transfer status: {status_data}")
    return status_data

@app.post("/api/transfer-stop")
async def stop_transfer():
    """Stop the transfer process"""
    global current_process, transfer_status
    
    # Check if process is actually running
    process_running = current_process and current_process.poll() is None
    if not process_running:
        raise HTTPException(status_code=400, detail="Kein Transfer läuft")
    
    try:
        if current_process and current_process.poll() is None:
            logger.info(f"Stopping transfer process (PID: {current_process.pid})")
            # Terminate the process
            current_process.terminate()
            
            # Wait for graceful termination (5 seconds)
            try:
                current_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if still running
                logger.warning(f"Force killing process {current_process.pid}")
                current_process.kill()
                current_process.wait()
            
            logger.info(f"Transfer process stopped (PID: {current_process.pid})")
        
        # Reset status
        transfer_status.is_running = False
        transfer_status.error_message = None
        current_process = None
        
        logger.info("Transfer status reset to not running")
        return {"status": "stopped", "message": "Transfer gestoppt"}
    except Exception as e:
        logger.error(f"Failed to stop transfer: {str(e)}")
        transfer_status.is_running = False
        raise HTTPException(status_code=500, detail=f"Fehler beim Stoppen: {str(e)}")

@app.post("/api/transfer-start")
async def start_transfer(config: Dict[str, Any]):
    """Start the transfer process"""
    global current_process, transfer_status
    
    # Check if process is actually running
    process_running = current_process and current_process.poll() is None
    if process_running:
        raise HTTPException(status_code=400, detail="Transfer läuft bereits")
    
    try:
        # Stop any existing process first
        if current_process:
            if current_process.poll() is None:
                current_process.terminate()
                try:
                    current_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    current_process.kill()
                    current_process.wait()
            current_process = None
        
        # Update transfer status BEFORE starting
        transfer_status.is_running = True
        transfer_status.start_time = datetime.now()
        transfer_status.progress = 0
        transfer_status.error_message = None
        
        # Build command arguments
        cmd = ["python", "zammad_to_qdrant.py"]
        
        if config.get("bm25_cache"):
            cmd.append("--use-cached-bm25")
        
        # Start process without capturing output to prevent blocking
        current_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,  # Don't capture output
            stderr=subprocess.DEVNULL,  # Don't capture errors
            cwd=os.getcwd()  # Ensure correct working directory
        )
        
        logger.info(f"Transfer process started with PID: {current_process.pid}")
        
        return {"status": "started", "message": f"Transfer gestartet (PID: {current_process.pid})"}
    except Exception as e:
        transfer_status.is_running = False
        transfer_status.error_message = str(e)
        logger.error(f"Failed to start transfer: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Fehler beim Starten: {str(e)}")

@app.get("/api/live-log")
async def get_live_log():
    """Get live log entries"""
    return {"entries": read_live_log_entries()}

@app.get("/api/config")
async def get_config():
    """Get current configuration from .env"""
    return load_env_config().__dict__

@app.post("/api/config")
async def save_config(config: ConfigData):
    """Save configuration to .env"""
    try:
        save_env_config(config)
        # Force reload environment variables for this process
        load_dotenv(override=True)
        logger.info("Configuration saved and environment reloaded")
        return {"status": "saved", "message": "Konfiguration gespeichert"}
    except Exception as e:
        logger.error(f"Error saving config: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Fehler beim Speichern: {str(e)}")

# UI Language endpoints
class LanguageUpdate(BaseModel):
    language: str

@app.get("/api/ui-language")
async def get_ui_language():
    try:
        lang = os.getenv("UI_LANGUAGE", "DE").upper()
        if lang not in ("DE", "EN"):
            lang = "DE"
        return {"language": lang}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ui-language")
async def set_ui_language(update: LanguageUpdate):
    try:
        lang = (update.language or "DE").upper()
        if lang not in ("DE", "EN"):
            raise HTTPException(status_code=400, detail="Invalid language")
        set_env_key("UI_LANGUAGE", lang)
        return {"status": "saved", "language": lang}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting UI language: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Fehler beim Speichern: {str(e)}")

# API Routes for Settings
@app.get("/api/zammad-status")
async def zammad_status():
    """Check Zammad connection status"""
    return check_zammad_connection()

@app.get("/api/qdrant-status")
async def qdrant_status():
    """Check Qdrant connection status"""
    return check_qdrant_connection()

@app.get("/api/qdrant-collections")
async def qdrant_collections():
    """Get Qdrant collections count"""
    return get_qdrant_collections()

@app.get("/api/bm25-stats")
async def bm25_stats():
    """Get BM25 statistics if available"""
    try:
        stats_file = Path("bm25_stats.json")
        if stats_file.exists():
            with open(stats_file, 'r', encoding='utf-8') as f:
                stats = json.load(f)
            return {
                "available": True,
                "vocabulary_size": len(stats.get("term_df", {})),
                "documents": stats.get("N_docs", 0),
                "avg_doc_length": stats.get("avgdl", 0.0),
                "last_rebuild": stats.get("last_rebuild", "Unknown")
            }
        else:
            return {"available": False, "message": "BM25-Statistiken nicht verfügbar"}
    except Exception as e:
        return {"available": False, "error": str(e)}

# KI-Einstellungen API Routes
@app.get("/api/ai-config")
async def get_ai_config():
    """Get current AI configuration from .env"""
    return load_env_config().__dict__

@app.post("/api/ai-config")
async def save_ai_config(config: ConfigData):
    """Save AI configuration to .env"""
    try:
        save_env_config(config)
        # Force reload environment variables for this process
        load_dotenv(override=True)
        logger.info("AI Configuration saved and environment reloaded")
        return {"status": "saved", "message": "KI-Konfiguration gespeichert"}
    except Exception as e:
        logger.error(f"Error saving AI config: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Fehler beim Speichern: {str(e)}")

@app.get("/api/ollama-models")
async def get_ollama_models():
    """Get available Ollama models"""
    try:
        config = load_env_config()
        ollama_url = config.ollama_url.rstrip('/') + "/api/tags"
        
        response = requests.get(ollama_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            models = []
            for model in data.get("models", []):
                models.append({
                    "name": model.get("name", ""),
                    "size": model.get("size", 0),
                    "modified_at": model.get("modified_at", "")
                })
            return {"models": models, "status": "connected"}
        else:
            return {"models": [], "status": "error", "message": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"models": [], "status": "error", "message": str(e)}

@app.post("/api/ai-test-connection")
async def test_ai_connection():
    """Test Ollama connection"""
    try:
        config = load_env_config()
        base_url = config.ollama_url.rstrip('/')
        
        # Schritt 1: Teste Ollama-Tags API
        tags_url = f"{base_url}/api/tags"
        logger.info(f"Testing Ollama tags API: {tags_url}")
        
        tags_response = requests.get(tags_url, timeout=10)
        if tags_response.status_code != 200:
            return {"status": "error", "message": f"Ollama Tags API nicht erreichbar (HTTP {tags_response.status_code})"}
        
        # Schritt 2: Prüfe ob Modell verfügbar ist
        tags_data = tags_response.json()
        available_models = [model.get("name", "") for model in tags_data.get("models", [])]
        
        if config.ollama_model not in available_models:
            return {"status": "error", "message": f"Modell '{config.ollama_model}' nicht verfügbar. Verfügbare Modelle: {', '.join(available_models)}"}
        
        # Schritt 3: Teste einfache Generierung
        generate_url = f"{base_url}/api/generate"
        payload = {
            "model": config.ollama_model,
            "prompt": "Hallo",
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 20
            }
        }
        
        logger.info(f"Testing Ollama generate API with model: {config.ollama_model}")
        
        response = requests.post(generate_url, json=payload, timeout=300)
        logger.info(f"Ollama generate response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            response_preview = result.get("response", "").strip()
            return {
                "status": "connected",
                "message": "Ollama-Verbindung erfolgreich",
                "available_models": available_models,
                "current_model": config.ollama_model,
                "response_preview": response_preview[:100] if response_preview else "Keine Antwort"
            }
        else:
            error_text = response.text
            logger.error(f"Ollama generate error response: {error_text}")
            return {"status": "error", "message": f"Generierung fehlgeschlagen (HTTP {response.status_code}): {error_text[:200]}"}
            
    except requests.exceptions.ConnectionError:
        error_msg = "Verbindung zu Ollama fehlgeschlagen. Stellen Sie sicher, dass Ollama läuft und die URL korrekt ist."
        logger.error(error_msg)
        return {"status": "error", "message": error_msg}
    except requests.exceptions.Timeout:
        error_msg = "Timeout beim Verbinden zu Ollama. Überprüfen Sie die URL und die Erreichbarkeit."
        logger.error(error_msg)
        return {"status": "error", "message": error_msg}
    except Exception as e:
        error_msg = f"Unerwarteter Fehler: {str(e)}"
        logger.error(error_msg)
        return {"status": "error", "message": error_msg}

# KI-Service Control API Routes
@app.get("/api/ai-service-status")
async def get_ai_service_status():
    """Get KI service status"""
    try:
        import zammad_ai
        status = zammad_ai.get_ai_service_status()
        return status
    except Exception as e:
        logger.error(f"Error getting AI service status: {str(e)}")
        return {
            "running": False,
            "error": str(e),
            "thread_alive": False,
            "processed_tickets_count": 0
        }

@app.post("/api/ai-service-start")
async def start_ai_service():
    """Start the KI service"""
    try:
        import zammad_ai
        result = zammad_ai.start_ai_service()
        if result:
            return {"status": "started", "message": "KI-Service erfolgreich gestartet"}
        else:
            return {"status": "error", "message": "KI-Service konnte nicht gestartet werden"}
    except Exception as e:
        logger.error(f"Error starting AI service: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.post("/api/ai-service-stop")
async def stop_ai_service():
    """Stop the KI service"""
    try:
        import zammad_ai
        zammad_ai.stop_ai_service()
        return {"status": "stopped", "message": "KI-Service erfolgreich gestoppt"}
    except Exception as e:
        logger.error(f"Error stopping AI service: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.post("/api/bm25-cache-clear")
async def clear_bm25_cache():
    """Clear BM25 cache by deleting bm25_stats.json file"""
    try:
        stats_file = Path("bm25_stats.json")
        if stats_file.exists():
            stats_file.unlink()
            logger.info("BM25 cache file deleted successfully")
            return {"status": "success", "message": "BM25-Cache erfolgreich gelöscht"}
        else:
            return {"status": "success", "message": "BM25-Cache-Datei existierte nicht"}
    except Exception as e:
        logger.error(f"Failed to delete BM25 cache: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Fehler beim Löschen des BM25-Cache: {str(e)}")

@app.post("/api/collection-reset")
async def reset_collection():
    """Reset Qdrant collection - delete existing and create new one"""
    try:
        config = load_env_config()
        if not config.qdrant_url:
            raise HTTPException(status_code=400, detail="Qdrant URL fehlt")
        
        # Import Qdrant client here to avoid circular imports
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qmodels
        
        # Create client
        qc = QdrantClient(url=config.qdrant_url, api_key=config.qdrant_api_key)
        
        # Get collection name from environment
        collection_name = os.getenv("COLLECTION_NAME", "zammad_tickets")
        
        # Delete existing collection if it exists
        try:
            qc.get_collection(collection_name)
            logger.info(f"Deleting existing collection: {collection_name}")
            qc.delete_collection(collection_name)
            logger.info(f"Collection {collection_name} deleted successfully")
        except Exception:
            # Collection doesn't exist, that's fine
            logger.info(f"Collection {collection_name} doesn't exist, no need to delete")
        
        # Create new collection with same configuration as in zammad_to_qdrant.py
        # Use known dimension for the default model
        try:
            # Get embedding model from environment
            embedding_model = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
            
            # For the default multilingual-e5-large model, the dimension is 1024
            # For other models, we use a default dimension and let zammad_to_qdrant.py handle it
            if "multilingual-e5-large" in embedding_model:
                dim = 1024
            elif "multilingual-e5-small" in embedding_model:
                dim = 384
            elif "all-MiniLM-L6-v2" in embedding_model:
                dim = 384
            else:
                # Default dimension for unknown models
                dim = 384
                logger.warning(f"Unknown embedding model {embedding_model}, using default dimension {dim}")
            
            # Create collection
            logger.info(f"Creating new collection: {collection_name} with dimension {dim}")
            qc.create_collection(
                collection_name=collection_name,
                vectors_config={"dense": qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE)},
                sparse_vectors_config={"sparse": qmodels.SparseVectorParams()},
            )
            logger.info(f"Collection {collection_name} created successfully")
            
            return {
                "status": "success",
                "message": f"Collection '{collection_name}' erfolgreich zurückgesetzt",
                "collection_name": collection_name,
                "dimension": dim
            }
            
        except Exception as e:
            logger.error(f"Failed to create collection: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Fehler beim Erstellen der Collection: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reset collection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Fehler beim Zurücksetzen der Collection: {str(e)}")

# MCP Server Control
@app.get("/api/mcp-status")
async def mcp_status():
    """Get MCP server status and information"""
    global mcp_server_process
    
    try:
        # Get local IP address
        local_ip = get_local_ip()
        
        # Check if process is still running
        process_running = mcp_server_process and mcp_server_process.poll() is None
        
        if process_running:
            # Try to get server status via HTTP
            try:
                response = requests.get("http://127.0.0.1:8083/health", timeout=2)
                server_status = "running" if response.status_code == 200 else "error"
                server_port = 8083
                server_url_local = "http://127.0.0.1:8083"
                server_url_local_ip = f"http://{local_ip}:8083"
            except:
                # Server might not be responding but still starting up
                server_status = "starting"
                server_port = 8083
                server_url_local = "http://127.0.0.1:8083"
                server_url_local_ip = f"http://{local_ip}:8083"
        else:
            server_status = "stopped"
            server_port = None
            server_url_local = None
            server_url_local_ip = None
        
        return {
            "status": "running" if process_running else "stopped",
            "server_status": server_status,
            "port": server_port,
            "url_local": server_url_local,
            "url_local_ip": server_url_local_ip,
            "local_ip": local_ip,
            "process_running": process_running,
            "process_pid": mcp_server_process.pid if mcp_server_process else None,
            "last_action": "Unknown" if not process_running else "Server läuft"
        }
    except Exception as e:
        return {
            "status": "error",
            "server_status": "error",
            "error": str(e),
            "port": None,
            "url_local": None,
            "url_local_ip": None,
            "local_ip": get_local_ip(),
            "process_running": False,
            "process_pid": None
        }

@app.post("/api/mcp-start")
async def mcp_start():
    """Start the MCP server"""
    global mcp_server_process
    
    # Check if process is already running
    if mcp_server_process and mcp_server_process.poll() is None:
        raise HTTPException(status_code=400, detail="MCP Server läuft bereits")
    
    try:
        # Stop any existing process first
        if mcp_server_process:
            if mcp_server_process.poll() is None:
                mcp_server_process.terminate()
                try:
                    mcp_server_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    mcp_server_process.kill()
                    mcp_server_process.wait()
            mcp_server_process = None
        
        # Start MCP server in HTTP mode
        mcp_server_process = subprocess.Popen(
            ["python", "rerank_search_mcp.py", "--http"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=os.getcwd()
        )
        
        logger.info(f"MCP Server started with PID: {mcp_server_process.pid}")
        
        # Wait a moment and check if it's running
        import time
        time.sleep(2)
        
        return {
            "status": "started",
            "message": f"MCP Server gestartet (PID: {mcp_server_process.pid})",
            "port": 8083,
            "url": "http://127.0.0.1:8083"
        }
    except Exception as e:
        logger.error(f"Failed to start MCP server: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Fehler beim Starten: {str(e)}")

@app.post("/api/mcp-stop")
async def mcp_stop():
    """Stop the MCP server"""
    global mcp_server_process
    
    if not mcp_server_process or mcp_server_process.poll() is not None:
        raise HTTPException(status_code=400, detail="MCP Server läuft nicht")
    
    try:
        logger.info(f"Stopping MCP server (PID: {mcp_server_process.pid})")
        
        # Terminate the process
        mcp_server_process.terminate()
        
        # Wait for graceful termination (5 seconds)
        try:
            mcp_server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # Force kill if still running
            logger.warning(f"Force killing MCP server {mcp_server_process.pid}")
            mcp_server_process.kill()
            mcp_server_process.wait()
        
        mcp_server_process = None
        logger.info("MCP Server stopped")
        
        return {"status": "stopped", "message": "MCP Server gestoppt"}
    except Exception as e:
        logger.error(f"Failed to stop MCP server: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Fehler beim Stoppen: {str(e)}")

@app.get("/api/search")
async def proxy_search(query: str, top_k: int = 100, top_tickets: int = 5):
    """Proxy endpoint for MCP search to avoid CORS issues"""
    global mcp_server_process
    
    # Check if MCP server is running
    if not mcp_server_process or mcp_server_process.poll() is not None:
        raise HTTPException(status_code=400, detail="MCP Server läuft nicht")
    
    try:
        # Get local IP for constructing the URL
        local_ip = get_local_ip()
        mcp_url = f"http://127.0.0.1:8083"  # Always use localhost for internal communication
        
        # Forward the request to MCP server
        search_url = f"{mcp_url}/search"
        params = {
            "query": query,
            "top_k": top_k,
            "top_tickets": top_tickets
        }
        
        logger.info(f"Proxying search request to {search_url} with params {params}")
        
        # Make request to MCP server
        response = requests.get(search_url, params=params, timeout=30)
        
        if response.status_code == 200:
            # Return the response from MCP server
            return response.json()
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"MCP Server Error: {response.text}"
            )
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error proxying search request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Verbindung zum MCP Server fehlgeschlagen: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error in search proxy: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unerwarteter Fehler: {str(e)}")

# Schedule API Routes
@app.get("/api/schedules")
async def get_schedules():
    """Get all schedules"""
    global scheduled_transfers
    try:
        with schedule_lock:
            # Lazy-load from .env if empty (first request edge case)
            if not scheduled_transfers:
                scheduled_transfers = load_schedules()
            schedules = list(scheduled_transfers.values())
        return {
            "schedules": [schedule.to_dict() for schedule in schedules],
            "count": len(schedules)
        }
    except Exception as e:
        logger.error(f"Error getting schedules: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Fehler beim Abrufen der Zeitpläne: {str(e)}")

@app.post("/api/schedules")
async def create_schedule(schedule_data: Dict[str, Any]):
    """Create a new schedule"""
    global scheduled_transfers
    try:
        # Generate unique ID
        import uuid
        schedule_id = str(uuid.uuid4())[:8]
        
        # Create schedule config
        schedule_config = ScheduleConfig(
            id=schedule_id,
            name=schedule_data.get("name", "Unbenannter Zeitplan"),
            interval=schedule_data.get("interval", "daily"),
            time=schedule_data.get("time"),
            days=schedule_data.get("days"),
            enabled=schedule_data.get("enabled", True),
            config=schedule_data.get("config", {})
        )
        
        # Calculate next run time
        now = datetime.now()
        if schedule_config.interval == "hourly":
            schedule_config.next_run = now + timedelta(hours=1)
        elif schedule_config.interval == "daily" and schedule_config.time:
            hour, minute = map(int, schedule_config.time.split(':'))
            target_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if now.time() > target_time.time():
                target_time += timedelta(days=1)
            schedule_config.next_run = target_time
        elif schedule_config.interval == "weekly" and schedule_config.time and schedule_config.days:
            hour, minute = map(int, schedule_config.time.split(':'))
            target_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if now.time() > target_time.time():
                days = [int(d) for d in schedule_config.days.split(',')]
                days_ahead = 1
                while (now.weekday() + days_ahead) % 7 not in days:
                    days_ahead += 1
                target_time += timedelta(days=days_ahead)
            schedule_config.next_run = target_time
        
        # Save schedule
        with schedule_lock:
            scheduled_transfers[schedule_id] = schedule_config
            save_schedules(scheduled_transfers)
        
        # Auto-start scheduler if not running
        global schedule_running, schedule_thread
        if not schedule_running:
            schedule_running = True
            schedule_thread = threading.Thread(target=scheduler_worker, daemon=True)
            schedule_thread.start()
            scheduler_message = " (Scheduler automatisch gestartet)"
        else:
            scheduler_message = ""
        
        logger.info(f"Schedule created: {schedule_config.name} (ID: {schedule_id}){scheduler_message}")
        return {
            "status": "created",
            "schedule": schedule_config.to_dict(),
            "message": f"Zeitplan '{schedule_config.name}' erstellt{scheduler_message}"
        }
    except Exception as e:
        logger.error(f"Error creating schedule: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Fehler beim Erstellen des Zeitplans: {str(e)}")

@app.put("/api/schedules/{schedule_id}")
async def update_schedule(schedule_id: str, schedule_data: Dict[str, Any]):
    """Update an existing schedule"""
    global scheduled_transfers
    try:
        with schedule_lock:
            if schedule_id not in scheduled_transfers:
                raise HTTPException(status_code=404, detail="Zeitplan nicht gefunden")
            
            schedule_config = scheduled_transfers[schedule_id]
            
            # Update fields
            if "name" in schedule_data:
                schedule_config.name = schedule_data["name"]
            if "interval" in schedule_data:
                schedule_config.interval = schedule_data["interval"]
            if "time" in schedule_data:
                schedule_config.time = schedule_data["time"]
            if "days" in schedule_data:
                schedule_config.days = schedule_data["days"]
            if "enabled" in schedule_data:
                schedule_config.enabled = schedule_data["enabled"]
            if "config" in schedule_data:
                schedule_config.config = schedule_data["config"]
            
            # Recalculate next run if timing changed
            if any(field in schedule_data for field in ["interval", "time", "days"]):
                now = datetime.now()
                if schedule_config.interval == "hourly":
                    schedule_config.next_run = now + timedelta(hours=1)
                elif schedule_config.interval == "daily" and schedule_config.time:
                    hour, minute = map(int, schedule_config.time.split(':'))
                    target_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                    if now.time() > target_time.time():
                        target_time += timedelta(days=1)
                    schedule_config.next_run = target_time
                elif schedule_config.interval == "weekly" and schedule_config.time and schedule_config.days:
                    hour, minute = map(int, schedule_config.time.split(':'))
                    target_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                    if now.time() > target_time.time():
                        days = [int(d) for d in schedule_config.days.split(',')]
                        days_ahead = 1
                        while (now.weekday() + days_ahead) % 7 not in days:
                            days_ahead += 1
                        target_time += timedelta(days=days_ahead)
                    schedule_config.next_run = target_time
            
            save_schedules(scheduled_transfers)
        
        logger.info(f"Schedule updated: {schedule_config.name} (ID: {schedule_id})")
        return {
            "status": "updated",
            "schedule": schedule_config.to_dict(),
            "message": f"Zeitplan '{schedule_config.name}' aktualisiert"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating schedule: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Fehler beim Aktualisieren des Zeitplans: {str(e)}")

@app.delete("/api/schedules/{schedule_id}")
async def delete_schedule(schedule_id: str):
    """Delete a schedule"""
    global scheduled_transfers
    try:
        with schedule_lock:
            if schedule_id not in scheduled_transfers:
                raise HTTPException(status_code=404, detail="Zeitplan nicht gefunden")
            
            schedule_config = scheduled_transfers.pop(schedule_id)
            save_schedules(scheduled_transfers)
        
        logger.info(f"Schedule deleted: {schedule_config.name} (ID: {schedule_id})")
        return {
            "status": "deleted",
            "message": f"Zeitplan '{schedule_config.name}' gelöscht"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting schedule: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Fehler beim Löschen des Zeitplans: {str(e)}")

@app.post("/api/schedules/{schedule_id}/toggle")
async def toggle_schedule(schedule_id: str):
    """Enable/disable a schedule"""
    global scheduled_transfers
    try:
        with schedule_lock:
            if schedule_id not in scheduled_transfers:
                raise HTTPException(status_code=404, detail="Zeitplan nicht gefunden")
            
            schedule_config = scheduled_transfers[schedule_id]
            schedule_config.enabled = not schedule_config.enabled
            save_schedules(scheduled_transfers)
        
        status_text = "aktiviert" if schedule_config.enabled else "deaktiviert"
        logger.info(f"Schedule {status_text}: {schedule_config.name} (ID: {schedule_id})")
        return {
            "status": "toggled",
            "enabled": schedule_config.enabled,
            "message": f"Zeitplan '{schedule_config.name}' {status_text}"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error toggling schedule: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Fehler beim Umschalten des Zeitplans: {str(e)}")

@app.post("/api/schedules/start")
async def start_scheduler():
    """Start the scheduler"""
    global schedule_thread, schedule_running
    try:
        if schedule_running and schedule_thread and schedule_thread.is_alive():
            raise HTTPException(status_code=400, detail="Scheduler läuft bereits")
        
        schedule_running = True
        schedule_thread = threading.Thread(target=scheduler_worker, daemon=True)
        schedule_thread.start()
        
        logger.info("Scheduler started")
        return {
            "status": "started",
            "message": "Scheduler gestartet"
        }
    except HTTPException:
        raise
    except Exception as e:
        schedule_running = False
        logger.error(f"Error starting scheduler: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Fehler beim Starten des Schedulers: {str(e)}")

@app.post("/api/schedules/stop")
async def stop_scheduler():
    """Stop the scheduler"""
    global schedule_thread, schedule_running
    try:
        if not schedule_running:
            raise HTTPException(status_code=400, detail="Scheduler läuft nicht")
        
        schedule_running = False
        if schedule_thread and schedule_thread.is_alive():
            schedule_thread.join(timeout=5)
        
        logger.info("Scheduler stopped")
        return {
            "status": "stopped",
            "message": "Scheduler gestoppt"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping scheduler: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Fehler beim Stoppen des Schedulers: {str(e)}")

@app.get("/api/schedules/status")
async def get_scheduler_status():
    """Get scheduler status"""
    global schedule_thread, schedule_running
    try:
        is_alive = schedule_thread and schedule_thread.is_alive() if schedule_thread else False
        return {
            "running": schedule_running and is_alive,
            "thread_alive": is_alive,
            "active_schedules": len([s for s in scheduled_transfers.values() if s.enabled])
        }
    except Exception as e:
        logger.error(f"Error getting scheduler status: {str(e)}")
        return {
            "running": False,
            "thread_alive": False,
            "active_schedules": 0
        }

@app.get("/health")
async def health_check():
    """Gesundheitscheck für Monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "mode": "integrated"
    }

# Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "status_code": 500,
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    # Initialize scheduler
    try:
        scheduled_transfers = load_schedules()
        logger.info(f"Loaded {len(scheduled_transfers)} schedules from .env")
        
        # Auto-start scheduler if schedules exist
        if scheduled_transfers and not schedule_running:
            schedule_running = True
            schedule_thread = threading.Thread(target=scheduler_worker, daemon=True)
            schedule_thread.start()
            logger.info(f"Scheduler auto-started with {len(scheduled_transfers)} schedules")
        elif not scheduled_transfers:
            logger.info("No schedules found in .env - scheduler will start when schedules are created")
        
    except Exception as e:
        logger.error(f"Failed to initialize scheduler: {str(e)}")
    
    print("Starting Zammad Qdrant Web Interface (Clean UI)...")
    print("UI available at: http://localhost:8000")
    print("Settings available at: http://localhost:8000/settings")
    print("API Documentation: http://localhost:8000/docs")
    print("-" * 50)
    
    uvicorn.run(
        "demo_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
