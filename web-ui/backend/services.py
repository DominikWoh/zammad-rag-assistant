import requests
import psutil
import subprocess
import os
import sys
import logging
from datetime import datetime
from qdrant_client import QdrantClient
from config import settings

logger = logging.getLogger("webui.services")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.setLevel(logging.INFO)

def _detect_executable(candidates):
    for p in candidates:
        try:
            if subprocess.run(["/usr/bin/env", "bash", "-lc", f"test -x {p}"], capture_output=True).returncode == 0:
                return p
        except Exception:
            continue
    return None

class ServiceManager:
    def __init__(self):
        self.qdrant_client = None

    def check_qdrant_status(self):
        try:
            if not self.qdrant_client:
                self.qdrant_client = QdrantClient(
                    settings.QDRANT_URL,
                    api_key=settings.QDRANT_API_KEY
                )
            
            collection_info = self.qdrant_client.get_collection(settings.COLLECTION_NAME)
            points_count = collection_info.points_count
            
            return {
                "status": "online",
                "points_count": points_count,
                "url": settings.QDRANT_URL
            }
        except Exception as e:
            return {
                "status": "offline",
                "error": str(e),
                "url": settings.QDRANT_URL
            }

    def check_ollama_status(self):
        try:
            # Korrekter API-Endpoint für Ollama
            api_url = settings.OLLAMA_URL.rstrip('/') + '/api/tags'
            response = requests.get(api_url, timeout=5)
            
            if response.status_code == 200:
                models = response.json().get('models', [])
                current_model = settings.OLLAMA_MODEL
                
                return {
                    "status": "online",
                    "current_model": current_model,
                    "available_models": [m['name'] for m in models],
                    "url": settings.OLLAMA_URL
                }
            else:
                return {"status": "offline", "error": f"API nicht erreichbar (HTTP {response.status_code})"}
                
        except Exception as e:
            return {"status": "offline", "error": str(e)}

    def check_zammad_status(self):
        try:
            headers = {"Authorization": f"Token token={settings.ZAMMAD_TOKEN}"}
            response = requests.get(
                f"{settings.ZAMMAD_URL}/api/v1/users/me",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                user_info = response.json()
                return {
                    "status": "online",
                    "user": user_info.get('login', 'Unbekannt'),
                    "url": settings.ZAMMAD_URL
                }
            else:
                return {"status": "offline", "error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            return {"status": "offline", "error": str(e)}

    def get_system_stats(self):
        try:
            return {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": str(e)}

    def get_recent_activities(self):
        try:
            if not self.qdrant_client:
                self.qdrant_client = QdrantClient(
                    settings.QDRANT_URL,
                    api_key=settings.QDRANT_API_KEY
                )
            
            result, _ = self.qdrant_client.scroll(
                collection_name=settings.COLLECTION_NAME,
                with_payload=True,
                limit=500
            )
            
            tickets = []
            for point in result:
                payload = point.payload
                ticket_id = payload.get("ticket_id", str(point.id))
                title = payload.get("kurzbeschreibung") or payload.get("title") or "Kein Titel"
                processed_at = payload.get("processed_at") or payload.get("erstelldatum") or "1970-01-01"
                status = payload.get("status", "unknown")
                
                tickets.append({
                    "ticket_id": ticket_id,
                    "title": title,
                    "processed_at": processed_at,
                    "status": status
                })
            
            tickets.sort(key=lambda x: int(x.get("ticket_id", 0)), reverse=True)
            return tickets[:5]
            
        except Exception as e:
            return {"error": str(e)}

    def check_systemd_service_status(self, service_name):
        """Prüft den Status eines systemd-Dienstes mit bevorzugtem sudo -n + absolutem systemctl und detailliertem Logging."""
        try:
            unit = f"{service_name}.service"

            # Erkenne systemctl und sudo robust
            systemctl_path = _detect_executable(["/usr/bin/systemctl", "/bin/systemctl"]) or "systemctl"
            sudo_path = _detect_executable(["/usr/bin/sudo", "/bin/sudo"])
            has_sudo = bool(sudo_path)

            env_path = {"PATH": "/usr/bin:/bin"}

            attempts = []

            def _log_try(cmd):
                try:
                    logger.info(f"status try: cmd={' '.join(cmd)}")
                except Exception:
                    pass

            # 1) sudo -n + absoluter Pfad
            if has_sudo and isinstance(systemctl_path, str) and systemctl_path.startswith("/"):
                cmd1 = [sudo_path, "-n", systemctl_path, "is-active", unit]
                attempts.append(cmd1); _log_try(cmd1)
                # 1b) Alternativer absoluter Pfad
                alt_path = "/bin/systemctl" if systemctl_path == "/usr/bin/systemctl" else "/usr/bin/systemctl"
                if _detect_executable([alt_path]):
                    cmd1b = [sudo_path, "-n", alt_path, "is-active", unit]
                    attempts.append(cmd1b); _log_try(cmd1b)

            # 2) sudo -n + unqualifiziert (falls kein absoluter Pfad verfügbar)
            if has_sudo and systemctl_path == "systemctl":
                cmd2 = [sudo_path, "-n", "systemctl", "is-active", unit]
                attempts.append(cmd2); _log_try(cmd2)

            # 3) ohne sudo, absoluter Pfad (Diagnose-Fallback)
            if isinstance(systemctl_path, str) and systemctl_path.startswith("/"):
                cmd3 = [systemctl_path, "is-active", unit]
                attempts.append(cmd3); _log_try(cmd3)

            # 4) ohne sudo, unqualifiziert (letzter Fallback)
            if systemctl_path == "systemctl":
                cmd4 = ["systemctl", "is-active", unit]
                attempts.append(cmd4); _log_try(cmd4)

            last = None
            chosen = None
            for cmd in attempts:
                chosen = cmd
                last = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env_path)
                state = (last.stdout or "").strip()
                if last.returncode == 0 and state in ("active", "inactive", "failed", "activating", "deactivating"):
                    break

            cmd_str = " ".join(chosen) if chosen else "n/a"
            try:
                out = (last.stdout or "").strip() if last else ""
                err = (last.stderr or "").strip() if last else ""
                logger.info(f"status selected: cmd={cmd_str}, rc={(last.returncode if last else -1)}, out={out[:200]}, err={err[:200]}")
            except Exception:
                pass

            if last and last.returncode == 0 and (last.stdout or "").strip() == "active":
                return {"status": "online"}
            else:
                return {"status": "offline"}
        except FileNotFoundError:
            return {"status": "unknown", "message": "systemctl nicht gefunden"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def control_service(self, service_name, action):
        """
        Steuert systemd-Services robust:
        - Akzeptiert service_name mit oder ohne '.service'
        - Erlaubt nur Whitelist-Services (Sicherheitsaspekt)
        - Bevorzugt sudo -n + absoluten systemctl-Pfad (/usr/bin, dann /bin)
        - Fällt kontrolliert zurück und gibt den exakten cmd zurück
        """
        try:
            # Whitelist zulässiger Services
            allowed = {"zammad_rag_poller"}
            # Normalisieren: '.service' entfernen, wenn vorhanden
            raw = (service_name or "").strip()
            normalized = raw[:-8] if raw.endswith(".service") else raw

            if normalized not in allowed:
                return {"success": False, "message": f"Unbekannter Service: {service_name}"}

            unit = f"{normalized}.service"

            if action not in ["start", "stop", "restart", "status"]:
                return {"success": False, "message": f"Unbekannte Aktion: {action}"}

            # Pfade robust bestimmen (bevorzugt /usr/bin)
            systemctl_candidates = ["/usr/bin/systemctl", "/bin/systemctl"]
            systemctl_path = None
            for p in systemctl_candidates:
                if subprocess.run(["/usr/bin/env", "bash", "-lc", f"test -x {p}"], capture_output=True).returncode == 0:
                    systemctl_path = p
                    break
            if systemctl_path is None:
                # Letzter Versuch ohne Pfad (nicht empfohlen, aber als Fallback)
                systemctl_path = "systemctl"

            sudo_path = "/usr/bin/sudo"
            has_sudo = subprocess.run(["/usr/bin/env", "bash", "-lc", f"test -x {sudo_path}"], capture_output=True).returncode == 0

            env_path = {"PATH": "/usr/bin:/bin"}

            # Erzwinge /usr/bin oder /bin falls vorhanden
            if subprocess.run(["/usr/bin/env", "bash", "-lc", "test -x /usr/bin/systemctl"], capture_output=True).returncode == 0:
                systemctl_path = "/usr/bin/systemctl"
            elif subprocess.run(["/usr/bin/env", "bash", "-lc", "test -x /bin/systemctl"], capture_output=True).returncode == 0:
                systemctl_path = "/bin/systemctl"

            cwd = os.getcwd()
            try:
                logger.info(f"control_service enter: unit={unit}, action={action}, sysctl_pref={systemctl_path}, has_sudo={has_sudo}, __file__={__file__}, cwd={cwd}, sys.path0={sys.path[0] if sys.path else ''}")
            except Exception:
                pass

            # Basiskommando (status nutzt is-active)
            if action == "status":
                base_cmd = [systemctl_path, "is-active", unit]
            else:
                base_cmd = [systemctl_path, action, unit]

            attempts = []

            def _log_try(cmd):
                try:
                    logger.info(f"control_service try: cmd={' '.join(cmd)}")
                except Exception:
                    pass

            # Immer bevorzugt: sudo -n + absoluter Pfad (/usr/bin oder /bin)
            if has_sudo and isinstance(systemctl_path, str) and systemctl_path.startswith("/"):
                cmd1 = [sudo_path, "-n"] + base_cmd
                attempts.append(cmd1); _log_try(cmd1)

            # Alternativer absoluter Pfad
            alt_path = "/bin/systemctl" if systemctl_path == "/usr/bin/systemctl" else "/usr/bin/systemctl"
            if has_sudo and subprocess.run(["/usr/bin/env", "bash", "-lc", f"test -x {alt_path}"], capture_output=True).returncode == 0:
                alt_base = [alt_path] + base_cmd[1:]
                cmd2 = [sudo_path, "-n"] + alt_base
                attempts.append(cmd2); _log_try(cmd2)

            # Fallback: sudo -n + unqualifiziert
            if has_sudo and systemctl_path == "systemctl":
                cmd3 = [sudo_path, "-n"] + base_cmd
                attempts.append(cmd3); _log_try(cmd3)

            # Harter Fallbacks ohne sudo (nur zur klaren Fehlermeldung)
            if isinstance(systemctl_path, str) and systemctl_path.startswith("/"):
                cmd4 = base_cmd
                attempts.append(cmd4); _log_try(cmd4)
            if systemctl_path == "systemctl":
                cmd5 = base_cmd
                attempts.append(cmd5); _log_try(cmd5)

            # Ausführung mit frühem Abbruch bei Erfolg
            last_cmd = None
            last_result = None
            for cmd in attempts:
                last_cmd = cmd
                last_result = subprocess.run(cmd, capture_output=True, text=True, env=env_path)
                if action == "status":
                    state = (last_result.stdout or "").strip()
                    if last_result.returncode == 0 and state in ("active", "inactive", "failed", "activating", "deactivating"):
                        break
                else:
                    if last_result.returncode == 0:
                        break

            cmd_str = " ".join(last_cmd) if last_cmd else "n/a"
            result = last_result

            try:
                out = (result.stdout or "").strip() if result else ""
                err = (result.stderr or "").strip() if result else ""
                logger.info(f"control_service selected: cmd={cmd_str}, rc={(result.returncode if result else -1)}, out={out[:200]}, err={err[:200]}")
            except Exception:
                pass

            if result and result.returncode != 0 and "Interactive authentication required" in ((result.stderr or "") + (result.stdout or "")):
                if "sudo -n" not in cmd_str or (" /usr/bin/systemctl " not in f" {cmd_str} " and " /bin/systemctl " not in f" {cmd_str} "):
                    hint = "Hinweis: Der Aufruf erfolgte ohne 'sudo -n' und/oder ohne absoluten systemctl-Pfad. Prüfe sudoers für www-data und erlaube /usr/bin/systemctl (und ggf. /bin) mit NOPASSWD."
                    if result.stderr:
                        result.stderr += f"\n{hint}\n"
                    else:
                        result.stdout = (result.stdout or "") + f"\n{hint}\n"

            if action == "status":
                ok = bool(result and result.returncode == 0 and result.stdout.strip() == "active")
                return {
                    "success": ok,
                    "message": (result.stdout or result.stderr) if result else "no result",
                    "unit": unit,
                    "action": action,
                    "code": (result.returncode if result else -1),
                    "cmd": cmd_str
                }

            return {
                "success": bool(result and result.returncode == 0),
                "message": (result.stdout or result.stderr) if result else "no result",
                "unit": unit,
                "action": action,
                "code": (result.returncode if result else -1),
                "cmd": cmd_str
            }

        except FileNotFoundError:
            return {
                "success": False,
                "message": "systemctl nicht gefunden (läuft dieses System mit systemd?)",
                "hint": "Prüfe, ob /usr/bin/systemctl oder /bin/systemctl existiert und ob sudo vorhanden ist."
            }
        except Exception as e:
            return {"success": False, "message": str(e)}

service_manager = ServiceManager()
