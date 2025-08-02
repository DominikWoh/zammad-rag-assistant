#!/usr/bin/env bash
set -euo pipefail

# Zammad RAG-UI Full Installer (idempotent)
# Ziele:
#  - Einrichtung unter /opt/ai-suite/RAG-UI (dieses Repo wird dorthin kopiert/platziert)
#  - Python venv unter /opt/ai-suite/venv, Abhängigkeiten installieren
#  - .env erzeugen/aktualisieren (inkl. zufälligem QDRANT_API_KEY)
#  - systemd Units: zammad_rag_webui.service, zammad_rag_poller.service
#  - sudoers Drop-in: www-data darf systemctl für zammad_rag_poller.service via sudo -n
#  - Optional: Docker Compose Setup für Qdrant + Ollama (inkl. API-Key und Pull von qwen3:8b)
#  - Firewall-Ports öffnen (5000, 6333, 11434) sofern ufw vorhanden
#
# Voraussetzungen:
#  - Skript wird als root ausgeführt.
#  - Projekt liegt unter /opt/ai-suite/RAG-UI (dieses Repo an diesen Ort kopieren).
#
# Interaktiv:
#  - Fragt, ob Docker Compose Stack eingerichtet werden soll (Qdrant/Ollama)
#  - Bei Ja: erzeugt docker-compose.yml und startet Container, setzt API-Key, pullt qwen3:8b

BASE="/opt/ai-suite"
PROJECT_DIR="${BASE}/RAG-UI"
WEBUI_DIR="${PROJECT_DIR}/web-ui"
BACKEND_DIR="${WEBUI_DIR}/backend"
SERVICES_DIR="${WEBUI_DIR}/Services"
VENV_DIR="${BASE}/venv"
PYTHON_BIN="${VENV_DIR}/bin/python"
PIP_BIN="${VENV_DIR}/bin/pip"
SYSTEMD_DIR="/etc/systemd/system"
SUDOERS_FILE="/etc/sudoers.d/zammad_rag"

SERVICE_WEBUI="zammad_rag_webui.service"
SERVICE_POLLER="zammad_rag_poller.service"

# Defaults (können im .env überschrieben werden, UI bietet ebenfalls Config)
DEFAULT_WEBUI_HOST="0.0.0.0"
DEFAULT_WEBUI_PORT="5000"
DEFAULT_QDRANT_URL="http://192.168.0.120:6333"
DEFAULT_QDRANT_API_KEY=""   # wird generiert wenn leer
DEFAULT_OLLAMA_URL="http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL="qwen3:8b"
DEFAULT_COLLECTION="zammad-collection"
DEFAULT_ZAMMAD_URL="http://127.0.0.1"
DEFAULT_ZAMMAD_TOKEN=""      # später per UI setzen
DEFAULT_ENABLE_ASKKI="false"
DEFAULT_ENABLE_RAG_NOTE="true"

require_root() {
  if [[ "$(id -u)" -ne 0 ]]; then
    echo "Dieses Skript muss als root laufen." >&2
    exit 1
  fi
}

confirm() {
  # usage: confirm "Frage?" default_yes|default_no
  local prompt="$1"
  local default="$2"
  local ans
  if [[ "$default" == "default_yes" ]]; then
    read -rp "$prompt [Y/n]: " ans || true
    ans="${ans:-Y}"
  else
    read -rp "$prompt [y/N]: " ans || true
    ans="${ans:-N}"
  fi
  [[ "$ans" == [Yy] ]]
}

install_packages() {
  echo "==> Pakete installieren/aktualisieren"
  apt-get update -y
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-venv python3-pip \
    curl jq unzip ca-certificates gnupg lsb-release

  # Docker + Compose Plugin installieren (falls nicht vorhanden)
  if ! command -v docker >/dev/null 2>&1; then
    echo "==> Docker installieren"
    install -m 0755 -d /etc/apt/keyrings
    if [[ ! -f /etc/apt/keyrings/docker.gpg ]]; then
      curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
      chmod a+r /etc/apt/keyrings/docker.gpg
    fi
    local codename
    codename="$(. /etc/os-release && echo "$VERSION_CODENAME")"
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      ${codename} stable" | tee /etc/apt/sources.list.d/docker.list >/dev/null
    apt-get update -y
    DEBIAN_FRONTEND=noninteractive apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    systemctl enable docker --now
  else
    echo "Docker bereits installiert, überspringe."
  fi

  # ufw optional
  if command -v ufw >/dev/null 2>&1; then
    echo "==> UFW Ports freigeben (5000, 6333, 11434)"
    ufw allow 5000/tcp || true
    ufw allow 6333/tcp || true
    ufw allow 11434/tcp || true
  fi
}

ensure_layout() {
  echo "==> Verzeichnisstruktur prüfen"
  for p in "$BASE" "$PROJECT_DIR" "$WEBUI_DIR" "$BACKEND_DIR" "$SERVICES_DIR"; do
    if [[ ! -d "$p" ]]; then
      echo "Fehlender Pfad: $p" >&2
      exit 1
    fi
  done
  if [[ ! -f "${BACKEND_DIR}/main.py" ]]; then
    echo "Fehlende Datei: ${BACKEND_DIR}/main.py" >&2
    exit 1
  fi
}

setup_venv() {
  echo "==> Python venv einrichten unter $VENV_DIR"
  if [[ ! -d "$VENV_DIR" ]]; then
    python3 -m venv "$VENV_DIR"
  fi
  "$PIP_BIN" install --upgrade pip wheel
  # requirements (falls vorhanden)
  if [[ -f "${PROJECT_DIR}/requirements.txt" ]]; then
    "$PIP_BIN" install -r "${PROJECT_DIR}/requirements.txt"
  else
    # Minimale Laufzeitabhängigkeiten
    "$PIP_BIN" install fastapi uvicorn[standard] python-dotenv requests psutil qdrant-client
  fi
}

gen_random() {
  # 32 Byte zufällig Base64-like
  python3 - <<'PY'
import secrets
print(secrets.token_urlsafe(32))
PY
}

write_env_file() {
  echo "==> .env schreiben/aktualisieren"
  local env_path="${PROJECT_DIR}/web-ui/backend/.env"
  local qdrant_api_key="${DEFAULT_QDRANT_API_KEY}"
  if [[ -z "$qdrant_api_key" ]]; then
    qdrant_api_key="$(gen_random)"
  fi

  # ticket_ingest.env (Ubuntu-Pfad) auch pflegen, falls verwendet
  local ingest_env="/opt/ai-suite/ticket_ingest.env"

  cat > "$env_path" <<ENV
WEBUI_HOST=${DEFAULT_WEBUI_HOST}
WEBUI_PORT=${DEFAULT_WEBUI_PORT}

QDRANT_URL=${DEFAULT_QDRANT_URL}
QDRANT_API_KEY=${qdrant_api_key}
COLLECTION_NAME=${DEFAULT_COLLECTION}

OLLAMA_URL=${DEFAULT_OLLAMA_URL}
OLLAMA_MODEL=${DEFAULT_OLLAMA_MODEL}

ZAMMAD_URL=${DEFAULT_ZAMMAD_URL}
ZAMMAD_TOKEN=${DEFAULT_ZAMMAD_TOKEN}

ENABLE_ASKKI=${DEFAULT_ENABLE_ASKKI}
ENABLE_RAG_NOTE=${DEFAULT_ENABLE_RAG_NOTE}
ENV

  chmod 0644 "$env_path"
  echo "Geschrieben: $env_path"

  # optional zweites Env (falls von Poller verwendet)
  cat > "$ingest_env" <<ENV
QDRANT_URL=${DEFAULT_QDRANT_URL}
QDRANT_API_KEY=${qdrant_api_key}
COLLECTION_NAME=${DEFAULT_COLLECTION}
ENV
  chmod 0644 "$ingest_env"
  echo "Geschrieben: $ingest_env"
}

write_webui_unit() {
  echo "==> systemd Unit für Web-UI schreiben"
  local unit_path="${SYSTEMD_DIR}/${SERVICE_WEBUI}"
  cat > "$unit_path" <<'UNIT'
[Unit]
Description=Zammad RAG WebUI
After=network.target

[Service]
Type=simple
User=www-data
Group=www-data
WorkingDirectory=/opt/ai-suite/RAG-UI/web-ui/backend
Environment=PYTHONPATH=/opt/ai-suite/RAG-UI/web-ui
Environment=PATH=/opt/ai-suite/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/bin
ExecStart=/opt/ai-suite/venv/bin/python /opt/ai-suite/RAG-UI/web-ui/backend/main.py
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
UNIT
  chmod 0644 "$unit_path"
}

write_poller_unit() {
  echo "==> systemd Unit für Poller schreiben"
  local unit_path="${SYSTEMD_DIR}/${SERVICE_POLLER}"
  cat > "$unit_path" <<'UNIT'
[Unit]
Description=Zammad RAG Poller
After=network.target

[Service]
Type=simple
User=www-data
Group=www-data
WorkingDirectory=/opt/ai-suite/RAG-UI/web-ui
Environment=PYTHONPATH=/opt/ai-suite/RAG-UI/web-ui
Environment=PATH=/opt/ai-suite/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/bin
ExecStart=/opt/ai-suite/venv/bin/python /opt/ai-suite/RAG-UI/web-ui/Services/zammad_rag_poller.py
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
UNIT
  chmod 0644 "$unit_path"
}

write_sudoers() {
  echo "==> sudoers Drop-in erstellen"
  rm -f /etc/sudoers.d/zammad-rag /etc/sudoers.d/zammad_rag_webui || true
  cat > "$SUDOERS_FILE" <<SUDO
Cmnd_Alias RAGCTL = /usr/bin/systemctl start ${SERVICE_POLLER}, /usr/bin/systemctl stop ${SERVICE_POLLER}, /usr/bin/systemctl restart ${SERVICE_POLLER}, /usr/bin/systemctl status ${SERVICE_POLLER}, /bin/systemctl start ${SERVICE_POLLER}, /bin/systemctl stop ${SERVICE_POLLER}, /bin/systemctl restart ${SERVICE_POLLER}, /bin/systemctl status ${SERVICE_POLLER}
www-data ALL=(root) NOPASSWD: RAGCTL
SUDO
  chmod 0440 "$SUDOERS_FILE"
  echo "==> visudo -c"
  visudo -c >/dev/null
}

docker_compose_setup() {
  echo "==> Docker Compose Setup (Qdrant + Ollama)"
  local docker_dir="${BASE}/docker"
  local data_qdrant="${BASE}/data/qdrant"
  local data_ollama="${BASE}/data/ollama"
  mkdir -p "$docker_dir" "$data_qdrant" "$data_ollama"

  # Aktuelle QDRANT_API_KEY aus .env lesen
  local env_path="${BACKEND_DIR}/.env"
  local qkey
  qkey="$(grep -E '^QDRANT_API_KEY=' "$env_path" | cut -d= -f2- || true)"
  if [[ -z "$qkey" ]]; then
    qkey="$(gen_random)"
    sed -i "s|^QDRANT_API_KEY=.*$|QDRANT_API_KEY=${qkey}|" "$env_path"
  fi

  cat > "${docker_dir}/docker-compose.yml" <<'YML'
services:
  qdrant:
    image: qdrant/qdrant:latest
    container_name: qdrant
    restart: unless-stopped
    environment:
      QDRANT__SERVICE__API_KEY: "${QDRANT_API_KEY}"
    volumes:
      - /opt/ai-suite/data/qdrant:/qdrant/storage
    ports:
      - "6333:6333"
  ollama:
    image: ollama/ollama:latest
    container_name: ollama
    restart: unless-stopped
    volumes:
      - /opt/ai-suite/data/ollama:/root/.ollama
    environment:
      - OLLAMA_KEEP_ALIVE=1h
    ports:
      - "11434:11434"
YML

  # .env Datei für Compose
  cat > "${docker_dir}/.env" <<ENV
QDRANT_API_KEY=${qkey}
ENV

  echo "==> docker compose up -d"
  (cd "$docker_dir" && docker compose --env-file .env up -d)

  echo "==> Warten bis Qdrant online ist..."
  for i in {1..30}; do
    if curl -sf "http://127.0.0.1:6333/healthz" >/dev/null; then
      echo "Qdrant ist online."
      break
    fi
    sleep 2
  done

  echo "==> Qdrant-API-Key in Backend .env gesetzt: ${qkey}"

  echo "==> Warten bis Ollama online ist..."
  for i in {1..30}; do
    if curl -sf "http://127.0.0.1:11434/api/tags" >/dev/null; then
      echo "Ollama ist online."
      break
    fi
    sleep 2
  done

  echo "==> Modell qwen3:8b über Ollama pullen (einmalig, kann dauern)..."
  curl -sS -X POST http://127.0.0.1:11434/api/pull -d '{"name":"qwen3:8b"}' || true
}

enable_and_start_units() {
  echo "==> systemd daemon-reload und Dienste starten"
  systemctl daemon-reload
  systemctl enable "${SERVICE_WEBUI}" --now
  systemctl enable "${SERVICE_POLLER}" --now

  systemctl status "${SERVICE_WEBUI}" --no-pager -l || true
  systemctl status "${SERVICE_POLLER}" --no-pager -l || true
}

smoke_tests() {
  echo "==> Smoke-Tests"
  # sudo Test als www-data
  if sudo -u www-data sudo -n /usr/bin/systemctl status "${SERVICE_POLLER}" >/dev/null 2>&1; then
    echo "OK: www-data darf ${SERVICE_POLLER} via sudo -n ausführen."
  else
    echo "WARNUNG: www-data kann ${SERVICE_POLLER} via sudo -n nicht ausführen. Prüfe ${SUDOERS_FILE} und visudo -c." >&2
  fi

  # Web-UI Reachability
  local host="${DEFAULT_WEBUI_HOST}"
  [[ "$host" == "0.0.0.0" ]] && host="127.0.0.1"
  if curl -sf "http://${host}:${DEFAULT_WEBUI_PORT}/api/status" >/dev/null 2>&1; then
    echo "OK: Web-UI API erreichbar."
  else
    echo "HINWEIS: Web-UI API nicht erreichbar. Prüfe Dienst-Logs: journalctl -u ${SERVICE_WEBUI} -n 100 --no-pager" >&2
  fi

  # Qdrant Health (nur Info)
  if curl -sf "${DEFAULT_QDRANT_URL}/healthz" >/dev/null 2>&1; then
    echo "OK: Qdrant (extern/konfiguriert) erreichbar: ${DEFAULT_QDRANT_URL}"
  else
    echo "Info: Qdrant unter ${DEFAULT_QDRANT_URL} nicht erreichbar (evtl. extern nicht aktiv). Falls via Docker installiert, prüfe Compose-Stack." >&2
  fi
}

maybe_compose() {
  if confirm "Qdrant + Ollama via Docker Compose installieren/aktualisieren?" default_yes; then
    docker_compose_setup
  else
    echo "Überspringe Docker Compose Setup."
  fi
}

main() {
  require_root
  install_packages
  ensure_layout
  setup_venv
  write_env_file
  write_webui_unit
  write_poller_unit
  write_sudoers
  enable_and_start_units
  maybe_compose
  smoke_tests

  echo
  echo "Installation abgeschlossen."
  echo "Web-UI: http://SERVER_IP:${DEFAULT_WEBUI_PORT}"
  echo "Dienste: ${SERVICE_WEBUI}, ${SERVICE_POLLER}"
  echo "Logs prüfen:"
  echo "  journalctl -u ${SERVICE_WEBUI} -n 120 --no-pager"
  echo "  journalctl -u ${SERVICE_POLLER} -n 120 --no-pager"
}

main "$@"