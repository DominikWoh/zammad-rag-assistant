#!/usr/bin/env bash
set -euo pipefail

# Idempotentes Setup-Skript für Zammad RAG Web-UI und sudo/systemd-Integration
# Ziel:
#  - Web-UI läuft als www-data
#  - Korrektes PATH und PYTHONPATH im Dienst
#  - www-data darf systemctl für zammad_rag_poller.service via sudo -n ausführen
#  - Alle notwendigen Dateien werden sauber überschrieben
#
# Annahmen (vom Benutzer bestätigt):
#  - Basis: /opt/ai-suite
#  - Web-UI: /opt/ai-suite/RAG-UI/web-ui
#  - Backend: /opt/ai-suite/RAG-UI/web-ui/backend/main.py
#  - venv: /opt/ai-suite/venv
#  - Port: 5000

BASE="/opt/ai-suite"
WEBUI_DIR="${BASE}/RAG-UI/web-ui"
BACKEND_DIR="${WEBUI_DIR}/backend"
BACKEND_MAIN="${BACKEND_DIR}/main.py"
VENV_BIN="${BASE}/venv/bin"
PYTHON_BIN="${VENV_BIN}/python"
SERVICE_WEBUI="zammad_rag_webui.service"
SERVICE_POLLER="zammad_rag_poller.service" # nur für sudoers-Freigabe
SUDOERS_FILE="/etc/sudoers.d/zammad_rag"
SYSTEMD_DIR="/etc/systemd/system"

require_root() {
  if [[ "$(id -u)" -ne 0 ]]; then
    echo "Dieses Skript muss als root laufen." >&2
    exit 1
  fi
}

ensure_paths() {
  for p in "$WEBUI_DIR" "$BACKEND_DIR" "$VENV_BIN" ; do
    if [[ ! -d "$p" ]]; then
      echo "Fehlender Pfad: $p" >&2
      exit 1
    fi
  done
  if [[ ! -f "$BACKEND_MAIN" ]]; then
    echo "Fehlende Datei: $BACKEND_MAIN" >&2
    exit 1
  fi
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python nicht gefunden/executabel: $PYTHON_BIN" >&2
    exit 1
  fi
}

write_webui_unit() {
  local unit_path="${SYSTEMD_DIR}/${SERVICE_WEBUI}"
  cat >"$unit_path" <<'UNIT'
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
  echo "Geschrieben: $unit_path"
}

write_sudoers() {
  # Entferne alte potenziell kollidierende Dateien
  rm -f /etc/sudoers.d/zammad-rag /etc/sudoers.d/zammad_rag_webui || true

  # Schreibe die konsolidierte Datei
  cat >"$SUDOERS_FILE" <<SUDO
Cmnd_Alias RAGCTL = /usr/bin/systemctl start ${SERVICE_POLLER}, /usr/bin/systemctl stop ${SERVICE_POLLER}, /usr/bin/systemctl restart ${SERVICE_POLLER}, /usr/bin/systemctl status ${SERVICE_POLLER}, /bin/systemctl start ${SERVICE_POLLER}, /bin/systemctl stop ${SERVICE_POLLER}, /bin/systemctl restart ${SERVICE_POLLER}, /bin/systemctl status ${SERVICE_POLLER}
www-data ALL=(root) NOPASSWD: RAGCTL
SUDO

  chmod 0440 "$SUDOERS_FILE"
  echo "Geschrieben: $SUDOERS_FILE"

  # Syntax-Check
  if ! visudo -c >/dev/null; then
    echo "WARNUNG: visudo -c meldet einen Fehler. Bitte Ausgabe manuell prüfen: visudo -c" >&2
    visudo -c || true
    exit 1
  fi
}

daemon_reload_and_restart() {
  systemctl daemon-reload
  systemctl enable "$SERVICE_WEBUI" --now
  systemctl restart "$SERVICE_WEBUI"
  systemctl status "$SERVICE_WEBUI" --no-pager -l || true
}

smoke_tests() {
  echo "Prüfe sudo aus Sicht www-data..."
  if ! sudo -u www-data which sudo >/dev/null; then
    echo "ACHTUNG: 'sudo' ist nicht im PATH von www-data. Die Unit setzt PATH, bitte Web-UI-Logs prüfen." >&2
  fi

  echo "Schnelltest systemctl (status) als www-data via sudo -n..."
  if sudo -u www-data sudo -n /usr/bin/systemctl status "${SERVICE_POLLER}" >/dev/null; then
    echo "OK: www-data darf ${SERVICE_POLLER} status via sudo -n ausführen."
  else
    echo "FEHLER: sudo -n Status-Test für ${SERVICE_POLLER} fehlgeschlagen." >&2
    echo "Bitte prüfen: Inhalt $SUDOERS_FILE, 'visudo -c', und ob /usr/bin/sudo existiert/ausführbar ist." >&2
    exit 1
  fi

  echo "Hinweis: Web-UI jetzt aufrufen und in der UI den Poller per Button starten/stoppen/restarten."
  echo "Kontrolliere dann die Logs: journalctl -u ${SERVICE_WEBUI} -n 80 --no-pager"
  echo "Erwartete Zeile: cmd=sudo -n /usr/bin/systemctl restart ${SERVICE_POLLER}, code=0, success=True"
}

main() {
  require_root
  ensure_paths
  write_webui_unit
  write_sudoers
  daemon_reload_and_restart
  smoke_tests
  echo "Fertig."
}

main "$@"