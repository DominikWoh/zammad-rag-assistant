#!/bin/bash
set -e

INSTALL_DIR="/opt/ai-suite"
REPO_URL="https://github.com/DominikWoh/zammad-rag-assistant.git"
PROJECT_NAME="zammad-rag-assistant"
PROJECT_PATH="$INSTALL_DIR/$PROJECT_NAME"
SCRIPT_PATH="$PROJECT_PATH/ZammadToQdrant.py"
ENV_FILE="$INSTALL_DIR/ticket_ingest.env"
PYTHON_ENV="$INSTALL_DIR/venv"
ZAMMAD_DOCKER_PATH="/opt/zammad-docker"
ZAMMAD_RAG_POLLER="$PROJECT_PATH/zammad_rag_poller.py"

echo "📦 Starte vollständiges Setup inkl. AI-Stack und (optional) Zammad ..."

echo "[0/8] System vorbereiten (Upgrade & Cleanup)..."
apt update && apt upgrade -y

echo "[1/8] Docker & Compose installieren..."
apt install -y \
  curl git apt-transport-https ca-certificates software-properties-common gnupg lsb-release

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

apt update
apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

echo "[2/8] Stoppe & bereinige alte Docker-Container..."
docker container stop $(docker ps -aq) 2>/dev/null || true
docker system prune -af
docker volume prune -f
rm -rf "$INSTALL_DIR" "$ZAMMAD_DOCKER_PATH"

# IP ermitteln, die für alle Container genutzt wird
IP=$(hostname -I | awk '{print $1}')

echo "[3/8] Zammad optional installieren..."
read -p "❓ Möchtest du Zammad mit Docker installieren? (y/n): " INSTALL_ZAMMAD
if [[ "$INSTALL_ZAMMAD" =~ ^[Yy]$ ]]; then
  git clone https://github.com/zammad/zammad-docker-compose.git "$ZAMMAD_DOCKER_PATH"
  cd "$ZAMMAD_DOCKER_PATH"
  docker compose up -d
  echo "⏳ Zammad gestartet unter http://$IP:8080, erstelle ein API Token unter einem neuen Benutzer z.B. KI-Assistant"
  ZAMMAD_URL="http://localhost:8080"
  read -p "🔑 Bitte gib dein Zammad API-Token ein: " ZAMMAD_TOKEN
else
  read -p "🌐 Zammad-URL (z. B. http://it.local:8080): " ZAMMAD_URL
  read -p "🔑 Zammad API-Token: " ZAMMAD_TOKEN
fi

echo "[4/8] AI-Umgebung vorbereiten (Qdrant, Ollama, OpenWebUI)..."
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

QDRANT_API_KEY=$(openssl rand -hex 32)

cat > docker-compose.yml <<EOF
version: "3.9"
services:
  qdrant:
    image: qdrant/qdrant
    container_name: qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    environment:
      QDRANT__SERVICE__API_KEY: ${QDRANT_API_KEY}
    volumes:
      - qdrant_data:/qdrant/storage

  ollama:
    image: ollama/ollama
    container_name: ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama

  openwebui:
    image: ghcr.io/open-webui/open-webui:main
    container_name: openwebui
    ports:
      - "3000:8080"
    environment:
      OLLAMA_BASE_URL: http://ollama:11434
    depends_on:
      - ollama
    volumes:
      - webui_data:/app/backend/data

volumes:
  qdrant_data:
  ollama_data:
  webui_data:
EOF

docker compose --env-file <(echo "QDRANT_API_KEY=${QDRANT_API_KEY}") -f docker-compose.yml up -d

# IP ermitteln, die für alle Container genutzt wird
IP=$(hostname -I | awk '{print $1}')

echo "[5/8] Konfiguration & .env-Datei schreiben..."
read -p "🤖 Ollama-Modell (z. B. gemma3n:latest): " OLLAMA_MODEL
read -p "📁 Name der Qdrant-Collection: " COLLECTION_NAME

cat > "$ENV_FILE" <<EOF
QDRANT_URL=http://$IP:6333
QDRANT_API_KEY=${QDRANT_API_KEY}
OLLAMA_URL=http://$IP:11434/api/chat
OLLAMA_MODEL=${OLLAMA_MODEL}
COLLECTION_NAME=${COLLECTION_NAME}
ZAMMAD_URL=${ZAMMAD_URL/localhost/$IP}
ZAMMAD_TOKEN=${ZAMMAD_TOKEN}
EOF

chmod 600 "$ENV_FILE"
echo "🔐 .env gespeichert unter $ENV_FILE"

echo "[6/8] Warte auf Ollama-Start & lade Modell '$OLLAMA_MODEL'..."
sleep 20
docker exec ollama ollama pull "$OLLAMA_MODEL" || echo "❌ Modell konnte nicht geladen werden"

echo "[7/8] Python-Umgebung vorbereiten..."
apt install -y python3 python3-pip python3-venv

python3 -m venv "$PYTHON_ENV"
source "$PYTHON_ENV/bin/activate"
pip install --upgrade pip

git clone "$REPO_URL" "$PROJECT_PATH"

cat > "$PROJECT_PATH/requirements.txt" <<EOF
python-dotenv
requests
beautifulsoup4
sentence-transformers
qdrant-client
EOF

pip install -r "$PROJECT_PATH/requirements.txt"
deactivate

echo "[8/8] UFW Freigabe und Links"

echo "🌐 Öffne Firewall für relevante Ports ..."
if ! command -v ufw >/dev/null; then
  echo "📦 Installiere ufw ..."
  apt install -y ufw
fi

ufw allow 8080/tcp
ufw allow 3000/tcp
ufw allow 11434/tcp
ufw allow 6333/tcp

echo "✅ Setup abgeschlossen!"
echo "Zammad:       http://$IP:8080"
echo "OpenWebUI:    http://$IP:3000"
echo "Ollama API:   http://$IP:11434"
echo "Qdrant API:   http://$IP:6333/Dashboard"
echo "Collection:   $COLLECTION_NAME"
echo "Script-Log:   /var/log/zammad_to_qdrant.log"
echo "Qdrant API-Key:     $QDRANT_API_KEY"


# Service für zammad_rag_poller.py erstellen
echo "[9/8] Erstellen des Services für zammad_rag_poller.py:"

SERVICE_FILE="/etc/systemd/system/zammad_rag_poller.service"

echo "[Unit]" > "$SERVICE_FILE"
echo "Description=Zammad RAG Poller Service" >> "$SERVICE_FILE"
echo "After=network.target" >> "$SERVICE_FILE"
echo "" >> "$SERVICE_FILE"
echo "[Service]" >> "$SERVICE_FILE"
echo "ExecStart=$PYTHON_ENV/bin/python $ZAMMAD_RAG_POLLER" >> "$SERVICE_FILE"
echo "WorkingDirectory=$INSTALL_DIR/$PROJECT_NAME" >> "$SERVICE_FILE"
echo "EnvironmentFile=$ENV_FILE" >> "$SERVICE_FILE"
echo "Restart=always" >> "$SERVICE_FILE"
echo "TimeoutSec=120" >> "$SERVICE_FILE"
echo "" >> "$SERVICE_FILE"
echo "[Install]" >> "$SERVICE_FILE"
echo "WantedBy=multi-user.target" >> "$SERVICE_FILE"

# Berechtigungen setzen
chmod 644 "$SERVICE_FILE"
chmod +x "$ZAMMAD_RAG_POLLER"

# Service aktivieren und starten
systemctl daemon-reload
systemctl enable zammad_rag_poller.service
systemctl start zammad_rag_poller.service

# Cronjob für das ZammadToQdrant.py Skript einrichten
echo "[10/8] Cronjob für ZammadToQdrant.py hinzufügen und Skript sofort ausführen:"

# Cronjob für das ZammadToQdrant.py Skript einrichten
CRON_JOB="0 1 * * * $PYTHON_ENV/bin/python $SCRIPT_PATH >> /var/log/zammad_to_qdrant.log 2>&1"
( crontab -l 2>/dev/null | grep -v "$SCRIPT_PATH" ; echo "$CRON_JOB" ) | crontab -

# Sofortige Ausführung von ZammadToQdrant.py
echo "🔄 Führe ZammadToQdrant.py jetzt sofort aus ..."
$PYTHON_ENV/bin/python "$SCRIPT_PATH"

echo "✅ Skript sofort ausgeführt und Cronjob sowie Service für zammad_rag_poller eingerichtet!"
