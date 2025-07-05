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

echo "ðŸ“¦ Starte vollstÃ¤ndiges Setup inkl. AI-Stack und (optional) Zammad ..."

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
echo "[0/8] System vorbereiten (Upgrade & Cleanup)..."
apt update && apt upgrade -y

echo "[0/8] Stoppe & bereinige alte Docker-Container..."
docker container stop $(docker ps -aq) 2>/dev/null || true
docker system prune -af
docker volume prune -f
rm -rf "$INSTALL_DIR" "$ZAMMAD_DOCKER_PATH"

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
echo "[1/8] Docker & Compose installieren..."
apt install -y \
  curl git apt-transport-https ca-certificates software-properties-common gnupg lsb-release

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

apt update
apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
echo "[2/8] Zammad optional installieren..."
read -p "â“ MÃ¶chtest du Zammad mit Docker installieren? (y/n): " INSTALL_ZAMMAD
if [[ "$INSTALL_ZAMMAD" =~ ^[Yy]$ ]]; then
  git clone https://github.com/zammad/zammad-docker-compose.git "$ZAMMAD_DOCKER_PATH"
  cd "$ZAMMAD_DOCKER_PATH"
  docker compose up -d
  echo "â³ Zammad gestartet unter http://localhost:8080"
  ZAMMAD_URL="http://localhost:8080"
  read -p "ðŸ”‘ Bitte gib dein Zammad API-Token ein: " ZAMMAD_TOKEN
else
  read -p "ðŸŒ Zammad-URL (z.â€¯B. http://it.local:8080): " ZAMMAD_URL
  read -p "ðŸ”‘ Zammad API-Token: " ZAMMAD_TOKEN
fi

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
echo "[3/8] AI-Umgebung vorbereiten (Qdrant, Ollama, OpenWebUI)..."
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

# ───────────────────────────────────────────────────────────────
echo "[4/8] Konfiguration & .env-Datei schreiben..."
read -p "🤖 Ollama-Modell (z. B. gemma3n:latest): " OLLAMA_MODEL
read -p "📁 Name der Qdrant-Collection: " COLLECTION_NAME

# .env schreiben
cat > "$ENV_FILE" <<EOF
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=${QDRANT_API_KEY}
OLLAMA_URL=http://localhost:11434/api/chat
OLLAMA_MODEL=${OLLAMA_MODEL}
COLLECTION_NAME=${COLLECTION_NAME}
ZAMMAD_URL=${ZAMMAD_URL}
ZAMMAD_TOKEN=${ZAMMAD_TOKEN}
EOF

chmod 600 "$ENV_FILE"
echo "🔐 .env gespeichert unter $ENV_FILE"

# ───────────────────────────────────────────────────────────────
echo "[5/8] Warte auf Ollama-Start & lade Modell '$OLLAMA_MODEL'..."
sleep 20
docker exec ollama ollama pull "$OLLAMA_MODEL" || echo "❌ Modell konnte nicht geladen werden"


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
echo "[6/8] Python-Umgebung vorbereiten..."
apt install -y python3 python3-pip python3-venv

python3 -m venv "$PYTHON_ENV"
source "$PYTHON_ENV/bin/activate"
pip install --upgrade pip

# Projekt klonen
git clone "$REPO_URL" "$PROJECT_PATH"

# Requirements erzeugen
cat > "$PROJECT_PATH/requirements.txt" <<EOF
python-dotenv
requests
beautifulsoup4
sentence-transformers
qdrant-client
EOF

pip install -r "$PROJECT_PATH/requirements.txt"
deactivate

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
echo "[7/8] Starte Script jetzt sofort..."
"$PYTHON_ENV/bin/python" "$SCRIPT_PATH"

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
echo "[8/8] Cronjob einrichten fÃ¼r tÃ¤gliche AusfÃ¼hrung (01:00 Uhr)..."
CRON_JOB="0 1 * * * $PYTHON_ENV/bin/python $SCRIPT_PATH >> /var/log/zammad_to_qdrant.log 2>&1"
( crontab -l 2>/dev/null | grep -v "$SCRIPT_PATH" ; echo "$CRON_JOB" ) | crontab -

# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Optional: Firewall Ã¶ffnen
if command -v ufw >/dev/null && ufw status | grep -q "Status: active"; then
  echo "ðŸŒ Ã–ffne Firewall fÃ¼r relevante Ports ..."
  ufw allow 8080/tcp
  ufw allow 3000/tcp
  ufw allow 11434/tcp
  ufw allow 6333/tcp
fi

IP=$(hostname -I | awk '{print $1}')
echo ""
echo "âœ… Setup abgeschlossen!"
echo "Zammad:       http://$IP:8080"
echo "OpenWebUI:    http://$IP:3000"
echo "Ollama API:   http://$IP:11434"
echo "Qdrant API:   http://$IP:6333"
echo "Collection:   $COLLECTION_NAME"
echo "Script-Log:   /var/log/zammad_to_qdrant.log"
echo "Qdrant API-Key:     $QDRANT_API_KEY"
