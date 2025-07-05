#!/bin/bash
set -e

echo "[0/6] System vorbereiten (Upgrade & Cleanup)..."

apt update && apt upgrade -y

docker container stop $(docker ps -aq) 2>/dev/null || true
docker system prune -af
docker volume prune -f
rm -rf /opt/ai-suite /opt/zammad-docker-compose

echo "[1/6] Docker & Compose installieren..."
apt install -y \
  curl git apt-transport-https ca-certificates software-properties-common gnupg lsb-release

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
$(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

apt update
apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

echo "[2/6] Zammad-Stack klonen..."
cd /opt
git clone https://github.com/zammad/zammad-docker-compose.git
cd zammad-docker-compose

echo "[3/6] Zammad-Stack starten..."
docker compose up -d

echo "[4/6] Zusätzliche Dienste vorbereiten (OpenWebUI, Ollama, Qdrant)..."

mkdir -p /opt/ai-suite && cd /opt/ai-suite

cat > docker-compose.yml <<EOF
version: "3.9"
services:
  qdrant:
    image: qdrant/qdrant
    container_name: qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
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

echo "[5/6] OpenWebUI + Qdrant + Ollama starten..."
docker compose -f /opt/ai-suite/docker-compose.yml up -d

echo "[6/6] granite3.3 LLM in Ollama laden..."
sleep 20
if docker exec ollama ollama pull granite3.3; then
  echo "✅ granite3.3 erfolgreich geladen."
else
  echo "❗ Modell konnte nicht geladen werden."
fi

IP=$(hostname -I | awk '{print $1}')
echo ""
echo "✅ Setup abgeschlossen!"
echo "Zammad:     http://$IP:8080"
echo "OpenWebUI:  http://$IP:3000"
echo "Ollama API: http://$IP:11434"
echo "Qdrant:     http://$IP:6333"
