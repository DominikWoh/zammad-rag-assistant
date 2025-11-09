# ⚡ 5-Minute Zammad Qdrant RAG Setup

**Quick installation guide for the Zammad Qdrant RAG system with Docker.**

## 🎯 What You'll Get

- **Complete RAG System**: Vector database + AI search + web interface
- **Docker Orchestration**: 5 services with health monitoring
- **Local AI Models**: Ollama integration for privacy
- **Web Dashboard**: Modern React-style interface

## 📋 Requirements

### Minimal Setup
- **Docker** 20.10+
- **RAM**: 4GB
- **Storage**: 20GB
- **Internet**: For model downloads

### Recommended Setup
- **RAM**: 8-16GB (for AI models)
- **CPU**: 4+ cores
- **Storage**: 50GB+ SSD
- **GPU**: Optional (CUDA support)

## 🚀 Step 1: Environment Setup

### Download the project
```bash
git clone https://github.com/your-username/zammad-qdrant-rag.git
cd zammad-qdrant-rag
```

### Create environment file
```bash
cp .env.example .env
```

### Configure Zammad credentials
Edit `.env` file:
```bash
# Your Zammad instance
ZAMMAD_URL="https://your-zammad.example.com/"
ZAMMAD_TOKEN="your_zammad_api_token_here"

# Optional: Qdrant API key (for remote instances)
QDRANT_API_KEY=""

# AI Model settings
OLLAMA_MODEL="qwen2.5"
AI_ENABLED="true"
```

## 🐳 Step 2: Start the System

### Pull and start all services
```bash
docker-compose up -d
```

### Check service status
```bash
docker-compose ps
```

### View logs (optional)
```bash
docker-compose logs -f
```

## 🌐 Step 3: Access the Interface

After 2-3 minutes, services will be available:

| Service | URL | Description |
|---------|-----|-------------|
| **Web Dashboard** | http://localhost:8000 | Main interface |
| **API Documentation** | http://localhost:8000/docs | Interactive API docs |
| **Qdrant Admin** | http://localhost:6333/dashboard | Vector DB admin |
| **Ollama API** | http://localhost:11434 | AI model API |
| **MCP Search** | http://localhost:8083 | Search endpoint |

## ✅ Step 4: Configure AI Models

### Download Ollama models
```bash
# Pull recommended models
docker-compose exec ollama ollama pull qwen2.5
docker-compose exec ollama ollama pull llama2
docker-compose exec ollama ollama pull mistral

# Check available models
docker-compose exec ollama ollama list
```

### Test AI connection
1. Open http://localhost:8000
2. Go to **AI Settings**
3. Click **"Test Connection"**
4. Should show: "✓ Ollama instance found!"

## 🎛️ Step 5: Transfer Configuration

### Basic settings
1. Open **Settings** page
2. Configure **Zammad API** credentials
3. Set **Transfer Parameters**:
   - Minimum age: 14 days
   - Start date: 2018-01-01
   - BM25 cache: Enabled

### Start first transfer
1. Go to **Dashboard**
2. Click **"Start Transfer"**
3. Monitor progress in live log
4. Check BM25 statistics

## 🔧 Common Commands

### Stop all services
```bash
docker-compose down
```

### Restart specific service
```bash
docker-compose restart zammad-rag-app
```

### Update system
```bash
git pull
docker-compose down
docker-compose up -d
```

### View real-time logs
```bash
docker-compose logs -f zammad-rag-app
```

### Access container shell
```bash
docker-compose exec zammad-rag-app bash
```

## 🚨 Troubleshooting

### Service won't start
```bash
# Check Docker status
docker info

# Check port conflicts
netstat -tlnp | grep 8000
```

### Memory issues
```bash
# Check resource usage
docker stats

# Increase Docker memory limit to 6GB
```

### Qdrant connection failed
```bash
# Wait for service startup (1-2 minutes)
docker-compose logs qdrant

# Check volume permissions
ls -la ./qdrant_data
```

### AI models not loading
```bash
# Check Ollama logs
docker-compose logs ollama

# Download models manually
docker-compose exec ollama ollama pull qwen2.5
```

## 📊 Performance Tips

### For better performance
- **SSD Storage**: Improves vector operations
- **More RAM**: Enables larger models
- **GPU Support**: Accelerated AI inference

### Monitoring
```bash
# System resources
docker stats

# Health checks
curl -f http://localhost:8000/health
curl -f http://localhost:6333/health
```

## 🎉 Success Indicators

### System is working when:
- ✅ All containers show "Up" status
- ✅ Web dashboard loads at http://localhost:8000
- ✅ AI connection test passes
- ✅ Transfer completes without errors
- ✅ Search returns results

## 📚 Next Steps

1. **Read full documentation**: [README-Docker-Deployment.md](README-Docker-Deployment.md)
2. **Explore API docs**: http://localhost:8000/docs
3. **Check vector database**: http://localhost:6333/dashboard
4. **Set up monitoring**: Configure alerts and health checks

## 🆘 Need Help?

- **Documentation**: [README-Docker-Deployment.md](README-Docker-Deployment.md)
- **Troubleshooting**: Check logs with `docker-compose logs`
- **Community**: GitHub Discussions
- **Issues**: Create GitHub Issue

---

**🎯 Your Zammad Qdrant RAG system should now be running!**

*Estimated setup time: 5-10 minutes (including model downloads)*