# 🔌 Open WebUI Integration

Diese Anleitung zeigt dir, wie du deine **zammad-lightrag** Ticketsuche in Open WebUI einbindest.

Es gibt **zwei Wege** — einer ist deutlich einfacher.

---

## ⭐ Weg A: Ollama-Integration (empfohlen, 1 Minute)

LightRAG hat eine **eingebaute Ollama-kompatible Schnittstelle**. Open WebUI kann LightRAG direkt als "Ollama-Model" einbinden — **ohne Custom-Function**.

### Setup

1. **Open WebUI öffnen** → Zahnrad-Icon (Admin Panel)
2. **Settings → Connections → Ollama API**
3. **URL eintragen:**
   ```
   http://<dein-lightrag-host>:9621
   ```
   Beispiele:
   - `http://localhost:9621` (wenn Open WebUI auf demselben Rechner)
   - `http://192.168.0.190:9621` (wenn Open WebUI auf einem anderen Rechner)
   - `http://zammad-lightrag:9621` (wenn Open WebUI im selben Docker-Netz läuft)
4. **Speichern**
5. **Im Chat:** Modell-Dropdown öffnen → `lightrag:latest` auswählen

### Query-Modi (als Chat-Command)

Im Open WebUI Chat einfach den Prefix vor die Frage setzen:

| Prefix | Was passiert | Beispiel |
|--------|--------------|----------|
| `/hybrid` (Default) | Vektor-Suche + Knowledge Graph | `/hybrid Wie wurde das VPN-Problem gelöst?` |
| `/local` | Nur lokaler Kontext | `/local Druckerfehler Maschine 307` |
| `/global` | Nur Knowledge Graph | `/global Welche Kunden nutzen Citrix?` |
| `/naive` | Nur Vektor-Ähnlichkeit | `/naive WLAN Passwort zurücksetzen` |
| `/mix` | Kombination aller Modi | `/mix Häufige Outlook Probleme` |
| `/bypass` | Direkt ans LLM (kein RAG) | `/bypass Erzähl mir einen Witz` |
| `/context` | Nur Kontext, kein LLM-Call | `/context VPN Probleme` |

### Vorteile
- ✅ Kein Code nötig
- ✅ Alle Open WebUI Features (Chat-Verlauf, Markdown, Code-Highlighting, etc.)
- ✅ Streaming-Support
- ✅ Funktioniert mit Multi-User-Sessions

### Nachteile
- ❌ Quellenangaben nicht so schön formatiert
- ❌ User muss den `/mode` Prefix kennen

---

## 🔧 Weg B: Custom Tool (für mehr Kontrolle)

Wenn du einen **🔍 Button im Chat** haben willst, der "Suche in Zammad Tickets" macht — mit schön formatierten Quellen — nutze die Custom-Function.

### Setup

1. **Open WebUI öffnen** → Zahnrad-Icon (Admin Panel)
2. **Functions → Add Function → Import from Link** (oder paste)
3. **Inhalt einfügen** aus [`zammad-lightrag-search.py`](./zammad-lightrag-search.py)
4. **Speichern**
5. **Valves konfigurieren** (Edit-Icon an der Function):
   - `LIGHTRAG_URL`: z.B. `http://host.docker.internal:9621`
   - `QUERY_MODE`: `hybrid` (Default)
   - `SHOW_SOURCES`: `true` (Quellen anzeigen)
   - `LANGUAGE`: `German`
6. **Im Chat:** 🔍 Button erscheint in der Eingabeleiste
7. **Klicken → Tool auswählen** → `search_zammad` oder `lightrag_status`

### Verfügbare Tools

#### 🔍 `search_zammad`
Durchsucht die Zammad-Ticket-Wissensdatenbank.

**Parameter:**
- `query` (required): Die Suchanfrage
- `mode` (optional): `local` / `global` / `hybrid` / `naive` / `mix`
- `top_k` (optional): Anzahl Ergebnisse

**Beispiel-Fragen:**
- "Welche Probleme gab es mit VPN?"
- "Wie wurde das Drucker-Problem von Frau Müller gelöst?"
- "Häufige Fehler bei der Passwort-Zurücksetzung"
- "Tickets zu Maschine 307"

#### 📊 `lightrag_status`
Zeigt LightRAG-Status und Stats.

**Output:**
```
✅ LightRAG ist erreichbar
• Status: healthy
• LLM: google/gemma-4-26b-a4b-qat
• Embedding: text-embedding-bge-m3
• Sprache: German
• Indexierte Dokumente: 1234
```

### Vorteile
- ✅ Schöne Quellenangaben
- ✅ Klare UI-Trennung zwischen "Chat mit LLM" und "Suche in Tickets"
- ✅ Konfigurierbar pro Setup
- ✅ Status-Check direkt im Chat

### Nachteile
- ❌ Manuelle Code-Pflege
- ❌ Kein Streaming (Antwort kommt auf einmal)
- ❌ User muss aktiv den Tool-Button klicken

---

## 💡 Welcher Weg für wen?

| Szenario | Empfehlung |
|----------|------------|
| **Support-Mitarbeiter sucht schnell Tickets** | Weg A (Ollama) |
| **Wissen "im Vorbeigehen" während Chat** | Weg A (Ollama) |
| **Dedizierte Ticket-Suche mit schönen Quellen** | Weg B (Custom Tool) |
| **Beides parallel** | Beides! Funktioniert nebeneinander |

**Mein Tipp:** Starte mit **Weg A**. Du kannst Weg B später jederzeit hinzufügen.

---

## 🐛 Troubleshooting

### "Connection refused" bei Weg A
- Prüfe: `curl http://localhost:9621/health` funktioniert?
- Wenn Open WebUI in Docker läuft: nutze `http://host.docker.internal:9621`
- Wenn auf anderer Maschine: nutze die LAN-IP `http://192.168.x.x:9621`

### "lightrag:latest" erscheint nicht
- LightRAG-Server muss laufen: `docker ps | grep lightrag`
- Open WebUI Cache leeren: Settings → Connections → Ollama API → "Refresh"
- LightRAG hat mind. einen Embedding- und LLM-Provider konfiguriert

### Quellenangaben leer
- LightRAG muss Dokumente indexiert haben (Sync lief)
- Test: Öffne `http://localhost:9621/webui` und stelle eine Frage

### Custom Tool funktioniert nicht
- `requests` muss installiert sein: in Open WebUI Container ggf. `pip install requests`
- Valves prüfen: ist `LIGHTRAG_URL` korrekt?
- Function Status in Open WebUI: sollte grün sein

---

## 🔗 Beispiel-Konfiguration

### Open WebUI im Docker-Container
```yaml
services:
  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    ports:
      - "3000:8080"
    extra_hosts:
      - "host.docker.internal:host-gateway"
    # ...
```

Dann in Open WebUI:
- Ollama URL: `http://host.docker.internal:9621`
- Tool Valve `LIGHTRAG_URL`: `http://host.docker.internal:9621`

### Open WebUI auf demselben Host
- Ollama URL: `http://localhost:9621`
- Tool Valve `LIGHTRAG_URL`: `http://localhost:9621`
