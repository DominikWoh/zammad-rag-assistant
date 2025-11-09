#!/usr/bin/env python3
"""
Zammad_AI.py - Automatische Ticket-Verarbeitung mit Ollama und RAG über MCP Server

Funktionalität:
1. Überwacht neue Tickets ohne Antworten in Zammad
2. Generiert mit Ollama intelligente Suchbegriffe für RAG-Suche
3. Führt RAG-Suche über MCP Server durch
4. Erstellt automatische Antworten und postet sie als Notizen in Zammad
"""

import os
import time
import json
import logging
import threading
import requests
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
ZAMMAD_URL = os.getenv("ZAMMAD_URL", "").rstrip("/")
ZAMMAD_TOKEN = os.getenv("ZAMMAD_TOKEN", "")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5")
AI_CHECK_INTERVAL = int(os.getenv("AI_CHECK_INTERVAL", "300"))  # 5 Minuten
AI_TICKET_MAX_AGE_DAYS = int(os.getenv("AI_TICKET_MAX_AGE_DAYS", "1"))  # Maximales Ticket-Alter
TOP_K = int(os.getenv("TOP_K", "5"))  # Anzahl der Vektor-Suchergebnisse
TOP_TICKETS = int(os.getenv("TOP_TICKETS", "5"))  # Anzahl der relevantesten Tickets
RAG_SEARCH_PROMPT = os.getenv("RAG_SEARCH_PROMPT", "Erstelle einen prägnanten Suchbegriff für die RAG-Suche basierend auf folgendem Ticket: {ticket_content}")
ZAMMAD_NOTE_PROMPT = os.getenv("ZAMMAD_NOTE_PROMPT", "Erstelle eine hilfreiche und professionelle Antwort für folgendes Zammad-Ticket basierend auf den verfügbaren Informationen: {ticket_content}\n\nRelevante Informationen:\n{search_results}")
AI_ENABLED = os.getenv("AI_ENABLED", "false").lower() == "true"

# MCP Server Configuration
MCP_SERVER_URL = "http://127.0.0.1:8083"
MCP_SEARCH_ENDPOINT = f"{MCP_SERVER_URL}/search"

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("zammad_ai.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TicketData:
    """Datenklasse für Ticket-Informationen"""
    id: int
    title: str
    content: str
    created_at: str
    customer_id: Optional[int] = None
    state: str = "new"
    
# AIResponse Klasse entfernt - wird nicht mehr benötigt

class OllamaClient:
    """Ollama-Client für KI-Operationen"""
    
    def __init__(self, base_url: str = OLLAMA_URL, model: str = OLLAMA_MODEL):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.session = requests.Session()
    
    def generate(self, prompt: str) -> str:
        """Generiert Text mit Ollama"""
        try:
            url = f"{self.base_url}/api/generate"
            
            logger.debug(f"[DEBUG Ollama] Modell: {self.model}")
            logger.debug(f"[DEBUG Ollama] Prompt: {prompt[:200]}{'...' if len(prompt) > 200 else ''}")
            
            # Einheitlicher Payload für alle Modelle
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7
                }
            }
            
            logger.debug(f"[DEBUG Ollama] Sende POST Request an {url}")
            response = self.session.post(url, json=payload, timeout=300)
            logger.debug(f"[DEBUG Ollama] HTTP Status: {response.status_code}")
            
            response.raise_for_status()
            
            result = response.json()
            logger.debug(f"[DEBUG Ollama] Vollständige Response: {result}")
            
            raw_response = result.get("response", "").strip()
            logger.debug(f"[DEBUG Ollama] RAW RESPONSE: '{raw_response}'")
            logger.debug(f"[DEBUG Ollama] Roh-Antwort (vor Filterung): '{raw_response}'")
            
            # Filtere "think" Inhalte aus der Antwort
            filtered_response = self._filter_thinking_content(raw_response)
            logger.debug(f"[DEBUG Ollama] RAW FILTERED: '{filtered_response}'")
            logger.debug(f"[DEBUG Ollama] Gefilterte Antwort: '{filtered_response}'")
            logger.debug(f"[DEBUG Ollama] Antwortlänge: {len(filtered_response)}")
            
            return filtered_response
            
        except requests.exceptions.RequestException as e:
            logger.error(f"[DEBUG Ollama] HTTP Request fehlgeschlagen: {e}")
            return ""
        except Exception as e:
            logger.error(f"[DEBUG Ollama] Unerwarteter Fehler: {e}")
            return ""
    
    def _filter_thinking_content(self, text: str) -> str:
        """Filtert und gibt den Text zurück"""
        if not text:
            return ""
        
        result = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        return result
    
    def test_connection(self) -> bool:
        """Testet die Ollama-Verbindung"""
        try:
            # Teste mit einfacher Model-Liste Anfrage
            response = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            return response.status_code == 200
        except Exception:
            return False

class MCPRAGClient:
    """RAG-Client für MCP Server-basierte Suche"""
    
    def __init__(self):
        self.base_url = MCP_SEARCH_ENDPOINT
        self.session = requests.Session()
    
    def search(self, query: str, limit: Optional[int] = None) -> List[Dict]:
        """Führt RAG-Suche über MCP Server durch"""
        try:
            if limit is None:
                limit = TOP_K
                
            params = {
                "query": query,
                "top_k": limit,
                "top_tickets": TOP_TICKETS
            }
            
            logger.debug(f"MCP-Server Suche: {query}")
            response = self.session.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            results = data.get("results", [])
            
            # Format für Kompatibilität mit altem Code
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "score": result.get("score", 0.0),
                    "text": result.get("text", ""),
                    "ticket_id": result.get("ticket_id"),
                    "article_id": None,  # MCP Server liefert nicht immer article_id
                    "ticket_url": ""
                })
            
            logger.debug(f"MCP-Server Ergebnisse: {len(formatted_results)} Treffer")
            return formatted_results
            
        except requests.exceptions.RequestException as e:
            logger.error(f"MCP-Server Anfrage fehlgeschlagen: {e}")
            return []
        except Exception as e:
            logger.error(f"Unerwarteter Fehler bei MCP-Server Suche: {e}")
            return []

class ZammadAIClient:
    """Zammad-Client mit KI-Funktionen - Verarbeitet nur Tickets mit genau einem Artikel"""
    
    def __init__(self):
        self.base_url = ZAMMAD_URL
        self.token = ZAMMAD_TOKEN
        self.headers = {
            "Authorization": f"Token token={self.token}",
            "Content-Type": "application/json"
        }
        self.ollama = OllamaClient()
        self.rag = MCPRAGClient()
    
    def get_headers(self) -> Dict[str, str]:
        """Gibt API-Header zurück"""
        return self.headers
    
    def get_open_tickets_without_articles(self, limit: int = 50) -> List[Dict]:
        """Holt offene Tickets mit genau einem Artikel (Erstartikel) - keine Antwort-Tickets"""
        try:
            # Verwende die globale AI_TICKET_MAX_AGE_DAYS Variable
            max_age_days = AI_TICKET_MAX_AGE_DAYS
            
            # URL-Encoding und Suchquery-Parameter
            from urllib.parse import quote
            import requests
            
            # KORREKTE Zammad Search Query basierend auf curl-Tests: state_id:1 OR state_id:2
            search_query = "state_id:1 OR state_id:2"
            encoded_query = quote(search_query)
            
            # RICHTIGER API-Endpoint für Such-Requests
            url = f"{self.base_url}/api/v1/tickets/search?query={encoded_query}"
            
            # Zusätzliche Parameter für Pagination
            params = {
                "limit": limit,
                "per_page": limit
            }
            
            logger.info(f"API-Call mit Search-Endpoint: {url} mit params: {params}")
            response = requests.get(url, headers=self.get_headers(), params=params, timeout=30)
            response.raise_for_status()
            
            # Search-Endpoint gibt laut curl-Tests direkt eine Liste zurück
            response_data = response.json()
            
            # Response ist direkt eine Liste von Tickets
            if isinstance(response_data, list):
                tickets = response_data
            else:
                # Fallback für andere Strukturen
                logger.warning(f"Unerwartete Search-API Response-Struktur: {type(response_data)}")
                logger.warning(f"Response-Content: {str(response_data)[:200]}...")
                tickets = []
            
            logger.info(f"Gefundene Tickets mit Search-Query: {len(tickets)}")
            
            # Client-seitige Filterung für Zeit und Artikel
            processed_tickets = []
            
            logger.info(f"Gefundene Tickets vor manueller Filterung: {len(tickets)}")
            
            for ticket in tickets:
                ticket_id = ticket.get("id")
                state_id = ticket.get("state_id")
                state_name = self._get_state_name(state_id) if state_id else "unknown"
                created_at = ticket.get("created_at", "")
                
                logger.debug(f"Ticket {ticket_id}: state_id={state_id} ({state_name}), erstellt: {created_at}")
                
                # 1. Zeitfilter: prüfe ob Ticket nicht zu alt ist
                is_new_enough = True
                if created_at and max_age_days > 0:
                    try:
                        from datetime import datetime, timezone
                        ticket_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        cutoff_date = datetime.now(timezone.utc) - timedelta(days=max_age_days)
                        is_new_enough = ticket_date >= cutoff_date
                        
                        if not is_new_enough:
                            logger.debug(f"Ticket {ticket_id} ist zu alt: {created_at} < {cutoff_date.isoformat()}")
                            continue
                    except Exception as e:
                        logger.warning(f"Fehler beim Datumsvergleich für Ticket {ticket_id}: {e}")
                        # Bei Datumsfehler, Ticket einschließen
                
                # 2. Artikelfilter: prüfe ob genau ein Artikel vorhanden ist (Erstartikel)
                if self._has_exactly_one_article(ticket_id):
                    processed_tickets.append(ticket)
                    logger.debug(f"Ticket {ticket_id} hat genau einen Artikel - wird verarbeitet")
                else:
                    # Artikelanzahl wird in _has_exactly_one_article bereits geloggt
                    pass
            
            logger.info(f"Verarbeitete Tickets mit genau einem Artikel (max. {max_age_days} Tage alt): {len(processed_tickets)}")
            return processed_tickets
            
        except Exception as e:
            logger.error(f"Fehler beim Abrufen offener Tickets: {e}")
            return []
    
    def _get_state_name(self, state_id: int) -> str:
        """Gibt State-Namen basierend auf State-ID zurück"""
        state_names = {
            1: "new",
            2: "open",
            3: "pending reminder",
            4: "closed",
            5: "merged",
            6: "pending close"
        }
        return state_names.get(state_id, f"unknown({state_id})")
    
    def _has_no_agent_articles(self, ticket_id: int) -> bool:
        """Prüft, ob ein Ticket keine Agent-Antworten hat (nur User-Artikel oder System-Artikel)"""
        try:
            url = f"{self.base_url}/api/v1/ticket_articles/by_ticket/{ticket_id}"
            response = requests.get(url, headers=self.get_headers(), timeout=15)
            
            if response.status_code != 200:
                return True  # Bei Fehlern vorsichtshalber ignorieren
            
            articles = response.json()
            # Prüfe, ob es Agent-Antworten gibt (echte Antworten)
            agent_articles = [a for a in articles if a.get("sender_type", "").lower() == "agent"]
            
            # Zusätzlich prüfen, ob bereits KI-Assistent Antworten vorhanden sind
            ai_responses = [a for a in agent_articles if "🤖 KI-Assistent Antwort" in a.get("body", "")]
            
            if ai_responses:
                logger.debug(f"Ticket {ticket_id} hat bereits KI-Assistent Antworten - wird übersprungen")
                return False  # Ticket hat bereits KI-Antworten
            
            return len(agent_articles) == 0
            
        except Exception as e:
            logger.warning(f"Fehler beim Prüfen der Agent-Artikel für Ticket {ticket_id}: {e}")
            return True
    
    def _has_exactly_one_article(self, ticket_id: int) -> bool:
        """Prüft, ob ein Ticket genau einen Artikel hat (Erstartikel)"""
        try:
            url = f"{self.base_url}/api/v1/ticket_articles/by_ticket/{ticket_id}"
            response = requests.get(url, headers=self.get_headers(), timeout=15)
            
            if response.status_code != 200:
                logger.warning(f"Fehler beim Abrufen der Artikel für Ticket {ticket_id}: HTTP {response.status_code}")
                return False  # Bei Fehlern vorsichtshalber überspringen
            
            articles = response.json()
            article_count = len(articles)
            
            logger.debug(f"Ticket {ticket_id} hat {article_count} Artikel")
            
            if article_count == 1:
                logger.debug(f"Ticket {ticket_id} hat genau einen Artikel - wird verarbeitet")
                return True
            else:
                logger.debug(f"Ticket {ticket_id} hat {article_count} Artikel (nicht genau 1) - wird übersprungen")
                return False
                
        except Exception as e:
            logger.warning(f"Fehler beim Prüfen der Artikelnanzahl für Ticket {ticket_id}: {e}")
            return False
    
    def get_ticket_content(self, ticket_id: int) -> TicketData:
        """Extrahiert Ticket-Inhalte für KI-Verarbeitung"""
        try:
            url = f"{self.base_url}/api/v1/tickets/{ticket_id}"
            response = requests.get(url, headers=self.get_headers(), timeout=15)
            response.raise_for_status()
            
            ticket = response.json()
            
            # Ticket-Informationen sammeln
            title = ticket.get("title", "")
            content = f"Titel: {title}\n"
            
            # Artikel abrufen
            articles_url = f"{self.base_url}/api/v1/ticket_articles/by_ticket/{ticket_id}"
            articles_response = requests.get(articles_url, headers=self.get_headers(), timeout=15)
            
            if articles_response.status_code == 200:
                articles = articles_response.json()
                for article in articles:
                    body = article.get("body", "")
                    if body.strip():
                        content += f"\n---\n{body}"
            
            return TicketData(
                id=ticket_id,
                title=title,
                content=content.strip(),
                created_at=ticket.get("created_at", ""),
                customer_id=ticket.get("customer_id"),
                state=ticket.get("state", "new")
            )
            
        except Exception as e:
            logger.error(f"Fehler beim Extrahieren des Ticket-Inhalts {ticket_id}: {e}")
            return TicketData(id=ticket_id, title="", content="", created_at="")
    
    def create_ai_response(self, ticket: TicketData) -> Optional[str]:
        """Erstellt KI-Antwort für ein Ticket und gibt nur generated_answer zurück"""
        try:
            # 1. Suchbegriff mit Ollama generieren
            search_prompt = RAG_SEARCH_PROMPT.format(ticket_content=ticket.content)
            logger.info(f"[DEBUG Ticket {ticket.id}] Generiere Suchbegriff...")
            search_query = self.ollama.generate(search_prompt)
            logger.info(f"[DEBUG Ticket {ticket.id}] Generierter Suchbegriff: '{search_query[:200]}{'...' if len(search_query) > 200 else ''}'")
            
            if not search_query:
                logger.warning(f"Konnte keinen Suchbegriff für Ticket {ticket.id} generieren")
                return None
            
            # 2. RAG-Suche über MCP Server durchführen
            logger.info(f"[DEBUG Ticket {ticket.id}] Führe RAG-Suche über MCP Server durch...")
            logger.info(f"[DEBUG Ticket {ticket.id}] Verwende TOP_K={TOP_K}, TOP_TICKETS={TOP_TICKETS}")
            rag_results = self.rag.search(search_query)  # Verwende TOP_K aus .env
            logger.info(f"[DEBUG Ticket {ticket.id}] MCP Server Antwort: {len(rag_results)} Ergebnisse erhalten")
            
            if not rag_results:
                logger.warning(f"Keine RAG-Ergebnisse für Ticket {ticket.id} gefunden")
                return None
            
            # 3. Relevante Informationen zusammenstellen
            search_results_text = ""
            for i, result in enumerate(rag_results[:3], 1):
                search_results_text += f"{i}. {result['text'][:200]}...\n"
                logger.info(f"[DEBUG Ticket {ticket.id}] RAG Ergebnis {i}: Score={result['score']:.4f}, Textlänge={len(result['text'])}")
            
            # 4. Antwort mit Ollama generieren
            note_prompt = ZAMMAD_NOTE_PROMPT.format(
                ticket_content=ticket.content,
                search_results=search_results_text
            )
            logger.info(f"[DEBUG Ticket {ticket.id}] Generiere finale Antwort...")
            generated_answer = self.ollama.generate(note_prompt)
            logger.info(f"[DEBUG Ticket {ticket.id}] Generierte Antwort (Länge: {len(generated_answer)} Zeichen)")
            
            if not generated_answer:
                logger.warning(f"Konnte keine Antwort für Ticket {ticket.id} generieren")
                return None
            
            # Nur die generierte Antwort zurückgeben
            return generated_answer
            
        except Exception as e:
            logger.error(f"Fehler bei der KI-Antwort-Erstellung für Ticket {ticket.id}: {e}")
            return None
    
    def add_ai_note(self, ticket_id: int, generated_answer: str) -> bool:
        try:
            url = f"{self.base_url}/api/v1/ticket_articles"
            payload = {
                "ticket_id": ticket_id,
                "subject": "AI Assistant",
                "body": generated_answer,
                "type": "note",              # Notiz
                "internal": True,            # interne Sichtbarkeit -> Rahmen
                "sender": "Agent",           # korrektes Feld laut API
                "content_type": "text/html", # konsistente Darstellung im UI
            }
            response = requests.post(url, headers=self.get_headers(), json=payload, timeout=30)
            response.raise_for_status()
            logger.info(f"KI-Notiz erfolgreich zu Ticket {ticket_id} hinzugefügt")
            return True
        except Exception as e:
            logger.error(f"Fehler beim Hinzufügen der KI-Notiz zu Ticket {ticket_id}: {e}")
            return False

class ZammadAIService:
    """Haupt-Service für automatische Ticket-Verarbeitung"""
    
    def __init__(self):
        self.client = ZammadAIClient()
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.check_interval = AI_CHECK_INTERVAL  # Aus .env lesen
        self.processed_tickets = set()
    
    def start(self):
        """Startet den KI-Service"""
        if self.running:
            logger.info("KI-Service läuft bereits")
            return
        
        # Verbindungstests durchführen
        if not self._test_connections():
            logger.error("Verbindungstests fehlgeschlagen - Service wird nicht gestartet")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        logger.info("Zammad KI-Service gestartet")
    
    def stop(self):
        """Stoppt den KI-Service"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=10)
        logger.info("Zammad KI-Service gestoppt")
    
    def _test_connections(self) -> bool:
        """Testet alle erforderlichen Verbindungen"""
        tests_passed = 0
        total_tests = 2  # Nur Ollama und Zammad testen (Qdrant läuft über MCP Server)
        
        # Test 1: Ollama
        if self.client.ollama.test_connection():
            logger.info("OK Ollama-Verbindung erfolgreich")
            tests_passed += 1
        else:
            logger.error("FAIL Ollama-Verbindung fehlgeschlagen")
        
        # Test 2: Zammad
        try:
            response = requests.get(
                f"{ZAMMAD_URL}/api/v1/tickets?page=1&per_page=1",
                headers=self.client.get_headers(),
                timeout=10
            )
            if response.status_code == 200:
                logger.info("OK Zammad-Verbindung erfolgreich")
                tests_passed += 1
            else:
                logger.error(f"✗ Zammad-Verbindung fehlgeschlagen: HTTP {response.status_code}")
        except Exception as e:
            logger.error(f"✗ Zammad-Verbindung fehlgeschlagen: {e}")
        
        # MCP Server Test (optional - nur warnen wenn nicht verfügbar)
        try:
            mcp_response = requests.get(f"{MCP_SERVER_URL}/health", timeout=5)
            if mcp_response.status_code == 200:
                logger.info("OK MCP Server verfügbar")
            else:
                logger.warning("⚠ MCP Server antwortet nicht korrekt")
        except Exception as e:
            logger.warning(f"⚠ MCP Server nicht erreichbar: {e}")
        
        return tests_passed == total_tests
    
    def _run_loop(self):
        """Haupt-Loop für kontinuierliche Ticket-Verarbeitung"""
        logger.info(f"KI-Service Loop gestartet (Intervall: {self.check_interval}s)")
        
        while self.running:
            try:
                self._process_tickets()
            except Exception as e:
                logger.error(f"Fehler in der Haupt-Loop: {e}")
            
            # Warten bis zum nächsten Check
            for _ in range(self.check_interval):
                if not self.running:
                    break
                time.sleep(1)
        
        logger.info("KI-Service Loop beendet")
    
    def _process_tickets(self):
        """Verarbeitet neue Tickets mit genau einem Artikel (Erstartikel)"""
        try:
            # Offene Tickets mit genau einem Artikel abrufen
            open_tickets = self.client.get_open_tickets_without_articles(limit=20)
            
            if not open_tickets:
                logger.debug("Keine Tickets mit genau einem Artikel gefunden")
                return
            
            logger.info(f"{len(open_tickets)} Tickets mit genau einem Artikel gefunden")
            
            for ticket in open_tickets:
                ticket_id = ticket.get("id")
                
                # Skip tickets with no valid ID
                if ticket_id is None:
                    continue
                
                # Bereits verarbeitete Tickets überspringen
                if ticket_id in self.processed_tickets:
                    logger.debug(f"Ticket {ticket_id} bereits verarbeitet - übersprungen")
                    continue
                
                # Ticket-Inhalt extrahieren
                ticket_data = self.client.get_ticket_content(int(ticket_id))
                
                if not ticket_data.content.strip():
                    logger.warning(f"Ticket {ticket_id} hat keinen verwertbaren Inhalt")
                    self.processed_tickets.add(ticket_id)
                    continue
                
                logger.info(f"Verarbeite Ticket {ticket_id} (Erstartikel)")
                
                # KI-Antwort generieren
                generated_answer = self.client.create_ai_response(ticket_data)
                
                if generated_answer:
                    # Notiz zu Ticket hinzufügen
                    if self.client.add_ai_note(int(ticket_id), generated_answer):
                        logger.info(f"Ticket {ticket_id} erfolgreich mit KI beantwortet")
                    else:
                        logger.error(f"Fehler beim Hinzufügen der KI-Notiz zu Ticket {ticket_id}")
                else:
                    logger.warning(f"Konnte keine KI-Antwort für Ticket {ticket_id} generieren")
                
                # Ticket als verarbeitet markieren
                self.processed_tickets.add(ticket_id)
                
                # Kurze Pause zwischen Tickets
                time.sleep(2)
            
            # Alte Einträge aus processed_tickets entfernen (max 1000)
            if len(self.processed_tickets) > 1000:
                self.processed_tickets = set(list(self.processed_tickets)[-500:])
                
        except Exception as e:
            logger.error(f"Fehler bei der Ticket-Verarbeitung: {e}")
    
    def get_status(self) -> Dict:
        """Gibt aktuellen Service-Status zurück"""
        return {
            "running": self.running,
            "thread_alive": self.thread.is_alive() if self.thread else False,
            "processed_tickets_count": len(self.processed_tickets),
            "check_interval": self.check_interval,
            "ollama_url": OLLAMA_URL,
            "ollama_model": OLLAMA_MODEL,
            "ai_enabled": AI_ENABLED,
            "mcp_server_url": MCP_SERVER_URL,
            "top_k": TOP_K,
            "top_tickets": TOP_TICKETS
        }

# Globale Service-Instanz
ai_service = ZammadAIService()

def start_ai_service():
    """Startet den KI-Service (für externe Aufrufe)"""
    if not AI_ENABLED:
        logger.info("KI-Features sind deaktiviert (AI_ENABLED=false)")
        return False
    
    ai_service.start()
    return True

def stop_ai_service():
    """Stoppt den KI-Service (für externe Aufrufe)"""
    ai_service.stop()

def get_ai_service_status():
    """Gibt KI-Service-Status zurück (für externe Aufrufe)"""
    return ai_service.get_status()

if __name__ == "__main__":
    # Direkter Aufruf - Service starten
    logger.info("Starte Zammad KI-Service...")
    
    if not AI_ENABLED:
        logger.error("KI-Features sind deaktiviert. Setzen Sie AI_ENABLED=true in .env")
        exit(1)
    
    if start_ai_service():
        logger.info("Zammad KI-Service läuft. Drücken Sie Ctrl+C zum Beenden.")
        try:
            # Haupt-Thread am Leben halten
            while True:
                time.sleep(60)
                # Status alle 5 Minuten loggen
                status = get_ai_service_status()
                logger.info(f"KI-Service Status: {status}")
        except KeyboardInterrupt:
            logger.info("Beende Zammad KI-Service...")
            stop_ai_service()
            logger.info("Zammad KI-Service beendet")
    else:
        logger.error("KI-Service konnte nicht gestartet werden")
        exit(1)