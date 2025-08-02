import os
import time
import requests
import json
import traceback
import re
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# === .env laden (plattformübergreifend, wie Web-UI) ===
WINDOWS_ENV_PATH = os.path.join(os.getcwd(), ".env")
UBUNTU_ENV_PATH = "/opt/ai-suite/ticket_ingest.env"

def detect_env_path():
    if os.path.isfile(WINDOWS_ENV_PATH):
        return WINDOWS_ENV_PATH
    elif os.path.isfile(UBUNTU_ENV_PATH):
        return UBUNTU_ENV_PATH
    else:
        open(WINDOWS_ENV_PATH, "a", encoding="utf-8").close()
        return WINDOWS_ENV_PATH

ENV_PATH = detect_env_path()
load_dotenv(ENV_PATH)

# Prompt-ENV-Dateien aus dem Services-Ordner laden (override=True)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_THIS_DIR, "askai.env"), override=True)
load_dotenv(os.path.join(_THIS_DIR, "rag_prompt.env"), override=True)

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OLLAMA_URL = os.getenv("OLLAMA_URL")
if OLLAMA_URL and not OLLAMA_URL.rstrip('/').endswith('/api/chat'):
  OLLAMA_URL = OLLAMA_URL.rstrip('/') + '/api/chat'
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
ZAMMAD_URL = os.getenv("ZAMMAD_URL")
ZAMMAD_TOKEN = os.getenv("ZAMMAD_TOKEN")
EMBED_MODEL = "intfloat/multilingual-e5-base"

# Feature-Flags aus .env mit Defaults: ENABLE_ASKKI=false, ENABLE_RAG_NOTE=true
def _to_bool(val, default=False):
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in ("true", "1", "yes", "on"):
        return True
    if s in ("false", "0", "no", "off"):
        return False
    return default

ENABLE_ASKKI = _to_bool(os.getenv("ENABLE_ASKKI"), default=False)
ENABLE_RAG_NOTE = _to_bool(os.getenv("ENABLE_RAG_NOTE"), default=True)
print(f"[Flags] ENABLE_ASKKI={ENABLE_ASKKI}, ENABLE_RAG_NOTE={ENABLE_RAG_NOTE}")

client = QdrantClient(QDRANT_URL, api_key=QDRANT_API_KEY)
model = SentenceTransformer(EMBED_MODEL, cache_folder='/opt/ai-suite/.cache/huggingface')
HEADERS = {"Authorization": f"Token token={ZAMMAD_TOKEN}"}

# Default-Prompts als Fallbacks
DEFAULT_ASKAI_PROMPT = (
    "Du bist ein IT-Support-Experte. Bitte beantworte die folgende Frage und erstelle für den IT Profi eine mögliche Lösung ohne Floskeln.\n\n"
    "Frage bzw. Anfrage:\n---\n{user_text}\n---\n\n"
    "Antworte möglichst konkret und hilfreich, mache kein Markdown, verwende nur Zeilenumbrüche."
)

DEFAULT_RAG_PROMPT = (
    "Du bist ein IT-Support-Experte. Antworte auf die folgende Anfrage so hilfreich wie möglich (auf Deutsch) ohne Floskeln.\n"
    "Du schreibst die (mögliche) Lösung für einen IT Profi, als Notiz, damit dieser das Problem schneller lösen kann.\n"
    "Mache keine Markdown. Es gibt nur Zeilenumbrüche.\n"
    "Nutze dein eigenes Wissen und, aber nur wenn relevant, das folgende Ticket-Wissen (Knowledge) aus ähnlichen Fällen:\n\n"
    "Benutzeranfrage:\n---\n{combined_text}\n---\n\n"
    "Hilfreiches Ticket-Wissen:\n---\n{knowledge}\n---\n\n"
    "Antworte möglichst konkret, aber nur wenn sinnvoll und fachlich korrekt."
)

def expand_query_with_llm(query, max_attempts=3):
    template = os.getenv("QUERY_EXPANSION_PROMPT",
        'Du bist ein IT‑Support‑Experte. Erzeuge drei alternative kurze Suchsätze (Suchanfragen), die das Problem aus unterschiedlichen Perspektiven oder mit anderen Formulierungen ausdrücken.\nFrage:\n"{query}"\n- Gib ausschließlich eine Python‑Liste aus, z. B.: ["Alternative 1", "Alternative 2", "Alternative 3"]\n- Kein Markdown, keine ```python, keine Codeblöcke.'
    )
    prompt = template.format(query=query)
    for attempt in range(1, max_attempts + 1):
        try:
            res = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [
                        {"role": "system", "content": "Du bist hilfreich."},
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False
                },
                timeout=220
            )
            res.raise_for_status()
            response_json = res.json()
            print("Ollama JSON Response:", response_json)
            raw = response_json['message']['content']
            raw = raw.strip()
            print("=== LLM-RAW-ANTWORT BEGIN ===")
            print(raw)
            print("=== LLM-RAW-ANTWORT ENDE ===")
            match = re.search(r'\[(?:.|\n)*?\]', raw)
            if not match:
                raise ValueError(f"Keine Python-Liste in der LLM-Antwort gefunden: {raw}")
            alt = json.loads(match.group(0))
            if isinstance(alt, list):
                print(f"🔎 Query-Expansion erfolgreich: {alt}")
                return alt
        except Exception as e:
            print(f"⚠️ Fehler bei Query-Expansion (Versuch {attempt}): {e}")
            print(traceback.format_exc())
            time.sleep(2)
    print("⛔️ Keine Query-Expansion erhalten.")
    return []

def multi_vector_search(vec):
    hits = []
    for field in ["kurzbeschreibung", "beschreibung", "lösung", "all"]:
        try:
            results = client.query_points(
                collection_name=COLLECTION_NAME,
                query=vec.tolist(),
                using=field,
                limit=5,
                with_payload=True
            ).points
            hits.extend(results)
        except Exception:
            print(f"⚠️ Fehler bei Multi-Vector-Search (Field: {field})")
            traceback.print_exc()
    unique = {}
    for h in hits:
        if h.id not in unique or h.score > unique[h.id].score:
            unique[h.id] = h
    return sorted(unique.values(), key=lambda x: x.score, reverse=True)[:5]

def search_qdrant(query: str):
    variants = [query] + expand_query_with_llm(query)
    all_hits = []
    for v in variants:
        vec = model.encode(v, normalize_embeddings=True)
        all_hits += multi_vector_search(vec)
    seen = set()
    final = []
    for h in sorted(all_hits, key=lambda x: x.score, reverse=True):
        if h.id not in seen:
            seen.add(h.id)
            final.append(h)
    return final[:5]  # Gib die ersten 5 Ergebnisse zurück

def clean_html_to_text(html):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    for bq in soup.find_all("blockquote"):
        bq.decompose()
    for img in soup.find_all("img"):
        img.replace_with("[Bildanhang entfernt]")
    text = soup.get_text(separator="\n")
    return "\n".join([line.strip() for line in text.splitlines() if line.strip()])

def fetch_ticket_articles(ticket):
    """Abrufen der Artikel für ein Ticket"""
    articles = []
    for article_id in ticket["article_ids"]:
        article_url = f"{ZAMMAD_URL}/api/v1/ticket_articles/{article_id}"
        try:
            response = requests.get(article_url, headers=HEADERS)
            if response.status_code == 200:
                articles.append(response.json())
            else:
                print(f"❌ Fehler beim Abrufen des Artikels {article_id}: {response.status_code}")
        except Exception as e:
            print(f"❌ Fehler bei der Anfrage des Artikels {article_id}: {e}")
    return articles


def build_full_ticket_text(ticket):
    """Baue einen vollständigen Verlaufstext aus allen Ticket-Artikeln."""
    if isinstance(ticket, dict):  # Sicherstellen, dass ticket ein Dictionary ist
        # Abrufen der Artikel
        articles = fetch_ticket_articles(ticket)
        
        if not articles:
            print(f"❌ Keine Artikel im Ticket {ticket['id']} gefunden.")
            return ""  # Rückgabe eines leeren Strings, wenn keine Artikel vorhanden sind

        out = []
        for a in articles:
            created = a.get("created_at", "")[:19]
            sender = a.get("from", "") or a.get("type", "")
            typ = a.get("type", "")
            subject = a.get("subject", "")
            body = clean_html_to_text(a.get("body", ""))
            out.append(
                f"[{created} | {sender} | {typ} | {subject}]\n{body}\n"
            )
        return "\n---\n".join(out)
    else:
        print(f"❌ Fehler: ticket ist ein {type(ticket)} und kein Dictionary. Ticket-Daten: {ticket}")
        return ""  # Rückgabe eines leeren Strings, falls ticket kein Dictionary ist


def build_rag_prompt(ticket, rag_hits):
    """Erstellt das RAG-Prompt unter Verwendung des gesamten Tickettextes und des Ticket-Titels."""
    ticket_title = ticket.get("title", "Kein Titel verfügbar")
    user_text = build_full_ticket_text(ticket)
    combined_text = f"Ticket-Titel: {ticket_title}\n\nTicket-Inhalt:\n{user_text}"

    hit_blocks = []
    for hit in rag_hits:
        pl = hit.payload
        kurz = pl.get("kurzbeschreibung", "")
        beschr = pl.get("beschreibung", "")
        loesung = pl.get("lösung", "")
        if kurz or beschr or loesung:
            hit_blocks.append(f"[Problem] {kurz}\n[Beschreibung] {beschr}\n[Lösung] {loesung}")

    knowledge = "\n\n".join(hit_blocks) if hit_blocks else "[Keine ähnlichen Tickets gefunden]"

    template = os.getenv("RAG_PROMPT", DEFAULT_RAG_PROMPT)
    prompt = template.format(combined_text=combined_text, knowledge=knowledge)
    print("User Text (inkl. Titel):", combined_text)
    print("Knowledge:", knowledge)
    return str(prompt).strip()

def call_llm(prompt, max_attempts=3):
    # Sicherstellen, dass 'prompt' ein String ist
    prompt = str(prompt)

    data = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": "Du bist ein IT-Support-Experte."},
            {"role": "user", "content": prompt}  # Sicherstellen, dass 'prompt' ein String ist
        ],
        "stream": False,
        "max_tokens": 800,
        "temperature": 0.1,
    }

    for attempt in range(1, max_attempts + 1):
        try:
            print("Sending request to Ollama...")
            res = requests.post(OLLAMA_URL, json=data, timeout=220)  # Timeout auf 220 Sekunden
            res.raise_for_status()
            response_json = res.json()
            print("Ollama JSON Response:", response_json)
            content = response_json['message']['content'].strip()
            if content:
                print("💬 LLM-Antwort erhalten.")
                return content
        except requests.exceptions.HTTPError as e:
            print(f"⚠️ HTTPError (Versuch {attempt}):", e)
            print(f"Response: {e.response.text}")
        except requests.exceptions.Timeout as e:
            print(f"⚠️ TimeoutError (Versuch {attempt}):", e)
        except Exception as e:
            print(f"⚠️ Fehler bei LLM (Versuch {attempt}):", e)
            print(traceback.format_exc())
            time.sleep(2)

    print("⛔️ Keine Antwort vom LLM erhalten.")
    return ""


def post_note_to_ticket(ticket_id, text):
    data = {
        "ticket_id": ticket_id,
        "subject": "Automatische KI-Antwort",
        "body": text,
        "type": "note",
        "internal": True  # "False" = öffentlich (Kunde sieht Antwort)
    }
    try:
        # Verwende den richtigen Endpunkt: /ticket_articles
        res = requests.post(
            f"{ZAMMAD_URL}/api/v1/ticket_articles",
            headers=HEADERS,
            json=data
        )
        if res.status_code == 201:
            print(f"✅ Notiz bei Ticket {ticket_id} gespeichert.")
        else:
            print(f"❌ Fehler beim Speichern der Notiz bei Ticket {ticket_id}: {res.status_code}, {res.text}")
    except Exception as e:
        print(f"❌ HTTP-Fehler beim Posten der Notiz: {e}")

def should_process_ticket(ticket):
    article_count = ticket.get("article_count", 0)
    ticket_state = ticket.get("state_id", None)

    # 1. Wenn nur ein Artikel existiert und das Ticket neu oder offen ist
    if article_count == 1 and ticket_state in [1, 2]:  # state.id == 1 (neu) oder 2 (offen)
        if ENABLE_RAG_NOTE:
            print(f"⚠️ Ticket {ticket['id']} hat nur einen Artikel und ist neu oder offen. Gehe zu RAG.")
            return "RAG"  # RAG aktivieren für Tickets mit einem Artikel
        else:
            print(f"ℹ️ RAG deaktiviert (ENABLE_RAG_NOTE=false) – Ticket {ticket['id']} wird übersprungen.")
            return False

    # 2. Wenn mehrere Artikel vorhanden sind, aber der letzte Artikel "AskAI" enthält
    elif article_count > 1:
        last_article_id = ticket["article_ids"][-1]
        last_article_url = f"{ZAMMAD_URL}/api/v1/ticket_articles/{last_article_id}"
        try:
            # Artikel abrufen und überprüfen
            response = requests.get(last_article_url, headers=HEADERS)
            if response.status_code == 200:
                last_article = response.json()
                last_body = last_article.get("body", "").lower()
                if "askai" in last_body:
                    if ENABLE_ASKKI:
                        print(f"⚠️ 'AskAI' im letzten Artikel von Ticket {ticket['id']} gefunden. Gehe zu LLM.")
                        prompt = build_askai_prompt(last_article)  # Übergabe des letzten Artikels an das LLM
                        return prompt  # Rückgabe des angepassten Prompts für den 'AskAI' Trigger
                    else:
                        print(f"ℹ️ ASKKI deaktiviert (ENABLE_ASKKI=false) – 'AskAI'-Trigger ignoriert (Ticket {ticket['id']}).")
                        return False
            else:
                print(f"❌ Fehler beim Abrufen des letzten Artikels von Ticket {ticket['id']}")

        except Exception as e:
            print(f"❌ Fehler bei der Anfrage des letzten Artikels von Ticket {ticket['id']}: {e}")

    return False  # Kein LLM erforderlich, falls kein 'AskAI' Trigger

def build_askai_prompt(last_article):
    """Erstellt den Prompt für die LLM-Anfrage, wenn 'AskAI' im letzten Artikel vorkommt."""
    user_text = f"{last_article['body']}"  # Nur der letzte Artikel
    user_text_cleaned = clean_html_to_text(user_text)
    template = os.getenv("ASKAI_PROMPT", DEFAULT_ASKAI_PROMPT)
    prompt = template.format(user_text=user_text_cleaned)
    print("User Text:", user_text_cleaned)
    return str(prompt).strip()

def fetch_new_and_open_tickets(api_url, api_token):
    headers = {
        "Authorization": f"Token token={api_token}"
    }
    url = f"{api_url}/api/v1/tickets/search?query=state.id%3A1%20OR%20state.id%3A2&fields=id,title"
    
    # Anfrage an die API
    response = requests.get(url, headers=headers)
    
    # Überprüfen, ob die Anfrage erfolgreich war
    if response.status_code == 200:
        tickets = response.json()  # Konvertiere die Antwort in JSON
        return tickets
    else:
        print(f"Fehler beim Abrufen der Tickets: {response.status_code}")
        return None

def process_tickets():
    print("👀 Starte Zammad RAG-Poller ...")
    while True:
        try:
            tickets = fetch_new_and_open_tickets(ZAMMAD_URL, ZAMMAD_TOKEN)
            if tickets:
                for ticket in tickets:
                    try:
                        ticket_id = ticket["id"]
                        print(f"📋 Ticket {ticket_id} - Title: {ticket['title']}")

                        # Falls 'AskAI' gefunden wird, gehe zu LLM
                        prompt_or_rag = should_process_ticket(ticket)
                        
                        if isinstance(prompt_or_rag, str) and prompt_or_rag == "RAG":  # RAG aktivieren
                            if not ENABLE_RAG_NOTE:
                                print(f"ℹ️ RAG deaktiviert – Ticket {ticket_id} wird ohne Notiz übersprungen.")
                            else:
                                user_text = build_full_ticket_text(ticket)
                                print("User Text (Ticket):", user_text)

                                # Suche nach ähnlichen Tickets und relevanten Daten
                                rag_hits = search_qdrant(user_text)

                                # Generiere das RAG-Prompt und rufe das LLM auf
                                prompt = build_rag_prompt(ticket, rag_hits)
                                reply = call_llm(prompt)  # LLM-Antwort erhalten
                                if reply:
                                    post_note_to_ticket(ticket_id, reply)
                                    print(f"✅ Antwort bei Ticket {ticket_id} gespeichert.")
                                else:
                                    print("❌ Keine KI-Antwort erhalten – Ticket bleibt unbearbeitet.")

                        elif prompt_or_rag:  # Wenn 'AskAI' gefunden wurde, wird hier das LLM direkt aufgerufen
                            if not ENABLE_ASKKI:
                                print(f"ℹ️ ASKKI deaktiviert – Ticket {ticket_id} wird ohne Notiz übersprungen.")
                            else:
                                reply = call_llm(prompt_or_rag)  # LLM-Antwort erhalten
                                if reply:
                                    post_note_to_ticket(ticket_id, reply)
                                    print(f"✅ Antwort bei Ticket {ticket_id} gespeichert.")
                                else:
                                    print("❌ Keine KI-Antwort erhalten – Ticket bleibt unbearbeitet.")
                    except Exception as inner_exception:
                        print(f"❌ Fehler beim Verarbeiten von Ticket {ticket.get('id', 'unbekannt')}: {inner_exception}")
                        print(traceback.format_exc())
        except Exception as e:
            print(f"❌ Fehler beim Abrufen der Tickets: {e}")
        time.sleep(60)  # Alle 60 Sekunden wiederholen

# Hauptprogramm ausführen
if __name__ == "__main__":
    process_tickets()
