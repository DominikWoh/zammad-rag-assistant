import os
import time
import requests
import json
import traceback
import re
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# === Einheitliches .env-Handling für Docker ===
ENV_FILE = os.getenv("ENV_FILE", "/data/config/ticket_ingest.env")
if os.path.isfile(ENV_FILE):
    load_dotenv(ENV_FILE)

# Prompt-ENV-Dateien aus /data/config/prompts laden (override=True)
PROMPTS_DIR = os.getenv("PROMPTS_DIR", "/data/config/prompts")
if os.path.isdir(PROMPTS_DIR):
    for fname in ("askai.env", "rag_prompt.env"):
        fpath = os.path.join(PROMPTS_DIR, fname)
        if os.path.isfile(fpath):
            load_dotenv(fpath, override=True)

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OLLAMA_URL = os.getenv("OLLAMA_URL")
if OLLAMA_URL:
    base = OLLAMA_URL.rstrip('/')
    # Nur anhängen, wenn klar kein API-Pfad vorhanden ist
    if not base.endswith('/api/chat') and '/api/' not in base:
        OLLAMA_URL = base + '/api/chat'
    else:
        OLLAMA_URL = base
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
ZAMMAD_URL = os.getenv("ZAMMAD_URL")
ZAMMAD_TOKEN = os.getenv("ZAMMAD_TOKEN")
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")
HUGGINGFACE_CACHE_DIR = os.getenv("HUGGINGFACE_CACHE_DIR", "/data/cache")

# Feature-Flags aus .env mit Defaults: ENABLE_ASKKI=false, ENABLE_RAG_NOTE=true
def _to_bool(val, default=False):
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in ("true", "1", "yes", "on", "y", "ja"):
        return True
    if s in ("false", "0", "no", "off", "n", "nein"):
        return False
    return default

def _read_flags_from_env():
    """Liest Flags aus aktueller ENV (robustes Parsing)."""
    return {
        "ENABLE_ASKKI": _to_bool(os.getenv("ENABLE_ASKKI"), default=False),
        "ENABLE_RAG_NOTE": _to_bool(os.getenv("ENABLE_RAG_NOTE"), default=True),
    }

# Initiale Flags laden
_flags = _read_flags_from_env()
ENABLE_ASKKI = _flags["ENABLE_ASKKI"]
ENABLE_RAG_NOTE = _flags["ENABLE_RAG_NOTE"]
print(f"[Flags@import] ENABLE_ASKKI={ENABLE_ASKKI}, ENABLE_RAG_NOTE={ENABLE_RAG_NOTE}")

client = QdrantClient(QDRANT_URL, api_key=QDRANT_API_KEY)
model = SentenceTransformer(EMBED_MODEL, cache_folder=HUGGINGFACE_CACHE_DIR)
HEADERS = {"Authorization": f"Token token={ZAMMAD_TOKEN}"}

# Default-Prompts als Fallbacks
DEFAULT_ASKAI_PROMPT = (
    "Du bist ein IT-Support-Experte. Bitte beantworte die folgende Frage und erstelle für den IT Profi eine mögliche Lösung ohne Floskeln.\n\n"
    "Frage bzw. Anfrage:\n---\n{user_text}\n---\n\n"
    "WICHTIG: Gib ausschließlich die finale Antwort zurück. Keine Meta-Erklärungen, keine Denkprozesse, keine Tags wie <think> ... </think>."
)

DEFAULT_RAG_PROMPT = (
    "Du bist ein IT-Support-Experte. Antworte auf die folgende Anfrage so hilfreich wie möglich (auf Deutsch) ohne Floskeln.\n"
    "Du schreibst die (mögliche) Lösung für einen IT Profi, als Notiz, damit dieser das Problem schneller lösen kann.\n"
    "Mache keine Markdown. Es gibt nur Zeilenumbrüche.\n"
    "Nutze dein eigenes Wissen und – nur wenn relevant – das folgende Ticket‑Wissen (Knowledge) aus ähnlichen Fällen:\n\n"
    "Benutzeranfrage:\n---\n{combined_text}\n---\n\n"
    "Hilfreiches Ticket‑Wissen:\n---\n{knowledge}\n---\n\n"
    "WICHTIG: Gib ausschließlich die finale Antwort zurück. Keine Meta-Erklärungen, keine Denkprozesse, keine Tags wie <think> ... </think>."
)

def expand_query_with_llm(query, max_attempts=3):
    template = os.getenv("QUERY_EXPANSION_PROMPT",
        'Du bist ein IT‑Support‑Experte. Erzeuge drei alternative kurze Suchsätze (Suchanfragen), die das Problem aus unterschiedlichen Perspektiven oder mit anderen Formulierungen ausdrücken, es sollten schon ein deutlicher unterschied sein aber mit gleicher intention.\nFrage:\n"{query}"\n- Gib ausschließlich eine Python‑Liste aus, z. B.: ["Alternative 1", "Alternative 2", "Alternative 3"]\n- Kein Markdown, keine ```python, keine Codeblöcke.'
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
    # Sicherheitsnetz: Keine Denk-Tags oder Meta-Inhalte zulassen
    prompt += "\n\nAntwort-Regel: Nur die finale Antwort zurückgeben. Keine Denkprozesse, keine <think>-Tags."
    return str(prompt).strip()

def _remove_thinking_blocks(text: str) -> str:
    """
    Entfernt Denk-/Reasoning-Blöcke wie:
    - XML/HTML: <think>...</think> (case-insensitive)
    - Präfixe: 'thinking:', 'reasoning:', 'thought:' etc. inkl. Folgeinhalt
    Gibt ausschließlich den bereinigten finalen Antworttext zurück.
    """
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return ""

    # 1) Alle <think>...</think> Blöcke entfernen
    text = re.sub(r"<\s*think\s*>.*?<\s*/\s*think\s*>", "", text, flags=re.IGNORECASE | re.DOTALL)

    # 2) Alles nach Indikatoren abschneiden
    indicators = re.compile(r"(?:^|\n|\r)(thinking|reasoning|thought|denk[^\s]*)\s*:?.*$", re.IGNORECASE | re.DOTALL)
    m = indicators.search(text)
    if m:
        text = text[:m.start()].rstrip()

    # 3) Whitespace aufräumen
    return text.strip()


def call_llm(prompt, max_attempts=3):
    # Sicherstellen, dass 'prompt' ein String ist
    prompt = str(prompt)

    data = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Du bist ein IT-Support-Experte. Antworte ausschließlich mit der finalen Lösung/Antwort "
                    "ohne Denkprozesse, Meta-Kommentare oder Tags wie <think>...</think>."
                )
            },
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "max_tokens": 800,
        "temperature": 0.1,
    }

    for attempt in range(1, max_attempts + 1):
        try:
            print("Sending request to Ollama...")
            res = requests.post(OLLAMA_URL, json=data, timeout=220)
            res.raise_for_status()
            response_json = res.json()
            print("Ollama JSON Response:", response_json)
            content = (response_json.get('message', {}) or {}).get('content', '')
            content = (content or '').strip()
            if content:
                content = _remove_thinking_blocks(content)
                print("💬 LLM-Antwort bereinigt.")
                return content
        except requests.exceptions.HTTPError as e:
            print(f"⚠️ HTTPError (Versuch {attempt}):", e)
            try:
                print(f"Response: {e.response.text}")
            except Exception:
                pass
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
    # Sicherheitsnetz: Keine Denk-Tags oder Meta-Inhalte zulassen
    prompt += "\n\nAntwort-Regel: Nur die finale Antwort zurückgeben. Keine Denkprozesse, keine <think>-Tags."
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

def process_tickets(stop_event=None, status_holder=None):
    global ENABLE_ASKKI, ENABLE_RAG_NOTE  # vor erster Nutzung deklarieren
    print("👀 Starte Zammad RAG-Poller ...")

    # Beim Start die effektiv geladenen Flags ausgeben
    try:
        print(f"[Flags@start] ENABLE_ASKKI={ENABLE_ASKKI}, ENABLE_RAG_NOTE={ENABLE_RAG_NOTE}, ENV_FILE={ENV_FILE}")
    except Exception:
        pass

    last_flag_reload = 0.0
    HOT_RELOAD_SECS = 10.0  # Flags alle 10s neu einlesen

    while True:
        if stop_event is not None and stop_event.is_set():
            print("⏹️ Stop-Event empfangen – Poller beendet.")
            break
        try:
            # Optionaler Live-Reload der ENV, damit Änderungen aus /data/config/ticket_ingest.env greifen
            try:
                if os.path.isfile(ENV_FILE):
                    load_dotenv(ENV_FILE, override=True)
            except Exception:
                pass

            # Flags hot-reloaden (alle 10s)
            now = time.time()
            if (now - last_flag_reload) >= HOT_RELOAD_SECS:
                _f = _read_flags_from_env()
                if _f["ENABLE_ASKKI"] != ENABLE_ASKKI or _f["ENABLE_RAG_NOTE"] != ENABLE_RAG_NOTE:
                    print(f"[Flags@reload] ENABLE_ASKKI {ENABLE_ASKKI} -> {_f['ENABLE_ASKKI']}, ENABLE_RAG_NOTE {ENABLE_RAG_NOTE} -> {_f['ENABLE_RAG_NOTE']}")
                ENABLE_ASKKI = _f["ENABLE_ASKKI"]
                ENABLE_RAG_NOTE = _f["ENABLE_RAG_NOTE"]
                last_flag_reload = now

            tickets = fetch_new_and_open_tickets(ZAMMAD_URL, ZAMMAD_TOKEN)
            if tickets:
                for ticket in tickets:
                    try:
                        ticket_id = ticket["id"]
                        print(f"📋 Ticket {ticket_id} - Title: {ticket['title']}")
                        prompt_or_rag = should_process_ticket(ticket)
                        if isinstance(prompt_or_rag, str) and prompt_or_rag == "RAG":
                            if not ENABLE_RAG_NOTE:
                                print(f"ℹ️ RAG deaktiviert – Ticket {ticket_id} wird ohne Notiz übersprungen.")
                            else:
                                user_text = build_full_ticket_text(ticket)
                                rag_hits = search_qdrant(user_text)
                                prompt = build_rag_prompt(ticket, rag_hits)
                                reply = call_llm(prompt)
                                if reply:
                                    post_note_to_ticket(ticket_id, reply)
                                    print(f"✅ Antwort bei Ticket {ticket_id} gespeichert.")
                                else:
                                    print("❌ Keine KI-Antwort erhalten – Ticket bleibt unbearbeitet.")
                        elif prompt_or_rag:
                            if not ENABLE_ASKKI:
                                print(f"ℹ️ ASKKI deaktiviert (ENABLE_ASKKI=false) – 'AskAI'-Trigger ignoriert (Ticket {ticket_id}).")
                            else:
                                reply = call_llm(prompt_or_rag)
                                if reply:
                                    post_note_to_ticket(ticket_id, reply)
                                    print(f"✅ Antwort bei Ticket {ticket_id} gespeichert.")
                                else:
                                    print("❌ Keine KI-Antwort erhalten – Ticket bleibt unbearbeitet.")
                    except Exception as inner_exception:
                        print(f"❌ Fehler beim Verarbeiten von Ticket {ticket.get('id', 'unbekannt')}: {inner_exception}")
                        print(traceback.format_exc())
            if status_holder is not None:
                status_holder["last_tick"] = time.time()
        except Exception as e:
            print(f"❌ Fehler beim Abrufen der Tickets: {e}")
        # Warten mit vorzeitigem Abbruch
        if stop_event is not None and stop_event.wait(60.0):
            print("⏹️ Stop-Event während Wartezeit – Poller beendet.")
            break

# Hauptprogramm ausführen
if __name__ == "__main__":
    process_tickets()
