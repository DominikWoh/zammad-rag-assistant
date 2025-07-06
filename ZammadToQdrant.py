import time
import json
import re
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from qdrant_client.models import VectorParams, Distance
from sentence_transformers import SentenceTransformer
import requests
from datetime import datetime, timedelta, timezone
from bs4 import BeautifulSoup

MIN_CLOSED_DAYS = 14  # Mindestanzahl Tage geschlossen

# .env laden
load_dotenv("/opt/ai-suite/ticket_ingest.env")

# ---- KONFIG AUS .env ----
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OLLAMA_URL = os.getenv("OLLAMA_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
ZAMMAD_URL = os.getenv("ZAMMAD_URL")
ZAMMAD_TOKEN = os.getenv("ZAMMAD_TOKEN")
EMBED_MODEL = "intfloat/multilingual-e5-base"
MIN_TICKET_DATE = datetime(2025, 1, 1, tzinfo=timezone.utc)
MAX_ATTEMPTS = 1

client = QdrantClient(QDRANT_URL, api_key=QDRANT_API_KEY)
model = SentenceTransformer(EMBED_MODEL)

if not client.collection_exists(collection_name=COLLECTION_NAME):
    print(f"📦 Erstelle Collection '{COLLECTION_NAME}' ...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "kurzbeschreibung": VectorParams(size=768, distance=Distance.COSINE),
            "beschreibung": VectorParams(size=768, distance=Distance.COSINE),
            "lösung": VectorParams(size=768, distance=Distance.COSINE),
            "all": VectorParams(size=768, distance=Distance.COSINE),
        }
    )
    print("Collection wurde neu erstellt.")
else:
    print(f"📦 Collection '{COLLECTION_NAME}' existiert bereits.")

print("Collection Info:", client.get_collection(COLLECTION_NAME))


def html_to_text(html):
    soup = BeautifulSoup(html, "html.parser")

    # Entferne blockquotes (z.B. Zitate aus vorherigen Mails)
    for bq in soup.find_all("blockquote"):
        bq.decompose()

    # Bilder durch Platzhalter ersetzen
    for img in soup.find_all("img"):
        img.replace_with("[Bildanhang entfernt]")

    # Text extrahieren – mit einfachem Zeilenumbruch
    lines = soup.get_text(separator="\n").splitlines()

    # Leere Zeilen entfernen, trimmen
    clean_lines = [line.strip() for line in lines if line.strip()]

    return clean_lines  # Liste von Zeilen, die dann später mit '\n'.join(...) verbunden werden


def call_llm(prompt, max_tokens=900, temperature=0.1, max_attempts=MAX_ATTEMPTS):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": "Du bist ein IT-Wissensdatenbank-Experte."},
            {"role": "user", "content": prompt}
        ],
        "stream": False,  # ganz wichtig!
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    for attempt in range(max_attempts):
        try:
            res = requests.post(OLLAMA_URL, headers=headers, json=payload, timeout=600)
            res.raise_for_status()

            try:
                response_json = res.json()
                content = response_json["message"]["content"].strip()
                print("🔍 LLM-Antwort:", content)
                return content
            except Exception as json_err:
                print("❌ JSON-Parsing-Fehler:", json_err)
                print("⬇️ Antwort war:")
                print(res.text)
                return None

        except Exception as e:
            print(f"⚠️ LLM Request fehlgeschlagen (Versuch {attempt+1}/{max_attempts}): {e}")
            if attempt < max_attempts - 1:
                print("→ Neuer Versuch in 2 Sekunden ...")
                time.sleep(2)
                continue
            else:
                print("❌ LLM Request endgültig fehlgeschlagen!")
                return None


def clean_llm_json_answer(answer):
    cleaned = re.sub(r"```[a-z]*\n?", "", answer, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```", "", cleaned).strip()
    return cleaned

def is_ticket_helpful(ticket):
    print(ticket['full_text'])
    prompt = f"""
Du bist IT-Support-Experte. Beantworte mit "JA" oder "NEIN":
- Ist dieses Ticket hilfreich für IT Admins, weil es eine konkrete Lösung, einen Workaround oder eine nachvollziehbare Fehlerursache oder zumindest weitere Details die zur Lösungsfindung beitragen können enthält?
- Wenn nur das Problem beschrieben ist, aber keine Lösung/Workaround/Fix genannt wird, antworte mit "NEIN".
- Wenn es im Ticket nur steht sinngemäß steht Software/Hardware XY wurde korrigiert, behoben, installiert, neu installiert, aktualisiert etc. aber keine Details genannt werden, antworte mit "NEIN".
- Wenn im Ticket von [SUSPECTED PHISHING] die Rede ist, antworte mit "NEIN".

Ticket-Text:
---
{ticket['full_text']}
---
Antworte ausschließlich mit "JA" oder "NEIN".
"""
    for attempt in range(MAX_ATTEMPTS):
        print(f"🔎 [LLM] Prüfe auf Relevanz/Lösung (Versuch {attempt+1}) ...")
        answer = call_llm(prompt, max_tokens=5, temperature=0.0)
        if not answer:
            print(f"⛔️ Keine Antwort vom LLM bei Relevanzprüfung (Versuch {attempt+1}) – wiederhole...")
            time.sleep(1)
            continue
        answer = answer.upper()
        if answer == "JA":
            print("✅ Ticket ist relevant/hilfreich.")
            return True
        if answer == "NEIN":
            print("✅ Ticket nicht hilfreich – wird übersprungen.")
            return False
        print(f"⛔️ Ungültige LLM-Antwort (Versuch {attempt+1}): '{answer}' – wiederhole...")
        time.sleep(1)
    print(f"⏹️ Konnte keine gültige JA/NEIN-Antwort vom LLM bekommen. Ticket wird übersprungen.")
    return False

def summarize_ticket_to_json(ticket):
    prompt = f"""
Du bist ein IT-Support-Experte und bereitest Tickets für eine Wissensdatenbank auf. Deine Aufgabe:

1. Extrahiere und fasse die wichtigsten Felder wie unten vorgegeben präzise zusammen.
2. Gib deine Antwort ausschließlich als korrektes, minimales JSON aus (keine Kommentare, keine Freitexte, Umlaute erlaubt).
3. Halte dich zu 100% an die vorgegebenen Feldnamen und -typen in der JSON-Struktur.
3. Wenn für ein Feld keine Information vorhanden ist, gib null aus ohne Anführungszeichen also ein echtes json null kein Text.

Feldbedeutungen und Beispiele:
- "ersteller": Name des Ticket-Erstellers. Beispiel: "Max Mustermann"
- "erstelldatum": Datum im Format YYYY-MM-DD. Beispiel: "2024-06-01"
- "kategorie": Oberkategorie. Beispiel: Client, Server, Andere Hardware, Software, Netzwerk, Benutzerverwaltung
- "kurzbeschreibung": Sehr kurze Zusammenfassung des Problems (1 Satz). Beispiel: "Outlook startet nicht"
- "beschreibung": Fehlermeldungen, Besonderheiten (keine Lösung, keine Workarounds, nur das Problem beschreiben, gerne mehr Text). Beispiel: "Beim Start von Outlook erscheint Fehlercode 0x800123."
- "lösung": Konkrete Lösung im Detail (ausführlich, gerne mehr Text) oder Workaround, was wurde gemacht um das Problem zu lösen. Beispiel: "Datei von XY nach Z kopiert dann PC neu gestartet, auf dem chaos server unter E:\Daten Ordner angelegt und Berechtigung für xy vergeben"
- "system": Betriebssystem/Produkt/Software inkl. Version, falls erkennbar. Beispiel: "Windows 10, Outlook 365, Teams, Drucker"
- "tags": Schlagworte als Liste (3-6 relevante Schlagworte). Beispiel: ["Teams", "VPN", "Drucker"]

Halte dich an das Beispiel-JSON:

{{
  "ersteller": "Max Mustermann",
  "erstelldatum": "2024-06-01",
  "kategorie": "Software",
  "kurzbeschreibung": "Outlook startet nicht",
  "beschreibung": "Beim Start von Outlook erscheint Fehlercode 0x800123. Und...",
  "lösung": "KB518511 installiert unter C:\temp Daten gelöscht danach PC neu gestartet. Und...",
  "system": "Windows 10, Outlook 365",
  "tags": ["Outlook", "E-Mail"]
}}

Ticket-Text:
---
{ticket['full_text']}
---
Antworte ausschließlich mit dem JSON-Objekt wie oben. Keine weiteren Erklärungen.
"""
    REQUIRED_FIELDS = [
        "ersteller", "erstelldatum", "kategorie",
        "kurzbeschreibung", "beschreibung", "lösung", "system", "tags"
    ]
    for attempt in range(MAX_ATTEMPTS):
        print(f"📝 [LLM] Zusammenfassung/Strukturierung (Versuch {attempt+1}) ...")
        answer = call_llm(prompt, max_tokens=900, temperature=0.1)
        if not answer:
            print("⏹️ Keine Antwort vom LLM – Ticket wird übersprungen.")
            return None
        cleaned_answer = clean_llm_json_answer(answer)
        try:
            fields = json.loads(cleaned_answer)
            # 🗑️ Feldnamen normalisieren
            keymap = {"loesung": "lösung", "l¶sung": "lösung", "kürzel": "kurzbeschreibung", "tag": "tags"}
            for wrong, correct in keymap.items():
                if wrong in fields:
                    fields[correct] = fields.pop(wrong)
            # Prüfe alle Keys
            if isinstance(fields, dict) and all(k in fields for k in REQUIRED_FIELDS):
                print(f"✅ Zusammenfassung erfolgreich (Versuch {attempt+1})")
                print("✅ JSON-Antwort (gültig):", json.dumps(fields, ensure_ascii=False, indent=2))
                return fields
            else:
                print("⛔️ JSON (falsch/inkomplett):", cleaned_answer)
        except Exception as e:
            print("⛔️ JSON-Parsing-Fehler:", e)
            print("⛔️ LLM Output (fehlerhaft):", cleaned_answer)
        print(f"⛔️ Ungültige LLM-JSON-Antwort (Versuch {attempt+1}) – wiederhole...")
        time.sleep(1)
    print(f"⏹️ Konnte keine gültige JSON-Antwort vom LLM bekommen. Ticket wird übersprungen.")
    return None

def get_ticket_fulltext(ticket_id):
    headers = {"Authorization": f"Token token={ZAMMAD_TOKEN}"}
    res = requests.get(
        f"{ZAMMAD_URL}/api/v1/ticket_articles/by_ticket/{ticket_id}",
        headers=headers
    )
    if res.status_code != 200:
        return ""
    articles = res.json()
    # HTML bereinigen
    clean_texts = []
    for a in articles:
        html_body = a.get("body", "")
        clean_body = html_to_text(html_body)
        clean_texts.extend(clean_body)
    return " ".join(clean_texts)


def build_ticket_for_llm(ticket):
    ticket_id = ticket.get("id", "")
    ersteller = str(ticket.get("created_by_id", ""))
    erstelldatum = ticket.get("created_at", "")[:10]
    kategorie = ticket.get("group", None) or ticket.get("group_id", "")
    title = ticket.get("title", "")
    full_text = get_ticket_fulltext(ticket_id)
    text = f"Ersteller: {ersteller}\nErstelldatum: {erstelldatum}\nKategorie: {kategorie}\nTitel: {title}\n{full_text}"
    return {
        "id": ticket_id,
        "full_text": text
    }

def process_and_store_ticket(ticket, point_id):
    print(f"\n--- [{point_id}] Bearbeite Ticket ID: {ticket.get('id', point_id)} ---")
    
    if not is_ticket_helpful(ticket):
        print(f"⏩ Ticket {ticket.get('id', point_id)} nicht relevant – wird übersprungen.")
        return

    fields = summarize_ticket_to_json(ticket)
    if not fields or not fields.get("lösung"):
        print(f"⏩ Ticket {ticket.get('id', point_id)} hat keine Lösung – wird übersprungen.")
        return

    # Kontextuell angereicherte Felder
    kurz = f"Kurzbeschreibung: {fields.get('kurzbeschreibung', '').strip()}"
    beschr = f"Beschreibung: {fields.get('beschreibung', '').strip()}"
    lösung = f"Lösung: {fields.get('lösung', '').strip()}"
    system = f"System: {fields.get('system', '').strip()}"
    kategorie = f"Kategorie: {fields.get('kategorie', '').strip()}"

    tags_list = fields.get("tags", []) or []
    if isinstance(tags_list, list):
        tags_text = ", ".join(tags_list)
    else:
        tags_text = str(tags_list)
    tags = f"Tags: {tags_text}"

    # Kombinierter Vektor (für 'all') mit allem relevanten
    alle = "\n".join([
        kurz,
        beschr,
        lösung,
        system,
        kategorie,
        tags
    ])

    print("🧠 Erstelle Embeddings ...")
    emb_kurz = model.encode(kurz, normalize_embeddings=True)
    emb_beschr = model.encode(beschr, normalize_embeddings=True)
    emb_lösung = model.encode(lösung, normalize_embeddings=True)
    emb_all = model.encode(alle, normalize_embeddings=True)

    print("💾 Speichere in Qdrant ...")
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=point_id,
                vector={
                    "kurzbeschreibung": emb_kurz.tolist(),
                    "beschreibung": emb_beschr.tolist(),
                    "lösung": emb_lösung.tolist(),
                    "all": emb_all.tolist()
                },
                payload={
                    "ticket_id": ticket.get('id'),
                    "ersteller": fields.get("ersteller"),
                    "erstelldatum": fields.get("erstelldatum"),
                    "kategorie": fields.get("kategorie"),
                    "kurzbeschreibung": fields.get("kurzbeschreibung"),
                    "beschreibung": fields.get("beschreibung"),
                    "lösung": fields.get("lösung"),
                    "system": fields.get("system"),
                    "tags": tags_list,
                    "has_lösung": True,
                    "is_relevant": True
                }
            )
        ]
    )
    print(f"✅ Ticket {ticket.get('id', point_id)} gespeichert.")

# -- NEU: IDs der bereits gespeicherten Tickets laden --
def get_existing_ticket_ids():
    """Lade alle bereits gespeicherten ticket_id aus der Qdrant-Collection."""
    ticket_ids = set()
    offset = 0
    limit = 1000
    while True:
        result = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=None,
            with_payload=True,
            offset=offset,
            limit=limit
        )
        points = result[0]
        if not points:
            break
        for point in points:
            tid = point.payload.get("ticket_id")
            if tid is not None:
                ticket_ids.add(str(tid))
        offset += limit
    return ticket_ids

# IDs einmal am Anfang holen
existing_ticket_ids = get_existing_ticket_ids()
print(f"🧾 Bereits indexierte Tickets: {len(existing_ticket_ids)}")

def fetch_and_process_zammad_tickets():
    headers = {"Authorization": f"Token token={ZAMMAD_TOKEN}"}
    point_id = 1
    page = 1
    total = 0
    print("🚃 Starte Ticket-Import & Verarbeitung (nur Tickets ab 01.01.2020, min. 14 Tage geschlossen) ...")

    while True:
        print(f"🔎 Hole Tickets (Seite {page}) ...")
        try:
            res = requests.get(
                f"{ZAMMAD_URL}/api/v1/tickets?per_page=100&page={page}&order_by=created_at:desc",
                headers=headers
            )
            res.raise_for_status()
            page_data = res.json()
        except Exception as e:
            print(f"⏹️ Fehler beim Abrufen der Tickets (Seite {page}): {e}")
            break

        if not page_data:
            print("✅ Keine weiteren Tickets gefunden.")
            break

        for t_stub in page_data:
            ticket_id = t_stub.get("id")
            if not ticket_id:
                continue
            if str(ticket_id) in existing_ticket_ids:
                print(f"⏭️ Ticket {ticket_id} bereits aufgenommen – wird übersprungen.")
                continue

            try:
                res_detail = requests.get(f"{ZAMMAD_URL}/api/v1/tickets/{ticket_id}", headers=headers)
                res_detail.raise_for_status()
                t = res_detail.json()
            except Exception as e:
                print(f"⛔️ Fehler beim Abrufen von Ticket {ticket_id}: {e}")
                continue

            updated_str = t.get("created_at", "")
            try:
                updated_str = updated_str.replace("Z", "+00:00")
                updated = datetime.fromisoformat(updated_str)
            except:
                print(f"⛔️ Kann Datum nicht parsen: {updated_str} – Ticket wird übersprungen.")
                continue
            print(f"📝 Ticket {ticket_id} | Updated At: {updated.isoformat()} ")

            # Nur Tickets ab MIN_TICKET_DATE prüfen
            if updated < MIN_TICKET_DATE:
                continue

            # Statusname per state_id holen
            state_id = t.get("state_id")
            if not state_id:
                print(f"⛔️ Keine state_id bei Ticket {ticket_id} – wird übersprungen.")
                continue

            try:
                res_state = requests.get(f"{ZAMMAD_URL}/api/v1/ticket_states/{state_id}", headers=headers)
                res_state.raise_for_status()
                state_data = res_state.json()
                state_raw = state_data.get("name", "").strip()
            except Exception as e:
                print(f"⛔️ Konnte Status für state_id {state_id} nicht laden: {e}")
                continue

            if not state_raw:
                continue
            state = state_raw.lower()

            # Muss 14 Tage geschlossen sein
            closed_since = datetime.now(timezone.utc) - updated
            if state not in ["closed", "closed successful", "closed_unsuccessful", "archived", "geschlossen"]:
                continue
            if closed_since.days < MIN_CLOSED_DAYS:
                print(f"⏩ Ticket {ticket_id} wurde vor weniger als {MIN_CLOSED_DAYS} Tagen geschlossen – übersprungen.")
                continue

            ticket_llm = build_ticket_for_llm(t)
            process_and_store_ticket(ticket_llm, point_id)
            point_id += 1
            total += 1

        page += 1

    print(f"\n🏁 Verarbeitung beendet: {total} Tickets bearbeitet & gespeichert.")


# ---- Haupt-Workflow ----
fetch_and_process_zammad_tickets()
