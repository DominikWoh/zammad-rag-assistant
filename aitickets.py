import requests
import time
import json
import random
import re

OLLAMA_URL = "http://192.168.0.120:11434/api/generate"
ZAMMAD_URL = "http://192.168.0.120:8080"
ZAMMAD_TOKEN = "utZBrfXUVSQaBIj9TpB2TWNGL9uUB14YeeXbTKsRDroad7YiWU5N2Efl8HAL23B_"
MODEL = "gemma3n:latest"

headers_zammad = {
    "Content-Type": "application/json",
    "Authorization": f"Token token={ZAMMAD_TOKEN}"
}

# 100 zufällige IT-Themen
TICKET_TOPICS = [
    "Outlook startet nicht", "Teams Anmeldung schlägt fehl", "VPN sehr langsam", "Drucker offline", "Monitor bleibt schwarz",
    "Bluescreen beim Hochfahren", "Audioausgabe funktioniert nicht", "Netzlaufwerk nicht verbunden",
    "Datei versehentlich gelöscht", "Tastatur schreibt falsche Zeichen", "Maus reagiert verzögert",
    "Internet bricht immer wieder ab", "Windows Update hängt", "Antivirensoftware blockiert Programm",
    "Bitlocker Wiederherstellungsschlüssel wird verlangt", "USB-Stick wird nicht erkannt",
    "PDF lässt sich nicht öffnen", "Excel Formel gibt falschen Wert", "Outlook zeigt keine Bilder",
    "OneDrive synchronisiert nicht", "Edge Browser startet langsam", "Druckausgabe fehlt Inhalte",
    "MFA Login geht nicht", "Exchange-Zugriff verweigert", "Dateizugriff verweigert", "E-Mail kommt nicht an",
    "Kalenderfreigabe funktioniert nicht", "Adobe Acrobat stürzt ab", "Kamera nicht verfügbar",
    "Zoom-Mikrofon geht nicht", "Bluetooth-Maus trennt sich", "SSD wird im BIOS nicht erkannt",
    "Citrix friert ein", "SAP GUI stürzt ab", "Autologin funktioniert nicht", "Passwort abgelaufen",
    "Benutzer kann sich nicht anmelden", "Drucker druckt nur leere Seiten", "Fax funktioniert nicht",
    "GPO wird nicht angewendet", "CMD deaktiviert", "Skript durch Sicherheitsrichtlinie blockiert",
    "Teams zeigt leeres Fenster", "Outlook-Regel nicht ausgeführt", "Standarddrucker ändert sich",
    "Mobilgerät verbindet nicht", "Windows Defender blockiert Datei", "Netzwerkdrucker lässt sich nicht hinzufügen",
    "PowerShell Fehler", "Bildschirmauflösung verstellt", "Dual Monitor funktioniert nicht",
    "Kein Zugriff auf Terminalserver", "Remote Desktop wird abgelehnt", "Druckwarteschlange hängt",
    "Excel-Add-In verursacht Absturz", "Java-Update benötigt Adminrechte", "Firefox speichert keine Passwörter",
    "Taskleiste reagiert nicht", "Explorer.exe stürzt ab", "System fährt von allein herunter",
    "Softwareverteilung schlägt fehl", "Adobe Lizenz nicht akzeptiert", "Lizenzschlüssel bereits verwendet",
    "Druckerfreigabe nicht verfügbar", "Freigabeordner nicht sichtbar", "Schattenkopien nicht verfügbar",
    "Excel Datei schreibgeschützt", "PDF mit Signatur defekt", "Systemzeit weicht ab",
    "Anmeldung dauert lange", "Windows Hello reagiert nicht", "Fingerprint wird nicht erkannt",
    "WLAN Authentifizierung schlägt fehl", "Netzwerkkarte deaktiviert sich", "Hyper-V startet VM nicht",
    "Teams-Meeting nicht möglich", "Outlook öffnet .ics-Datei nicht", "E-Mail-Signatur fehlt",
    "Bildschirm dreht sich automatisch", "Falsche Sprache im Login", "Excel startet im Safe-Mode",
    "Schriftart nicht verfügbar", "Word Absturz beim Speichern", "Drag & Drop defekt", "RDP bleibt schwarz",
    "Grafiktreiberproblem", "Hardwarebeschleunigung flackert", "Autostart Einträge fehlen",
    "Explorer startet mit falschem Pfad", "Kein Zugriff auf Gruppenlaufwerk", "Citrix StoreFront lädt nicht",
    "Thunderbolt Dock erkennt Geräte nicht", "DNS-Auflösung schlägt fehl", "Hosts-Datei nicht speicherbar",
    "Proxy-Einstellungen weg", "Outlook-Kontakte verschwunden", "Numlock beim Start deaktiviert",
    "Excel Datei lässt sich nicht schließen", "Skriptabbruch in Outlook VBA", "Makros werden blockiert"
]

def extract_json_from_response(text):
    try:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            return json.loads(match.group())
        else:
            print("[❌] Kein JSON gefunden.")
            return None
    except Exception as e:
        print("[❌] Fehler beim JSON-Parsing:", e)
        return None

def generate_ticket_via_ollama(index):
    num_entries = random.randint(2, 5)
    topic = random.choice(TICKET_TOPICS)

    system_prompt = f"""
Du bist ein IT-Support-Agent. Erstelle ein realistisches IT-Support-Ticket zum Thema: **{topic}**.

- Ein kurzer Titel
- {num_entries} Nachrichten (abwechselnd User und IT)
- Die letzte Nachricht enthält die Lösung, gerne 2–3 Sätze lang - Die Lösung darf sehr technisch sein und individuell für ein unternhmen mit individuallen pfaden und servernamen und lösungsansätzen.
- Die Nachrichten sollen realistisch und hilfreich sein
- Gib das Ergebnis als korrektes JSON zurück – ohne Fließtext, ohne Anführungszeichen außen!

Beispiel:

{{
  "title": "Teams startet nicht",
  "conversation": [
    {{"role": "User", "message": "Teams startet nicht."}},
    {{"role": "IT", "message": "Bitte starten Sie Teams im abgesicherten Modus."}}
  ]
}}

Nur JSON – keine Erklärung oder zusätzliche Kommentare!
"""

    payload = {
        "model": MODEL,
        "prompt": system_prompt,
        "stream": False,
        "temperature": 0.7
    }

    try:
        print(f"\n[🔄] Ticket {index}: Anfrage an Ollama (Thema: {topic})...")
        r = requests.post(OLLAMA_URL, json=payload)
        r.raise_for_status()
        response = r.json()["response"]
        print(f"[📥] Antwort von Ollama:\n{response.strip()[:500]}...\n")

        ticket_json = extract_json_from_response(response)
        return ticket_json

    except Exception as e:
        print(f"[❌] Fehler bei Ticket {index}: {e}")
        print("Antwort (raw):", r.text if 'r' in locals() else "Keine Antwort")
        return None

def create_zammad_ticket(ticket_data, index):
    title = ticket_data.get("title", f"Ticket {index}")
    conversation = ticket_data.get("conversation", [])
    customer_email = "dominik.wohnhas@gmail.com"

    if not conversation or not isinstance(conversation, list):
        print(f"[⚠️] Ungültige Konversation bei Ticket {index}")
        return

    data = {
        "title": title,
        "group": "Users",  # Stelle sicher, dass diese Gruppe existiert
        "customer": customer_email,
        "article": {
            "subject": title,
            "body": conversation[0]["message"],
            "type": "note",
            "internal": False
        }
    }

    print(f"[📨] Ticket {index}: Sende an Zammad...")
    r = requests.post(f"{ZAMMAD_URL}/api/v1/tickets", headers=headers_zammad, json=data)

    if not r.ok:
        print(f"[❌] Fehler beim Erstellen von Ticket #{index}")
        print("Status Code:", r.status_code)
        print("Antwort:", r.text)
        print("Gesendete Daten:", json.dumps(data, indent=2))
        return

    ticket_id = r.json()["id"]
    print(f"[✅] Ticket #{ticket_id} erstellt: {title}")

    for msg in conversation[1:]:
        article = {
            "ticket_id": ticket_id,
            "subject": f"{msg['role']} schreibt",
            "body": msg["message"],
            "type": "note",
            "internal": False
        }

        print(f"[➕] Artikel hinzufügen: {msg['role']} – {msg['message'][:60]}...")
        ra = requests.post(f"{ZAMMAD_URL}/api/v1/ticket_articles", headers=headers_zammad, json=article)
        if not ra.ok:
            print(f"[⚠️] Fehler beim Artikel: {ra.status_code} – {ra.text}")
        time.sleep(0.3)

    # Ticket schließen
    print(f"[🔒] Ticket #{ticket_id} auf 'closed' setzen...")
    close_payload = {
        "state": "closed"
    }
    rc = requests.put(f"{ZAMMAD_URL}/api/v1/tickets/{ticket_id}", headers=headers_zammad, json=close_payload)
    if rc.ok:
        print(f"[✅] Ticket #{ticket_id} erfolgreich geschlossen.")
    else:
        print(f"[⚠️] Fehler beim Schließen: {rc.status_code} – {rc.text}")

def main():
    for i in range(1, 500):  # mind. 40 Tickets
        ticket = generate_ticket_via_ollama(i)
        if ticket:
            create_zammad_ticket(ticket, i)
        else:
            print(f"[⛔] Ticket {i} wurde übersprungen.")
        time.sleep(1)

if __name__ == "__main__":
    main()
