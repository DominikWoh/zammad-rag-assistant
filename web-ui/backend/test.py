import os
from dotenv import load_dotenv

# Lade deine .env-Datei (hier Pfad anpassen, z.B. './.env' oder absoluten Pfad)
load_dotenv(dotenv_path='./.env')

# Beispiel: lies die Variablen aus der Umgebung
qdrant_url = os.getenv("QDRANT_URL")
zammad_token = os.getenv("ZAMMAD_TOKEN")
collection = os.getenv("COLLECTION_NAME")

print("QDRANT_URL:", qdrant_url)
print("ZAMMAD_TOKEN:", zammad_token)
print("COLLECTION_NAME:", collection)

# Prüfung, ob eine Variable fehlt
if not all([qdrant_url, zammad_token, collection]):
    print("WARNUNG: Nicht alle Umgebungsvariablen konnten geladen werden!")
else:
    print("Alle Umgebungsvariablen erfolgreich geladen.")
