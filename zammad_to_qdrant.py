from __future__ import annotations

"""
Zammad → Qdrant Ingestion mit BM25-Steuerung (seitenweise Verarbeitung)

- Standard (use_cached_bm25=False):
  1) Tickets seitenweise aus Zammad holen → pro Ticket Dense in Qdrant upserten (Sparse leer).
  2) Nach allen Tickets: Alle Chunks aus Qdrant scrollen → BM25 global neu berechnen → Sparse für ALLE Punkte aktualisieren.
  3) BM25-Index (Stats) in bm25_stats.json persistieren.

- Trigger (use_cached_bm25=True):
  1) BM25-Index aus bm25_stats.json laden (ohne Rebuild).
  2) Nur neue Tickets ingestieren → Dense + Sparse (aus Cache) direkt mitschreiben.
  3) Keine Neu-Berechnung/Neu-Zuweisung für bestehende Punkte.
"""

import os
import math
import time
import json
import logging
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional

import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import regex as re
from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

# -------------------------
# UUID Helper
# -------------------------
def make_uuid_from_string(s: str) -> str:
    """Deterministische UUID aus einem String erzeugen (gleiches s → gleiche UUID)."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, s))

# -------------------------
# Setup & Config
# -------------------------
load_dotenv()

# Zammad
ZAMMAD_URL = os.getenv("ZAMMAD_URL", "").rstrip("/")
ZAMMAD_TOKEN = os.getenv("ZAMMAD_TOKEN", "")

# Qdrant
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "zammad_tickets")

# Filter aus .env
TICKET_MIN_AGE_DAYS = int(os.getenv("TICKET_MIN_AGE_DAYS", "14"))
START_DATE = os.getenv("START_DATE", "2018-01-01")
USE_CACHED_BM25 = os.getenv("USE_CACHED_BM25", "false").lower() == "true"

# Zeit-Logger für beide Cutoff-Date
# MIN_AGE: Tickets dürfen nur nach X Tagen Inaktivität aufgenommen werden (Filter nach close_at)
MIN_AGE_CUTOFF_DATE = datetime.now(timezone.utc) - timedelta(days=TICKET_MIN_AGE_DAYS)
# START_DATE: Nur Tickets ab diesem Erstellungsdatum aufnehmen (Filter nach created_at)
try:
    START_DATE_CUTOFF_DATE = datetime.fromisoformat(START_DATE).replace(tzinfo=timezone.utc)
except Exception as e:
    logging.warning("Konnte START_DATE nicht parsen (%s), verwende Standard: 2018-01-01", e)
    START_DATE_CUTOFF_DATE = datetime(2018, 1, 1, tzinfo=timezone.utc)

# Dense Embeddings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")

# Sparse BM25
BM25_K1 = float(os.getenv("BM25_K1", "0.9"))
BM25_B = float(os.getenv("BM25_B", "0.4"))
BM25_STATS_FILE = os.getenv("BM25_STATS_FILE", "bm25_stats.json")

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "450"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
CHUNK_MAX = int(os.getenv("CHUNK_MAX", "512"))

logging.basicConfig(
    level=logging.INFO,  # für Debug: logging.DEBUG
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# Text-Log-Datei
LOG_FILE = f"zammad_ingest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
# -------------------------
# Stopwords + Regex
# -------------------------
STOPWORDS_DE = set(
    """aber alle allem allen aller alles als also am an ander andere anderem anderen
    anderer anderes anderm andern anderr auch auf aus bei bin bis bist da dadurch
    dafür dagegen daher damit dann das dasselbe dass dazu dein deine deinem deinen
    deiner deines dem den denn der dessen deshalb die dies diese diesem diesen
    dieser dieses doch dort du durch ein eine einem einen einer eines einig einige
    einigem einigen einiger einiges einmal er es etwas euer eure eurem euren eurer
    eures für gegen gewesen hab habe haben hat hatte hatten hattest hattet hier hin
    hinter ich ihm ihn ihnen ihr ihre ihrem ihren ihrer ihres im in ist ja jede
    jedem jeden jeder jedes je jetzt kann kein keine keinem keinen keiner keines
    können könnte machen man manche manchem manchen mancher manches mein meine
    meinem meinen meiner meines mich mir mit muss musste nach nicht nichts noch nun
    nur ob oder ohne sehr sein seine seinem seinen seiner seines selbst sich sie
    sind so solche solchem solchen solcher solches soll sollte sondern sonst um und
    uns unse unser unsrem unsren unsrer unsres unter vom von vor wann war waren
    warst was weg weil weiter welche welchem welchen welcher welches wenn wer wird
    wirst wo wollen wollte würde würden zu zum zur zwar zwischen""".split()
)
STOPWORDS = STOPWORDS_DE

WORD_RE = re.compile(r"[\p{L}\p{N}]+", re.IGNORECASE)

QUOTE_PATTERNS = [
    r"^-{2,}\s*Ursprüngliche Nachricht\s*-{2,}$",
    r"^Von: .+ schrieb:$",
    r"^Am .+ schrieb:$",
]
QUOTE_REGEXES = [re.compile(pat, re.IGNORECASE | re.MULTILINE) for pat in QUOTE_PATTERNS]

HEADER_PATTERNS = [
    r"^Von:.*$",
    r"^An:.*$",
    r"^Datum:.*$",
    r"^Betreff:.*$",
]
HEADER_REGEXES = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in HEADER_PATTERNS]

SIGNATURE_TRIGGERS_RAW = [
    "mit freundlichen grüßen",
    "viele grüße",
    "best regards",
    "kind regards",
    "i.v.",
    "i.a.",
    "tel.:",
    "telefon:",
    "fax:",                         # konsistent lowercased
    "sitz der gesellschaft",
    "geschäftsführer",
    "amtsgericht",
    "www.",                                        # generisch
    "besuchen sie uns",
    "im internet",
    "-- it service",               # <--- hinzugefügt
    "it service",
]
# konsequent normalisiert für den Vergleich
SIGNATURE_TRIGGERS = [s.strip().casefold() for s in SIGNATURE_TRIGGERS_RAW]

# Inline-Signatur-Trenner wie " -- ", " – ", " — "
INLINE_SPLIT_RE = re.compile(r"\s--+\s?.*")  # alles nach -- entfernen

COMPANY_RE = re.compile(r"\b(gmbh|ag|kg|gesmbh|inc|ltd)\b", re.I)

# Domains/Kontakt (fängt www., http(s) und bare domains inkl. .local ab)
DOMAIN_RE  = re.compile(r"(?:https?://|www\.)\S+|\b[a-z0-9][\w.-]*\.(?:local|lan|de|com|net|org)\b", re.I)
CONTACT_RE = re.compile(r"\b(?:tel\.?|telefon|fax|e-?mail|email|kontakt)\b", re.I)
ZIP_RE     = re.compile(r"\b(?:D-\d{5}|\d{5})\b", re.I)

def remove_mail_headers(text: str) -> str:
    for regex in HEADER_REGEXES:
        text = regex.sub("", text)
    return text.strip()

def remove_signature_blocks(text: str) -> str:
    out = []
    for line in text.splitlines():
        raw = line.strip()
        if not raw:
            continue

        # 1) Alles nach "--" entfernen
        raw = INLINE_SPLIT_RE.sub("", raw).strip()

        # 2) Trigger innerhalb der Zeile → ab da abschneiden
        lcf = raw.casefold()
        for trig in SIGNATURE_TRIGGERS:
            pos = lcf.find(trig)
            if pos != -1:
                raw = raw[:pos].rstrip()
                break

        # 3) Generische Muster
        for regex in (DOMAIN_RE, CONTACT_RE, ZIP_RE):
            m = regex.search(raw)
            if m:
                raw = raw[:m.start()].rstrip()

        # 4) Firmenzeilen komplett verwerfen
        if COMPANY_RE.search(raw):
            logging.debug("Firmen-Signatur entfernt: %s", raw)
            continue

        if raw:
            out.append(raw)

    return "\n".join(out).strip()

def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    for sel in ["blockquote", "div.gmail_quote", "cite"]:
        for tag in soup.select(sel):
            tag.decompose()
    return soup.get_text("\n")

def strip_quoted_lines(text: str) -> str:
    lines = text.splitlines()
    return "\n".join(line for line in lines if not any(r.search(line) for r in QUOTE_REGEXES))

def dedupe_lines(text: str) -> str:
    seen, out = set(), []
    for line in text.splitlines():
        key = line.strip()
        if not key:
            out.append("")
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
    return "\n".join(out)

def normalize_text(raw: str) -> str:
    if not raw:
        return ""
    txt = html_to_text(raw)
    txt = strip_quoted_lines(txt)
    txt = remove_mail_headers(txt)
    txt = remove_signature_blocks(txt)
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    txt = dedupe_lines(txt)
    return txt.strip()
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import math
import unicodedata

# -------------------------
# Chunking
# -------------------------
@dataclass
class Chunk:
    text: str
    index: int


def tokenize(text: str, tokenizer: AutoTokenizer) -> List[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def detokenize(ids: List[int], tokenizer: AutoTokenizer) -> str:
    return tokenizer.decode(ids)


def window_split(ids: List[int], size: int, overlap: int = 0) -> List[List[int]]:
    out, i = [], 0
    step = max(1, size - overlap)
    while i < len(ids):
        out.append(ids[i:i + size])
        i += step
    return out


def chunk_by_tokens(
    text: str,
    tokenizer: AutoTokenizer,
    chunk_size: int = 450,
    overlap: int = 50,
    hard_max: int = 512
) -> List[Chunk]:
    if not text:
        return []
    tokens = tokenize(text, tokenizer)
    if len(tokens) <= chunk_size:
        return [Chunk(detokenize(tokens, tokenizer), 0)]
    chunks, idx = [], 0
    windows = window_split(tokens, chunk_size, overlap)
    for w in windows:
        if len(w) > hard_max:
            for sw in window_split(w, hard_max, 0):
                chunks.append(Chunk(detokenize(sw, tokenizer), idx))
                idx += 1
        else:
            chunks.append(Chunk(detokenize(w, tokenizer), idx))
            idx += 1
    return chunks


# -------------------------
# BM25: Tokenization & Stats
# -------------------------
def normalize_token(token: str) -> str:
    return unicodedata.normalize("NFKC", token).casefold()


def tokenize_words_for_bm25(text: str) -> List[str]:
    out: List[str] = []
    for m in WORD_RE.finditer(text):
        t = normalize_token(m.group(0))
        if t and t not in STOPWORDS:
            out.append(t)
    return out


def build_corpus_stats(texts: List[str]) -> Tuple[Dict[str, int], float, Dict[int, int], Dict[str, int]]:
    term_df: Dict[str, int] = {}
    doc_len_map: Dict[int, int] = {}
    for idx, text in enumerate(texts):
        terms = tokenize_words_for_bm25(text)
        doc_len_map[idx] = len(terms)
        for t in set(terms):
            term_df[t] = term_df.get(t, 0) + 1
    N = max(1, len(texts))
    avgdl = (sum(doc_len_map.values()) / N) if N else 1.0
    vocab = {t: i for i, t in enumerate(sorted(term_df.keys()))}
    return term_df, avgdl, doc_len_map, vocab


def bm25_sparse_vector(
    text: str,
    chunk_index: int,
    term_df: Dict[str, int],
    avgdl: float,
    doc_len_map: Optional[Dict[int, int]],
    vocab: Dict[str, int],
    N_docs: int,
    k1: float,
    b: float,
) -> Dict[str, List]:
    terms = tokenize_words_for_bm25(text)
    if not terms:
        return {"indices": [], "values": []}

    dl = len(terms) if (doc_len_map is None or chunk_index == -1) else max(1, doc_len_map.get(chunk_index, len(terms)))

    tf_counts: Dict[str, int] = {}
    for t in terms:
        tf_counts[t] = tf_counts.get(t, 0) + 1

    indices: List[int] = []
    values: List[float] = []
    for t, tf in tf_counts.items():
        df = term_df.get(t, 0)
        if df <= 0 or t not in vocab:
            continue
        idf = math.log((N_docs - df + 0.5) / (df + 0.5) + 1.0)
        denom = tf + k1 * (1 - b + b * (dl / max(1e-9, avgdl)))
        score = idf * (tf * (k1 + 1)) / max(1e-9, denom)
        indices.append(vocab[t])
        values.append(float(score))
    return {"indices": indices, "values": values}
# -------------------------
# Embeddings & Qdrant
# -------------------------
_embedder: Optional[SentenceTransformer] = None
_tokenizer: Optional[AutoTokenizer] = None


def torch_cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def load_models() -> None:
    global _embedder, _tokenizer
    if _embedder is None:
        device = "cuda" if torch_cuda_available() else "cpu"
        logging.info("Lade Embedding-Modell (%s) auf %s ...", EMBEDDING_MODEL, device)
        _embedder = SentenceTransformer(EMBEDDING_MODEL, device=device)
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)


def embed_texts_batched(texts: List[str], batch_size: int = 64) -> List[List[float]]:
    assert _embedder is not None
    out: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        logging.debug("  → Embedding-Batch (%d Texte)", len(chunk))
        out.extend(_embedder.encode(
            chunk,
            normalize_embeddings=True,
            convert_to_numpy=True
        ).tolist())
    return out


def ensure_collection(client: QdrantClient, dim: int) -> None:
    try:
        client.get_collection(COLLECTION_NAME)
        logging.info("Collection '%s' existiert bereits.", COLLECTION_NAME)
    except Exception:
        logging.info("Erstelle Collection '%s' ...", COLLECTION_NAME)
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={"dense": qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE)},
            sparse_vectors_config={"sparse": qmodels.SparseVectorParams()},
        )


# -------------------------
# BM25-Stats persistieren
# -------------------------
def save_bm25_stats(term_df, avgdl, vocab, N_docs) -> None:
    stats = {
        "term_df": term_df,
        "avgdl": avgdl,
        "vocab": vocab,
        "N_docs": N_docs,
        "last_rebuild": datetime.utcnow().isoformat() + "Z",
    }
    with open(BM25_STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f)
    logging.info("BM25-Stats gespeichert (%d Terms, %d Docs).", len(term_df), N_docs)


def load_bm25_stats():
    if not os.path.exists(BM25_STATS_FILE):
        raise RuntimeError(f"BM25-Stats-Datei fehlt: {BM25_STATS_FILE}")
    with open(BM25_STATS_FILE, "r", encoding="utf-8") as f:
        stats = json.load(f)
    logging.info("BM25-Stats geladen (%d Terms, %d Docs).", len(stats["term_df"]), stats["N_docs"])
    return stats


# -------------------------
# Qdrant Utilities
# -------------------------
def qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def qdrant_scroll_all_points_text_only(client: QdrantClient, limit: int = 4096) -> List[Tuple[str, str]]:
    """Scrollt alle Points und liefert (point_id, text) zurück."""
    all_items: List[Tuple[str, str]] = []
    next_page = None
    while True:
        pts, next_page = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=None,
            with_payload=True,
            with_vectors=False,
            limit=limit,
            offset=next_page,
        )
        for p in pts:
            pid = str(p.id)
            payload = getattr(p, "payload", {}) or {}
            text = payload.get("text", "")
            if isinstance(text, str) and text.strip():
                all_items.append((pid, text))
        if next_page is None:
            break
    logging.info("Qdrant-Scroll: %d Texte gefunden.", len(all_items))
    return all_items


def qdrant_update_sparse_vectors(client: QdrantClient, id_to_sparse: Dict[str, qmodels.SparseVector], batch_size: int = 256):
    """Aktualisiert NUR den 'sparse'-Vektor in Qdrant."""
    ids = list(id_to_sparse.keys())
    for i in tqdm(range(0, len(ids), batch_size), desc="Update sparse vectors"):
        batch_ids = ids[i:i + batch_size]
        points = [
            qmodels.PointStruct(
                id=pid,
                vector={"sparse": id_to_sparse[pid]},
                payload=None
            )
            for pid in batch_ids
        ]
        # Retries
        for retry in range(3):
            try:
                logging.debug("  → Sparse-Upsert Batch (%d Punkte)", len(points))
                client.upsert(collection_name=COLLECTION_NAME, points=points)
                break
            except Exception as e:
                logging.warning("Sparse-Upsert fehlgeschlagen (Try %s/3): %s", retry + 1, e)
                time.sleep(1 + retry)
        else:
            raise RuntimeError("Sparse-Upsert nach 3 Versuchen gescheitert.")
# -------------------------
# Zammad API Helpers
# -------------------------
def zammad_headers() -> Dict[str, str]:
    if not ZAMMAD_TOKEN:
        raise RuntimeError("ZAMMAD_TOKEN is missing")
    return {"Authorization": f"Token token={ZAMMAD_TOKEN}", "Content-Type": "application/json"}


def _parse_dt_any(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def get_ticket_articles(ticket_id: int) -> List[dict]:
    """
    Holt alle Artikel zu einem Ticket.
    Nutzt den stabilen Endpoint: /api/v1/ticket_articles/by_ticket/{id}
    """
    url = f"{ZAMMAD_URL}/api/v1/ticket_articles/by_ticket/{ticket_id}"
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=zammad_headers(), timeout=60)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "articles" in data:
                return data["articles"]
            return []
        except Exception as e:
            logging.warning("Artikelabruf fehlgeschlagen für %s (Try %s/3): %s",
                            ticket_id, attempt + 1, e)
            time.sleep(0.5 * (attempt + 1))
    return []


def iter_ticket_pages(min_age_days: int):
    page = 1
    per_page = 100

    while True:
        logging.info("[PAGE %s] Lade Tickets ...", page)
        url = f"{ZAMMAD_URL}/api/v1/tickets?page={page}&per_page={per_page}"
        resp = requests.get(url, headers=zammad_headers(), timeout=60)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            logging.info("[PAGE %s] Keine weiteren Tickets.", page)
            break

        filtered: List[dict] = []
        for t in batch:
            tid = t.get("id")
            close_at = _parse_dt_any(t.get("close_at"))
            created_at = _parse_dt_any(t.get("created_at"))
            logging.info("Ticket %s: close_at=%s, created_at=%s, MIN_AGE_CUTOFF=%s, START_DATE_CUTOFF=%s", 
                        tid, close_at, created_at, MIN_AGE_CUTOFF_DATE, START_DATE_CUTOFF_DATE)
            
            if not close_at:
                continue
                
            # Filter 1: TICKET_MIN_AGE_DAYS (nur Tickets, die seit X Tagen geschlossen sind)
            if close_at > MIN_AGE_CUTOFF_DATE:
                logging.info("FILTER 1: Ticket %s übersprungen (zu neu, close_at=%s, cutoff=%s)",
                           tid, close_at, MIN_AGE_CUTOFF_DATE)
                continue
                
            # Filter 2: START_DATE (nur Tickets ab einem bestimmten Erstellungsdatum)
            if created_at and created_at < START_DATE_CUTOFF_DATE:
                logging.debug("Ticket %s übersprungen (created_at vor START_DATE, created=%s, cutoff=%s)",
                            tid, created_at, START_DATE_CUTOFF_DATE)
                continue
                
            filtered.append(t)

        logging.info("[PAGE %s] %d gültige Tickets gefunden.", page, len(filtered))
        if filtered:
            yield filtered

        page += 1
        time.sleep(0.05)


# -------------------------
# Titel-Blacklist & Pattern-Skip
# -------------------------
SKIP_TITLES = [
    "Returned mail: see transcript for details",
    "Undelivered Mail",
    "Alarm Notice!",
    "Mercury Managed Print Services Fehler",
]

SKIP_KEYWORDS = [
    "[suspected phishing]",
    "[suspected spam]",
    "SQL Server-Warnungssystem:",
    "[Fehler] SQL Server-Auftragssystem:",
    "Undelivered Mail",
    "Alarm Notice!",   
    "USV-SR1-01 - Periodic report",
    "Periodic report - Eaton 9PX 5000i",
    "Mercury Managed Print Services Fehler",
]

def should_skip_ticket(title: str) -> bool:
    if not title:
        return False
    clean = title.strip()
    if clean in SKIP_TITLES:
        return True
    lower = clean.lower()
    return any(kw in lower for kw in SKIP_KEYWORDS)


# -------------------------
# Metriken & Logging
# -------------------------
@dataclass
class IngestionMetrics:
    """Metriken für die Ticket-Verarbeitung."""
    total_tickets_seen: int = 0
    tickets_skipped_title: int = 0
    tickets_skipped_existing: int = 0
    tickets_processed: int = 0
    total_chunks_created: int = 0
    articles_processed: int = 0
    errors: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def duration(self) -> Optional[timedelta]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    def to_text_log(self) -> str:
        """Erstellt Text-Log-Eintrag für finale Ausgabe."""
        lines = []
        lines.append(f"=== Zammad → Qdrant Ingestion ===")
        lines.append(f"Start: {self.start_time.strftime('%Y-%m-%d %H:%M:%S') if self.start_time else 'N/A'}")
        lines.append(f"Ende: {self.end_time.strftime('%Y-%m-%d %H:%M:%S') if self.end_time else 'N/A'}")
        lines.append(f"Dauer: {self.duration}")
        lines.append("")
        lines.append("STATISTIKEN:")
        lines.append(f"  Tickets gesehen: {self.total_tickets_seen}")
        lines.append(f"  Tickets übersprungen (Titel): {self.tickets_skipped_title}")
        lines.append(f"  Tickets übersprungen (bereits vorhanden): {self.tickets_skipped_existing}")
        lines.append(f"  Tickets verarbeitet: {self.tickets_processed}")
        lines.append(f"  Artikel verarbeitet: {self.articles_processed}")
        lines.append(f"  Chunks erstellt: {self.total_chunks_created}")
        lines.append(f"  Fehler: {self.errors}")
        return "\n".join(lines)

# Globale Metriken
metrics = IngestionMetrics()

def clean_old_logs() -> None:
    """Löscht alle alten zammad_ingest_*.log Dateien."""
    import glob
    try:
        old_logs = glob.glob("zammad_ingest_*.log")
        for log_file in old_logs:
            try:
                os.remove(log_file)
                logging.info("Alte Log-Datei gelöscht: %s", log_file)
            except Exception as e:
                logging.warning("Konnte alte Log-Datei nicht löschen %s: %s", log_file, e)
    except Exception as e:
        logging.warning("Fehler beim Suchen alter Log-Dateien: %s", e)

def write_text_log(message: str, level: str = "INFO") -> None:
    """Schreibt in Text-Log-Datei (nur das Nötigste)."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_line = f"[{timestamp}] {level}: {message}\n"
    
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(log_line)
    except Exception as e:
        logging.warning("Konnte nicht in Text-Log schreiben: %s", e)

def log_current_ticket(ticket_id: int, message: str) -> None:
    """Loggt aktuell bearbeitetes Ticket."""
    write_text_log(f"Ticket {ticket_id}: {message}")

# -------------------------
# Hauptlogik
# -------------------------
def ingest(use_cached_bm25: bool = False) -> None:
    """
    - use_cached_bm25=False (Standard):
        1) Tickets seitenweise aus Zammad ingestieren (Dense-Vektoren, Sparse leer).
        2) Nach allen Seiten: BM25 global NEU berechnen & Sparse aktualisieren.
        3) bm25_stats.json speichern.

    - use_cached_bm25=True (Trigger):
        1) bm25_stats.json laden.
        2) Tickets seitenweise ingestieren → Dense + Sparse aus Cache.
        3) Keine Updates für bestehende Punkte.
    """
    global metrics
    
    # Alte Logs löschen
    clean_old_logs()
    
    # Text-Log neu erstellen
    try:
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write(f"=== Zammad → Qdrant Ingestion gestartet ===\n")
            f.write(f"Datum: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"START_DATE: {START_DATE}\n")
            f.write(f"START_DATE_CUTOFF: {START_DATE_CUTOFF_DATE}\n")
            f.write(f"TICKET_MIN_AGE_DAYS: {TICKET_MIN_AGE_DAYS}\n")
            f.write(f"MIN_AGE_CUTOFF_DATE: {MIN_AGE_CUTOFF_DATE}\n")
            f.write(f"use_cached_bm25: {use_cached_bm25}\n")
            f.write(f"========================================\n\n")
    except Exception as e:
        logging.warning("Konnte Text-Log nicht erstellen: %s", e)
    
    metrics.start_time = datetime.now()
    write_text_log("Ingestion gestartet")
    
    if not ZAMMAD_URL:
        write_text_log("FEHLER: ZAMMAD_URL fehlt", "ERROR")
        raise RuntimeError("ZAMMAD_URL is missing")
    if not ZAMMAD_TOKEN:
        write_text_log("FEHLER: ZAMMAD_TOKEN fehlt", "ERROR")
        raise RuntimeError("ZAMMAD_TOKEN is missing")

    load_models()
    tokenizer = _tokenizer
    embedder = _embedder
    
    if embedder is None:
        raise RuntimeError("Embedding-Modell konnte nicht geladen werden")
    if tokenizer is None:
        raise RuntimeError("Tokenizer konnte nicht geladen werden")
    
    dim = embedder.get_sentence_embedding_dimension()
    if dim is None:
        raise RuntimeError("Embedding-Dimension nicht ermittelt")
    dim = int(dim)

    qc = qdrant_client()
    ensure_collection(qc, dim)

    logging.info("Starte seitenweise Ingestion ...")

    # --- 1) Tickets seitenweise holen & direkt verarbeiten ---
    for page_num, tickets in enumerate(iter_ticket_pages(TICKET_MIN_AGE_DAYS), start=1):
        logging.info("==== Verarbeite Seite %d (%d Tickets) ====", page_num, len(tickets))

        for t in tickets:
            metrics.total_tickets_seen += 1
            tid = int(t["id"])
            title_raw = str(t.get("title", ""))
            if should_skip_ticket(title_raw):
                metrics.tickets_skipped_title += 1
                log_msg = f"Ticket {tid} übersprungen (Titel: {title_raw[:50]}{'...' if len(title_raw) > 50 else ''})"
                logging.info("→ %s", log_msg)
                write_text_log(log_msg)
                continue

            logging.info("→ Ticket %s (Seite %d)", tid, page_num)

            base = {
                "ticket_id": tid,
                "ticket_url": f"{ZAMMAD_URL}/#ticket/zoom/{tid}",
                "status": "closed",                 # 'state' gibt's in deiner Instanz nicht
                "closed_at": t.get("close_at"),     # wichtig fürs spätere Debugging
                "created_at": t.get("created_at"),
            }

            # Titel → 1 Chunk
            title_text = normalize_text(title_raw)
            ticket_chunks: List[Tuple[str, dict, str]] = []
            if title_text:
                meta = {
                    **base,
                    "article_id": 0,
                    "sender_type": "User",
                    "content_type": "title",
                    "is_first_user_article": False,
                    "article_position": -1,
                    "chunk_index": 0,
                    "text": title_text,
                }
                pid_raw = f"{base['ticket_id']}_0_0"
                pid = make_uuid_from_string(pid_raw)
                ticket_chunks.append((title_text, meta, pid))

            # Artikel laden & chunken
            articles = get_ticket_articles(base["ticket_id"]) or []

            for pos, a in enumerate(articles):
                norm = normalize_text(a.get("body") or "")
                if not norm:
                    continue
                chunks = chunk_by_tokens(norm, tokenizer, CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_MAX)
                for ci, ch in enumerate(chunks):
                    meta = {
                        **base,
                        "article_id": int(a.get("id", 0)),
                        "sender_type": "User" if str(a.get("sender_type", "")).lower() in {"customer", "user"} else "Agent",
                        "content_type": "article",
                        "is_first_user_article": (pos == 0),   # <--- nur erster Artikel!
                        "article_position": pos,
                        "chunk_index": ci,
                        "text": ch.text,
                    }
                    pid_raw = f"{base['ticket_id']}_{meta['article_id']}_{ci}"
                    pid = make_uuid_from_string(pid_raw)
                    ticket_chunks.append((ch.text, meta, pid))


            if not ticket_chunks:
                logging.info("  Ticket %s: keine verwertbaren Inhalte.", tid)
                continue

            # Deduplizierung: existierende Punkt-IDs überspringen
            ids_for_ticket = [pid for _, _, pid in ticket_chunks]
            existing = set()
            for i in range(0, len(ids_for_ticket), 256):
                try:
                    res = qc.retrieve(collection_name=COLLECTION_NAME, ids=ids_for_ticket[i:i + 256])
                    for r in res:
                        if r is not None and getattr(r, "id", None) is not None:
                            existing.add(str(r.id))
                except Exception:
                    pass
            pending = [(txt, meta, pid) for (txt, meta, pid) in ticket_chunks if pid not in existing]
            if not pending:
                metrics.tickets_skipped_existing += 1
                log_msg = f"Ticket {tid}: alle Chunks existieren bereits"
                logging.info("  %s", log_msg)
                write_text_log(log_msg)
                continue

            logging.info("  Ticket %s: %d neue Chunks zu verarbeiten.", tid, len(pending))

            # Dense Embeddings
            dense_vecs = embed_texts_batched([p[0] for p in pending])

            # Sparse (abhängig vom Modus)
            stats = load_bm25_stats() if use_cached_bm25 else None
            points: List[qmodels.PointStruct] = []
            for (text, meta, pid), dense in zip(pending, dense_vecs):
                if use_cached_bm25 and stats:
                    sparse_calc = bm25_sparse_vector(
                        text=text,
                        chunk_index=-1,
                        term_df=stats["term_df"],
                        avgdl=stats["avgdl"],
                        doc_len_map=None,
                        vocab=stats["vocab"],
                        N_docs=stats["N_docs"],
                        k1=BM25_K1,
                        b=BM25_B,
                    )
                    sparse_vec = qmodels.SparseVector(indices=sparse_calc["indices"], values=sparse_calc["values"])
                else:
                    sparse_vec = qmodels.SparseVector(indices=[], values=[])

                points.append(
                    qmodels.PointStruct(
                        id=pid,
                        vector={"dense": dense, "sparse": sparse_vec},
                        payload=meta,
                    )
                )

            # Upsert mit Retries
            for retry in range(3):
                try:
                    qc.upsert(collection_name=COLLECTION_NAME, points=points)
                    metrics.tickets_processed += 1
                    metrics.total_chunks_created += len(points)
                    metrics.articles_processed += len(articles)
                    log_msg = f"Ticket {tid} OK ({len(points)} Chunks, {len(articles)} Art.)"
                    logging.info("  %s", log_msg)
                    write_text_log(log_msg)
                    break
                except Exception as e:
                    metrics.errors += 1
                    log_msg = f"Ticket {tid} Upsert fehlgeschlagen (Try {retry + 1}/3): {e}"
                    logging.warning("  %s", log_msg)
                    write_text_log(log_msg, "WARNING")
                    time.sleep(1 + retry)
            else:
                metrics.errors += 1
                log_msg = f"Ticket {tid} konnte NICHT gespeichert werden!"
                logging.error("  %s", log_msg)
                write_text_log(log_msg, "ERROR")

    # --- 2) BM25 global rebuild (nur im Standardmodus) ---
    if not use_cached_bm25:
        logging.info("Standardmodus: BM25 wird GLOBAL neu berechnet ...")
        pid_texts = qdrant_scroll_all_points_text_only(qc, limit=4096)
        if not pid_texts:
            logging.info("Keine Points in Qdrant gefunden – nichts zu tun.")
            return

        ids_order = [pid for (pid, _) in pid_texts]
        texts_order = [txt for (_, txt) in pid_texts]

        term_df, avgdl, doc_len_map, vocab = build_corpus_stats(texts_order)
        N_docs = max(1, len(texts_order))
        save_bm25_stats(term_df, avgdl, vocab, N_docs)

        id_to_sparse: Dict[str, qmodels.SparseVector] = {}
        for gidx, pid in enumerate(ids_order):
            text = texts_order[gidx]
            sparse_calc = bm25_sparse_vector(
                text=text,
                chunk_index=gidx,
                term_df=term_df,
                avgdl=avgdl,
                doc_len_map=doc_len_map,
                vocab=vocab,
                N_docs=N_docs,
                k1=BM25_K1,
                b=BM25_B,
            )
            id_to_sparse[pid] = qmodels.SparseVector(indices=sparse_calc["indices"], values=sparse_calc["values"])

        qdrant_update_sparse_vectors(qc, id_to_sparse, batch_size=256)
        logging.info("BM25-Update abgeschlossen: %d Punkte aktualisiert.", len(id_to_sparse))
    else:
        logging.info("Trigger-Modus: BM25 wurde NICHT neu berechnet.")
    
    # Metriken finalisieren
    metrics.end_time = datetime.now()
    write_text_log("Ingestion abgeschlossen")
    write_text_log("\n" + metrics.to_text_log())
    
    # Text-Log auch in Konsole ausgeben
    print("\n" + "="*50)
    print("FINALE STATISTIKEN:")
    print("="*50)
    print(metrics.to_text_log())
    print("="*50)
    print(f"Vollständiger Log verfügbar in: {LOG_FILE}")
    print("="*50)
# -------------------------
# Entry Point
# -------------------------
if __name__ == "__main__":
    try:
        logging.info("Starte Zammad → Qdrant Ingestion Script ...")
        logging.info("USE_CACHED_BM25: %s", USE_CACHED_BM25)

        # Verwende die Einstellung aus .env
        ingest(use_cached_bm25=USE_CACHED_BM25)

        logging.info("Ingestion abgeschlossen ✅")

    except Exception as e:
        logging.exception("Fataler Fehler: %s", e)
        raise
