"""
Zammad → LightRAG Sync Engine
=============================
Syncs closed Zammad tickets into LightRAG's knowledge graph.

Runs as a daemon: sync → wait (SYNC_INTERVAL) → sync → ...
If SYNC_TIME is set (e.g. "02:00"), syncs at that time every day.

Environment variables (all read from environment, set via .env / docker-compose):
    ZAMMAD_URL          - Zammad base URL
    ZAMMAD_TOKEN        - Zammad API token (ticket.agent permission)
    LIGHTRAG_URL        - LightRAG server URL
    TICKET_MIN_AGE_DAYS - Only sync tickets closed >= N days ago (default: 7)
    START_DATE          - Only sync tickets created after this date (default: 2020-01-01)
    SYNC_INTERVAL       - hourly, daily, weekly (default: daily)
    SYNC_TIME           - Time to run daily/weekly sync in HH:MM format (default: 02:00)
"""

import os
import sys
import json
import time
import sqlite3
import hashlib
import logging
import signal
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ZAMMAD_URL = os.getenv("ZAMMAD_URL", "http://localhost:8080").rstrip("/")
ZAMMAD_TOKEN = os.getenv("ZAMMAD_TOKEN", "")
LIGHTRAG_URL = os.getenv("LIGHTRAG_URL", "http://lightrag:9621").rstrip("/")
TICKET_MIN_AGE_DAYS = int(os.getenv("TICKET_MIN_AGE_DAYS", "7"))
START_DATE = os.getenv("START_DATE", "2020-01-01")
SYNC_INTERVAL = os.getenv("SYNC_INTERVAL", "daily").lower()
SYNC_TIME = os.getenv("SYNC_TIME", "02:00")
SYNC_LIMIT = int(os.getenv("SYNC_LIMIT", "0"))  # 0 = unlimited
PROGRESS_DB = os.getenv("PROGRESS_DB", "data/sync_progress.db")

INTERVAL_SECONDS = {
    "hourly": 3600,
    "daily": 86400,
    "weekly": 604800,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("sync")

# Graceful shutdown flag
_shutting_down = False


def _handle_signal(signum, frame):
    global _shutting_down
    _shutting_down = True
    log.info(f"Received signal {signum}, shutting down after current operation...")


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


# ---------------------------------------------------------------------------
# Progress Database
# ---------------------------------------------------------------------------

def init_db():
    os.makedirs(os.path.dirname(PROGRESS_DB) or ".", exist_ok=True)
    conn = sqlite3.connect(PROGRESS_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sync_progress (
            ticket_id    INTEGER PRIMARY KEY,
            status       TEXT DEFAULT 'done',
            content_hash TEXT,
            synced_at    TEXT DEFAULT (datetime('now')),
            attempts     INTEGER DEFAULT 0,
            error        TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sync_meta (
            key   TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    conn.commit()
    conn.close()


def is_synced(ticket_id: int) -> bool:
    conn = sqlite3.connect(PROGRESS_DB)
    row = conn.execute(
        "SELECT status FROM sync_progress WHERE ticket_id = ?", (ticket_id,)
    ).fetchone()
    conn.close()
    return row is not None and row[0] == "done"


def get_hash(ticket_id: int) -> str | None:
    conn = sqlite3.connect(PROGRESS_DB)
    row = conn.execute(
        "SELECT content_hash FROM sync_progress WHERE ticket_id = ?", (ticket_id,)
    ).fetchone()
    conn.close()
    return row[0] if row else None


def mark_done(ticket_id: int, content_hash: str):
    conn = sqlite3.connect(PROGRESS_DB)
    conn.execute("""
        INSERT INTO sync_progress (ticket_id, status, content_hash)
        VALUES (?, 'done', ?)
        ON CONFLICT(ticket_id) DO UPDATE SET
            status='done', content_hash=?, synced_at=datetime('now')
    """, (ticket_id, content_hash, content_hash))
    conn.commit()
    conn.close()


def mark_failed(ticket_id: int, error: str):
    conn = sqlite3.connect(PROGRESS_DB)
    conn.execute("""
        INSERT INTO sync_progress (ticket_id, status, error, attempts)
        VALUES (?, 'failed', ?, 1)
        ON CONFLICT(ticket_id) DO UPDATE SET
            status='failed', error=?, attempts=attempts+1
    """, (ticket_id, error, error))
    conn.commit()
    conn.close()


def get_failed_tickets(limit: int = 50) -> list[int]:
    conn = sqlite3.connect(PROGRESS_DB)
    rows = conn.execute(
        "SELECT ticket_id FROM sync_progress WHERE status='failed' AND attempts < 5"
    ).fetchall()
    conn.close()
    return [r[0] for r in rows[:limit]]


def get_stats() -> dict:
    conn = sqlite3.connect(PROGRESS_DB)
    total = conn.execute("SELECT COUNT(*) FROM sync_progress WHERE status='done'").fetchone()[0]
    failed = conn.execute("SELECT COUNT(*) FROM sync_progress WHERE status='failed'").fetchone()[0]
    conn.close()
    return {"synced": total, "failed": failed}


def get_last_sync() -> str | None:
    conn = sqlite3.connect(PROGRESS_DB)
    row = conn.execute("SELECT value FROM sync_meta WHERE key='last_sync'").fetchone()
    conn.close()
    return row[0] if row else None


def save_last_sync():
    conn = sqlite3.connect(PROGRESS_DB)
    conn.execute(
        "INSERT OR REPLACE INTO sync_meta (key, value) VALUES ('last_sync', ?)",
        (datetime.now(timezone.utc).isoformat(),),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Zammad API
# ---------------------------------------------------------------------------

def zammad_get(path: str, params: dict = None) -> dict | list:
    url = f"{ZAMMAD_URL}/api/v1/{path.lstrip('/')}"
    headers = {"Authorization": f"Token token={ZAMMAD_TOKEN}"}
    with httpx.Client(timeout=30) as client:
        resp = client.get(url, headers=headers, params=params)
        resp.raise_for_status()
        return resp.json()


def fetch_tickets(page: int = 1, per_page: int = 50, updated_after: str = None) -> list[dict]:
    """Fetch closed tickets. Uses search API for delta sync if updated_after is set."""
    if updated_after:
        return zammad_get("tickets/search", params={
            "query": "*", "state": "closed",
            "updated_after": updated_after,
            "per_page": per_page, "page": page,
        })
    return zammad_get("tickets", params={
        "page": page, "per_page": per_page, "state": "closed",
        "sort_by": "updated_at", "order_by": "asc",
    })


def fetch_articles(ticket_id: int) -> list[dict]:
    return zammad_get(f"ticket_articles/by_ticket/{ticket_id}")


def fetch_ticket_by_id(ticket_id: int) -> dict:
    return zammad_get(f"tickets/{ticket_id}")


# ---------------------------------------------------------------------------
# Content Formatting
# ---------------------------------------------------------------------------

def format_ticket(ticket: dict, articles: list[dict]) -> str:
    lines = [
        f"# Ticket #{ticket.get('id', '?')}: {ticket.get('title', 'Ohne Titel')}",
        "",
        f"- Datum: {ticket.get('created_at', 'unbekannt')}",
        f"- Status: {ticket.get('state', 'unbekannt')}",
        f"- Priorität: {ticket.get('priority', 'unbekannt')}",
        f"- Gruppe: {ticket.get('group', 'unbekannt')}",
        f"- Kunde: {ticket.get('customer', 'unbekannt')}",
    ]

    tags = ticket.get("tags", "")
    if tags:
        lines.append(f"- Tags: {tags}")

    lines.extend(["", "## Verlauf", ""])

    for article in articles:
        sender = article.get("sender", "Unbekannt")
        created = article.get("created_at", "")
        body = article.get("body", "").strip()
        if not body:
            continue
        body = clean_body(body)
        if not body:
            continue
        lines.append(f"### [{sender}] ({created})")
        lines.append(body)
        lines.append("")

    formatted = "\n".join(lines)
    # Cap total ticket size — long tickets cause LLM timeouts during entity extraction
    if len(formatted) > 6000:
        formatted = formatted[:6000] + "\n\n[... ticket truncated for processing ...]"
    return formatted


def clean_body(text: str) -> str:
    """Remove email signatures, quoted replies, HTML, excessive whitespace."""
    # Strip HTML tags if any
    import re
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'&[a-z]+;', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    for marker in ["-- ", "--\n", "Von:", "Gesendet:", "From:", "Sent:"]:
        idx = text.find(marker)
        if idx > 50:
            text = text[:idx]

    lines = [l for l in text.split("\n") if not l.startswith(">")]

    result = []
    empty = 0
    for line in lines:
        if line.strip() == "":
            empty += 1
            if empty <= 1:
                result.append(line)
        else:
            empty = 0
            result.append(line)

    cleaned = "\n".join(result).strip()
    # Truncate very long articles to keep LLM context manageable
    # (most ticket content fits in 2000 chars, longer articles are usually noise)
    if len(cleaned) > 3000:
        cleaned = cleaned[:3000] + "\n[... truncated ...]"
    return cleaned


def content_hash(ticket: dict, articles: list[dict]) -> str:
    raw = json.dumps(
        {"t": ticket.get("updated_at", ""), "a": len(articles)},
        sort_keys=True,
    )
    return hashlib.md5(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# LightRAG API
# ---------------------------------------------------------------------------

def lightrag_insert(content: str, file_source: str) -> bool:
    url = f"{LIGHTRAG_URL}/documents/text"

    for attempt in range(5):
        try:
            with httpx.Client(timeout=120) as client:
                resp = client.post(url, json={"text": content, "file_source": file_source})
                if resp.status_code in (200, 202):
                    return True
                if resp.status_code == 409:
                    wait = (2 ** attempt) * 5
                    log.warning(f"LightRAG busy (409), retry in {wait}s...")
                    time.sleep(wait)
                    continue
                log.error(f"LightRAG error {resp.status_code}: {resp.text[:200]}")
                return False
        except httpx.HTTPError as e:
            wait = (2 ** attempt) * 5
            log.warning(f"Insert failed (attempt {attempt+1}/5): {e}, retry in {wait}s...")
            time.sleep(wait)

    return False


def wait_for_lightrag():
    """Wait until LightRAG is healthy before starting sync."""
    log.info("Waiting for LightRAG to be ready...")
    for i in range(60):
        try:
            r = httpx.get(f"{LIGHTRAG_URL}/health", timeout=5)
            if r.status_code == 200 and r.json().get("status") == "healthy":
                log.info("LightRAG is ready.")
                return True
        except Exception:
            pass
        time.sleep(2)
    log.error("LightRAG not ready after 120s, starting anyway.")
    return False


# ---------------------------------------------------------------------------
# Sync Logic
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Ticket Filtering
# ---------------------------------------------------------------------------

# Exact title matches to skip (lowercase comparison)
SKIP_TITLES = [
    "usv-sr1-01 - periodic report",
    "periodic report - eaton 9px 5000i",
    "mercury managed print services fehler",
]

# Substring matches in title (lowercase comparison)
SKIP_KEYWORDS = [
    "[suspected phishing]",
    "[suspected spam]",
    "sql server-warnungssystem:",
    "[fehler] sql server-auftragssystem:",
    "undelivered mail",
    "alarm notice!",
]


def should_skip_title(title: str) -> bool:
    """Return True if ticket should be skipped based on title."""
    if not title:
        return False
    clean = title.strip()
    if clean.lower() in SKIP_TITLES:
        return True
    lower = clean.lower()
    return any(kw in lower for kw in SKIP_KEYWORDS)


def should_sync(ticket: dict) -> bool:
    created = ticket.get("created_at", "")[:10]
    if created < START_DATE:
        return False

    closed_at = ticket.get("close_at") or ticket.get("close_escalation_at") or ""
    if closed_at and TICKET_MIN_AGE_DAYS > 0:
        try:
            closed_date = datetime.fromisoformat(closed_at.replace("Z", "+00:00"))
            age = datetime.now(timezone.utc) - closed_date
            if age.days < TICKET_MIN_AGE_DAYS:
                return False
        except (ValueError, TypeError):
            pass

    return True


def sync_ticket(ticket_id: int) -> bool:
    try:
        ticket = fetch_ticket_by_id(ticket_id)
        if should_skip_title(ticket.get("title", "")):
            return True
        if not should_sync(ticket):
            return True

        articles = fetch_articles(ticket_id)
        formatted = format_ticket(ticket, articles)
        chash = content_hash(ticket, articles)

        old_hash = get_hash(ticket_id)
        if old_hash == chash:
            return True

        if lightrag_insert(formatted, f"zammad-ticket-{ticket_id}"):
            mark_done(ticket_id, chash)
            log.info(f"✓ Ticket #{ticket_id} synced ({len(formatted)} chars)")
            return True
        else:
            mark_failed(ticket_id, "LightRAG insert failed")
            return False

    except Exception as e:
        log.error(f"✗ Ticket #{ticket_id} failed: {e}")
        mark_failed(ticket_id, str(e))
        return False


def run_sync(limit: int = None):
    log.info(f"Starting sync (min_age={TICKET_MIN_AGE_DAYS}d, start={START_DATE})")
    init_db()

    # Retry failed tickets
    failed = get_failed_tickets()
    if failed:
        log.info(f"Retrying {len(failed)} failed tickets...")
        for tid in failed:
            if _shutting_down:
                return
            sync_ticket(tid)
            time.sleep(0.5)

    # Delta sync
    last_sync = get_last_sync()
    if last_sync:
        log.info(f"Delta sync: tickets updated after {last_sync}")
    else:
        log.info("Initial sync: fetching all closed tickets")
    
    if SYNC_LIMIT > 0:
        log.info(f"TEST MODE: limited to {SYNC_LIMIT} tickets")

    synced = 0
    skipped = 0
    page = 1

    while not _shutting_down:
        try:
            tickets = fetch_tickets(
                page=page, per_page=50,
                updated_after=last_sync,
            )
        except httpx.HTTPStatusError as e:
            log.error(f"Zammad API error page {page}: {e}")
            break

        if not tickets:
            break

        for ticket in tickets:
            if _shutting_down:
                break

            tid = ticket.get("id")
            if not tid:
                continue

            if is_synced(tid):
                skipped += 1
                continue

            if should_skip_title(ticket.get("title", "")):
                skipped += 1
                continue

            if not should_sync(ticket):
                skipped += 1
                continue

            if limit and synced >= limit:
                save_last_sync()
                _summary(synced, skipped)
                return
            
            if not limit and SYNC_LIMIT > 0 and synced >= SYNC_LIMIT:
                log.info(f"SYNC_LIMIT reached ({SYNC_LIMIT}), stopping.")
                save_last_sync()
                _summary(synced, skipped)
                return

            sync_ticket(tid)
            synced += 1
            time.sleep(1.0)

        page += 1

    save_last_sync()
    _summary(synced, skipped)


def _summary(synced: int, skipped: int):
    stats = get_stats()
    log.info(
        f"Sync complete: {synced} new, {skipped} skipped, "
        f"total={stats['synced']}, failed={stats['failed']}"
    )


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def seconds_until_next_sync() -> float:
    """Calculate seconds to wait until the next sync should run.

    For hourly: next occurrence of SYNC_TIME minute.
    For daily/weekly: next occurrence of SYNC_TIME HH:MM.
    """
    if SYNC_INTERVAL == "hourly":
        # Sync at :MM of every hour
        try:
            minute = int(SYNC_TIME.split(":")[1])
        except (IndexError, ValueError):
            minute = 0
        now = datetime.now()
        target = now.replace(minute=minute, second=0, microsecond=0)
        if target <= now:
            target += timedelta(hours=1)
        return (target - now).total_seconds()

    # daily / weekly: sync at HH:MM
    try:
        hour, minute = map(int, SYNC_TIME.split(":"))
    except (ValueError, AttributeError):
        hour, minute = 2, 0

    now = datetime.now()
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        if SYNC_INTERVAL == "weekly":
            target += timedelta(days=7)
        else:
            target += timedelta(days=1)

    return (target - now).total_seconds()


def run_daemon():
    """Main daemon loop: wait → sync → repeat."""
    log.info(f"Sync daemon started (interval={SYNC_INTERVAL}, time={SYNC_TIME})")

    # Wait for LightRAG first
    wait_for_lightrag()

    # Run initial sync on first start
    init_db()
    if get_last_sync() is None:
        log.info("First run — starting initial sync immediately.")
        run_sync()
    else:
        log.info("Data already exists — waiting for next scheduled sync.")

    while not _shutting_down:
        wait = seconds_until_next_sync()
        next_time = datetime.now() + timedelta(seconds=wait)
        log.info(f"Next sync at {next_time.strftime('%Y-%m-%d %H:%M:%S')} (in {wait/3600:.1f}h)")

        # Sleep in small increments for graceful shutdown
        end = time.monotonic() + wait
        while time.monotonic() < end and not _shutting_down:
            time.sleep(min(10, end - time.monotonic()))

        if _shutting_down:
            break

        try:
            run_sync()
        except Exception as e:
            log.error(f"Sync cycle failed: {e}")

    log.info("Sync daemon stopped.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if "--once" in sys.argv:
        wait_for_lightrag()
        limit = None
        for arg in sys.argv:
            if arg.startswith("--limit="):
                limit = int(arg.split("=")[1])
        run_sync(limit=limit)
    else:
        run_daemon()
