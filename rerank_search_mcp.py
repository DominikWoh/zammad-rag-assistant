#!/usr/bin/env python3
# rerank_search.py
from __future__ import annotations

import os
import sys
import json
import math
import logging
import threading
import secrets
import re
import html
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer, CrossEncoder
from fastmcp import FastMCP
from fastmcp.tools import Tool
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

# ======================================================
# Setup & Config
# ======================================================
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "zammad_tickets")
MCP_API_KEY = os.getenv("MCP_API_KEY", None)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

BM25_STATS_FILE = os.getenv("BM25_STATS_FILE", "bm25_stats.json")
BM25_K1 = float(os.getenv("BM25_K1", "0.9"))
BM25_B = float(os.getenv("BM25_B", "0.4"))

QDRANT_SEARCH_HNSW_EF = int(os.getenv("QDRANT_SEARCH_HNSW_EF", "128"))
TOP_K = int(os.getenv("TOP_K", "100"))
TOP_TICKETS = int(os.getenv("TOP_TICKETS", "10"))

ZAMMAD_URL = os.getenv("ZAMMAD_URL", "")
ZAMMAD_TOKEN = os.getenv("ZAMMAD_TOKEN", "")
ZAMMAD_SEARCH_PREVIEW_WORDS = int(os.getenv("ZAMMAD_SEARCH_PREVIEW_WORDS", "200"))
ZAMMAD_SEARCH_DEFAULT_LIMIT = int(os.getenv("ZAMMAD_SEARCH_DEFAULT_LIMIT", "20"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ======================================================
# Zammad API Helpers
# ======================================================
def _get_zammad_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Token token={ZAMMAD_TOKEN}",
        "Content-Type": "application/json"
    }

def _html_to_text(html_content: str) -> str:
    if not html_content:
        return ""
    text = html_content
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</p>', '\n\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<div[^>]*>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</div>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '', text)
    text = html.unescape(text)
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def _truncate_words(text: str, max_words: int) -> str:
    if not text:
        return ""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."

def _get_ticket_from_zammad(ticket_id: int) -> Optional[Dict[str, Any]]:
    try:
        url = f"{ZAMMAD_URL.rstrip('/')}/api/v1/tickets/{ticket_id}?expand=true"
        resp = requests.get(url, headers=_get_zammad_headers(), timeout=30)
        if resp.status_code == 200:
            return resp.json()
        logging.warning("Zammad ticket %s not found: HTTP %s", ticket_id, resp.status_code)
        return None
    except Exception as e:
        logging.error("Error fetching ticket %s from Zammad: %s", ticket_id, e)
        return None

def _get_ticket_articles_from_zammad(ticket_id: int) -> List[Dict[str, Any]]:
    try:
        ticket = _get_ticket_from_zammad(ticket_id)
        if not ticket or "article_ids" not in ticket:
            return []
        
        articles = []
        for article_id in ticket.get("article_ids", []):
            try:
                url = f"{ZAMMAD_URL.rstrip('/')}/api/v1/ticket_articles/{article_id}"
                resp = requests.get(url, headers=_get_zammad_headers(), timeout=30)
                if resp.status_code == 200:
                    articles.append(resp.json())
            except Exception as e:
                logging.warning("Error fetching article %s: %s", article_id, e)
        
        articles.sort(key=lambda a: a.get("created_at", ""))
        return articles
    except Exception as e:
        logging.error("Error fetching articles for ticket %s: %s", ticket_id, e)
        return []

def _search_zammad_tickets(query: str, limit: int = 20) -> List[Dict[str, Any]]:
    try:
        url = f"{ZAMMAD_URL.rstrip('/')}/api/v1/tickets/search"
        params = {"query": query, "limit": limit}
        resp = requests.get(url, headers=_get_zammad_headers(), params=params, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        logging.warning("Zammad search failed: HTTP %s", resp.status_code)
        return []
    except Exception as e:
        logging.error("Error searching Zammad: %s", e)
        return []

# ======================================================
# QdrantClient Singleton (Thread-Safe)
# ======================================================
_qdrant_client: Optional[QdrantClient] = None
_qdrant_lock = threading.Lock()

def get_qdrant_client() -> QdrantClient:
    """Get or create a singleton QdrantClient instance (thread-safe)"""
    global _qdrant_client
    if _qdrant_client is None:
        with _qdrant_lock:
            if _qdrant_client is None:
                _qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return _qdrant_client

# ======================================================
# Models
# ======================================================
embedder = SentenceTransformer(EMBEDDING_MODEL)
reranker = CrossEncoder(RERANKER_MODEL)

# ======================================================
# BM25 Helpers
# ======================================================
def normalize_token(token: str) -> str:
    import unicodedata
    return unicodedata.normalize("NFKC", token).casefold()

def tokenize_words_for_bm25(text: str) -> List[str]:
    import regex as re
    WORD_RE = re.compile(r"[\p{L}\p{N}]+", re.IGNORECASE)
    
    STOPWORDS_DE = set(
        """aber alle allem allen aller alles als also am an ander andere anderem anderen
        anderer anderes anderm andern anderr auch auf aus bei bin bis bist da dadurch
        dafür dagegen daher damit dann das dasselbe dass dazu dein deine deinem deinen
        deiner deines dem den denn der dessen deshalb die dies diese diesem diesen
        dieser dieses doch dort du durch ein eine einem einen einer eines einig einige
        einigem einigen einiger einiges einmal er es etwas euer eure eurem euren eurer
        eures für gegen gewesen hab habe haben hat hatte hatten hattest hattet hier hin
        hinter ich ihm ihn ihnen ihr ihre ihrem ihren ihrer eures im in ist ja jede
        jedem jeden jeder jedes je jetzt kann kein keine keinem keinen keiner keines
        können könnte machen man manche manchem manchen mancher manches mein meine
        meinem meinen meiner meines mich mir mit muss musste nach nicht nichts noch nun
        nur ob oder ohne sehr sein seine seinem seinen seiner seines selbst sich sie
        sind so solche solchem solchen solcher solches soll sollte sondern sonst um und
        uns unse unser unsrem unsren unsrer unsres unter vom von vor wann war waren
        warst was weg weil weiter welche welchem welchen welcher welches wenn wer wird
        wirst wo wollen wollte würde würden zu zum zur zwar zwischen""".split()
    )
    
    STOPWORDS_EN = set(
        """a about above after again against all am an and any are aren't as at be
        because been before being below between both but by can't cannot could couldn't
        did didn't do does doesn't doing don't down during each few for from further
        had hadn't has hasn't have haven't having he he'd he'll he's her here here's
        hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't
        it it's its itself let's me more most mustn't my myself no nor not of off on
        once only or other ought our ours ourselves out over own same shan't she she'd
        she'll she's should shouldn't so some such than that that's the their theirs
        them themselves then there there's these they they'd they'll they're they've
        this those through to too under until up very was wasn't we we'd we'll we're
        we've were weren't what what's when when's where where's which while who who's
        whom why why's with won't would wouldn't you you'd you'll you're you've your
        yours yourself yourselves""".split()
    )
    
    MCP_LANGUAGE = os.getenv("MCP_LANGUAGE", "DE").upper()
    STOPWORDS = STOPWORDS_EN if MCP_LANGUAGE == "EN" else STOPWORDS_DE
    
    out: List[str] = []
    for m in WORD_RE.finditer(text):
        t = normalize_token(m.group(0))
        if t and t not in STOPWORDS:
            out.append(t)
    return out

def load_bm25_stats() -> Dict[str, Any]:
    if not os.path.exists(BM25_STATS_FILE):
        raise RuntimeError(f"BM25 stats file not found: {BM25_STATS_FILE}")
    with open(BM25_STATS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def bm25_sparse_vector(
    text: str,
    term_df: Dict[str, int],
    avgdl: float,
    vocab: Dict[str, int],
    N_docs: int,
    k1: float,
    b: float,
) -> qmodels.SparseVector:
    terms = tokenize_words_for_bm25(text)
    if not terms:
        return qmodels.SparseVector(indices=[], values=[])
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
        denom = tf + k1 * (1 - b + b * (len(terms) / max(1e-9, avgdl)))
        score = idf * (tf * (k1 + 1)) / max(1e-9, denom)
        indices.append(vocab[t])
        values.append(float(score))
    return qmodels.SparseVector(indices=indices, values=values)

# ======================================================
# RRF Fusion
# ======================================================
def rrf_fusion(results_dense, results_sparse, k: int = 60, top_k: int = 100):
    scores: Dict[str, float] = {}
    payloads: Dict[str, dict] = {}
    for rank, r in enumerate(results_dense, start=1):
        pid = str(r.id)
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
        if pid not in payloads:
            payloads[pid] = r.payload
    for rank, r in enumerate(results_sparse, start=1):
        pid = str(r.id)
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
        if pid not in payloads:
            payloads[pid] = r.payload
    fused = [(payloads[pid].get("text", ""), score, payloads[pid]) for pid, score in scores.items()]
    fused.sort(key=lambda x: x[1], reverse=True)
    return fused[:top_k]

# ======================================================
# Hybrid Search
# ======================================================
def hybrid_search(query: str, top_k: int = TOP_K):
    qc = get_qdrant_client()
    query_vec = embedder.encode(query, normalize_embeddings=True).tolist()
    
    # Dense vector search
    results_dense = qc.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        using="dense",
        with_payload=True,
        limit=top_k,
        search_params=qmodels.SearchParams(hnsw_ef=QDRANT_SEARCH_HNSW_EF, exact=False),
    )
    
    use_sparse = True
    try:
        stats = load_bm25_stats()
        sparse_vec = bm25_sparse_vector(
            text=query,
            term_df=stats["term_df"],
            avgdl=stats["avgdl"],
            vocab=stats["vocab"],
            N_docs=stats["N_docs"],
            k1=BM25_K1,
            b=BM25_B,
        )
    except Exception as e:
        logging.warning("BM25 deaktiviert: %s", e)
        use_sparse = False
        sparse_vec = None
    
    if use_sparse and sparse_vec is not None:
        results_sparse = qc.query_points(
            collection_name=COLLECTION_NAME,
            query=qmodels.SparseVector(indices=sparse_vec.indices, values=sparse_vec.values),
            using="sparse",
            with_payload=True,
            limit=top_k,
            search_params=qmodels.SearchParams(hnsw_ef=QDRANT_SEARCH_HNSW_EF, exact=False),
        )
    else:
        results_sparse = []
    
    return rrf_fusion(results_dense.points, results_sparse.points if hasattr(results_sparse, 'points') else results_sparse, k=60, top_k=top_k)

# ======================================================
# Chunks sortieren
# ======================================================
def _fetch_all_ticket_chunks_for_ordered_concat(qc: QdrantClient, ticket_id: int) -> str:
    all_items: List[Tuple[int, int, str]] = []
    texts_fallback: List[str] = []
    next_page = None
    while True:
        pts, next_page = qc.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=qmodels.Filter(must=[qmodels.FieldCondition(key="ticket_id", match=qmodels.MatchValue(value=ticket_id))]),
            with_payload=True,
            with_vectors=False,
            limit=256,
            offset=next_page,
        )
        for p in pts:
            payload = p.payload or {}
            text = payload.get("text", "")
            if not text:
                continue
            ap = payload.get("article_position")
            ci = payload.get("chunk_index")
            if isinstance(ap, int) and isinstance(ci, int):
                all_items.append((ap, ci, text))
            else:
                texts_fallback.append(text)
        if next_page is None:
            break
    if all_items:
        all_items.sort(key=lambda t: (t[0], t[1]))
        parts = [t[2] for t in all_items]
        if texts_fallback:
            parts.extend(texts_fallback)
        return "\n".join(parts)
    return "\n".join(texts_fallback)

# ======================================================
# Rerank & Group
# ======================================================
def rerank_and_group(query: str, docs: List[Tuple[str, float, dict]], top_k_tickets: int = TOP_TICKETS):
    if not docs:
        return []
    pairs = [(query, d[0]) for d in docs]
    scores = reranker.predict(pairs)
    reranked = [(docs[i][0], float(scores[i]), docs[i][2]) for i in range(len(docs))]
    reranked.sort(key=lambda x: x[1], reverse=True)
    ticket_scores: Dict[int, float] = {}
    for _, score, payload in reranked:
        tid = payload.get("ticket_id")
        if tid is not None:
            ticket_scores[tid] = max(ticket_scores.get(tid, 0.0), score)
    top_tickets = sorted(ticket_scores.items(), key=lambda x: x[1], reverse=True)[:top_k_tickets]
    qc = get_qdrant_client()
    results = []
    for tid, score in top_tickets:
        full_text = _fetch_all_ticket_chunks_for_ordered_concat(qc, tid)
        if not full_text.strip():
            parts = [txt for (txt, _, pl) in reranked if pl.get("ticket_id") == tid and txt]
            full_text = "\n".join(parts)
        results.append((tid, score, full_text))
    return results

# ======================================================
# MCP-Integration (kompatibel mit fastmcp 2.12.3)
# ======================================================
def _run_mcp_server(mode: str = "stdio"):
    from fastmcp import FastMCP
    from fastmcp.tools import Tool

    mcp = FastMCP("rerank_search")

    def search(query: str, top_k: int = 100, top_tickets: int = 10):
        """MCP search function for finding relevant tickets via Qdrant hybrid search"""
        logging.info("MCP search gestartet | query=%s", query)
        initial = hybrid_search(query, top_k=top_k)
        final = rerank_and_group(query, initial, top_k_tickets=top_tickets)
        return [{"ticket_id": tid, "score": score, "text": text} for tid, score, text in final]

    def get_ticket(ticket_id: int):
        """Get complete ticket with all articles directly from Zammad API"""
        logging.info("MCP get_ticket gestartet | ticket_id=%s", ticket_id)
        ticket = _get_ticket_from_zammad(ticket_id)
        if not ticket:
            return {"error": f"Ticket {ticket_id} nicht gefunden", "ticket_id": ticket_id}
        
        articles = _get_ticket_articles_from_zammad(ticket_id)
        
        formatted_articles = []
        for a in articles:
            body_text = _html_to_text(a.get("body", ""))
            formatted_articles.append({
                "from": a.get("from", ""),
                "to": a.get("to", ""),
                "subject": a.get("subject", ""),
                "body": body_text,
                "created_at": a.get("created_at", ""),
                "internal": a.get("internal", False)
            })
        
        return {
            "ticket_id": ticket_id,
            "number": ticket.get("number", ""),
            "title": ticket.get("title", ""),
            "state_id": ticket.get("state_id"),
            "priority_id": ticket.get("priority_id"),
            "customer_id": ticket.get("customer_id"),
            "owner_id": ticket.get("owner_id"),
            "created_at": ticket.get("created_at", ""),
            "updated_at": ticket.get("updated_at", ""),
            "articles": formatted_articles,
            "article_count": len(formatted_articles)
        }

    def zammad_search(query: str, limit: int = ZAMMAD_SEARCH_DEFAULT_LIMIT):
        """Search tickets using Zammad's built-in search (returns previews)"""
        logging.info("MCP zammad_search gestartet | query=%s | limit=%s", query, limit)
        tickets = _search_zammad_tickets(query, limit)
        
        results = []
        for t in tickets:
            ticket_id = t.get("id")
            articles = _get_ticket_articles_from_zammad(ticket_id)
            
            all_body_text = ""
            for a in articles:
                if not a.get("internal", False):
                    body = _html_to_text(a.get("body", ""))
                    all_body_text += body + " "
            
            preview = _truncate_words(all_body_text.strip(), ZAMMAD_SEARCH_PREVIEW_WORDS)
            
            first_article = articles[0] if articles else {}
            
            results.append({
                "ticket_id": ticket_id,
                "number": t.get("number", ""),
                "title": t.get("title", ""),
                "from": first_article.get("from", ""),
                "preview": preview,
                "created_at": t.get("created_at", "")
            })
        
        return {
            "tickets": results,
            "query": query,
            "count": len(results)
        }

    def health_check():
        """Health check endpoint for monitoring"""
        return {"status": "healthy", "service": "rerank_search_mcp", "timestamp": datetime.now().isoformat()}

    mcp.add_tool(Tool.from_function(search))
    mcp.add_tool(Tool.from_function(get_ticket))
    mcp.add_tool(Tool.from_function(zammad_search))
    mcp.add_tool(Tool.from_function(health_check))

    if mode == "http":
        bind_host = "127.0.0.1" if not MCP_API_KEY else "0.0.0.0"
        logging.info(f"MCP-Server läuft per HTTP auf http://{bind_host}:8083")
        if MCP_API_KEY:
            logging.info("MCP API-Key Authentifizierung aktiviert")
        else:
            logging.warning("Kein MCP_API_KEY gesetzt - Server nur lokal erreichbar")
        
        app = FastAPI(title="Rerank Search MCP Server", version="1.0.0")
        
        @app.middleware("http")
        async def verify_api_key(request: Request, call_next):
            if request.url.path == "/health":
                return await call_next(request)
            if MCP_API_KEY:
                provided_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
                if provided_key != MCP_API_KEY:
                    return JSONResponse(
                        status_code=401,
                        content={"error": "Unauthorized", "message": "Invalid or missing API key"}
                    )
            return await call_next(request)
        
        @app.get("/health")
        async def health():
            return JSONResponse({
                "status": "healthy",
                "service": "rerank_search_mcp",
                "timestamp": datetime.now().isoformat(),
                "auth_enabled": MCP_API_KEY is not None
            })
        
        @app.get("/search")
        async def http_search(query: str, top_k: int = 100, top_tickets: int = 10):
            """HTTP endpoint for hybrid search (Qdrant + Reranking)"""
            try:
                result = search(query, top_k, top_tickets)
                return {"results": result, "query": query}
            except Exception as e:
                import traceback
                return JSONResponse(
                    status_code=500,
                    content={"error": str(e), "traceback": traceback.format_exc(), "query": query}
                )
        
        @app.get("/ticket/{ticket_id}")
        async def http_get_ticket(ticket_id: int):
            """HTTP endpoint to get complete ticket from Zammad"""
            try:
                result = get_ticket(ticket_id)
                if "error" in result:
                    return JSONResponse(status_code=404, content=result)
                return result
            except Exception as e:
                import traceback
                return JSONResponse(
                    status_code=500,
                    content={"error": str(e), "traceback": traceback.format_exc(), "ticket_id": ticket_id}
                )
        
        @app.get("/zammad-search")
        async def http_zammad_search(query: str, limit: int = ZAMMAD_SEARCH_DEFAULT_LIMIT):
            """HTTP endpoint for Zammad built-in search"""
            try:
                result = zammad_search(query, limit)
                return result
            except Exception as e:
                import traceback
                return JSONResponse(
                    status_code=500,
                    content={"error": str(e), "traceback": traceback.format_exc(), "query": query}
                )
        
        uvicorn.run(app, host=bind_host, port=8083, log_level="error")
    else:
        logging.info("MCP-Server läuft per STDIO")
        mcp.run()


# ======================================================
# Main
# ======================================================
if __name__ == "__main__":
    if len(sys.argv) >= 2:
        arg = sys.argv[1].lower()
        if arg in {"--mcp", "-m"}:
            _run_mcp_server("stdio")
            sys.exit(0)
        elif arg in {"--http", "-h"}:
            _run_mcp_server("http")
            sys.exit(0)

    if len(sys.argv) < 2:
        print("Usage:")
        print("  MCP-Server (STDIO): python rerank_search.py --mcp")
        print("  MCP-Server (HTTP) : python rerank_search.py --http")
        print('  CLI-Suche         : python rerank_search.py "Suchanfrage"')
        sys.exit(1)

    query = sys.argv[1]
    logging.info("Suche nach: %s", query)
    initial = hybrid_search(query, top_k=TOP_K)
    final = rerank_and_group(query, initial, top_k_tickets=TOP_TICKETS)

    print("\n=== Top Tickets ===")
    for i, (tid, score, text) in enumerate(final, 1):
        print(f"\n[{i}] Ticket {tid} | Score={score:.4f}")
        print("=" * 60)
        print(text[:4000])
        print("=" * 60)