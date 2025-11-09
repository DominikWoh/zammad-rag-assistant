#!/usr/bin/env python3
# rerank_search.py
from __future__ import annotations

import os
import sys
import json
import math
import logging
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer, CrossEncoder
from fastmcp import FastMCP
from fastmcp.tools import Tool
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

# ======================================================
# Setup & Config
# ======================================================
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "zammad_tickets")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

BM25_STATS_FILE = os.getenv("BM25_STATS_FILE", "bm25_stats.json")
BM25_K1 = float(os.getenv("BM25_K1", "0.9"))
BM25_B = float(os.getenv("BM25_B", "0.4"))

QDRANT_SEARCH_HNSW_EF = int(os.getenv("QDRANT_SEARCH_HNSW_EF", "128"))
TOP_K = int(os.getenv("TOP_K", "100"))
TOP_TICKETS = int(os.getenv("TOP_TICKETS", "10"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

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
    STOPWORDS = STOPWORDS_DE
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
    qc = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    query_vec = embedder.encode(query, normalize_embeddings=True).tolist()
    results_dense = qc.search(
        collection_name=COLLECTION_NAME,
        query_vector={"name": "dense", "vector": query_vec},
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
        results_sparse = qc.search(
            collection_name=COLLECTION_NAME,
            query_vector={"name": "sparse", "vector": {"indices": sparse_vec.indices, "values": sparse_vec.values}},
            with_payload=True,
            limit=top_k,
            search_params=qmodels.SearchParams(hnsw_ef=QDRANT_SEARCH_HNSW_EF, exact=False),
        )
    else:
        results_sparse = []
    return rrf_fusion(results_dense, results_sparse, k=60, top_k=top_k)

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
    qc = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
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
        """MCP search function for finding relevant tickets"""
        logging.info("MCP search gestartet | query=%s", query)
        initial = hybrid_search(query, top_k=top_k)
        final = rerank_and_group(query, initial, top_k_tickets=top_tickets)
        return [{"ticket_id": tid, "score": score, "text": text} for tid, score, text in final]

    def health_check():
        """Health check endpoint for monitoring"""
        return {"status": "healthy", "service": "rerank_search_mcp", "timestamp": datetime.now().isoformat()}

    mcp.add_tool(Tool.from_function(search))
    mcp.add_tool(Tool.from_function(health_check))

    if mode == "http":
        logging.info("MCP-Server läuft per HTTP auf http://127.0.0.1:8083")
        
        # Create FastAPI app for HTTP mode
        app = FastAPI(title="Rerank Search MCP Server", version="1.0.0")
        
        @app.get("/health")
        async def health():
            return JSONResponse({
                "status": "healthy",
                "service": "rerank_search_mcp",
                "timestamp": datetime.now().isoformat()
            })
        
        @app.get("/search")
        async def http_search(query: str, top_k: int = 100, top_tickets: int = 10):
            """HTTP endpoint for search functionality"""
            try:
                result = search(query, top_k, top_tickets)
                return {"results": result, "query": query}
            except Exception as e:
                return JSONResponse(
                    status_code=500,
                    content={"error": str(e), "query": query}
                )
        
        # Run the server
        uvicorn.run(app, host="127.0.0.1", port=8083, log_level="error")
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