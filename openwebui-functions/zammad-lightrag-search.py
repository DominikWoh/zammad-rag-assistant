"""
title: Zammad-LightRAG Search
author: duftertyp
author_url: https://github.com/duftertyp/zammad-lightrag
version: 2.0.0
license: MIT
description: Search Zammad helpdesk tickets via LightRAG knowledge graph. Adds a search button to Open WebUI that queries your synced ticket knowledge base with German-friendly responses and source attribution.
requirements: requests
"""

import json
import requests
from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Valves (Configuration)
# ---------------------------------------------------------------------------
# These appear in the Open WebUI admin panel when you add the Function.
# Each field can be edited in the UI; users don't have to touch the code.

class Valves(BaseModel):
    """User-configurable settings (shown in Open WebUI admin panel)."""

    LIGHTRAG_URL: str = Field(
        default="http://host.docker.internal:9621",
        description="URL of the LightRAG server. Examples: http://localhost:9621, http://192.168.0.153:9621",
    )

    QUERY_MODE: str = Field(
        default="hybrid",
        description="RAG query mode: local, global, hybrid, naive, mix. Hybrid combines vector search and knowledge graph.",
    )

    TOP_K: int = Field(
        default=20,
        description="Number of top entities/relations to retrieve from knowledge graph.",
    )

    CHUNK_TOP_K: int = Field(
        default=5,
        description="Number of text chunks per entity to include in the LLM context.",
    )

    MAX_TOKENS: int = Field(
        default=4000,
        description="Maximum tokens for the LLM response.",
    )

    LANGUAGE: str = Field(
        default="German",
        description="Language for LightRAG responses. Examples: English, German, French.",
    )

    SHOW_SOURCES: bool = Field(
        default=True,
        description="Show ticket source references such as zammad-ticket-123.",
    )

    REQUEST_TIMEOUT: int = Field(
        default=120,
        description="HTTP request timeout in seconds. LightRAG can be slow on first query.",
    )


# ---------------------------------------------------------------------------
# Main Function Class
# ---------------------------------------------------------------------------

class Tools:
    """Open WebUI Tool for searching Zammad tickets via LightRAG."""

    def __init__(self):
        self.valves = Valves()
        self.citation = False  # Open WebUI internal: enables source rendering

    # -----------------------------------------------------------------------
    # Tool: search_zammad
    # -----------------------------------------------------------------------
    async def search_zammad(
        self,
        query: str,
        mode: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> str:
        """
        Search the Zammad ticket knowledge base and answer the question.
        Use this when the user asks about past support tickets, known issues,
        customer problems, or how something was resolved.

        :param query: The search query, e.g. VPN Probleme Mueller or Drucker Maschine 307
        :param mode: Override query mode (local/global/hybrid/naive/mix). Default = auto
        :param top_k: Override number of results. Default = valves setting
        :return: Answer with optional source references
        """
        try:
            # Try requested/preferred mode first
            preferred = mode or self.valves.QUERY_MODE
            result = self._query_lightrag(query, preferred, top_k or self.valves.TOP_K)

            # If hybrid/global returns nothing, fall back to naive (better for short terms)
            if "no relevant context" in result.lower() and preferred in ("hybrid", "global", "mix"):
                naive_result = self._query_lightrag(query, "naive", top_k or self.valves.TOP_K)
                if "[SOURCES]" in naive_result or (
                    "no relevant" not in naive_result.lower() and len(naive_result) > 100
                ):
                    result = naive_result

            return result
        except Exception as e:
            return f"[ERROR] LightRAG query failed: {str(e)}"

    # -----------------------------------------------------------------------
    # Internal: query LightRAG API
    # -----------------------------------------------------------------------
    def _query_lightrag(self, query: str, mode: str, top_k: int) -> str:
        url = f"{self.valves.LIGHTRAG_URL.rstrip('/')}/query"
        payload = {
            "query": query,
            "mode": mode,
            "top_k": top_k,
            "chunk_top_k": self.valves.CHUNK_TOP_K,
            "max_total_tokens": self.valves.MAX_TOKENS,
            "response_type": "Multiple Paragraphs",
            "stream": False,
            "only_need_context": False,
        }

        resp = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=self.valves.REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        answer = data.get("response", "No answer received.")

        # Add source references if enabled
        if self.valves.SHOW_SOURCES:
            references = data.get("references", [])
            if references:
                sources = ", ".join(
                    ref.get("file_path", "unknown")
                    .replace("zammad-ticket-", "#")
                    for ref in references[:5]
                )
                answer += f"\n\n[SOURCES] {sources}"

        return answer

    # -----------------------------------------------------------------------
    # Internal: health check (used by the action button)
    # -----------------------------------------------------------------------
    async def lightrag_status(self) -> str:
        """
        Check whether the LightRAG server is reachable and report stats.
        Use this to verify the connection or see how many tickets are indexed.
        """
        try:
            url = f"{self.valves.LIGHTRAG_URL.rstrip('/')}/health"
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            data = resp.json()

            status = data.get("status", "unknown")
            config = data.get("configuration", {})

            llm = config.get("llm_model", "unknown")
            embedding = config.get("embedding_model", "unknown")
            doc_count = 0

            # Count docs via separate endpoint
            try:
                docs = requests.get(
                    f"{self.valves.LIGHTRAG_URL.rstrip('/')}/documents",
                    timeout=5,
                ).json()
                if isinstance(docs, list):
                    doc_count = len(docs)
            except Exception:
                pass

            return (
                f"[OK] LightRAG is reachable\n"
                f"- Status: {status}\n"
                f"- LLM: {llm}\n"
                f"- Embedding: {embedding}\n"
                f"- Language: {config.get('summary_language', '?')}\n"
                f"- Indexed documents: {doc_count}"
            )
        except requests.exceptions.ConnectionError:
            return (
                f"[ERROR] LightRAG not reachable at {self.valves.LIGHTRAG_URL}\n"
                f"Check if container runs: docker ps | grep lightrag"
            )
        except Exception as e:
            return f"[ERROR] {str(e)}"
