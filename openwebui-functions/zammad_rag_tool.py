"""
Zammad RAG MCP Tool for Open WebUI
Ermöglicht Suche in Zammad-Tickets via Hybrid Search und Zammad API
"""

from pydantic import BaseModel, Field
from typing import Optional
import requests

MCP_URL = "http://192.168.1.28:8083"
MCP_API_KEY = "mcp_Xk9zPm4vQr2tYw5nBc8fHj1lMa7sUe0g"


class Tools:
    class Valves(BaseModel):
        mcp_url: str = Field(default=MCP_URL, description="MCP Server URL")
        mcp_api_key: str = Field(default=MCP_API_KEY, description="MCP API Key")
        pass

    def __init__(self):
        self.valves = self.Valves()
        pass

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        top_tickets: int = 5,
    ) -> str:
        """
        Durchsucht Qdrant Vektordatenbank mit Hybrid Search (Dense + Sparse + Reranking).
        Findet semantisch ähnliche Tickets basierend auf dem Inhalt.
        Verwende dies für intelligente Suche nach Ticket-Inhalten.

        :param query: Suchbegriff oder Frage (z.B. "Passwort zurücksetzen", "VPN Probleme")
        :param top_k: Anzahl der Chunks die durchsucht werden (default: 10)
        :param top_tickets: Anzahl der zurückgegebenen Tickets (default: 5)
        :return: JSON mit relevanten Tickets inkl. Scores und Text-Ausschnitten
        """
        try:
            response = requests.get(
                f"{self.valves.mcp_url}/search",
                params={"query": query, "top_k": top_k, "top_tickets": top_tickets},
                headers={"X-API-Key": self.valves.mcp_api_key},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            if not data.get("results"):
                return f"Keine Ergebnisse für: {query}"

            results = []
            for r in data["results"]:
                results.append(
                    f"Ticket #{r['ticket_id']} (Score: {r['score']:.2f})\n{r['text'][:500]}..."
                )

            return f"Hybrid Search Ergebnisse für '{query}':\n\n" + "\n\n---\n\n".join(
                results
            )

        except Exception as e:
            return f"Fehler bei Hybrid Search: {str(e)}"

    def zammad_search(
        self,
        query: str,
        limit: int = 5,
    ) -> str:
        """
        Durchsucht Zammad direkt mit der eingebauten Suche.
        Verwende dies für schnelle Suche nach Ticket-Titeln, Nummern oder Stichworten.

        :param query: Suchbegriff (z.B. Ticket-Nummer, Titel, Stichwort)
        :param limit: Maximale Anzahl Ergebnisse (default: 5)
        :return: JSON mit gefundenen Tickets inkl. Preview
        """
        try:
            response = requests.get(
                f"{self.valves.mcp_url}/zammad-search",
                params={"query": query, "limit": limit},
                headers={"X-API-Key": self.valves.mcp_api_key},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            if not data.get("tickets"):
                return f"Keine Zammad-Tickets gefunden für: {query}"

            results = []
            for t in data["tickets"]:
                results.append(
                    f"Ticket #{t['ticket_id']} ({t['number']}) - {t['title']}\n"
                    f"Von: {t['from']} | Erstellt: {t['created_at']}\n"
                    f"Preview: {t['preview'][:300]}..."
                )

            return f"Zammad Search Ergebnisse für '{query}' ({data['count']} gefunden):\n\n" + "\n\n---\n\n".join(
                results
            )

        except Exception as e:
            return f"Fehler bei Zammad Search: {str(e)}"

    def get_ticket(
        self,
        ticket_id: int,
    ) -> str:
        """
        Holt ein spezifisches Zammad-Ticket mit allen Artikeln/Kommunikation.
        Verwende dies um Details zu einem bestimmten Ticket abzurufen.

        :param ticket_id: Die Zammad Ticket-ID (z.B. 18348)
        :return: Vollständiges Ticket mit allen Artikeln
        """
        try:
            response = requests.get(
                f"{self.valves.mcp_url}/ticket/{ticket_id}",
                headers={"X-API-Key": self.valves.mcp_api_key},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            if "error" in data:
                return f"Fehler: {data['error']}"

            articles = []
            for a in data.get("articles", []):
                articles.append(
                    f"[{a['created_at']}] {a['from']}\n"
                    f"Betreff: {a.get('subject', 'N/A')}\n"
                    f"{a['body'][:500]}..."
                )

            return (
                f"Ticket #{data['ticket_id']} ({data['number']})\n"
                f"Titel: {data['title']}\n"
                f"Status: {data.get('state_id', 'N/A')} | Priorität: {data.get('priority_id', 'N/A')}\n"
                f"Erstellt: {data['created_at']} | Aktualisiert: {data['updated_at']}\n"
                f"Artikel ({data.get('article_count', 0)}):\n\n"
                + "\n\n---\n\n".join(articles)
            )

        except Exception as e:
            return f"Fehler beim Abrufen des Tickets: {str(e)}"
