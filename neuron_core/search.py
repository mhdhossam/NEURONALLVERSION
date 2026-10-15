"""Low-cost external search as an input to Neuron's exploration loop.

No language model is required. Providers return sourced observations; search
snippets are not promoted to verified facts. Use SQLite caching to avoid repeat
network requests, and keep retrieval/exploration bounded.
"""
from __future__ import annotations

from contextlib import closing
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, urlencode, urljoin, urlparse
from urllib.request import Request, urlopen
import hashlib

from .engine import NeuronGraph
from .processing import trace_processing


class SearchError(RuntimeError):
    """A search provider failed or returned unusable data."""


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    provider: str


class SearchProvider(Protocol):
    """Minimal provider contract so online search can be swapped without changing Neuron."""

    @property
    def cache_key(self) -> str: ...

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]: ...


def _validate_search(query: str, limit: int) -> None:
    if not query or not query.strip():
        raise ValueError("query must not be empty")
    if limit < 1 or limit > 20:
        raise ValueError("limit must be between 1 and 20")


class _DuckDuckGoParser(HTMLParser):
    """Small tolerant parser for DuckDuckGo's HTML result page."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self.snippets: list[str] = []
        self._capture: tuple[str, str, int] | None = None
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        classes = set((values.get("class") or "").split())
        if tag == "a" and "result__a" in classes:
            self.results.append({
                "title": "", "url": values.get("href") or "", "snippet": ""
            })
            self._capture = ("title", tag, len(self.results) - 1)
            self._parts = []
        elif "result__snippet" in classes:
            self._capture = ("snippet", tag, len(self.snippets))
            self._parts = []

    def handle_data(self, data: str) -> None:
        if self._capture is not None:
            self._parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._capture is None:
            return
        kind, ending_tag, index = self._capture
        if tag != ending_tag:
            return
        value = " ".join(" ".join(self._parts).split()).strip()
        if kind == "title" and index < len(self.results):
            self.results[index]["title"] = value
        elif kind == "snippet":
            self.snippets.append(value)
        self._capture = None
        self._parts = []


class DuckDuckGoHTMLProvider:
    """Free general-web search adapter; may be rate-limited or changed upstream."""

    endpoint = "https://html.duckduckgo.com/html/"

    def __init__(self, *, timeout: float = 8.0, max_response_bytes: int = 2_000_000,
                 user_agent: str = "NeuronGraphResearch/0.1 (local graph search)") -> None:
        if timeout <= 0 or max_response_bytes < 1024:
            raise ValueError("timeout must be positive and response limit at least 1024 bytes")
        self.timeout = timeout
        self.max_response_bytes = max_response_bytes
        self.user_agent = user_agent

    @property
    def cache_key(self) -> str:
        return "duckduckgo-html-v1"

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        _validate_search(query, limit)
        url = self.endpoint + "?" + urlencode({"q": query.strip()})
        request = Request(url, headers={"User-Agent": self.user_agent, "Accept": "text/html"})
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read(self.max_response_bytes + 1)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise SearchError(f"DuckDuckGo search failed: {exc}") from exc
        if len(raw) > self.max_response_bytes:
            raise SearchError("search response exceeded configured byte limit")
        parser = _DuckDuckGoParser()
        try:
            parser.feed(raw.decode("utf-8", errors="replace"))
        except Exception as exc:
            raise SearchError(f"could not parse DuckDuckGo result HTML: {exc}") from exc

        output = []
        for index, row in enumerate(parser.results):
            target = row["url"]
            parsed = urlparse(target)
            if parsed.netloc.endswith("duckduckgo.com"):
                redirect = parse_qs(parsed.query).get("uddg", [])
                if redirect:
                    target = redirect[0]
            target = urljoin(self.endpoint, target)
            if urlparse(target).scheme not in {"http", "https"}:
                continue
            title = row["title"].strip()
            if not title:
                continue
            snippet = parser.snippets[index] if index < len(parser.snippets) else ""
            output.append(SearchResult(title[:300], target, snippet[:1200], self.cache_key))
            if len(output) >= limit:
                break
        return output


class WikipediaSearchProvider:
    """No-key search provider for Wikipedia, useful as a dependable fallback."""

    endpoint = "https://en.wikipedia.org/w/rest.php/v1/search/page"

    def __init__(self, *, timeout: float = 8.0, max_response_bytes: int = 1_000_000) -> None:
        if timeout <= 0 or max_response_bytes < 1024:
            raise ValueError("timeout must be positive and response limit at least 1024 bytes")
        self.timeout = timeout
        self.max_response_bytes = max_response_bytes

    @property
    def cache_key(self) -> str:
        return "wikipedia-rest-v1"

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        _validate_search(query, limit)
        url = self.endpoint + "?" + urlencode({"q": query.strip(), "limit": limit})
        request = Request(url, headers={
            "User-Agent": "NeuronGraphResearch/0.1 (local graph search)",
            "Accept": "application/json",
        })
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read(self.max_response_bytes + 1)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise SearchError(f"Wikipedia search failed: {exc}") from exc
        if len(raw) > self.max_response_bytes:
            raise SearchError("search response exceeded configured byte limit")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SearchError("Wikipedia returned invalid JSON") from exc

        pages = payload.get("pages", [])
        if isinstance(pages, dict):
            pages = list(pages.values())
        output = []
        for page in pages:
            title = str(page.get("title") or "").strip()
            if not title:
                continue
            key = str(page.get("key") or title.replace(" ", "_"))
            output.append(SearchResult(
                title=title[:300],
                url="https://en.wikipedia.org/wiki/" + quote(key.replace(" ", "_"), safe="/()'"),
                snippet=" ".join(str(page.get("description") or page.get("excerpt") or "").split())[:1200],
                provider=self.cache_key,
            ))
            if len(output) >= limit:
                break
        return output


class FallbackSearchProvider:
    """Try providers in order; returns the first non-empty result set."""

    def __init__(self, *providers: SearchProvider) -> None:
        if not providers:
            raise ValueError("at least one search provider is required")
        self.providers = providers

    @property
    def cache_key(self) -> str:
        return "fallback:" + ",".join(provider.cache_key for provider in self.providers)

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        errors = []
        for provider in self.providers:
            try:
                results = provider.search(query, limit=limit)
            except SearchError as exc:
                errors.append(str(exc))
                continue
            if results:
                return results[:limit]
        if errors:
            raise SearchError("all search providers failed: " + " | ".join(errors))
        return []


class SQLiteCachedSearchProvider:
    """Small bounded TTL cache around any provider; only titles/snippets are cached."""

    def __init__(self, provider: SearchProvider, path: str | Path = ".neuron_cache/search.sqlite3",
                 *, ttl_seconds: float = 86_400, max_entries: int = 5_000) -> None:
        if ttl_seconds <= 0 or max_entries < 1:
            raise ValueError("ttl_seconds and max_entries must be positive")
        self.provider = provider
        self.path = Path(path)
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries

    @property
    def cache_key(self) -> str:
        return "sqlite-cache:" + self.provider.cache_key

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=3.0)
        connection.execute(
            "CREATE TABLE IF NOT EXISTS search_cache ("
            "cache_id TEXT PRIMARY KEY, created_at REAL NOT NULL, payload TEXT NOT NULL)"
        )
        return connection

    def search(self, query: str, *, limit: int = 5) -> list[SearchResult]:
        _validate_search(query, limit)
        normal = " ".join(query.casefold().split())
        cache_id = hashlib.sha256(
            f"{self.provider.cache_key}\0{normal}\0{limit}".encode("utf-8")
        ).hexdigest()
        now = time.time()
        with closing(self._connect()) as db:
            db.execute("BEGIN")
            row = db.execute(
                "SELECT created_at, payload FROM search_cache WHERE cache_id = ?",
                (cache_id,),
            ).fetchone()
            if row and now - float(row[0]) <= self.ttl_seconds:
                payload = json.loads(row[1])
                return [SearchResult(**item) for item in payload]
            db.execute("DELETE FROM search_cache WHERE created_at < ?", (now - self.ttl_seconds,))
        results = self.provider.search(query, limit=limit)
        payload = json.dumps([asdict(item) for item in results], ensure_ascii=False)
        with closing(self._connect()) as db:
            db.execute("BEGIN")
            db.execute(
                "INSERT OR REPLACE INTO search_cache(cache_id, created_at, payload) VALUES (?, ?, ?)",
                (cache_id, now, payload),
            )
            db.execute(
                "DELETE FROM search_cache WHERE cache_id NOT IN "
                "(SELECT cache_id FROM search_cache ORDER BY created_at DESC LIMIT ?)",
                (self.max_entries,),
            )
        return results


def think_with_search(
    graph: NeuronGraph,
    query: str,
    provider: SearchProvider,
    *,
    result_limit: int = 5,
    snippet_char_limit: int = 1200,
    max_hops: int = 2,
    force_search: bool = False,
) -> dict[str, Any]:
    """Use local graph knowledge first; search only when no lexical local seed exists.

    Search observations are stored as source nodes and linked to a query node.
    They are not marked as verified facts and are not converted into invented
    subject-predicate-object claims. A processing trace explores local and newly
    ingested sources after retrieval.

    Set force_search=True to supplement existing local knowledge deliberately.
    """
    _validate_search(query, result_limit)
    if snippet_char_limit < 1 or max_hops < 0 or max_hops > 5:
        raise ValueError("snippet_char_limit must be positive and max_hops must be between 0 and 5")

    local_candidates = graph.find_candidates(query, top_k=5)
    should_search = force_search or not local_candidates
    results: list[SearchResult] = provider.search(query, limit=result_limit) if should_search else []

    query_node_id = None
    source_node_ids: list[str] = []
    if results:
        for neuron_id, neuron in graph.neurons.items():
            if neuron.kind == "search_query" and neuron.metadata.get("search_query_key") == query.casefold().strip():
                query_node_id = neuron_id
                break
        if query_node_id is None:
            query_node_id = graph.add_concept(
                f"Search query: {query.strip()}",
                kind="search_query",
                content=query.strip(),
                metadata={"search_query_key": query.casefold().strip(), "query": query.strip()},
            )

        for result in results:
            if not result.url.startswith(("http://", "https://")):
                continue
            source_id = next((
                neuron_id for neuron_id, neuron in graph.neurons.items()
                if neuron.kind == "external_source" and neuron.metadata.get("url") == result.url
            ), None)
            if source_id is None:
                source_id = graph.add_concept(
                    result.title or result.url,
                    kind="external_source",
                    content=result.snippet[:snippet_char_limit],
                    attributes={"url": result.url, "provider": result.provider},
                    metadata={
                        "url": result.url,
                        "provider": result.provider,
                        "search_queries": [query.strip()],
                        "source_kind": "external_search_result",
                    },
                )
            else:
                neuron = graph.neurons[source_id]
                searches = neuron.metadata.setdefault("search_queries", [])
                if query.strip() not in searches:
                    searches.append(query.strip())
                    graph.refresh_neuron_metadata(source_id)
            source_node_ids.append(source_id)
            existing_edge = any(
                graph.synapses[sid].target_id == source_id
                for sid in graph._outgoing_by_relation.get(query_node_id, {}).get("found_by_search", ())
            )
            if not existing_edge:
                graph.connect(query_node_id, source_id, "found_by_search", weight=0.8)

    trace = trace_processing(
        graph, query, max_hops=max_hops, seed_limit=10,
        max_nodes=250, max_context_branches=50,
    )
    return {
        "query": query,
        "local_candidate_count": len(local_candidates),
        "search_triggered": should_search,
        "search_reason": "forced" if force_search else ("no_local_lexical_seed" if not local_candidates else "local_candidates_found"),
        "provider": provider.cache_key,
        "search_results": [asdict(result) for result in results],
        "query_node_id": query_node_id,
        "source_node_ids": source_node_ids,
        "processing_trace": trace,
        "decision": None,
        "note": (
            "Search is external observation, not verified truth. Neuron explores "
            "the resulting graph state; this function does not invent extracted facts "
            "or select a final answer."
        ),
    }
