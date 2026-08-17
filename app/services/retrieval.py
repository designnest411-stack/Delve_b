"""
Delve Retrieval Services
─────────────────────────
Fetches papers from ArXiv, Semantic Scholar, and web via Tavily.
All methods are async and return normalized paper dicts.
"""

import asyncio
import logging
from typing import Any
from datetime import datetime
import random
import re

import httpx

from app.core.config import settings

logger = logging.getLogger("delve.retrieval")


def _author_string_from_parts(names: list[str], limit: int = 5) -> str:
    cleaned = [str(name).strip() for name in names if str(name).strip()]
    if not cleaned:
        return "Unknown"
    short = cleaned[:limit]
    text = ", ".join(short)
    if len(cleaned) > limit:
        text += " et al."
    return text


def _strip_html_tags(text: str) -> str:
    cleaned = re.sub(r"<[^>]+>", " ", str(text or ""))
    return re.sub(r"\s+", " ", cleaned).strip()


def _openalex_abstract(abstract_index: dict[str, list[int]] | None) -> str:
    if not isinstance(abstract_index, dict) or not abstract_index:
        return ""
    slots: dict[int, str] = {}
    for token, positions in abstract_index.items():
        if not isinstance(positions, list):
            continue
        for pos in positions:
            if isinstance(pos, int) and pos not in slots:
                slots[pos] = token
    if not slots:
        return ""
    return " ".join(slots[i] for i in sorted(slots))


# ── Paper Schema ──────────────────────────────────────────────────────────

def _make_paper(
    title: str,
    authors: str,
    abstract: str,
    year: int | str,
    source: str,
    url: str = "",
    doi: str = "",
    citation_count: int = 0,
    arxiv_id: str = "",
) -> dict[str, Any]:
    """Create a normalized paper dict."""
    return {
        "title": title.strip(),
        "authors": authors.strip(),
        "abstract": abstract.strip(),
        "year": str(year),
        "source": source,
        "url": url,
        "doi": doi,
        "citation_count": citation_count,
        "arxiv_id": arxiv_id,
        "paper_id": doi or arxiv_id or title[:60].lower().replace(" ", "_"),
    }


# ── ArXiv ─────────────────────────────────────────────────────────────────

async def fetch_arxiv(query: str, max_results: int = 10) -> list[dict]:
    """
    Fetch papers from arXiv API.
    Uses the `arxiv` library in a thread executor since it's synchronous.
    """
    try:
        import arxiv

        def _search():
            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            papers = []
            for result in client.results(search):
                authors = ", ".join(a.name for a in result.authors[:5])
                if len(result.authors) > 5:
                    authors += " et al."

                papers.append(_make_paper(
                    title=result.title,
                    authors=authors,
                    abstract=result.summary,
                    year=result.published.year if result.published else "",
                    source="arxiv",
                    url=result.entry_id,
                    arxiv_id=result.entry_id.split("/")[-1] if result.entry_id else "",
                ))
            return papers

        results = await asyncio.to_thread(_search)
        logger.info("ArXiv returned %d papers for '%s'", len(results), query)
        return results

    except Exception as e:
        logger.error("ArXiv fetch failed: %s", e)
        return []


# ── Semantic Scholar ──────────────────────────────────────────────────────

async def fetch_semantic_scholar(query: str, max_results: int = 10) -> list[dict]:
    """
    Fetch papers from Semantic Scholar free API.
    Endpoint: https://api.semanticscholar.org/graph/v1/paper/search
    """
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": max_results,
        "fields": "title,authors,abstract,year,citationCount,externalIds,url",
    }

    retries = max(1, settings.semantic_scholar_max_retries)
    backoff_base = max(0.5, settings.semantic_scholar_base_backoff_seconds)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            for attempt in range(1, retries + 1):
                resp = await client.get(url, params=params)

                if resp.status_code == 200:
                    data = resp.json()
                    papers = []
                    for p in data.get("data", []):
                        if not p.get("abstract"):
                            continue

                        authors = ", ".join(
                            a.get("name", "") for a in (p.get("authors") or [])[:5]
                        )
                        ext_ids = p.get("externalIds") or {}

                        papers.append(_make_paper(
                            title=p.get("title", ""),
                            authors=authors,
                            abstract=p.get("abstract", ""),
                            year=p.get("year", ""),
                            source="semantic_scholar",
                            url=p.get("url", ""),
                            doi=ext_ids.get("DOI", ""),
                            citation_count=p.get("citationCount", 0),
                            arxiv_id=ext_ids.get("ArXiv", ""),
                        ))

                    logger.info("Semantic Scholar returned %d papers for '%s'", len(papers), query)
                    return papers

                if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                    jitter = random.uniform(0, 0.5)
                    delay = backoff_base * (2 ** (attempt - 1)) + jitter
                    logger.warning(
                        "Semantic Scholar HTTP %s (attempt %d/%d), backing off %.1fs",
                        resp.status_code,
                        attempt,
                        retries,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue

                logger.error("Semantic Scholar HTTP %s", resp.status_code)
                return []

    except Exception as e:
        logger.error("Semantic Scholar fetch failed: %s", e)
        return []


async def fetch_openalex(query: str, max_results: int = 10) -> list[dict]:
    """Fetch papers from OpenAlex."""
    url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "per-page": max_results,
        "filter": "has_abstract:true",
    }

    try:
        async with httpx.AsyncClient(timeout=30.0, headers={"User-Agent": "Delve/1.0"}) as client:
            resp = await client.get(url, params=params)
            if resp.status_code != 200:
                logger.error("OpenAlex HTTP %s", resp.status_code)
                return []

            data = resp.json()
            papers = []
            for item in data.get("results", []):
                abstract = _openalex_abstract(item.get("abstract_inverted_index"))
                if not abstract:
                    continue
                authors = _author_string_from_parts([
                    (authorship.get("author") or {}).get("display_name", "")
                    for authorship in (item.get("authorships") or [])
                ])
                ids = item.get("ids") or {}
                doi = str(ids.get("doi", "")).replace("https://doi.org/", "")
                primary_location = item.get("primary_location") or {}
                url_value = primary_location.get("landing_page_url") or primary_location.get("pdf_url") or item.get("id", "")

                papers.append(_make_paper(
                    title=item.get("title", ""),
                    authors=authors,
                    abstract=abstract,
                    year=item.get("publication_year", ""),
                    source="openalex",
                    url=url_value,
                    doi=doi,
                    citation_count=item.get("cited_by_count", 0),
                ))

            logger.info("OpenAlex returned %d papers for '%s'", len(papers), query)
            return papers
    except Exception as e:
        logger.error("OpenAlex fetch failed: %s", e)
        return []


async def fetch_crossref(query: str, max_results: int = 10) -> list[dict]:
    """Fetch metadata-rich papers from Crossref."""
    url = "https://api.crossref.org/works"
    params = {
        "query.bibliographic": query,
        "rows": max_results,
    }
    contact = (settings.crossref_contact_email or "").strip()
    headers = {
        "User-Agent": f"Delve/1.0 (mailto:{contact})" if contact else "Delve/1.0",
    }

    try:
        async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
            resp = await client.get(url, params=params)
            if resp.status_code != 200:
                logger.error("Crossref HTTP %s", resp.status_code)
                return []

            message = (resp.json() or {}).get("message", {})
            papers = []
            for item in message.get("items", []):
                abstract = _strip_html_tags(item.get("abstract", ""))
                if not abstract:
                    title_text = " ".join(item.get("title") or [])
                    subtitle = " ".join(item.get("subtitle") or [])
                    container = " ".join(item.get("container-title") or [])
                    abstract = ". ".join(part for part in [title_text, subtitle, container] if part)
                if len(abstract.strip()) < 40:
                    continue

                authors = _author_string_from_parts([
                    " ".join(part for part in [a.get("given", ""), a.get("family", "")] if part)
                    for a in (item.get("author") or [])
                ])
                year_parts = (
                    (item.get("published-print") or {}).get("date-parts")
                    or (item.get("published-online") or {}).get("date-parts")
                    or []
                )
                year = year_parts[0][0] if year_parts and year_parts[0] else ""
                title = " ".join(item.get("title") or []) or "Untitled"
                doi = item.get("DOI", "")
                url_value = item.get("URL", "")

                papers.append(_make_paper(
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    year=year,
                    source="crossref",
                    url=url_value,
                    doi=doi,
                    citation_count=int(item.get("is-referenced-by-count", 0) or 0),
                ))

            logger.info("Crossref returned %d papers for '%s'", len(papers), query)
            return papers
    except Exception as e:
        logger.error("Crossref fetch failed: %s", e)
        return []


async def fetch_github_repositories(query: str, max_results: int = 5) -> list[dict]:
    """Fetch notable GitHub repositories related to the topic."""
    url = "https://api.github.com/search/repositories"
    params = {
        "q": query,
        "sort": "stars",
        "order": "desc",
        "per_page": max_results,
    }
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "Delve/1.0",
    }

    try:
        async with httpx.AsyncClient(timeout=20.0, headers=headers) as client:
            resp = await client.get(url, params=params)
            if resp.status_code != 200:
                logger.warning("GitHub repository search HTTP %s", resp.status_code)
                return []

            data = resp.json() or {}
            repos = []
            for item in data.get("items", []):
                description = str(item.get("description", "") or "").strip()
                if len(description) < 24:
                    continue
                owner = (item.get("owner") or {}).get("login", "GitHub")
                created_year = str(item.get("created_at", "")).split("-", 1)[0]
                repos.append(_make_paper(
                    title=item.get("full_name", "") or item.get("name", "Repository"),
                    authors=owner,
                    abstract=description,
                    year=created_year,
                    source="github_repo",
                    url=item.get("html_url", ""),
                    citation_count=int(item.get("stargazers_count", 0) or 0),
                    arxiv_id=item.get("node_id", ""),
                ))

            logger.info("GitHub returned %d repositories for '%s'", len(repos), query)
            return repos
    except Exception as e:
        logger.error("GitHub repository fetch failed: %s", e)
        return []


# ── Tavily Web Search ─────────────────────────────────────────────────────

async def fetch_web_tavily(query: str, api_key: str, max_results: int = 5) -> list[dict]:
    """
    Fetch web results via Tavily API.
    Returns normalized paper-like dicts for web articles.
    """
    if not api_key:
        logger.warning("Tavily API key not set, skipping web search.")
        return []

    try:
        from tavily import AsyncTavilyClient

        client = AsyncTavilyClient(api_key=api_key)
        response = await client.search(
            query=f"academic research {query}",
            max_results=max_results,
            search_depth="advanced",
        )

        papers = []
        for result in response.get("results", []):
            papers.append(_make_paper(
                title=result.get("title", ""),
                authors="Web Source",
                abstract=result.get("content", ""),
                year=str(datetime.now().year),
                source="web_tavily",
                url=result.get("url", ""),
            ))

        logger.info("Tavily returned %d results for '%s'", len(papers), query)
        return papers

    except Exception as e:
        logger.error("Tavily fetch failed: %s", e)
        return []


# ── Deduplication ─────────────────────────────────────────────────────────

def deduplicate_papers(papers: list[dict]) -> list[dict]:
    """
    Deduplicate papers based on title similarity and DOI.
    Keeps the entry with higher citation count.
    """
    seen: dict[str, dict] = {}

    for paper in papers:
        # Create a normalized key from title
        key = str(paper.get("title", "")).lower().strip()[:80]
        doi = paper.get("doi", "")

        # If we have a DOI, prefer that as the key
        if doi:
            key = doi

        if not key:
            # No title and no DOI -> can't dedup meaningfully, keep as-is.
            seen[id(paper)] = paper
            continue

        if key in seen:
            # Keep the one with higher citation count
            if paper.get("citation_count", 0) > seen[key].get("citation_count", 0):
                seen[key] = paper
        else:
            seen[key] = paper

    return list(seen.values())
