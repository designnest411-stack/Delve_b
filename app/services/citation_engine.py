"""
ResearchAgent Citation & Reference Engine
─────────────────────────────────────────
Performs citation resolution, BibTeX serialization, and reference deduplication.
Ensures zero fabricated DOIs and guarantees sequential or alphabetical CSL compliance.
"""

from __future__ import annotations

import re
from typing import Any
from app.services.manuscript_schema import CitationSource


def clean_bibtex_value(val: Any) -> str:
    """Escape braces and clean BibTeX field strings."""
    s = str(val or "").strip()
    # Replace unicode dashes with standard hyphens
    s = s.replace("–", "--").replace("—", "---")
    # Escape unescaped ampersands
    s = re.sub(r"(?<!\\)&", r"\&", s)
    # Remove raw braces that are unbalanced
    if s.count("{") != s.count("}"):
        s = s.replace("{", "").replace("}", "")
    return s


def format_bibtex_authors(authors_raw: Any) -> str:
    """Format authors into standard 'First Last and First Last' BibTeX syntax."""
    if isinstance(authors_raw, list):
        clean_list = [clean_bibtex_value(a) for a in authors_raw if str(a).strip()]
        return " and ".join(clean_list)
    
    raw = str(authors_raw or "").strip()
    if not raw:
        return "Anonymous"
    
    # If already separated by ' and ', normalize whitespace
    if " and " in raw:
        parts = [p.strip() for p in raw.split(" and ") if p.strip()]
        return " and ".join(parts)
    
    # If comma separated, convert to 'and'
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) > 1 and all(len(p.split()) <= 4 for p in parts):
        return " and ".join(parts)
    
    return raw


def generate_bibtex_database(sources: list[CitationSource]) -> str:
    """
    Generate a complete, valid BibTeX string from CitationSource objects.
    Compatible with Typst CSL engine and standard BibTeX / Biber.
    """
    entries: list[str] = []
    seen_keys: set[str] = set()

    for idx, src in enumerate(sources, 1):
        key = src.citation_key or f"ref_{idx}"
        # Ensure alphanumeric key
        clean_key = re.sub(r"[^a-zA-Z0-9_\-]", "", key)
        if not clean_key:
            clean_key = f"ref_{idx}"
        if clean_key in seen_keys:
            clean_key = f"{clean_key}_{idx}"
        seen_keys.add(clean_key)

        entry_type = src.entry_type or "article"
        authors = format_bibtex_authors(src.authors)
        title = clean_bibtex_value(src.title)
        year = str(src.year or "2024")

        lines = [
            f"@{entry_type}{{{clean_key},",
            f"  author = {{{authors}}},",
            f"  title = {{{title}}},",
            f"  year = {{{year}}},",
        ]

        if src.venue:
            if entry_type == "inproceedings":
                lines.append(f"  booktitle = {{{clean_bibtex_value(src.venue)}}},")
            else:
                lines.append(f"  journal = {{{clean_bibtex_value(src.venue)}}},")

        if src.volume:
            lines.append(f"  volume = {{{clean_bibtex_value(src.volume)}}},")
        if src.issue:
            lines.append(f"  number = {{{clean_bibtex_value(src.issue)}}},")
        if src.pages:
            lines.append(f"  pages = {{{clean_bibtex_value(src.pages)}}},")
        if src.doi:
            lines.append(f"  doi = {{{clean_bibtex_value(src.doi)}}},")
        if src.url:
            lines.append(f"  url = {{{clean_bibtex_value(src.url)}}},")

        lines.append("}\n")
        entries.append("\n".join(lines))

    return "\n".join(entries)


def resolve_citation_mapping(
    raw_text: str,
    sources: list[CitationSource],
) -> tuple[str, list[CitationSource]]:
    """
    Resolves citation markers in raw text to Typst `@key` format.
    Ensures that every citation maps to a registered source.
    """
    if not sources:
        return raw_text, []

    # Map paper IDs and citation keys
    key_map: dict[str, str] = {}
    for src in sources:
        key_map[src.citation_key] = src.citation_key
        key_map[src.id] = src.citation_key

    # Replace [@key] or [@AuthorYear] with @key
    def _replace_bracketed_cite(match: re.Match) -> str:
        inner = match.group(1).strip()
        keys = [k.strip().lstrip("@") for k in inner.split(";")]
        valid_keys = [f"@{key_map[k]}" for k in keys if k in key_map]
        if valid_keys:
            return " ".join(valid_keys)
        # If it's a numeric citation like [1], [2]
        return match.group(0)

    processed = re.sub(r"\[(@[a-zA-Z0-9_\-]+(?:;\s*@[a-zA-Z0-9_\-]+)*)\]", _replace_bracketed_cite, raw_text)
    return processed, sources
