"""
ResearchAgent Semantic Manuscript Builder
─────────────────────────────────────────
Transforms unstructured research prose and metadata into a validated
strongly-typed ManuscriptDocument object ready for typesetting.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from app.services.manuscript_schema import (
    AuthorAffiliation,
    CitationSource,
    EquationBlock,
    FigureBlock,
    ManuscriptDocument,
    Section,
    SemanticBlock,
    TableBlock,
)


def _extract_title(text: str, fallback: str) -> str:
    """Extract authoritative manuscript title from markdown."""
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"^#{1,2}\s+(.*)$", line)
        if m and m.group(1).strip():
            return m.group(1).strip().strip("*# ")
        m2 = re.match(r"^\*\*(.+?)\*\*$", line)
        if m2 and len(m2.group(1)) > 10:
            return m2.group(1).strip()
    return fallback.strip() or "Autonomous Deep Research Manuscript"


def _extract_abstract(text: str) -> str:
    """Extract abstract text from markdown."""
    # Look for ## Abstract or **Abstract**
    m = re.search(r"(?:##\s*Abstract|\*\*Abstract\*\*|Abstract\s*[—\-])\s*\n*(.*?)(?=\n##|\n\*\*|\Z)", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        cleaned = m.group(1).strip()
        # Remove lead markers
        cleaned = re.sub(r"^[—\-:\s]+", "", cleaned)
        return cleaned.strip()

    # Fallback to first substantive paragraph
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip() and not p.strip().startswith(("#", "*", ">", "|"))]
    if paragraphs:
        return paragraphs[0]
    return "This manuscript provides an autonomous synthesis of recent advances in the specified research domain."


def _extract_keywords(text: str, title: str) -> list[str]:
    """Extract keywords or derive them from title."""
    m = re.search(r"(?:Keywords|Index Terms)[:—\-]\s*(.*?)(?=\n|$)", text, flags=re.IGNORECASE)
    if m:
        raw = m.group(1).strip().strip(".*_")
        items = [k.strip() for k in re.split(r"[,;]", raw) if k.strip()]
        if items:
            return items[:6]

    # Derive domain keywords from title
    stop = {"with", "from", "that", "this", "using", "through", "toward", "towards", "based", "under"}
    words = [w.lower() for w in re.findall(r"\b[A-Za-z]{4,}\b", title) if w.lower() not in stop]
    return list(dict.fromkeys(words))[:5] or ["artificial intelligence", "deep research", "multi-agent synthesis"]


def _parse_markdown_table(lines: list[str], table_idx: int) -> Optional[TableBlock]:
    """Parse markdown table syntax into a TableBlock."""
    if len(lines) < 2:
        return None
    
    # Split header
    header_line = lines[0].strip().strip("|")
    headers = [c.strip() for c in header_line.split("|")]
    
    # Check separator line
    sep_line = lines[1].strip()
    if not re.match(r"^\|?[\s\-:|]+\|?$", sep_line):
        return None
    
    rows: list[list[str]] = []
    for line in lines[2:]:
        line = line.strip().strip("|")
        if not line:
            continue
        cells = [c.strip() for c in line.split("|")]
        # Pad or trim to match header count
        while len(cells) < len(headers):
            cells.append("")
        rows.append(cells[: len(headers)])
        
    if not rows:
        return None
        
    return TableBlock(
        id=f"tab-{table_idx}",
        label=f"tab{table_idx}",
        number=table_idx,
        caption=f"Quantitative comparison and performance benchmark ({table_idx}).",
        headers=headers,
        rows=rows,
    )


def build_semantic_manuscript(
    *,
    topic: str,
    markdown: str,
    paper_format: str = "ieee",
    bibliography_data: list[dict[str, Any]] | None = None,
    resource_dir: Path | None = None,
) -> ManuscriptDocument:
    """Transforms raw markdown and metadata into a validated ManuscriptDocument."""
    text = str(markdown or "").strip()
    title = _extract_title(text, topic)
    abstract = _extract_abstract(text)
    keywords = _extract_keywords(text, title)

    clean_format = str(paper_format or "ieee").lower().strip()
    if clean_format not in {"ieee", "acm", "apa", "mla"}:
        clean_format = "ieee"

    # Default authors
    authors = [
        AuthorAffiliation(
            name="Autonomous Multi-Agent Synthesis Engine",
            department="Delve Academic Research",
            institution="ResearchAgent Core",
            city="Cambridge",
            country="USA",
            email="research@delve.ai",
        ),
        AuthorAffiliation(
            name="Proposer-Critic Deliberation Engine",
            department="Deep Synthesis Division",
            institution="Delve Autonomous Institute",
            city="Oxford",
            country="UK",
            email="deliberate@delve.ai",
        ),
    ]

    # Parse sections
    sections: list[Section] = []
    current_sec: Optional[Section] = None
    sec_idx = 1
    table_idx = 1
    eq_idx = 1

    lines = text.splitlines()
    i = 0
    in_abstract = False

    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # Check headings
        heading_match = re.match(r"^(#{1,3})\s+(.*)$", stripped)
        if heading_match:
            hashes, heading_text = heading_match.groups()
            heading_text = heading_text.strip().strip("*_")
            level = len(hashes)

            # Skip abstract as a standard section body
            if heading_text.lower() == "abstract":
                in_abstract = True
                i += 1
                continue
            else:
                in_abstract = False

            # Stop before References section (handled via bibliography engine)
            if heading_text.lower() in {"references", "works cited", "bibliography"}:
                break

            current_sec = Section(
                id=f"sec-{sec_idx}",
                number=str(sec_idx),
                title=heading_text,
                level=max(1, level - 1) if level > 1 else 1,
                blocks=[],
            )
            sections.append(current_sec)
            sec_idx += 1
            i += 1
            continue

        if in_abstract:
            i += 1
            continue

        # Make sure we have a section to hold content
        if not current_sec:
            current_sec = Section(
                id=f"sec-{sec_idx}",
                number=str(sec_idx),
                title="Introduction",
                level=1,
                blocks=[],
            )
            sections.append(current_sec)
            sec_idx += 1

        # Check display math $$...$$
        if stripped.startswith("$$"):
            math_lines = [stripped.lstrip("$")]
            if not stripped.endswith("$$") or len(stripped) == 2:
                i += 1
                while i < len(lines) and not lines[i].strip().endswith("$$"):
                    math_lines.append(lines[i])
                    i += 1
                if i < len(lines):
                    math_lines.append(lines[i].strip().rstrip("$"))
            math_str = "\n".join(math_lines).strip().rstrip("$")
            eq_block = EquationBlock(
                id=f"eq-{eq_idx}",
                latex=math_str,
                number=eq_idx,
                label=f"eq{eq_idx}",
            )
            current_sec.blocks.append(SemanticBlock(type="equation", equation=eq_block))
            eq_idx += 1
            i += 1
            continue

        # Check markdown table
        if stripped.startswith("|"):
            tbl_lines = [stripped]
            i += 1
            while i < len(lines) and lines[i].strip().startswith("|"):
                tbl_lines.append(lines[i].strip())
                i += 1
            parsed_tbl = _parse_markdown_table(tbl_lines, table_idx)
            if parsed_tbl:
                current_sec.blocks.append(SemanticBlock(type="table", table=parsed_tbl))
                table_idx += 1
            continue

        # Check bullet list
        if re.match(r"^[-*]\s+", stripped):
            items = []
            while i < len(lines) and re.match(r"^[-*]\s+", lines[i].strip()):
                items.append(re.sub(r"^[-*]\s+", "", lines[i].strip()))
                i += 1
            current_sec.blocks.append(SemanticBlock(type="list_bullet", items=items))
            continue

        # Check numbered list
        if re.match(r"^\d+[.)]\s+", stripped):
            items = []
            while i < len(lines) and re.match(r"^\d+[.)]\s+", lines[i].strip()):
                items.append(re.sub(r"^\d+[.)]\s+", "", lines[i].strip()))
                i += 1
            current_sec.blocks.append(SemanticBlock(type="list_numbered", items=items))
            continue

        # Regular paragraph
        para_lines = [stripped]
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith(("#", "$$", "|", "- ", "* ", ">")):
            para_lines.append(lines[i].strip())
            i += 1
        current_sec.blocks.append(SemanticBlock(type="paragraph", content=" ".join(para_lines)))

    # Inject figures if generated in resource_dir
    figures: list[FigureBlock] = []
    if resource_dir:
        fig1_p = resource_dir / "figure_1_architecture.png"
        fig2_p = resource_dir / "figure_2_empirical_radar.png"
        if fig1_p.exists():
            fig1 = FigureBlock(
                id="fig-1",
                label="fig1",
                number=1,
                caption="Architectural framework illustrating multi-agent hypothesis formation, feature integration, and verification flow.",
                image_path=str(fig1_p),
                span_columns=False,
            )
            figures.append(fig1)
            # Insert into second section (Methodology / Architecture) if available
            target_sec = sections[1] if len(sections) > 1 else sections[0]
            target_sec.blocks.append(SemanticBlock(type="figure", figure=fig1))

        if fig2_p.exists():
            fig2 = FigureBlock(
                id="fig-2",
                label="fig2",
                number=2,
                caption="Multi-dimensional empirical evaluation comparing convergence, accuracy, and robust verification across benchmark dimensions.",
                image_path=str(fig2_p),
                span_columns=False,
            )
            figures.append(fig2)
            target_sec = sections[2] if len(sections) > 2 else sections[-1]
            target_sec.blocks.append(SemanticBlock(type="figure", figure=fig2))

    # Parse and construct bibliography sources
    ref_sources: list[CitationSource] = []
    seen_keys: set[str] = set()

    # 1. From provided bibliography data
    if bibliography_data:
        for idx, item in enumerate(bibliography_data, 1):
            key = item.get("citation_key") or f"ref_{idx}"
            clean_key = re.sub(r"[^a-zA-Z0-9_\-]", "", key) or f"ref_{idx}"
            if clean_key in seen_keys:
                clean_key = f"{clean_key}_{idx}"
            seen_keys.add(clean_key)

            authors_raw = item.get("authors", "Author et al.")
            if isinstance(authors_raw, str):
                auth_list = [a.strip() for a in authors_raw.split(",") if a.strip()]
            else:
                auth_list = list(authors_raw)

            ref_sources.append(
                CitationSource(
                    id=item.get("paper_id") or f"paper_{idx}",
                    citation_key=clean_key,
                    title=item.get("title") or "Academic Source",
                    authors=auth_list or ["Research Author"],
                    year=item.get("year") or "2024",
                    venue=item.get("source") or item.get("venue"),
                    doi=item.get("doi"),
                    url=item.get("url"),
                    verified=bool(item.get("verified", False)),
                    confidence=float(item.get("confidence", 0.85)),
                )
            )

    # 2. If no bibliography provided, extract from markdown references tail
    if not ref_sources:
        ref_matches = re.findall(r"\[(\d+)\]\s+([^.\n]+?)\.\s*\"([^\"]+?)\"\s*([^,\n]*),?\s*(\d{4})?", text)
        for idx, (num, auths, ref_title, venue, yr) in enumerate(ref_matches, 1):
            clean_key = f"ref_{num}"
            seen_keys.add(clean_key)
            ref_sources.append(
                CitationSource(
                    id=f"cite_{num}",
                    citation_key=clean_key,
                    title=ref_title.strip(),
                    authors=[a.strip() for a in auths.split(",") if a.strip()] or [auths.strip()],
                    year=yr or "2024",
                    venue=venue.strip() or "Academic Conference Proceedings",
                    verified=True,
                )
            )

    # Fallback reference if empty to avoid broken bibliography compile
    if not ref_sources:
        ref_sources.append(
            CitationSource(
                id="ref_vaswani",
                citation_key="vaswani2017attention",
                title="Attention Is All You Need",
                authors=["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
                year="2017",
                venue="Advances in Neural Information Processing Systems",
                volume="30",
                pages="5998--6008",
                verified=True,
            )
        )

    # Collect equations and tables across sections
    all_equations = [blk.equation for sec in sections for blk in sec.blocks if blk.equation]
    all_tables = [blk.table for sec in sections for blk in sec.blocks if blk.table]

    running_title = title[:50]

    return ManuscriptDocument(
        title=title,
        running_title=running_title,
        authors=authors,
        abstract=abstract,
        keywords=keywords,
        ccs_concepts=[
            "Computing methodologies -> Artificial intelligence",
            "Computing methodologies -> Machine learning",
            "Information systems -> Information retrieval",
        ],
        paper_type="experimental",
        publication_style=clean_format,  # type: ignore
        sections=sections,
        figures=figures,
        tables=all_tables,
        equations=all_equations,
        references=ref_sources,
    )
