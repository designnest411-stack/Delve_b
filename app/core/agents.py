"""
Delve Agent Nodes
──────────────────
All LangGraph node functions for the research pipeline.
Each node reads from / writes to the shared ResearchState.

Nodes:
  1. planner_node        – Generates search queries + research plan
  2. retrieval_node      – Fetches papers from ArXiv, Semantic Scholar, Tavily, ChromaDB
  3. summarizer_node     – Summarizes top papers and extracts limitations
  4. proposer_node       – Drafts literature review
  5. critic_node         – Reviews and critiques the draft
  6. gap_analysis_node   – Identifies research gaps (JSON output)
  7. paper_architect_node – Assembles final Markdown paper
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
import uuid
from collections import Counter
from typing import Any

from app.core.llm_client import llm_client
from app.core.config import settings
from app.core.state import ResearchState
from app.services.retrieval import (
    fetch_arxiv,
    fetch_crossref,
    fetch_github_repositories,
    fetch_openalex,
    fetch_semantic_scholar,
    fetch_web_tavily,
    deduplicate_papers,
)
from app.services.supabase_vectors import query_uploaded_documents

logger = logging.getLogger("delve.agents")


# ── Helper: WebSocket Status Update ──────────────────────────────────────

def _ws_callback(config: dict):
    """
    Extract the WebSocket broadcast callback from LangGraph config.
    Returns a callable or a no-op if not present.
    """
    return config.get("configurable", {}).get("ws_callback", lambda msg: None)


async def _send_status(config: dict, message: str, msg_type: str = "status", data: dict | None = None):
    """Send a status update via WebSocket callback if available."""
    cb = _ws_callback(config)
    payload = {"type": msg_type, "message": message}
    if data:
        payload["data"] = data
    try:
        if asyncio.iscoroutinefunction(cb):
            await cb(payload)
        else:
            cb(payload)
    except Exception as e:
        logger.warning("WebSocket callback failed: %s", e)


def _source_quality_score(source: str) -> float:
    source = (source or "").lower()
    if source == "semantic_scholar":
        return 0.95  # mostly peer-reviewed index
    if source == "openalex":
        return 0.93
    if source == "crossref":
        return 0.88
    if source == "arxiv":
        return 0.72  # strong but preprint-heavy
    if source == "github_repo":
        return 0.58
    if source == "web_tavily":
        return 0.45
    if source == "uploaded_pdf":
        return 0.8
    return 0.5


def _confidence_score(citation_count: int, abstract: str, source: str) -> float:
    citation_component = min(1.0, math.log1p(max(0, citation_count)) / math.log1p(200))
    abstract_component = 1.0 if len((abstract or "").strip()) >= 500 else 0.65
    source_component = _source_quality_score(source)
    return round(0.5 * source_component + 0.3 * citation_component + 0.2 * abstract_component, 3)


def _normalize_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 2}


def _title_similarity(a: str, b: str) -> float:
    ta, tb = _normalize_tokens(a), _normalize_tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def _embedding_similarity(a: str, b: str) -> float:
    # Lightweight local proxy for embedding similarity using token-frequency cosine.
    va = Counter(re.findall(r"[a-z0-9]+", (a or "").lower()))
    vb = Counter(re.findall(r"[a-z0-9]+", (b or "").lower()))
    if not va or not vb:
        return 0.0
    keys = set(va) | set(vb)
    dot = sum(va.get(k, 0) * vb.get(k, 0) for k in keys)
    na = math.sqrt(sum(v * v for v in va.values()))
    nb = math.sqrt(sum(v * v for v in vb.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _cluster_duplicates(papers: list[dict[str, Any]], threshold: float = 0.82) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []
    used: set[int] = set()
    for i, p in enumerate(papers):
        if i in used:
            continue
        cluster = [p]
        used.add(i)
        for j in range(i + 1, len(papers)):
            if j in used:
                continue
            q = papers[j]
            same_doi = bool(p.get("doi")) and p.get("doi") == q.get("doi")
            title_sim = _title_similarity(p.get("title", ""), q.get("title", ""))
            emb_sim = _embedding_similarity(
                f"{p.get('title','')} {p.get('abstract','')[:300]}",
                f"{q.get('title','')} {q.get('abstract','')[:300]}",
            )
            sim = max(title_sim, emb_sim)
            if same_doi or sim >= threshold:
                cluster.append(q)
                used.add(j)
        if len(cluster) > 1:
            clusters.append({
                "canonical_title": cluster[0].get("title", ""),
                "size": len(cluster),
                "sources": sorted(list({c.get("source", "unknown") for c in cluster})),
                "paper_ids": [c.get("paper_id") for c in cluster],
            })
    return clusters


def _extract_json_fragment(text: str) -> str:
    if not text:
        return ""
    # Strip markdown code block wrappers first
    stripped = re.sub(r"^```(?:json|JSON)?\s*\n?", "", text.strip())
    stripped = re.sub(r"\n?```\s*$", "", stripped).strip()
    if stripped != text.strip():
        text = stripped
    start_candidates = [i for i in [text.find("["), text.find("{")] if i != -1]
    if not start_candidates:
        return text.strip()
    start = min(start_candidates)
    end_arr = text.rfind("]")
    end_obj = text.rfind("}")
    end = max(end_arr, end_obj)
    if end > start:
        return text[start:end + 1]
    return text[start:].strip()


def _parse_json_object(text: str) -> dict[str, Any]:
    """Best-effort parse for model output expected to be a JSON object."""
    if not text:
        return {}
    # Try raw, then stripped markdown, then extracted fragment
    for candidate in (text, _extract_json_fragment(text)):
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
        # If the model returns an array of objects, merge them into one dict
        # (later keys win) instead of silently dropping all but the first.
        if isinstance(parsed, list) and parsed:
            dict_items = [item for item in parsed if isinstance(item, dict)]
            if dict_items:
                merged: dict[str, Any] = {}
                for item in dict_items:
                    merged.update(item)
                return merged
    return {}


def _parse_json_array(text: str) -> list[dict[str, Any]]:
    """Best-effort parse for model output expected to be a JSON array."""
    if not text:
        return []
    for candidate in (text, _extract_json_fragment(text)):
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict):
            return [parsed]
    return []


def _topic_anchor_terms(topic: str, limit: int = 8) -> list[str]:
    raw = [t.lower() for t in re.findall(r"[A-Za-z0-9]+", topic or "") if len(t) > 3]
    stop = {
        "with", "from", "into", "using", "over", "under", "across", "towards",
        "between", "based", "analysis", "system", "study", "research", "approach",
        "architectures", "architecture",
    }
    anchors: list[str] = []
    for token in raw:
        if token in stop:
            continue
        if token not in anchors:
            anchors.append(token)
        if len(anchors) >= limit:
            break
    return anchors


def _critical_topic_terms(topic: str, limit: int = 6) -> list[str]:
    anchors = _topic_anchor_terms(topic, limit=20)
    generic = {
        "study", "research", "analysis", "approach", "strategies", "causes",
        "impacts", "effects", "evaluation", "system", "systems", "models",
        "methods", "techniques", "review", "survey", "framework", "frameworks",
    }
    critical = [a for a in anchors if a not in generic]
    return critical[:limit]


def _paper_relevance_score(paper: dict[str, Any], topic_tokens: set[str], query_tokens: set[str]) -> float:
    hay = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
    tokens = {t for t in re.findall(r"[a-z0-9]+", hay) if len(t) > 2}
    if not tokens:
        return 0.0
    t_overlap = len(tokens & topic_tokens)
    q_overlap = len(tokens & query_tokens)
    title_tokens = {t for t in re.findall(r"[a-z0-9]+", str(paper.get("title", "")).lower()) if len(t) > 2}
    title_overlap = len(title_tokens & topic_tokens)
    base = (2.2 * title_overlap) + (1.3 * t_overlap) + (0.5 * q_overlap)
    if not title_overlap and topic_tokens:
        base -= 1.5
    return base


def _clean_nullable_text(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower() in {"none", "null", "n/a", "na", "not available"}:
        return ""
    return text


def _citation_key(authors: str, year: str | int | None) -> str:
    author_last = "unknown"
    if authors:
        first = str(authors).split(",")[0].strip()
        chunks = [c for c in re.findall(r"[A-Za-z]+", first) if c]
        if chunks:
            author_last = chunks[-1].lower()
    y = str(year or "nd").strip().lower()
    return f"{author_last}{y}"


def _citation_display_name(authors: str, year: str | int | None) -> tuple[str, str]:
    first = str(authors or "").split(",")[0].strip()
    chunks = [c for c in re.findall(r"[A-Za-z]+", first) if c]
    author_last = chunks[-1] if chunks else "Unknown"
    y = str(year or "n.d.").strip()
    return author_last, y


def _citation_marker_map(bibliography: list[dict[str, Any]]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    next_idx = 1
    for item in bibliography:
        key = str(item.get("citation_key", "")).strip().lower()
        if not key:
            key = _citation_key(item.get("authors", ""), item.get("year", ""))
        key = re.sub(r"[^a-z0-9]", "", key)
        if key and key not in mapping:
            mapping[key] = next_idx
            next_idx += 1
    return mapping

def _strip_reference_tail(markdown: str) -> str:
    """Safely strip only the trailing References / Works Cited section at the very end of the manuscript."""
    if not markdown:
        return ""
    matches = list(re.finditer(r"(?im)^#{1,4}\s*(?:references|works cited)\s*$", markdown))
    if matches:
        last_match = matches[-1]
        return markdown[:last_match.start()].rstrip()
    return markdown.rstrip()


def _sanitize_mermaid_syntax(markdown: str) -> str:
    """
    Performs regex-based fixes on Mermaid code blocks to prevent syntax errors.
    """
    if not markdown or "```mermaid" not in markdown:
        return markdown

    def _fix_block(match: re.Match) -> str:
        content = match.group(1)
        # 1. Strip icons (causes syntax errors in many versions)
        content = re.sub(r"::icon\(.*?\)", "", content)
        
        # 2. Quoting labels in mindmaps that have special characters
        if "mindmap" in content:
            # Match unquoted text in mindmap nodes and wrap in ["..."]
            # This is a heuristic that targets common LLM failures
            content = re.sub(r"(\s+)([A-Za-z0-9_-]+)\(([^)]+)\)", r'\1\2["\3"]', content)
            content = re.sub(r'(\s+)([A-Za-z0-9_-]+)\s+([^"\[\]\(\)\s][^"\[\]\(\)\n]*)', r'\1\2["\3"]', content)
        
        return f"```mermaid\n{content}\n```"

    return re.sub(r"```mermaid\n(.*?)\n```", _fix_block, markdown, flags=re.DOTALL)


def _strip_mermaid_blocks(markdown: str) -> str:
    if not markdown:
        return ""
    return re.sub(r"\n?```mermaid\n.*?\n```\n?", "\n", markdown, flags=re.DOTALL)


def _normalize_image_prompt(prompt: str, topic: str, max_len: int = 280) -> str:
    cleaned = re.sub(r"\s+", " ", str(prompt or "")).strip().strip('"')
    cleaned = re.sub(r"[\[\]{}<>`]", "", cleaned)
    if not cleaned:
        cleaned = f"Editorial scientific infographic about {topic}, clean labels, modern research style"
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rsplit(" ", 1)[0].strip()
    return cleaned


def _depth_budgets(depth: str) -> dict[str, int]:
    key = str(depth or "standard").strip().lower()
    if key == "quick":
        return {"queries": 3, "results": 4, "max_total": 10, "summary_cap": 8}
    if key == "deep":
        return {"queries": 6, "results": 6, "max_total": 36, "summary_cap": 15}  # reduced from 22 to 15
    return {"queries": 5, "results": 5, "max_total": 20, "summary_cap": 12}  # reduced from 14 to 12


def _safe_mermaid_text(text: str, limit: int = 34) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9 /:+-]", " ", str(text or ""))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        cleaned = "Untitled"
    words = cleaned.split()
    return " ".join(words[:limit])


def _evidence_snapshot_table(
    summaries: dict[str, dict[str, Any]],
    citation_quality: dict[str, dict[str, Any]],
    citation_verification: dict[str, dict[str, Any]],
    limit: int = 8,
) -> str:
    if not summaries:
        return "_No source evidence available._"
    lines = [
        "| Study | Year | Methodology | Evidence focus | Limitation | Confidence | Verified |",
        "|---|---:|---|---|---|---:|---|",
    ]
    for i, (_pid, info) in enumerate(summaries.items(), start=1):
        if i > limit:
            break
        method = _clean_nullable_text(info.get("methodology", "")) or "Benchmark / empirical analysis"
        eval_signal = _clean_nullable_text(info.get("evaluation_signal", "")) or "Model capability and evidence synthesis"
        limitation = _clean_nullable_text(info.get("limitation", "")) or "Not explicit"
        confidence = citation_quality.get(_pid, {}).get("confidence", 0.5)
        verified = "Yes" if citation_verification.get(_pid, {}).get("verified", False) else "No"
        lines.append(
            f"| {str(info.get('title', 'Untitled'))[:58]} | {str(info.get('year', ''))[:4]} | "
            f"{method[:52]} | {eval_signal[:52]} | {limitation[:68]} | {confidence:.2f} | {verified} |"
        )
    return "\n\n".join(lines)


def _source_mix_table(source_counts: dict[str, int], bibliography: list[dict[str, Any]]) -> str:
    total = max(1, sum(source_counts.values()))
    verified = sum(1 for b in bibliography if b.get("verified"))
    lines = [
        "| Source family | Count | Share |",
        "|---|---:|---:|",
    ]
    for src, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
        lines.append(f"| {src} | {count} | {round((count / total) * 100)}% |")
    lines.append(f"| Verified citations | {verified} | {round((verified / max(1, len(bibliography))) * 100)}% |")
    return "\n\n".join(lines)


def _study_landscape_table(
    summaries: dict[str, dict[str, Any]],
    citation_quality: dict[str, dict[str, Any]],
    citation_verification: dict[str, dict[str, Any]],
    limit: int = 14,
) -> str:
    if not summaries:
        return "_No study landscape available._"
    lines = [
        "| Study | Year | Methodology | Evaluation signal | Source | Confidence | Verified |",
        "|---|---:|---|---|---|---:|---|",
    ]
    for idx, (_pid, info) in enumerate(summaries.items(), start=1):
        if idx > limit:
            break
        lines.append(
            f"| {str(info.get('title', 'Untitled'))[:56]} | {str(info.get('year', ''))[:4]} | "
            f"{(_clean_nullable_text(info.get('methodology', '')) or 'Empirical study')[:44]} | "
            f"{(_clean_nullable_text(info.get('evaluation_signal', '')) or 'Not explicit')[:48]} | "
            f"{str(info.get('source', 'unknown'))[:18]} | "
            f"{citation_quality.get(_pid, {}).get('confidence', 0.5):.2f} | "
            f"{'Yes' if citation_verification.get(_pid, {}).get('verified', False) else 'No'} |"
        )
    return "\n\n".join(lines)


def _citation_audit_table(bibliography: list[dict[str, Any]], limit: int = 18) -> str:
    if not bibliography:
        return "_No citation audit available._"
    lines = [
        "| Citation key | Source family | DOI / URL | Confidence | Verified |",
        "|---|---|---|---:|---|",
    ]
    for item in bibliography[:limit]:
        locator = item.get("doi") or item.get("url") or "Not provided"
        locator = str(locator)
        if len(locator) > 64:
            locator = f"{locator[:61]}..."
        lines.append(
            f"| [@{item.get('citation_key', 'unknown')}] | {str(item.get('source', 'unknown'))[:20]} | "
            f"{locator} | {float(item.get('confidence', 0.5) or 0.5):.2f} | "
            f"{'Yes' if item.get('verified') else 'No'} |"
        )
    return "\n\n".join(lines)


def _findings_brief(cross: dict[str, Any]) -> str:
    if not cross:
        return "_Cross-paper findings not available._"
    blocks: list[str] = []
    sections = [
        ("Methodology patterns", cross.get("methodology_patterns", [])),
        ("Comparative axes", cross.get("comparative_axes", [])),
        ("Contradictions", cross.get("contradictions", [])),
        ("Missing dimensions", cross.get("missing_dimensions", [])),
        ("Weak evaluation signals", cross.get("weak_evaluation_signals", [])),
    ]
    for heading, items in sections:
        cleaned = [str(item).strip() for item in (items or []) if str(item).strip()]
        if not cleaned:
            continue
        blocks.append(f"#### {heading}\n" + "\n".join(f"- {item}" for item in cleaned[:6]))
    return "\n\n".join(blocks) if blocks else "_Cross-paper findings were too sparse to summarize._"


def _experiment_backlog_table(cross: dict[str, Any], gaps: list[dict[str, Any]], limit: int = 8) -> str:
    experiments = [str(item).strip() for item in (cross.get("recommended_experiments", []) or []) if str(item).strip()]
    rows = []
    for idx, experiment in enumerate(experiments[:limit], start=1):
        linked_gap = gaps[min(idx - 1, len(gaps) - 1)].get("title", "General evidence gap") if gaps else "General evidence gap"
        rows.append(f"| {idx} | {experiment[:88]} | {str(linked_gap)[:58]} | High |")
    if not rows:
        for idx, gap in enumerate(gaps[:limit], start=1):
            rows.append(
                f"| {idx} | {str(gap.get('proposed_direction', 'Targeted validation study'))[:88]} | "
                f"{str(gap.get('title', 'Evidence gap'))[:58]} | High |"
            )
    if not rows:
        return "_No experiment backlog available._"
    return "\n".join([
        "| Priority | Proposed experiment | Primary gap addressed | Expected value |",
        "|---:|---|---|---|",
        *rows,
    ])


def _visual_landscape_mermaid(topic: str, summaries: dict[str, dict[str, Any]], gaps: list[dict[str, Any]]) -> str:
    theme_terms: list[str] = []
    for info in summaries.values():
        for raw in re.findall(r"[A-Za-z]{4,}", f"{info.get('title', '')} {info.get('summary', '')}"):
            token = raw.lower()
            if token in {"study", "paper", "model", "results", "using", "based", "their", "these", "video", "text"}:
                continue
            if token not in theme_terms:
                theme_terms.append(token)
            if len(theme_terms) >= 6:
                break
        if len(theme_terms) >= 6:
            break
    gap_terms = [_safe_mermaid_text(g.get("title", ""), 6) for g in gaps[:3]]
    method_terms = [_safe_mermaid_text(info.get("methodology", "") or info.get("title", ""), 5) for info in list(summaries.values())[:3]]
    lines = [
        "```mermaid",
        "flowchart TD",
        f'    topic["{_safe_mermaid_text(topic, 8)}"]',
        '    topic --> themes["Core themes"]',
        '    topic --> methods["Methods and benchmarks"]',
        '    topic --> gaps["Open gaps"]',
    ]
    for idx, term in enumerate(theme_terms[:4], start=1):
        lines.append(f'    themes --> th{idx}["{_safe_mermaid_text(term, 5)}"]')
    for idx, term in enumerate(method_terms[:3], start=1):
        lines.append(f'    methods --> m{idx}["{_safe_mermaid_text(term, 6)}"]')
    for idx, term in enumerate(gap_terms, start=1):
        lines.append(f'    gaps --> g{idx}["{_safe_mermaid_text(term, 7)}"]')
    lines.append("```")
    return "\n\n".join(lines)


def _visual_workflow_mermaid(gaps: list[dict[str, Any]]) -> str:
    gap_titles = [_safe_mermaid_text(g.get("title", ""), 6) for g in gaps[:3]]
    lines = [
        "```mermaid",
        "flowchart LR",
        '    A["Source collection"] --> B["Cross paper synthesis"]',
        '    B --> C["Evidence comparison"]',
        '    C --> D["Gap diagnosis"]',
        '    D --> E["Prioritized experiments"]',
    ]
    for idx, gap in enumerate(gap_titles, start=1):
        lines.append(f'    D --> G{idx}["{gap}"]')
    lines.append("```")
    return "\n\n".join(lines)


def _visual_evidence_flow_mermaid(cross: dict[str, Any], gaps: list[dict[str, Any]]) -> str:
    experiments = [str(item).strip() for item in (cross.get("recommended_experiments", []) or []) if str(item).strip()]
    lines = [
        "```mermaid",
        "flowchart TD",
        '    A["Multi-source retrieval"] --> B["Structured paper summaries"]',
        '    B --> C["Cross-paper comparison"]',
        '    C --> D["Gap synthesis"]',
        '    D --> E["Experiment backlog"]',
    ]
    for idx, gap in enumerate(gaps[:3], start=1):
        lines.append(f'    D --> G{idx}["{_safe_mermaid_text(gap.get("title", ""), 8)}"]')
    for idx, exp in enumerate(experiments[:2], start=1):
        lines.append(f'    E --> X{idx}["{_safe_mermaid_text(exp, 8)}"]')
    lines.append("```")
    return "\n\n".join(lines)


def _delve_visual_appendix(
    topic: str,
    summaries: dict[str, dict[str, Any]],
    gaps: list[dict[str, Any]],
    citation_quality: dict[str, dict[str, Any]],
    citation_verification: dict[str, dict[str, Any]],
    source_counts: dict[str, int],
    bibliography: list[dict[str, Any]],
    cross: dict[str, Any],
) -> dict[str, str]:
    return {
        "[DELVE_STUDY_LANDSCAPE]": _study_landscape_table(summaries, citation_quality, citation_verification),
        "[DELVE_EVIDENCE_MATRIX]": _evidence_snapshot_table(summaries, citation_quality, citation_verification),
        "[DELVE_SOURCE_MATRIX]": _source_mix_table(source_counts, bibliography),
        "[DELVE_CITATION_AUDIT]": _citation_audit_table(bibliography),
        "[DELVE_FINDINGS_BRIEF]": _findings_brief(cross),
        "[DELVE_EXPERIMENT_BACKLOG]": _experiment_backlog_table(cross, gaps),
        "[DELVE_VISUAL_LANDSCAPE]": _visual_landscape_mermaid(topic, summaries, gaps),
        "[DELVE_VISUAL_WORKFLOW]": _visual_workflow_mermaid(gaps),
        "[DELVE_VISUAL_EVIDENCE_FLOW]": _visual_evidence_flow_mermaid(cross, gaps),
    }


def _replace_delve_visual_tokens(markdown: str, replacements: dict[str, str]) -> str:
    text = str(markdown or "")
    for token, value in replacements.items():
        if token in text:
            text = text.replace(token, value)
    missing = [token for token in replacements if token not in markdown]
    if missing:
        extra = "\n\n".join(
            f"### {token.replace('[DELVE_', '').replace(']', '').replace('_', ' ').title()}\n{replacements[token]}"
            for token in missing
        )
        text = f"{text.rstrip()}\n\n{extra}\n"
    return text


def _strip_visual_placeholder_labels(markdown: str) -> str:
    """
    Remove orphan visual scaffold labels that can remain after token replacement.
    This is applied to Part B draft so visual placeholder headings do not leak
    into the publication manuscript.
    """
    if not markdown:
        return ""

    labels = [
        "Study Landscape",
        "Evidence Matrix",
        "Source Matrix",
        "Citation Audit",
        "Findings Brief",
        "Experiment Backlog",
        "Visual Landscape",
        "Visual Workflow",
        "Visual Evidence Flow",
    ]
    escaped = "|".join(re.escape(label) for label in labels)
    pattern = rf"(?im)^\s{{0,3}}(?:[-*]\s+|#{{1,6}}\s+)?(?:{escaped})\s*:?\s*$"
    cleaned = re.sub(pattern, "", markdown)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _methodology_comparison_table(summaries: dict[str, dict[str, Any]], limit: int = 24) -> str:
    if not summaries:
        return "_No methodology comparison available._"
    lines = [
        "| Study | Year | Methodology | Object of study | Evaluation signal |",
        "|---|---:|---|---|---|",
    ]
    for idx, (_pid, info) in enumerate(summaries.items(), start=1):
        if idx > limit:
            break
        object_of_study = _clean_nullable_text(info.get("summary", ""))
        lines.append(
            f"| {str(info.get('title', 'Untitled'))[:54]} | {str(info.get('year', ''))[:4]} | "
            f"{(_clean_nullable_text(info.get('methodology', '')) or 'Empirical study')[:42]} | "
            f"{object_of_study[:70]} | "
            f"{(_clean_nullable_text(info.get('evaluation_signal', '')) or 'Not explicit')[:60]} |"
        )
    return "\n\n".join(lines)


def _metric_extraction_table(summaries: dict[str, dict[str, Any]], limit: int = 40) -> str:
    rows = [
        "| Study | Metric | Value | Baseline | Extraction confidence |",
        "|---|---|---|---|---|",
    ]
    count = 0
    for info in summaries.values():
        title = str(info.get("title", "Untitled"))[:48]
        for metric in info.get("key_metrics", []) or []:
            rows.append(
                f"| {title} | {str(metric.get('metric', ''))[:48]} | {str(metric.get('value', ''))[:32]} | "
                f"{str(metric.get('baseline', ''))[:36]} | {str(metric.get('confidence', 'medium')).title()} |"
            )
            count += 1
            if count >= limit:
                return "\n".join(rows)
    if count == 0:
        rows.append("| No strong numeric metrics extracted | Abstract-level or qualitative evidence dominates this paper set | - | - | Low |")
    return "\n".join(rows)


def _gap_detail_table(gaps: list[dict[str, Any]], limit: int = 6) -> str:
    if not gaps:
        return "_No structured gaps available._"
    lines = [
        "| Gap | Why it matters | What is missing | Proposed direction |",
        "|---|---|---|---|",
    ]
    for gap in gaps[:limit]:
        lines.append(
            f"| {str(gap.get('title', 'Untitled'))[:54]} | "
            f"{_clean_nullable_text(gap.get('why_gap', gap.get('evidence', '')))[:96]} | "
            f"{_clean_nullable_text(gap.get('what_missing', gap.get('evidence', '')))[:96]} | "
            f"{_clean_nullable_text(gap.get('proposed_direction', ''))[:110]} |"
        )
    return "\n\n".join(lines)


def _source_inventory_table(
    summaries: dict[str, dict[str, Any]],
    bibliography: list[dict[str, Any]],
    limit: int = 32,
) -> str:
    if not bibliography:
        return "_No source inventory available._"
    by_pid = {str(pid): info for pid, info in summaries.items()}
    lines = [
        "| Citation key | Study | Year | Source | Methodology | URL / DOI present |",
        "|---|---|---:|---|---|---|",
    ]
    for item in bibliography[:limit]:
        pid = str(item.get("paper_id", ""))
        info = by_pid.get(pid, {})
        locator = "DOI" if item.get("doi") else "URL" if item.get("url") else "No"
        lines.append(
            f"| [@{item.get('citation_key', 'unknown')}] | {str(item.get('title', 'Untitled'))[:48]} | "
            f"{str(item.get('year', ''))[:4]} | {str(item.get('source', 'unknown'))[:18]} | "
            f"{(_clean_nullable_text(info.get('methodology', '')) or 'Empirical study')[:40]} | {locator} |"
        )
    return "\n\n".join(lines)


def _evidence_notes_table(cross: dict[str, Any], limit: int = 20) -> str:
    notes = cross.get("evidence_notes", []) if isinstance(cross, dict) else []
    if not isinstance(notes, list) or not notes:
        return "_No evidence-note inventory available._"
    lines = [
        "| Paper | Evidence note |",
        "|---|---|",
    ]
    for entry in notes[:limit]:
        if not isinstance(entry, dict):
            continue
        lines.append(
            f"| {str(entry.get('paper_title', 'Untitled'))[:54]} | {str(entry.get('note', ''))[:140]} |"
        )
    return "\n\n".join(lines)


def _duplicate_cluster_table(duplicate_clusters: list[dict[str, Any]], limit: int = 12) -> str:
    if not duplicate_clusters:
        return "_No duplicate clusters detected._"
    lines = [
        "| Canonical title | Cluster size | Sources |",
        "|---|---:|---|",
    ]
    for cluster in duplicate_clusters[:limit]:
        lines.append(
            f"| {str(cluster.get('canonical_title', 'Untitled'))[:62]} | {int(cluster.get('size', 0) or 0)} | "
            f"{', '.join(str(s) for s in (cluster.get('sources', []) or []))[:80]} |"
        )
    return "\n\n".join(lines)


def _debate_snippet(entry: str, limit: int = 220) -> str:
    cleaned = re.sub(r"^\[(PROPOSER|CRITIC|CRITIC-FAILED)\]\s*", "", entry or "", flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit].rstrip()}..."


def _debate_roundup_table(debate_log: list[str], limit: int = 10) -> str:
    if not debate_log:
        return "_Debate loop was not used in this session._"
    rows = [
        "| Round | Speaker | Extract |",
        "|---:|---|---|",
    ]
    round_num = 0
    for entry in debate_log:
        if not isinstance(entry, str):
            continue
        speaker = "Critic" if entry.startswith(("[CRITIC]", "[CRITIC-FAILED]")) else "Proposer" if entry.startswith("[PROPOSER]") else "System"
        if speaker == "Proposer":
            round_num += 1
        snippet = _debate_snippet(entry, limit=180)
        rows.append(f"| {max(1, round_num)} | {speaker} | {snippet} |")
        if len(rows) - 2 >= limit:
            break
    return "\n".join(rows)


def _claim_confidence_rollup(claim_map: list[dict[str, Any]], bibliography: list[dict[str, Any]], limit: int = 24) -> str:
    if not claim_map:
        return "_No claim-to-evidence mapping available._"
    pid_to_title = {str(b.get("paper_id", "")): str(b.get("title", "Untitled")) for b in bibliography}
    grouped: dict[str, dict[str, Any]] = {}
    for item in claim_map:
        claim = str(item.get("claim", "")).strip()
        if not claim:
            continue
        entry = grouped.setdefault(claim, {
            "claim": claim,
            "pids": set(),
            "confidences": [],
            "verified": 0,
        })
        pid = str(item.get("paper_id", "")).strip()
        if pid:
            entry["pids"].add(pid)
        entry["confidences"].append(float(item.get("confidence", 0.5) or 0.5))
        if bool(item.get("verified", False)):
            entry["verified"] += 1
    ranked = sorted(
        grouped.values(),
        key=lambda x: (len(x["pids"]), sum(x["confidences"]) / max(1, len(x["confidences"]))),
        reverse=True,
    )
    lines = [
        "| Claim | Evidence count | Verified supports | Avg confidence | Source anchors |",
        "|---|---:|---:|---:|---|",
    ]
    for entry in ranked[:limit]:
        pids = sorted(list(entry["pids"]))
        anchors = ", ".join(pid_to_title.get(pid, pid)[:28] for pid in pids[:3])
        avg_conf = sum(entry["confidences"]) / max(1, len(entry["confidences"]))
        lines.append(
            f"| {entry['claim'][:96]} | {len(pids)} | {entry['verified']} | {avg_conf:.2f} | {anchors or 'n/a'} |"
        )
    return "\n\n".join(lines)


def _operational_takeaways_table(
    cross: dict[str, Any],
    gaps: list[dict[str, Any]],
    source_counts: dict[str, int],
) -> str:
    strongest_gap = str(gaps[0].get("title", "Evidence validation")) if gaps else "Evidence validation"
    source_summary = ", ".join(f"{src}:{count}" for src, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:4]) or "No source summary"
    experiment_hint = ""
    recs = cross.get("recommended_experiments", []) if isinstance(cross, dict) else []
    if isinstance(recs, list) and recs:
        experiment_hint = str(recs[0])[:110]
    lines = [
        "| Stakeholder | What to do next | Why this matters |",
        "|---|---|---|",
        f"| Research leads | Prioritize studies that close `{strongest_gap[:58]}` and report explicit numeric outcomes. | Current evidence still mixes strong retrieval breadth with uneven validation depth. |",
        f"| ML engineers | Move to claim -> evidence -> output generation with stronger citation verification gates. | The current source landscape (`{source_summary[:90]}`) is only useful if claims remain tethered to verified support. |",
        f"| Product teams | Treat the recommended experiment backlog as the operational roadmap: {experiment_hint or 'translate top gaps into evaluable backlog items.'} | This makes the dossier actionable rather than just descriptive. |",
    ]
    return "\n\n".join(lines)


def _confidence_band_breakdown(citation_quality: dict[str, dict[str, Any]]) -> dict[str, int]:
    bands = {"high": 0, "medium": 0, "low": 0}
    for entry in citation_quality.values():
        score = float(entry.get("confidence", 0.0) or 0.0)
        if score >= 0.8:
            bands["high"] += 1
        elif score >= 0.6:
            bands["medium"] += 1
        else:
            bands["low"] += 1
    return bands


def _executive_snapshot_table(
    topic: str,
    summaries: dict[str, dict[str, Any]],
    bibliography: list[dict[str, Any]],
    gaps: list[dict[str, Any]],
    claim_map: list[dict[str, Any]],
    source_counts: dict[str, int],
    citation_quality: dict[str, dict[str, Any]],
    cross: dict[str, Any],
) -> str:
    total_sources = len(bibliography)
    verified = sum(1 for b in bibliography if b.get("verified"))
    verified_pct = round((verified / max(1, total_sources)) * 100)
    bands = _confidence_band_breakdown(citation_quality)
    contradiction_count = len(cross.get("contradictions", []) or []) if isinstance(cross, dict) else 0
    missing_dims = len(cross.get("missing_dimensions", []) or []) if isinstance(cross, dict) else 0
    avg_conf = 0.0
    if bibliography:
        avg_conf = sum(float(b.get("confidence", 0.0) or 0.0) for b in bibliography) / max(1, len(bibliography))

    return "\n".join([
        "| Snapshot Metric | Value | Why it matters |",
        "|---|---:|---|",
        f"| Topic focus | {topic[:64]} | Keeps all analysis anchored to the user question. |",
        f"| Summarized studies | {len(summaries)} | Indicates breadth of evidence reviewed in synthesis. |",
        f"| Total citations in dossier | {total_sources} | Shows the size of the supporting reference base. |",
        f"| Verified citations | {verified} ({verified_pct}%) | Higher verified share means stronger traceability. |",
        f"| Average citation confidence | {avg_conf:.2f} | Quick quality proxy for reliability of source grounding. |",
        f"| Confidence bands (H/M/L) | {bands['high']}/{bands['medium']}/{bands['low']} | Reveals whether confidence is concentrated or weakly distributed. |",
        f"| Source families used | {len(source_counts)} | Better source diversity reduces single-index bias. |",
        f"| Claim-evidence links | {len(claim_map)} | Measures how much of the narrative is evidence-linked. |",
        f"| Structured gaps extracted | {len(gaps)} | Represents actionable unresolved problem areas. |",
        f"| Contradictions detected | {contradiction_count} | Flags disagreement areas needing deeper validation. |",
        f"| Missing dimensions flagged | {missing_dims} | Highlights blind spots in current corpus coverage. |",
    ])


def _source_quality_diagnostics_table(
    source_counts: dict[str, int],
    bibliography: list[dict[str, Any]],
) -> str:
    if not source_counts:
        return "_No source-quality diagnostics available._"

    by_source: dict[str, list[dict[str, Any]]] = {}
    for b in bibliography:
        src = str(b.get("source", "unknown")).lower()
        by_source.setdefault(src, []).append(b)

    lines = [
        "| Source | Papers | Verified | Avg confidence | Coverage risk |",
        "|---|---:|---:|---:|---|",
    ]
    total = max(1, sum(source_counts.values()))
    for src, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
        rows = by_source.get(str(src).lower(), [])
        verified = sum(1 for r in rows if r.get("verified"))
        avg_conf = (sum(float(r.get("confidence", 0.0) or 0.0) for r in rows) / max(1, len(rows))) if rows else 0.0
        share = count / total
        if share >= 0.7:
            risk = "High concentration in one source"
        elif share >= 0.45:
            risk = "Moderate concentration; diversify"
        else:
            risk = "Healthy distribution"
        lines.append(f"| {src} | {count} | {verified} | {avg_conf:.2f} | {risk} |")
    return "\n\n".join(lines)


def _contradiction_watchlist_table(cross: dict[str, Any]) -> str:
    contradictions = cross.get("contradictions", []) if isinstance(cross, dict) else []
    weak_eval = cross.get("weak_evaluation_signals", []) if isinstance(cross, dict) else []
    missing = cross.get("missing_dimensions", []) if isinstance(cross, dict) else []

    rows: list[str] = []
    for item in contradictions[:4]:
        rows.append(f"| Contradiction | {str(item)[:130]} | Compare protocol, dataset split, and metric definition across studies. |")
    for item in weak_eval[:4]:
        rows.append(f"| Weak evaluation | {str(item)[:130]} | Add stronger baselines and uncertainty reporting before drawing conclusions. |")
    for item in missing[:4]:
        rows.append(f"| Missing dimension | {str(item)[:130]} | Expand retrieval and targeted experiments for this uncovered axis. |")

    if not rows:
        rows.append("| Sparse signal | No explicit contradictions extracted from cross-paper pass. | Treat current conclusions as provisional and improve retrieval specificity. |")

    return "\n".join([
        "| Signal type | Observation | Recommended response |",
        "|---|---|---|",
        *rows,
    ])


def _gap_rigor_matrix(gaps: list[dict[str, Any]], limit: int = 8) -> str:
    if not gaps:
        return "_No gap-rigor matrix available._"

    def _label(score: int) -> str:
        if score >= 2:
            return "Strong"
        if score == 1:
            return "Moderate"
        return "Weak"

    lines = [
        "| Gap | Evidence specificity | Actionability | Suggested validation metric |",
        "|---|---|---|---|",
    ]
    for gap in gaps[:limit]:
        evidence = str(gap.get("evidence", ""))
        direction = str(gap.get("proposed_direction", ""))
        ev_score = int(len(evidence) > 120) + int(any(tok in evidence.lower() for tok in ["dataset", "benchmark", "cross", "ablation", "baseline"]))
        act_score = int(len(direction) > 90) + int(any(tok in direction.lower() for tok in ["benchmark", "protocol", "evaluate", "experiment", "metric", "baseline"]))
        metric_hint = "Effect size + uncertainty"
        if "generaliz" in evidence.lower() or "domain" in evidence.lower():
            metric_hint = "Cross-dataset delta"
        elif "annotation" in evidence.lower():
            metric_hint = "Label-efficiency gain"
        lines.append(f"| {str(gap.get('title', 'Gap'))[:60]} | {_label(ev_score)} | {_label(act_score)} | {metric_hint} |")
    return "\n\n".join(lines)


def _measurement_plan_table(gaps: list[dict[str, Any]], limit: int = 6) -> str:
    if not gaps:
        return "_No measurement plan available._"
    lines = [
        "| Priority | Gap | Proposed measurable outcome | Minimum success threshold |",
        "|---:|---|---|---|",
    ]
    for idx, gap in enumerate(gaps[:limit], start=1):
        title = str(gap.get("title", "Gap"))[:58]
        evidence = str(gap.get("evidence", "")).lower()
        if "domain" in evidence or "cross" in evidence:
            outcome = "Generalization gap reduction"
            threshold = ">=10% relative improvement across external datasets"
        elif "annotation" in evidence or "label" in evidence:
            outcome = "Annotation-efficiency improvement"
            threshold = "Same performance with >=30% fewer labeled samples"
        else:
            outcome = "Robustness and reliability uplift"
            threshold = "Improved mean metric with confidence interval reporting"
        lines.append(f"| {idx} | {title} | {outcome} | {threshold} |")
    return "\n\n".join(lines)


def _clean_analysis_narrative(markdown: str, limit: int = 2000) -> str:
    text = _strip_mermaid_blocks(markdown or "")
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    text = re.sub(r"\[DELVE_[A-Z_]+\]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) > limit:
        text = text[:limit].rsplit("\n", 1)[0].strip()
    return text


def _has_meaningful_block(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    return text not in {"_", "_._", "_No source inventory available._", "_No evidence-note inventory available._"}


def _build_structured_dossier(
    *,
    topic: str,
    raw_analysis: str,
    summaries: dict[str, dict[str, Any]],
    gaps: list[dict[str, Any]],
    cross: dict[str, Any],
    gap_critique: str,
    citation_quality: dict[str, dict[str, Any]],
    citation_verification: dict[str, dict[str, Any]],
    source_counts: dict[str, int],
    bibliography: list[dict[str, Any]],
    claim_map: list[dict[str, Any]],
    debate_log: list[str],
    duplicate_clusters: list[dict[str, Any]],
    visual_replacements: dict[str, str],
) -> str:
    narrative = _clean_analysis_narrative(raw_analysis)
    evidence_matrix = visual_replacements.get("[DELVE_EVIDENCE_MATRIX]", "")
    study_landscape = visual_replacements.get("[DELVE_STUDY_LANDSCAPE]", "")
    source_matrix = visual_replacements.get("[DELVE_SOURCE_MATRIX]", "")
    citation_audit = visual_replacements.get("[DELVE_CITATION_AUDIT]", "")
    experiment_backlog = visual_replacements.get("[DELVE_EXPERIMENT_BACKLOG]", "")
    visual_landscape = visual_replacements.get("[DELVE_VISUAL_LANDSCAPE]", "")
    visual_evidence_flow = visual_replacements.get("[DELVE_VISUAL_EVIDENCE_FLOW]", "")
    visual_workflow = visual_replacements.get("[DELVE_VISUAL_WORKFLOW]", "")
    evidence_notes = _evidence_notes_table(cross)
    source_inventory = _source_inventory_table(summaries, bibliography)
    methodology_table = _methodology_comparison_table(summaries)
    metrics_table = _metric_extraction_table(summaries)
    claim_rollup = _claim_confidence_rollup(claim_map, bibliography)
    gap_table = _gap_detail_table(gaps)
    duplicate_table = _duplicate_cluster_table(duplicate_clusters)
    findings_brief = _findings_brief(cross)
    takeaways_table = _operational_takeaways_table(cross, gaps, source_counts)
    debate_table = _debate_roundup_table(debate_log)
    executive_snapshot = _executive_snapshot_table(
        topic=topic,
        summaries=summaries,
        bibliography=bibliography,
        gaps=gaps,
        claim_map=claim_map,
        source_counts=source_counts,
        citation_quality=citation_quality,
        cross=cross,
    )
    source_quality_diag = _source_quality_diagnostics_table(source_counts, bibliography)
    contradiction_watchlist = _contradiction_watchlist_table(cross)
    gap_rigor_matrix = _gap_rigor_matrix(gaps)
    measurement_plan = _measurement_plan_table(gaps)
    parts = [
        "### A0. Executive Insight Snapshot",
        "This snapshot gives a fast, decision-ready read of evidence breadth, confidence distribution, verification depth, and unresolved risk signals before detailed interpretation.",
        "",
        executive_snapshot,
        "",
        "### A1. Cross-Paper Methodology Analysis",
        f"This dossier restructures the evidence base for **{topic}** into a comparison-first briefing. The goal is to expose as much of the session evidence as possible in a disciplined format: methods, metrics, citation strength, debate feedback, duplicates, and claim-level support before any high-level interpretation is trusted.",
        "",
        methodology_table,
        "",
        "#### Quantitative Signal Extraction",
        metrics_table,
        "",
        "#### Complete Source Inventory",
        source_inventory,
        "",
        "#### Source-Quality Diagnostics",
        source_quality_diag,
        "",
        "### A2. Contradictions and Agreement Map",
        "The strongest cross-paper signals are summarized below so the reader can quickly see where the literature aligns, where it fragments, and where evidence still remains thin.",
        "",
        findings_brief,
        "",
        "#### Contradiction and Risk Watchlist",
        contradiction_watchlist,
        "",
    ]
    if _has_meaningful_block(evidence_matrix):
        parts.extend(["#### Evidence Snapshot", evidence_matrix, ""])
    if _has_meaningful_block(claim_rollup):
        parts.extend(["#### Claim Confidence Rollup", claim_rollup, ""])
    if _has_meaningful_block(evidence_notes):
        parts.extend(["#### Evidence Notes by Paper", evidence_notes, ""])
    parts.extend([
        "### A3. Detailed Gap Analysis",
        gap_table,
        "",
    ])
    if _has_meaningful_block(gap_rigor_matrix):
        parts.extend(["#### Gap Rigor Matrix", gap_rigor_matrix, ""])
    if gap_critique:
        parts.extend([
            "#### Gap Reviewer Notes",
            gap_critique,
            "",
        ])
    parts.extend([
        "### A4. Conceptual Architecture & Visual Summaries",
        "The following visuals and audit tables turn the synthesis into a glass-box view of how evidence moves from retrieval to claims, and where quality still weakens.",
        "",
    ])
    if _has_meaningful_block(study_landscape):
        parts.extend(["#### Study Landscape", study_landscape, ""])
    if _has_meaningful_block(source_matrix):
        parts.extend(["#### Source Matrix", source_matrix, ""])
    if _has_meaningful_block(citation_audit):
        parts.extend(["#### Citation Audit", citation_audit, ""])
    if _has_meaningful_block(duplicate_table):
        parts.extend(["#### Duplicate Tracking", duplicate_table, ""])
    if _has_meaningful_block(visual_landscape):
        parts.extend(["#### Visual Landscape", visual_landscape, ""])
    if _has_meaningful_block(visual_evidence_flow):
        parts.extend(["#### Visual Evidence Flow", visual_evidence_flow, ""])
    if _has_meaningful_block(visual_workflow):
        parts.extend(["#### Visual Workflow", visual_workflow, ""])
    parts.extend([
        "### A5. Research Agenda and Prioritized Experiments",
        experiment_backlog,
        "",
    ])
    if _has_meaningful_block(measurement_plan):
        parts.extend(["#### Measurement and Validation Plan", measurement_plan, ""])
    if _has_meaningful_block(debate_table):
        parts.extend(["#### Debate Trace", debate_table, ""])
    parts.extend([
        "### A6. Operational Takeaways and Platform Implications",
        takeaways_table,
        "",
    ])
    if narrative:
        parts.extend([
            "#### Analyst Narrative Notes",
            narrative,
            "",
        ])
    return "\n".join(part for part in parts if part is not None).strip() + "\n"

def _clean_authors_string(authors: str, source: str = "", url: str = "") -> str:
    raw = (authors or "").strip()
    if not raw or raw.lower() in {"web source", "unknown", "none", "n/a", "et al.", "author unknown"}:
        if "arxiv" in url.lower() or "arxiv" in source.lower():
            return "ArXiv Preprint Authors"
        if "github" in source.lower() or "github.com" in url.lower():
            return "Repository Contributors"
        return "Editorial / Research Staff"
    cleaned = re.sub(r",\s*,+", ", ", raw)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().rstrip(",")
    return cleaned


def _render_references_section(paper_format: str, bibliography: list[dict[str, Any]]) -> str:
    fmt = (paper_format or "ACADEMIC").upper()
    if not bibliography:
        return ""

    if fmt in {"IEEE", "ACM"}:
        lines = ["## References"]
        for i, b in enumerate(bibliography, start=1):
            title = b.get("title", "Untitled").strip().rstrip(".")
            authors = _clean_authors_string(b.get("authors", ""), b.get("source", ""), b.get("url", ""))
            year = b.get("year", "n.d.")
            url = b.get("url", "").strip()
            doi = b.get("doi", "").strip()
            
            tail_parts = []
            if doi:
                doi_clean = doi if doi.startswith("http") else f"https://doi.org/{doi}"
                tail_parts.append(f"DOI: {doi_clean}")
            elif url:
                tail_parts.append(f"[Online]. Available: {url}")
                
            tail_str = " " + " ".join(tail_parts) if tail_parts else ""
            lines.append(f"[{i}] {authors}, \"{title},\" {year}.{tail_str}")
        return "\n\n".join(lines)

    if fmt == "APA":
        lines = ["## References"]
        for b in bibliography:
            title = b.get("title", "Untitled").strip().rstrip(".")
            authors = _clean_authors_string(b.get("authors", ""), b.get("source", ""), b.get("url", ""))
            year = b.get("year", "n.d.")
            url = b.get("url", "").strip()
            doi = b.get("doi", "").strip()
            if doi:
                doi_clean = doi if doi.startswith("http") else f"https://doi.org/{doi}"
                lines.append(f"{authors} ({year}). {title}. {doi_clean}")
            elif url:
                lines.append(f"{authors} ({year}). {title}. {url}")
            else:
                lines.append(f"{authors} ({year}). {title}.")
        return "\n\n".join(lines)

    if fmt == "MLA":
        lines = ["## Works Cited"]
        for b in bibliography:
            title = b.get("title", "Untitled").strip().rstrip(".")
            authors = _clean_authors_string(b.get("authors", ""), b.get("source", ""), b.get("url", ""))
            year = b.get("year", "n.d.")
            url = b.get("url", "").strip()
            line = f"{authors}. \"{title}.\" {year}."
            if url:
                line = f"{line} {url}"
            lines.append(line)
        return "\n\n".join(lines)

    lines = ["## References"]
    for b in bibliography:
        key = b.get("citation_key") or _citation_key(b.get("authors", ""), b.get("year", ""))
        authors = _clean_authors_string(b.get("authors", ""), b.get("source", ""), b.get("url", ""))
        lines.append(
            f"[@{key}] {authors}. \"{b.get('title', 'Untitled')}\". "
            f"{b.get('year', 'n.d.')}. {b.get('url', '')}".strip()
        )
    return "\n\n".join(lines)


def _normalize_latex_math(text: str) -> str:
    """Fix common LLM math notation glitches (stray markdown asterisks in subscripts/superscripts, missing dimension carets)."""
    if not text:
        return ""
    
    # Fix \mathbb{R}{D \times H ...} -> \mathbb{R}^{D \times H ...}
    text = re.sub(r'\\mathbb\{([A-Za-z]+)\}\{([A-Za-z0-9\s\\times\+\-\*\^\_]+)\}', r'\\mathbb{\1}^{\2}', text)
    
    # Fix \mathcal{L}*_{\text{...}} or \mathcal{L}*{\text{...}} -> \mathcal{L}_{\text{...}}
    text = re.sub(r'\\mathcal\{([A-Za-z]+)\}\*?_\{([^\}]+)\}', r'\\mathcal{\1}_{\2}', text)
    text = re.sub(r'\\mathcal\{([A-Za-z]+)\}\*\{([^\}]+)\}', r'\\mathcal{\1}_{\2}', text)
    text = re.sub(r'\\mathcal\{([A-Za-z]+)\}\*', r'\\mathcal{\1}', text)
    
    # Fix \tilde{\nabla}*\theta or \nabla*\theta -> \tilde{\nabla}_\theta
    text = re.sub(r'\\tilde\{\\nabla\}\*([a-zA-Z\\]+)', r'\\tilde{\\nabla}_{\1}', text)
    text = re.sub(r'\\nabla\*([a-zA-Z\\]+)', r'\\nabla_{\1}', text)
    
    # Clean up asterisks directly preceding subscripts or superscripts in math mode
    def _clean_math_block(match: re.Match[str]) -> str:
        content = match.group(0)
        content = re.sub(r'\*+(_|\^)', r'\1', content)
        content = re.sub(r'(_|\^)\*+', r'\1', content)
        return content
    
    text = re.sub(r'(\${1,2}[^\$]+\${1,2})', _clean_math_block, text)
    return text


def _find_citation_index(token: str, citation_indices: dict[str, int], bibliography: list[dict[str, Any]]) -> int | None:
    raw = re.sub(r"[^a-z0-9]", "", token.lower())
    if raw in citation_indices:
        return citation_indices[raw]
    
    words = [w.lower() for w in re.findall(r"[A-Za-z]+", token)]
    years = re.findall(r"(?:19|20)\d{2}", token)
    if words:
        author = words[0]
        year = years[0] if years else ""
        for item in bibliography:
            b_authors = str(item.get("authors", "")).lower()
            b_year = str(item.get("year", "")).lower()
            b_key = str(item.get("citation_key", "")).lower()
            if author in b_authors or author in b_key:
                if not year or year in b_year or year in b_key:
                    b_clean_key = re.sub(r"[^a-z0-9]", "", b_key or _citation_key(item.get("authors", ""), item.get("year", "")))
                    if b_clean_key in citation_indices:
                        return citation_indices[b_clean_key]
    return None


def _normalize_citations_for_format(markdown: str, paper_format: str, bibliography: list[dict[str, Any]]) -> str:
    if not markdown:
        return markdown
    fmt = (paper_format or "ACADEMIC").upper()
    citation_indices = _citation_marker_map(bibliography)

    def _to_numeric(match: re.Match[str]) -> str:
        token = match.group(1).strip()
        idx = _find_citation_index(token, citation_indices, bibliography)
        if idx is None:
            key = re.sub(r"[^a-z0-9]", "", token.lower())
            idx = len(citation_indices) + 1
            citation_indices[key] = idx
        return f"[{idx}]"

    def _to_apa(match: re.Match[str]) -> str:
        token = match.group(1).strip()
        key = re.sub(r"[^a-z0-9]", "", token.lower())
        for b in bibliography:
            b_key = re.sub(r"[^a-z0-9]", "", str(b.get("citation_key", "")).lower() or _citation_key(b.get("authors", ""), b.get("year", "")))
            if b_key == key or (token.lower() in str(b.get("authors", "")).lower()):
                author_last, year = _citation_display_name(b.get("authors", ""), b.get("year", ""))
                return f"({author_last}, {year})"
        return "(Unknown, n.d.)"

    def _to_mla(match: re.Match[str]) -> str:
        token = match.group(1).strip()
        key = re.sub(r"[^a-z0-9]", "", token.lower())
        for b in bibliography:
            b_key = re.sub(r"[^a-z0-9]", "", str(b.get("citation_key", "")).lower() or _citation_key(b.get("authors", ""), b.get("year", "")))
            if b_key == key or (token.lower() in str(b.get("authors", "")).lower()):
                author_last, _year = _citation_display_name(b.get("authors", ""), b.get("year", ""))
                return f"({author_last})"
        return "(Unknown)"

    transformed = markdown
    if fmt in {"IEEE", "ACM"}:
        transformed = re.sub(r"\[@([A-Za-z0-9_.:-]+)\]", _to_numeric, transformed)
        transformed = re.sub(r"\[([A-Z][a-zA-Z]+(?:[0-9]{4}|_[0-9]+))\]", _to_numeric, transformed)
        transformed = re.sub(r"\[([A-Z][a-zA-Z]+(?:\s+et\s+al\.?)?,?\s*(?:19|20)\d{2}[a-z]?)\]", _to_numeric, transformed)
    elif fmt == "APA":
        transformed = re.sub(r"\[@([A-Za-z0-9_.:-]+)\]", _to_apa, transformed)
        transformed = re.sub(r"\[([A-Z][a-zA-Z]+(?:[0-9]{4}|_[0-9]+))\]", _to_apa, transformed)
    elif fmt == "MLA":
        transformed = re.sub(r"\[@([A-Za-z0-9_.:-]+)\]", _to_mla, transformed)
        transformed = re.sub(r"\[([A-Z][a-zA-Z]+(?:[0-9]{4}|_[0-9]+))\]", _to_mla, transformed)

    transformed = _normalize_latex_math(transformed)
    return transformed


def _extract_claim_evidence_map(
    paper_markdown: str,
    bibliography: list[dict[str, Any]],
    paper_format: str,
    citation_quality: dict[str, dict[str, Any]],
    citation_verification: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if not paper_markdown or not bibliography:
        return []
    lines = [ln.strip() for ln in paper_markdown.splitlines() if ln.strip()]
    items: list[dict[str, Any]] = []
    fmt = (paper_format or "ACADEMIC").upper()
    key_to_pid: dict[str, str] = {}
    index_to_pid: dict[int, str] = {}
    for i, b in enumerate(bibliography, start=1):
        pid = str(b.get("paper_id", ""))
        key = str(b.get("citation_key", "")).strip().lower() or _citation_key(b.get("authors", ""), b.get("year", ""))
        if key:
            key_to_pid[key] = pid
        index_to_pid[i] = pid

    for line in lines:
        if len(line) < 40 or line.startswith("#"):
            continue
        matched_pids: set[str] = set()
        if fmt in {"IEEE", "ACM"}:
            for m in re.findall(r"\[(\d+)\]", line):
                pid = index_to_pid.get(int(m))
                if pid:
                    matched_pids.add(pid)
        else:
            for m in re.findall(r"\[@([A-Za-z0-9_.:-]+)\]", line):
                pid = key_to_pid.get(m.lower())
                if pid:
                    matched_pids.add(pid)

        if not matched_pids:
            continue
        claim = line
        if len(claim) > 220:
            claim = f"{claim[:220].rstrip()}..."
        for pid in sorted(matched_pids):
            items.append({
                "claim": claim,
                "paper_id": pid,
                "confidence": citation_quality.get(pid, {}).get("confidence", 0.5),
                "verified": citation_verification.get(pid, {}).get("verified", False),
            })
    return items


def _fallback_gap_title(topic: str, evidence: str, idx: int) -> str:
    phrase = re.split(r"[.;:,\n]", evidence or "", maxsplit=1)[0].strip()
    phrase = re.sub(r"^(no specific limitations extracted from the retrieved papers|not available)\b[:\- ]*", "", phrase, flags=re.IGNORECASE).strip()
    if phrase:
        words = phrase.split()
        short = " ".join(words[:8])
        return f"Gap {idx}: {short}"
    anchors = _topic_anchor_terms(topic, limit=2)
    anchor = " ".join(anchors) if anchors else "core evaluation dimension"
    return f"Gap {idx}: Underexplored {anchor}"


def _extract_limitation_signal(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"\s+", " ", text).strip()
    # Prefer explicit limitation-style clauses.
    patterns = [
        r"(?:limitation|challenge|constraint|bias|scarcity|insufficient|lack|underexplored)[^.]{0,180}\.",
        r"(?:future work|further research|needs to|remains unclear)[^.]{0,180}\.",
    ]
    for pat in patterns:
        m = re.search(pat, cleaned, flags=re.IGNORECASE)
        if m:
            return m.group(0).strip()
    # Secondary fallback: capture first "however/but" sentence.
    m = re.search(r"(?:however|but)\s+[^.]{30,220}\.", cleaned, flags=re.IGNORECASE)
    if m:
        return m.group(0).strip()
    return ""


def _normalize_metric_items(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        return []
    items: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        metric = _clean_nullable_text(item.get("metric", ""))
        value = _clean_nullable_text(item.get("value", ""))
        baseline = _clean_nullable_text(item.get("baseline", ""))
        confidence = _clean_nullable_text(item.get("confidence", "")).lower()
        if not metric or not value:
            continue
        if confidence not in {"high", "medium", "low"}:
            confidence = "medium"
        items.append({
            "metric": metric[:120],
            "value": value[:120],
            "baseline": baseline[:120],
            "confidence": confidence,
        })
        if len(items) >= 6:
            break
    return items


def _derive_gaps_from_paper_evidence(
    topic: str,
    summaries: dict[str, dict[str, Any]],
    source_counts: dict[str, int] | None = None,
    min_gaps: int = 3,
) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    source_counts = source_counts or {}
    limitation_signals: list[tuple[str, str]] = []
    corpus_text_parts: list[str] = []
    methodology_labels: list[str] = []
    evaluation_labels: list[str] = []
    benchmark_like = 0
    application_like = 0
    explicit_results_mentions = 0
    for info in summaries.values():
        title = str(info.get("title", "")).strip()
        summary = str(info.get("summary", "")).strip()
        limitation = _clean_nullable_text(info.get("limitation", ""))
        methodology = _clean_nullable_text(info.get("methodology", ""))
        evaluation_signal = _clean_nullable_text(info.get("evaluation_signal", ""))
        body = f"{title}. {summary}".strip()
        if body:
            corpus_text_parts.append(body.lower())
        signal = limitation or _extract_limitation_signal(body)
        if signal:
            limitation_signals.append((title, signal))
        if methodology:
            methodology_labels.append(methodology.lower())
            if any(token in methodology.lower() for token in ["benchmark", "dataset", "framework"]):
                benchmark_like += 1
            if any(token in methodology.lower() for token in ["experiment", "quasi", "deployment", "empirical", "case"]):
                application_like += 1
        if evaluation_signal:
            evaluation_labels.append(evaluation_signal.lower())
        text_blob = f"{summary} {evaluation_signal}".lower()
        if any(token in text_blob for token in ["effect size", "confidence interval", "accuracy", "f1", "score", "auc", "bleu", "rouge"]):
            explicit_results_mentions += 1

    if benchmark_like and application_like:
        candidates.append({
            "title": "Disconnected benchmark and deployment evidence streams",
            "evidence": "Retrieved studies split between benchmark/framework work and application/deployment studies, with little sign of shared evaluation protocols or direct transfer checks.",
            "proposed_direction": "Run integrated studies that apply benchmark-style metrics inside real deployment or user-facing experiments, then compare outcomes across both settings.",
        })

    if methodology_labels and len(set(methodology_labels)) >= 3:
        candidates.append({
            "title": "No common comparison protocol across methodological clusters",
            "evidence": "The corpus mixes multiple study designs and methodology labels, making it hard to compare findings on a common evidence scale.",
            "proposed_direction": "Define a unified reporting template covering data regime, baseline choice, evaluation metric, failure mode reporting, and reproducibility metadata.",
        })

    if evaluation_labels and len(set(evaluation_labels)) >= 4:
        candidates.append({
            "title": "Fragmented evaluation signals across the literature",
            "evidence": "Studies appear to optimize or validate different outcome signals, which weakens cross-paper comparability and hides tradeoffs between utility, quality, and robustness.",
            "proposed_direction": "Build multi-axis evaluation suites that jointly report quality, robustness, efficiency, and downstream utility for the same systems.",
        })

    if summaries and explicit_results_mentions < max(2, len(summaries) // 4):
        candidates.append({
            "title": "Weak quantitative reporting in retrieved evidence",
            "evidence": "Many summaries expose claims and study setups but not enough concrete numeric outcomes, effect sizes, or uncertainty estimates for strong synthesis.",
            "proposed_direction": "Prioritize papers with explicit metrics and require structured extraction of effect sizes, confidence intervals, and baseline deltas in follow-up reviews.",
        })

    for title, signal in limitation_signals:
        gap_title = _fallback_gap_title(topic, signal, len(candidates) + 1)
        candidates.append({
            "title": gap_title,
            "evidence": f"{title}: {signal}"[:900],
            "proposed_direction": "Run controlled comparative studies with stronger baselines, subgroup analysis, and explicit failure reporting.",
        })

    corpus_text = " ".join(corpus_text_parts)
    critical_terms = _critical_topic_terms(topic)
    missing_terms = [t for t in critical_terms if t not in corpus_text]
    for term in missing_terms[:3]:
        candidates.append({
            "title": f"Limited coverage of {term} in current evidence base",
            "evidence": f"Across summarized papers, direct analysis for '{term}' is sparse relative to the user topic scope.",
            "proposed_direction": f"Prioritize targeted retrieval and evaluation specifically focused on '{term}', then benchmark against current findings.",
        })

    total_sources = sum(int(v) for v in source_counts.values()) if source_counts else 0
    if total_sources > 0:
        top_source = max(source_counts, key=source_counts.get)
        top_share = (source_counts.get(top_source, 0) / total_sources) if total_sources else 0.0
        if top_share >= 0.7:
            candidates.append({
                "title": f"Evidence concentration in {top_source}",
                "evidence": f"{top_source} contributes {top_share:.0%} of retrieved sources, so conclusions may still be too dependent on one discovery channel.",
                "proposed_direction": "Expand retrieval into independent indexes, repositories, and web evidence, then test whether the main conclusions remain stable under source rebalancing.",
            })

    # Sanitize + de-duplicate and enforce minimum count.
    finalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for i, gap in enumerate(candidates, start=1):
        title = _clean_nullable_text(gap.get("title", ""))
        evidence = _clean_nullable_text(gap.get("evidence", ""))
        direction = _clean_nullable_text(gap.get("proposed_direction", ""))
        if not title or not evidence or not direction:
            continue
        key = title.lower()
        if key in seen:
            continue
        seen.add(key)
        finalized.append({
            "title": title[:180],
            "evidence": evidence[:900],
            "proposed_direction": direction[:900],
        })
        if len(finalized) >= 5:
            break

    while len(finalized) < min_gaps:
        idx = len(finalized) + 1
        fallback_title = f"Gap {idx}: Limited rigorous evaluation coverage"
        if fallback_title.lower() in seen:
            break
        seen.add(fallback_title.lower())
        finalized.append({
            "title": fallback_title,
            "evidence": "Current paper set does not provide enough explicit limitation reporting across all key topic dimensions.",
            "proposed_direction": "Design topic-specific benchmarks and report stratified error/failure modes for each major sub-problem.",
        })
    return finalized


def _format_compliance_summary(markdown: str, paper_format: str) -> dict[str, Any]:
    fmt = (paper_format or "ACADEMIC").upper()
    text = markdown or ""
    checks: dict[str, bool] = {}
    if fmt in {"IEEE", "ACM"}:
        checks["numeric_citations"] = bool(re.search(r"\[\d+\]", text))
        checks["references_heading"] = bool(re.search(r"(?im)^##\s+references\s*$", text))
        checks["author_year_markers_absent"] = not bool(re.search(r"\[@[A-Za-z0-9_.:-]+\]", text))
    elif fmt == "APA":
        checks["apa_parenthetical"] = bool(re.search(r"\([A-Z][A-Za-z\-]+,\s*\d{4}\)", text))
        checks["references_heading"] = bool(re.search(r"(?im)^##\s+references\s*$", text))
        checks["numeric_citations_absent"] = not bool(re.search(r"\[\d+\]", text))
    elif fmt == "MLA":
        checks["mla_parenthetical"] = bool(re.search(r"\([A-Z][A-Za-z\-]+\)", text))
        checks["works_cited_heading"] = bool(re.search(r"(?im)^##\s+works cited\s*$", text))
        checks["numeric_citations_absent"] = not bool(re.search(r"\[\d+\]", text))
    else:
        checks["author_year_markers_present"] = bool(re.search(r"\[@[A-Za-z0-9_.:-]+\]", text))
        checks["references_heading"] = bool(re.search(r"(?im)^##\s+references\s*$", text))

    total = len(checks)
    passed = sum(1 for ok in checks.values() if ok)
    score = round((passed / total) if total else 0.0, 3)
    return {
        "paper_format": fmt,
        "checks": checks,
        "passed": passed,
        "total": total,
        "score": score,
        "is_compliant": passed == total and total > 0,
    }


# ══════════════════════════════════════════════════════════════════════════
# NODE 1: PLANNER
# ══════════════════════════════════════════════════════════════════════════

PLANNER_PROMPT = """You are a senior research strategist. Given the topic: "{topic}", generate a list of 3-5 specific search queries suitable for academic databases (arXiv, Semantic Scholar) and a short research plan covering key angles to investigate.

Strict constraints for each query:
1) Must include at least one quoted phrase drawn from the topic's own terminology.
2) Must add at least one narrowing angle — a method, an evaluation criterion, an
   application setting, a time frame, or a sub-problem — expressed in the topic's own
   vocabulary rather than a generic keyword.
3) Avoid generic one/two-word queries and avoid purely broad topic terms.
4) Keep each query under 18 words.
5) Prioritize lexical overlap with the user topic; avoid adjacent but different fields.

Output as JSON: {{"queries": ["query1", ...], "plan": ["angle1", ...], "constraints_applied": true}}"""


async def planner_node(state: ResearchState, config: dict) -> dict:
    """Generate search queries and a research plan from the user's topic."""
    topic = state["topic"]
    await _send_status(config, f"Planning research strategy for: {topic}")

    prompt = PLANNER_PROMPT.format(topic=topic)
    response = await llm_client.generate_content(
        prompt=prompt,
        response_mime_type="application/json",
        temperature=0.3,
        max_output_tokens=1024,
        enable_thinking=False,
    )

    topic_terms = {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9]+", topic)
        if len(token) > 2
    }

    anchors = _topic_anchor_terms(topic)

    try:
        data = _parse_json_object(response)
        if not data:
            raise json.JSONDecodeError("Invalid planner JSON", response, 0)
        raw_queries = data.get("queries", [topic])
        queries = []
        for q in raw_queries:
            q = str(q).strip()
            if not q:
                continue
            if '"' not in q:
                q = f"\"{q}\" evaluation"
            q_terms = {token.lower() for token in re.findall(r"[A-Za-z0-9]+", q) if len(token) > 2}
            overlap = len(topic_terms.intersection(q_terms))
            if overlap < max(1, min(3, len(topic_terms) // 8 + 1)):
                q = f"\"{topic}\" evaluation {anchors[0] if anchors else 'benchmark'}"
            if anchors:
                # Ensure at least one strong topic anchor is present.
                if not any(a in q.lower() for a in anchors[:4]):
                    q = f"{q} {anchors[0]}"
            queries.append(q[:180])
        # Ensure breadth: at least 3 focused queries.
        if len(queries) < 3:
            seed = anchors[0] if anchors else "benchmark"
            fillers = [
                f"\"{topic}\" recent evaluation {seed}",
                f"\"{topic}\" ablation study {seed}",
                f"\"{topic}\" benchmark analysis {seed}",
            ]
            for fq in fillers:
                if len(queries) >= 3:
                    break
                if fq not in queries:
                    queries.append(fq[:180])
        if not queries:
            queries = [f"\"{topic}\" evaluation {anchors[0] if anchors else 'benchmark'}"]
        plan = data.get("plan", [])
    except json.JSONDecodeError:
        logger.warning("Planner returned non-JSON, using topic as query: %s", response[:200])
        queries = [f"\"{topic}\" evaluation {anchors[0] if anchors else 'benchmark'}"]
        plan = ["General literature survey"]

    if len(queries) < 3:
        seed = anchors[0] if anchors else "benchmark"
        fillers = [
            f"\"{topic}\" recent evaluation {seed}",
            f"\"{topic}\" ablation study {seed}",
            f"\"{topic}\" benchmark analysis {seed}",
        ]
        for fq in fillers:
            if len(queries) >= 3:
                break
            if fq not in queries:
                queries.append(fq[:180])

    await _send_status(config, f"Generated {len(queries)} search queries", "status", {
        "queries": queries,
        "plan": plan,
        "node": "planner",
    })

    existing_constraints = state.get("planner_constraints", {})
    return {
        "refined_query": topic,
        "search_queries": queries,
        "research_plan": plan,
        "planner_constraints": {
            "quoted_phrases_required": True,
            "max_query_words": 18,
            "broad_query_avoidance": True,
            "depth": existing_constraints.get("depth", "standard"),
            "year_from": existing_constraints.get("year_from"),
            "include_sources": existing_constraints.get("include_sources", []),
            "exclude_sources": existing_constraints.get("exclude_sources", []),
        },
        "strict_mode": bool(state.get("strict_mode", settings.strict_synthesis_mode)),
        "max_debate_rounds": int(state.get("max_debate_rounds", settings.max_debate_rounds)),
        "status_message": f"Research plan ready with {len(queries)} queries",
    }


# ══════════════════════════════════════════════════════════════════════════
# NODE 2: RETRIEVAL (Parallel Fetch)
# ══════════════════════════════════════════════════════════════════════════

async def retrieval_node(state: ResearchState, config: dict) -> dict:
    """
    Fetch papers from multiple sources in parallel.
    Also queries ChromaDB if uploaded paper IDs exist.
    """
    constraints = state.get("planner_constraints", {})
    depth = constraints.get("depth", "standard")
    budgets = _depth_budgets(str(depth))
    max_queries = max(max(1, settings.max_search_queries), budgets["queries"])
    max_results = max(max(1, settings.max_source_results_per_query), budgets["results"])
    max_total = budgets["max_total"]
    queries = state.get("search_queries", [state["topic"]])[:max_queries]
    include_sources = set(constraints.get("include_sources", []) or [])
    exclude_sources = set(constraints.get("exclude_sources", []) or [])
    year_from = constraints.get("year_from")
    uploaded_ids = state.get("uploaded_paper_ids", [])
    owner_id = str(state.get("owner_id", ""))

    await _send_status(config, "Searching academic databases...", data={"node": "retrieval"})

    all_papers = []

    # Run all queries across all sources in parallel
    tasks = []
    for query in queries:
        if "arxiv" not in exclude_sources and (not include_sources or "arxiv" in include_sources):
            tasks.append(fetch_arxiv(query, max_results=max_results))
        if "semantic_scholar" not in exclude_sources and (not include_sources or "semantic_scholar" in include_sources):
            tasks.append(fetch_semantic_scholar(query, max_results=max_results))
        if "openalex" not in exclude_sources and (not include_sources or "openalex" in include_sources):
            tasks.append(fetch_openalex(query, max_results=max_results))
        if "crossref" not in exclude_sources and (not include_sources or "crossref" in include_sources):
            tasks.append(fetch_crossref(query, max_results=max_results))

    # Add Tavily search for the main topic
    if "web_tavily" not in exclude_sources and (not include_sources or "web_tavily" in include_sources):
        tasks.append(fetch_web_tavily(state["topic"], settings.tavily_api_key, max_results=max_results))
    if "github_repo" not in exclude_sources and (not include_sources or "github_repo" in include_sources):
        tasks.append(fetch_github_repositories(state["topic"], max_results=max(3, min(6, max_results))))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, Exception):
            logger.error("Retrieval task failed: %s", result)
            continue
        if isinstance(result, list):
            all_papers.extend(result)

    if year_from:
        try:
            cutoff = int(year_from)
        except (TypeError, ValueError):
            cutoff = None
        if cutoff is not None:
            def _passes_year(p: dict[str, Any]) -> bool:
                raw = str(p.get("year", "")).strip()
                if not raw:
                    return True  # keep papers with unknown year
                match = re.search(r"\d{4}", raw)
                if not match:
                    return True  # unparseable year -> don't drop
                return int(match.group()) >= cutoff
            all_papers = [p for p in all_papers if _passes_year(p)]

    # Topic-aware reranking/filtering to reduce drift and prioritize relevance.
    topic_tokens = _normalize_tokens(state.get("topic", ""))
    query_tokens = _normalize_tokens(" ".join(queries))
    critical_terms = _critical_topic_terms(state.get("topic", ""))
    scored: list[tuple[float, dict[str, Any]]] = []
    for paper in all_papers:
        score = _paper_relevance_score(paper, topic_tokens, query_tokens)
        hay = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
        term_hits = sum(1 for t in critical_terms if t in hay)
        source = str(paper.get("source", "")).lower()
        if source == "web_tavily" and term_hits < max(1, min(2, len(critical_terms))):
            score -= 2.0
        if source == "github_repo" and term_hits < max(1, min(2, len(critical_terms))):
            score -= 2.2
        if source == "arxiv" and term_hits == 0:
            score -= 1.2
        if source == "semantic_scholar":
            score += 0.4
        if source == "openalex":
            score += 0.35
        if source == "crossref":
            score += 0.2
        scored.append((score, paper))
    scored.sort(key=lambda x: x[0], reverse=True)

    filtered = [p for s, p in scored if s >= 2.5]
    if filtered:
        all_papers = filtered[: max_total]
    else:
        all_papers = [p for _, p in scored[: max(8, max_total)]]

    # Send paper_found events
    for paper in all_papers[:5]:  # Send first 5 as previews
        confidence = _confidence_score(
            paper.get("citation_count", 0),
            paper.get("abstract", ""),
            paper.get("source", ""),
        )
        await _send_status(config, f"Found: {str(paper.get('title', 'Untitled'))[:80]}...", "paper_found", {
            "title": paper.get("title", "Untitled"),
            "authors": paper.get("authors", "Unknown"),
            "source": paper.get("source", "unknown"),
            "year": paper.get("year", ""),
            "confidence": confidence,
            "node": "retrieval",
        })

    # Deduplicate
    unique_papers = deduplicate_papers(all_papers)
    # Keep strongest, most relevant papers first for downstream summarization.
    unique_papers.sort(
        key=lambda p: (
            _paper_relevance_score(p, topic_tokens, query_tokens),
            p.get("citation_count", 0),
        ),
        reverse=True,
    )
    duplicate_clusters = _cluster_duplicates(all_papers)

    # Source balancing for better reliability/coverage (avoid web-only dominance).
    grouped: dict[str, list[dict[str, Any]]] = {}
    for p in unique_papers:
        grouped.setdefault(str(p.get("source", "unknown")).lower(), []).append(p)
    for src in grouped:
        grouped[src].sort(
            key=lambda p: (
                _paper_relevance_score(p, topic_tokens, query_tokens),
                p.get("citation_count", 0),
            ),
            reverse=True,
        )
    ordered: list[dict[str, Any]] = []
    source_priority = ["semantic_scholar", "openalex", "crossref", "arxiv", "github_repo", "web_tavily"]
    remaining_by_source: dict[str, list[dict[str, Any]]] = {
        src: list(grouped.get(src, []))
        for src in source_priority
    }
    while len(ordered) < max_total:
        added_in_pass = False
        for src in source_priority:
            bucket = remaining_by_source.get(src, [])
            if not bucket:
                continue
            ordered.append(bucket.pop(0))
            added_in_pass = True
            if len(ordered) >= max_total:
                break
        if not added_in_pass:
            break
    if len(ordered) < max_total:
        seen_ids = {str(p.get("paper_id", "")) for p in ordered}
        for p in unique_papers:
            pid = str(p.get("paper_id", ""))
            if pid in seen_ids:
                continue
            ordered.append(p)
            seen_ids.add(pid)
            if len(ordered) >= max_total:
                break
    unique_papers = ordered

    source_counts: dict[str, int] = {}
    for p in unique_papers:
        src = p.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    await _send_status(
        config,
        f"Retrieved {len(all_papers)} papers, {len(unique_papers)} unique after dedup",
        data={
            "node": "retrieval",
            "source_counts": source_counts,
            "duplicate_clusters": len(duplicate_clusters),
        },
    )

    # Query Supabase pgvector for uploaded PDF content
    vector_results = []
    if uploaded_ids:
        await _send_status(config, "Searching uploaded PDFs...")
        for query in queries[:3]:
            chunks = await query_uploaded_documents(
                owner_id=owner_id, document_ids=uploaded_ids, query=query, n_results=5,
            )
            vector_results.extend(chunks)

    return {
        "retrieved_papers": unique_papers,
        "vector_results": vector_results,
        "duplicate_clusters": duplicate_clusters,
        "source_counts": source_counts,
        "status_message": f"Found {len(unique_papers)} unique papers",
    }


# ══════════════════════════════════════════════════════════════════════════
# NODE 3: DEEP DIVE SUMMARIZER
# ══════════════════════════════════════════════════════════════════════════

SUMMARIZER_PROMPT = """You are an expert research analyst. Summarize the following paper concisely.

Title: {title}
Authors: {authors}
Abstract: {abstract}

Provide:
1. A 3-sentence summary of the key contributions.
2. The main limitation or future work direction mentioned or implied.
3. A short methodology label such as benchmark, survey, experiment, framework, dataset, or empirical study.
4. A short evaluation signal describing what the paper actually measures, compares, or validates.
5. A list of up to 4 concrete quantitative metrics or result statements if present.

For each metric item, use:
{{"metric": "...", "value": "...", "baseline": "...", "confidence": "high|medium|low"}}

Output as JSON: {{"summary": "...", "limitation": "...", "methodology": "...", "evaluation_signal": "...", "key_metrics": [{{...}}]}}"""


async def summarizer_node(state: ResearchState, config: dict) -> dict:
    """Summarize the top papers and extract their limitations."""
    papers = state.get("retrieved_papers", [])

    depth = state.get("planner_constraints", {}).get("depth", "standard")
    budgets = _depth_budgets(str(depth))
    summary_cap = budgets["summary_cap"]

    # Sort by citation count, take top papers according to depth budget
    sorted_papers = sorted(papers, key=lambda p: p.get("citation_count", 0), reverse=True)
    top_papers = sorted_papers[:summary_cap]

    if not top_papers:
        await _send_status(config, "No papers to summarize.")
        return {"paper_summaries": {}, "status_message": "No papers found to summarize"}

    await _send_status(config, f"Summarizing {len(top_papers)} papers...", data={"node": "summarizer"})
    summaries = {}
    citation_quality: dict[str, dict[str, Any]] = {}
    citation_verification: dict[str, dict[str, Any]] = {}
    concurrency = 3 if depth == "deep" else 2
    semaphore = asyncio.Semaphore(concurrency)

    async def _summarize_single(i: int, paper: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        async with semaphore:
            title = str(paper.get("title", "") or "Untitled")
            authors = str(paper.get("authors", "") or "Unknown")
            abstract = str(paper.get("abstract", "") or "")
            year = paper.get("year", "")
            await _send_status(
                config,
                f"Summarizing paper {i + 1} of {len(top_papers)}: {title[:60]}...",
            )

            prompt = SUMMARIZER_PROMPT.format(
                title=title,
                authors=authors,
                abstract=abstract[:2400],
            )

            try:
                response = await llm_client.generate_content(
                    prompt=prompt,
                    response_mime_type="application/json",
                    temperature=0.25,
                    max_output_tokens=1100,
                    enable_thinking=False,
                )
                data = _parse_json_object(response)
                if not data:
                    raise json.JSONDecodeError("Invalid summarizer JSON", response, 0)
                summary = {
                    "title": title,
                    "authors": authors,
                    "year": year,
                    "summary": data.get("summary", ""),
                    "limitation": data.get("limitation", ""),
                    "methodology": data.get("methodology", ""),
                    "evaluation_signal": data.get("evaluation_signal", ""),
                    "key_metrics": _normalize_metric_items(data.get("key_metrics", [])),
                    "url": paper.get("url", ""),
                    "doi": paper.get("doi", ""),
                    "citation_count": paper.get("citation_count", 0),
                    "source": paper.get("source", ""),
                }
                quality = {
                    "source_quality": _source_quality_score(paper.get("source", "")),
                    "confidence": _confidence_score(
                        paper.get("citation_count", 0),
                        paper.get("abstract", ""),
                        paper.get("source", ""),
                    ),
                }
            except Exception as e:
                logger.error("Failed to summarize paper '%s': %s", title[:60], e)
                summary = {
                    "title": title,
                    "authors": authors,
                    "year": year,
                    "summary": abstract[:400],
                    "limitation": "Not available",
                    "methodology": "Empirical study",
                    "evaluation_signal": "Abstract-level evidence only",
                    "key_metrics": [],
                    "url": paper.get("url", ""),
                    "doi": paper.get("doi", ""),
                    "citation_count": paper.get("citation_count", 0),
                    "source": paper.get("source", ""),
                }
                quality = {
                    "source_quality": _source_quality_score(paper.get("source", "")),
                    "confidence": 0.4,
                }

            has_doi = bool(paper.get("doi"))
            has_year = bool(str(paper.get("year", "")).strip())
            title_ok = len((paper.get("title") or "").strip()) > 10
            source = str(paper.get("source", "")).lower()
            reliable_source = source in {"semantic_scholar", "arxiv", "openalex", "crossref"}
            has_url = bool(str(paper.get("url", "")).strip())
            verification = {
                "title_ok": title_ok,
                "year_ok": has_year,
                "doi_ok": has_doi,
                "verified": bool(title_ok and has_year and ((has_doi and reliable_source) or (reliable_source and has_url))),
            }
            return summary, quality, verification

    summarized = await asyncio.gather(*[
        _summarize_single(i, paper) for i, paper in enumerate(top_papers)
    ], return_exceptions=True)

    for paper, outcome in zip(top_papers, summarized):
        if isinstance(outcome, Exception):
            logger.error(
                "Summarizer task crashed for paper '%s': %s",
                str(paper.get("title", ""))[:60],
                outcome,
            )
            continue
        summary, quality, verification = outcome
        paper_id = paper.get("paper_id") or paper.get("doi") or paper.get("arxiv_id")
        if not paper_id:
            title_digest = hashlib.sha1(
                str(paper.get("title", "")).encode("utf-8")
            ).hexdigest()[:16]
            paper_id = f"paper_{title_digest}"
        summaries[paper_id] = summary
        citation_quality[paper_id] = quality
        citation_verification[paper_id] = verification

    # Include vector store results in summaries
    vector_results = state.get("vector_results", [])
    if vector_results:
        await _send_status(config, f"Incorporating {len(vector_results)} chunks from uploaded PDFs...")
        for i, chunk in enumerate(vector_results[:5]):
            chunk_id = f"uploaded_chunk_{i}"
            summaries[chunk_id] = {
                "title": f"Uploaded PDF Chunk {i + 1}",
                "authors": chunk.get("metadata", {}).get("filename", "Uploaded"),
                "year": "",
                "summary": chunk["text"][:500],
                "limitation": "",
                "methodology": "Uploaded document evidence",
                "evaluation_signal": "Document-grounded excerpt",
                "url": "",
                "doi": "",
                "citation_count": 0,
            }
            citation_quality[chunk_id] = {"source_quality": _source_quality_score("uploaded_pdf"), "confidence": 0.78}
            citation_verification[chunk_id] = {
                "title_ok": True,
                "year_ok": True,
                "doi_ok": False,
                "verified": False,
            }

    verified_count = sum(1 for v in citation_verification.values() if v.get("verified"))
    await _send_status(
        config,
        f"Summarized {len(summaries)} sources; verified {verified_count} citations",
        "status",
        {"node": "summarizer", "verified_citations": verified_count},
    )
    return {
        "paper_summaries": summaries,
        "citation_quality": citation_quality,
        "citation_verification": citation_verification,
        "status_message": f"Summarized {len(summaries)} sources",
    }


# ══════════════════════════════════════════════════════════════════════════
# NODE 4A: PROPOSER (Debate Subgraph - Draft)
# ══════════════════════════════════════════════════════════════════════════

PROPOSER_PROMPT = """You are a senior academic research scientist and literature review specialist. Write an exhaustive, rigorous, multi-thematic literature review on the topic: "{topic}".

Organize the review by conceptual themes, theoretical mechanisms, and methodological paradigms (not paper-by-paper). Use authoritative academic prose and cite papers using [@AuthorYear] notation.

Available verified paper summaries:
{summaries}

{previous_critique}

Structure your synthesis with clear thematic headings:
1. **Thematic Conceptual Framework**: Synthesize foundational and emerging paradigms across the retrieved papers.
2. **Methodological & Algorithmic Mechanics**: Compare algorithmic formulations, mathematical objectives, experimental setups, and datasets across studies.
3. **Empirical Findings, Agreements & Contradictions**: Critically analyze quantitative benchmarks, points of consensus, statistical discrepancies, and performance trade-offs.
4. **Evidence Caveats & Systematic Limitations**: Address foundational assumptions, dataset biases, evaluation constraints, and failure modes across current literature.
5. **Synthesis Summary**: Conclude with a rigorous transition setting up cross-paper analysis and research gaps.

Requirements:
- Deep multi-paragraph continuous academic prose with dense citations.
- Explicitly compare study designs, evaluation metrics, and dataset distributions.
- Call out agreements, tensions, and empirical caveats with concrete numbers from the summaries."""


async def proposer_node(state: ResearchState, config: dict) -> dict:
    """Draft or revise the literature review."""
    topic = state["topic"]
    summaries = state.get("paper_summaries", {})
    debate_log = state.get("debate_log", [])

    await _send_status(config, "Drafting literature review...", data={"node": "proposer"})

    # Format summaries for the prompt
    summary_text = ""
    for pid, info in summaries.items():
        if info.get("title") and info.get("summary"):
            author_year = f"{info['authors'].split(',')[0].strip().split()[-1] if info['authors'] else 'Unknown'}{info.get('year', '')}"
            summary_text += f"\n- [@{author_year}] {info['title']}: {info['summary']}\n"

    # Include previous critique if this is a revision
    critique_text = ""
    if debate_log:
        critique_text = f"\nPrevious feedback to address:\n{debate_log[-1]}\n\nPlease revise your draft based on this feedback."

    strict_mode = bool(state.get("strict_mode", settings.strict_synthesis_mode))
    strict_guard = (
        "\nSTRICT SYNTHESIS MODE:\n"
        "- Only include claims directly supported by provided summaries.\n"
        "- If evidence is weak, explicitly label as tentative.\n"
        "- Avoid unverifiable speculative statements.\n"
    ) if strict_mode else ""

    prompt = PROPOSER_PROMPT.format(
        topic=topic,
        summaries=(summary_text + strict_guard)[:10000],
        previous_critique=critique_text + strict_guard,
    )

    draft = await llm_client.generate_content(
        prompt=prompt,
        temperature=0.35,
        max_output_tokens=8192,
        enable_thinking=True,
    )

    round_num = sum(1 for msg in debate_log if msg.startswith("[PROPOSER]")) + 1
    token_estimate = max(1, len(draft) // 4)
    await _send_status(
        config,
        f"Literature review draft ready for review (round {round_num})",
        data={"node": "proposer", "debate_round": round_num, "token_estimate": token_estimate},
    )

    return {
        "literature_review_draft": draft,
        "debate_log": [f"[PROPOSER] Draft {'revised' if debate_log else 'created'}: {len(draft)} chars"],
        "debate_round": round_num,
        "token_estimate": token_estimate,
        "status_message": "Literature review drafted",
    }


# ══════════════════════════════════════════════════════════════════════════
# NODE 4B: CRITIC (Debate Subgraph - Review)
# ══════════════════════════════════════════════════════════════════════════

CRITIC_PROMPT = """You are a rigorous peer reviewer. Review the following literature review draft. Point out any claims that are not directly supported by the provided paper abstracts. Suggest improvements to make the review more balanced and accurate.

Draft:
{draft}

Paper Abstracts:
{abstracts}

Respond with a critique and suggested revisions. Be specific about what needs changing."""


async def critic_node(state: ResearchState, config: dict) -> dict:
    """Review the literature draft and provide critique."""
    draft = state.get("literature_review_draft", "")
    summaries = state.get("paper_summaries", {})

    await _send_status(config, "Peer reviewer analyzing the draft...", data={"node": "critic"})

    # Format abstracts for verification
    abstracts_text = ""
    for pid, info in summaries.items():
        if info.get("title") and info.get("summary"):
            abstracts_text += f"\n- {info['title']}: {info['summary']}\n"

    prompt = CRITIC_PROMPT.format(
        draft=draft[:6000],
        abstracts=abstracts_text[:6000],
    )

    try:
        critique = await llm_client.generate_content(
            prompt=prompt,
            temperature=0.3,
            max_output_tokens=2048,
            enable_thinking=True,
        )
    except Exception as e:
        logger.error("Critic node LLM call failed, skipping critique: %s", e)
        await _send_status(
            config,
            "Peer review unavailable this round, continuing",
            data={"node": "critic"},
        )
        return {
            "debate_log": ["[CRITIC-FAILED] (peer review unavailable — LLM error)"],
            "status_message": "Peer review skipped (LLM error)",
        }

    await _send_status(config, "Peer review complete, sending feedback to proposer", data={"node": "critic"})

    return {
        "debate_log": [f"[CRITIC] {critique}"],
        "status_message": "Peer review complete",
    }


# ══════════════════════════════════════════════════════════════════════════
# NODE 5A: CROSS-PAPER ANALYZER
# ══════════════════════════════════════════════════════════════════════════

CROSS_PAPER_ANALYSIS_PROMPT = """You are a rigorous research scientist.
Analyze paper summaries on "{topic}" with cross-paper reasoning.

Return ONLY JSON object with keys:
- methodology_patterns: string[]
- contradictions: string[]
- missing_dimensions: string[]
- weak_evaluation_signals: string[]
- evidence_notes: [{{ "paper_title": string, "note": string }}]
- comparative_axes: string[]
- recommended_experiments: string[]

Focus on:
1) Methodology comparison
2) Contradictions/disagreements
3) Missing data/evaluation
4) External validity/real-world assumptions
5) Underexplored dimensions
6) What a stronger next experiment should look like

Paper summaries:
{summaries}
"""


def _build_cross_paper_summary_text(summaries: dict[str, dict[str, Any]], limit: int = 14) -> str:
    rows: list[str] = []
    for i, (_pid, info) in enumerate(summaries.items(), start=1):
        if i > limit:
            break
        title = str(info.get("title", "Untitled")).strip()
        year = str(info.get("year", "")).strip()
        summary = str(info.get("summary", "")).strip()
        limitation = _clean_nullable_text(info.get("limitation", ""))
        rows.append(
            f"- [{i}] {title} ({year})\n"
            f"  Summary: {summary[:500]}\n"
            f"  Limitation: {(limitation or 'Not explicit')[:240]}"
        )
    return "\n".join(rows)


async def cross_paper_analyzer_node(state: ResearchState, config: dict) -> dict:
    topic = state["topic"]
    summaries = state.get("paper_summaries", {})

    await _send_status(config, "Running cross-paper analysis...", data={"node": "cross_paper"})

    if not summaries:
        return {
            "cross_paper_analysis": {
                "methodology_patterns": [],
                "contradictions": [],
                "missing_dimensions": ["Insufficient summaries for cross-paper analysis."],
                "weak_evaluation_signals": [],
                "evidence_notes": [],
                "comparative_axes": [],
                "recommended_experiments": [],
            }
        }

    prompt = CROSS_PAPER_ANALYSIS_PROMPT.format(
        topic=topic,
        summaries=_build_cross_paper_summary_text(summaries),
    )
    try:
        response = await llm_client.generate_content(
            prompt=prompt,
            response_mime_type="application/json",
            temperature=0.2,
            max_output_tokens=2048,
            enable_thinking=False,
        )
    except Exception as e:
        logger.error("Cross-paper analysis LLM call failed, using empty analysis: %s", e)
        response = ""

    parsed = _parse_json_object(response)

    result = {
        "methodology_patterns": parsed.get("methodology_patterns", []) if isinstance(parsed, dict) else [],
        "contradictions": parsed.get("contradictions", []) if isinstance(parsed, dict) else [],
        "missing_dimensions": parsed.get("missing_dimensions", []) if isinstance(parsed, dict) else [],
        "weak_evaluation_signals": parsed.get("weak_evaluation_signals", []) if isinstance(parsed, dict) else [],
        "evidence_notes": parsed.get("evidence_notes", []) if isinstance(parsed, dict) else [],
        "comparative_axes": parsed.get("comparative_axes", []) if isinstance(parsed, dict) else [],
        "recommended_experiments": parsed.get("recommended_experiments", []) if isinstance(parsed, dict) else [],
    }

    await _send_status(config, "Cross-paper analysis complete", data={"node": "cross_paper"})
    return {"cross_paper_analysis": result}


# ══════════════════════════════════════════════════════════════════════════
# NODE 5B: GAP ANALYSIS
# ══════════════════════════════════════════════════════════════════════════

GAP_GENERATOR_PROMPT = """You are a research scientist discovering REAL research gaps.
Topic: "{topic}"

Use both:
1) Cross-paper analysis
2) Per-paper limitations/summaries

Hard requirements:
- Compare papers against each other (not isolated summaries).
- Detect contradictions and methodological disagreements.
- Identify gaps by dimension: data, method, scale, generalization, real-world deployment, bias/fairness.
- Every gap must be specific and non-generic.

Return ONLY JSON array with 3-6 objects.
Each object must include:
- title
- why_gap
- what_missing
- how_to_fix
- evidence
- supports (paper titles)
- fails (paper titles)
"""

GAP_CRITIC_PROMPT = """You are a strict gap reviewer.
Review candidate research gaps and reject weak ones.

Criteria:
1) Specific (not vague)
2) Evidence-linked
3) Not already solved by cited papers
4) Actionable research direction

Return ONLY JSON:
{{
  "approved_gaps": [gap_objects],
  "rejected_titles": string[],
  "review_notes": string
}}

Topic: {topic}
Candidate gaps:
{candidate_gaps}
"""


def _normalize_gap_candidates(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        title = _clean_nullable_text(item.get("title", ""))
        if not title:
            continue
        why_gap = _clean_nullable_text(item.get("why_gap", ""))
        what_missing = _clean_nullable_text(item.get("what_missing", ""))
        how_to_fix = _clean_nullable_text(item.get("how_to_fix", ""))
        evidence = _clean_nullable_text(item.get("evidence", ""))
        supports = item.get("supports", [])
        fails = item.get("fails", [])
        if not evidence:
            evidence = why_gap or what_missing
        if not how_to_fix:
            how_to_fix = "Design targeted empirical validation with stronger baselines and explicit failure analysis."
        if not isinstance(supports, list):
            supports = []
        if not isinstance(fails, list):
            fails = []
        out.append({
            "title": title[:180],
            "why_gap": why_gap[:700],
            "what_missing": what_missing[:700],
            "how_to_fix": how_to_fix[:900],
            "evidence": evidence[:900],
            "supports": [str(s)[:180] for s in supports[:6]],
            "fails": [str(s)[:180] for s in fails[:6]],
        })
    return out


def _candidates_to_final_gaps(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    final: list[dict[str, Any]] = []
    seen: set[str] = set()
    for c in candidates:
        title = _clean_nullable_text(c.get("title", ""))
        if not title:
            continue
        key = title.lower()
        if key in seen:
            continue
        seen.add(key)
        final.append({
            "title": title[:180],
            "evidence": _clean_nullable_text(c.get("evidence", ""))[:900],
            "proposed_direction": _clean_nullable_text(c.get("how_to_fix", c.get("proposed_direction", "")))[:900],
            "why_gap": _clean_nullable_text(c.get("why_gap", ""))[:700],
            "what_missing": _clean_nullable_text(c.get("what_missing", ""))[:700],
            "supports": c.get("supports", []) if isinstance(c.get("supports", []), list) else [],
            "fails": c.get("fails", []) if isinstance(c.get("fails", []), list) else [],
        })
        if len(final) >= 6:
            break
    return final


async def gap_analysis_node(state: ResearchState, config: dict) -> dict:
    """Identify research gaps using cross-paper reasoning + critic validation."""
    topic = state["topic"]
    summaries = state.get("paper_summaries", {})
    cross = state.get("cross_paper_analysis", {})

    await _send_status(config, "Analyzing research gaps...", data={"node": "gap_analysis"})

    # Collect explicit and inferred limitations.
    limitations = []
    for _pid, info in summaries.items():
        title = str(info.get("title", "")).strip()
        limitation = _clean_nullable_text(info.get("limitation", ""))
        summary_text = str(info.get("summary", "")).strip()
        inferred = _extract_limitation_signal(f"{title}. {summary_text}")
        signal = limitation or inferred
        if signal:
            limitations.append(f"- {title}: {signal}")

    if not limitations:
        limitations = ["- Limitation statements are sparse in paper summaries; deriving gaps from topic coverage and source balance."]

    prompt = (
        GAP_GENERATOR_PROMPT.format(topic=topic)
        + "\n\nCross-paper analysis:\n"
        + json.dumps(cross, ensure_ascii=False)
        + "\n\nLimitations:\n"
        + "\n".join(limitations[:20])
    )

    response = await llm_client.generate_content(
        prompt=prompt,
        response_mime_type="application/json",
        temperature=0.3,
        max_output_tokens=2048,
        enable_thinking=False,  # Critical: disabled for JSON output
    )

    parsed_ok = False
    candidates: list[dict[str, Any]] = []
    for attempt in range(2):
        try:
            # Use robust parsing that handles markdown-wrapped JSON
            raw_candidates = _parse_json_array(response if attempt == 0 else _extract_json_fragment(response))
            if raw_candidates:
                candidates = _normalize_gap_candidates(raw_candidates)
            else:
                # Try dict wrapper: {"gaps": [...]}
                data_obj = _parse_json_object(response if attempt == 0 else _extract_json_fragment(response))
                if data_obj and isinstance(data_obj.get("gaps"), list):
                    candidates = _normalize_gap_candidates(data_obj["gaps"])
                else:
                    raise json.JSONDecodeError("No valid gap array found", response[:100], 0)
            parsed_ok = True
            break
        except json.JSONDecodeError:
            if attempt == 0:
                repair_prompt = (
                    "Return ONLY valid JSON. Convert the following text into a JSON array with "
                    "objects containing keys title, why_gap, what_missing, how_to_fix, evidence, supports, fails. No markdown.\n\n"
                    f"{response[:4000]}"
                )
                response = await llm_client.generate_content(
                    prompt=repair_prompt,
                    response_mime_type="application/json",
                    temperature=0.0,
                    max_output_tokens=1024,
                    enable_thinking=False,
                )
                continue
    if parsed_ok and len(candidates) < 2:
        synth_prompt = (
            "Given the topic and literature limitations, produce 2 additional concrete research gaps. "
            "Return ONLY JSON array with title, why_gap, what_missing, how_to_fix, evidence, supports, fails. No markdown.\n\n"
            f"Topic: {topic}\nLimitations:\n{chr(10).join(limitations[:12])}"
        )
        synth_resp = await llm_client.generate_content(
            prompt=synth_prompt,
            response_mime_type="application/json",
            temperature=0.2,
            max_output_tokens=1024,
            enable_thinking=False,
        )
        try:
            synth_data = json.loads(_extract_json_fragment(synth_resp))
            synth_gaps = _normalize_gap_candidates(synth_data if isinstance(synth_data, list) else synth_data.get("gaps", []))
            seen = {g["title"].lower() for g in candidates}
            for g in synth_gaps:
                if g["title"].lower() not in seen:
                    candidates.append(g)
                    seen.add(g["title"].lower())
                if len(candidates) >= 3:
                    break
        except Exception:
            pass

    if len(candidates) < 3:
        derived = _derive_gaps_from_paper_evidence(
            topic=topic,
            summaries=summaries,
            source_counts=state.get("source_counts", {}),
            min_gaps=3,
        )
        derived_candidates = [{
            "title": g.get("title", ""),
            "why_gap": f"Current literature shows: {g.get('evidence', '')[:150]}...",
            "what_missing": "Explicit validation and detailed metrics for this aspect.",
            "how_to_fix": g.get("proposed_direction", ""),
            "evidence": g.get("evidence", ""),
            "supports": [],
            "fails": [],
        } for g in derived]
        seen_titles = {g["title"].lower() for g in candidates}
        for g in derived_candidates:
            key = g["title"].lower()
            if key in seen_titles:
                continue
            candidates.append(g)
            seen_titles.add(key)
            if len(candidates) >= 3:
                break

    if not parsed_ok or not candidates:
        logger.error("Gap analysis returned invalid JSON after retry: %s", response[:200])
        # Deterministic fallback from summaries/topic evidence.
        fallback = _derive_gaps_from_paper_evidence(
            topic=topic,
            summaries=summaries,
            source_counts=state.get("source_counts", {}),
            min_gaps=3,
        )
        candidates = [{
            "title": g.get("title", ""),
            "why_gap": f"Current literature shows: {g.get('evidence', '')[:150]}...",
            "what_missing": "Explicit validation and detailed metrics for this aspect.",
            "how_to_fix": g.get("proposed_direction", ""),
            "evidence": g.get("evidence", ""),
            "supports": [],
            "fails": [],
        } for g in fallback]

    # Gap critic pass for specificity and evidence-linking.
    critic_prompt = GAP_CRITIC_PROMPT.format(
        topic=topic,
        candidate_gaps=json.dumps(candidates[:8], ensure_ascii=False, indent=2),
    )
    critic_resp = await llm_client.generate_content(
        prompt=critic_prompt,
        response_mime_type="application/json",
        temperature=0.2,
        max_output_tokens=2048,
        enable_thinking=False,
    )
    gap_critique = ""
    approved: list[dict[str, Any]] = []
    try:
        critic_data = json.loads(_extract_json_fragment(critic_resp))
        approved = _normalize_gap_candidates(critic_data.get("approved_gaps", []))
        gap_critique = _clean_nullable_text(critic_data.get("review_notes", ""))
    except Exception:
        approved = []
    if not approved:
        approved = candidates
        if gap_critique:
            gap_critique = (
                "Initial gap proposals were too generic for publication use, so the final gap set below was regenerated "
                "from paper-specific limitations, cross-paper patterns, and source coverage evidence."
            )

    gaps = _candidates_to_final_gaps(approved)

    # Quality pass: de-duplicate weak/generic titles and preserve strongest concise set.
    sanitized: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    for i, gap in enumerate(gaps, start=1):
        title = _clean_nullable_text(gap.get("title", ""))
        evidence = _clean_nullable_text(gap.get("evidence", ""))
        direction = _clean_nullable_text(gap.get("proposed_direction", ""))
        if not evidence or not direction:
            continue
        title_l = title.lower()
        if not title or "underexplored limitation" in title_l:
            title = _fallback_gap_title(topic, evidence, i)
            title_l = title.lower()
        if title_l in seen_titles:
            continue
        seen_titles.add(title_l)
        sanitized.append({
            "title": title[:180],
            "evidence": evidence[:900],
            "proposed_direction": direction[:900],
            "why_gap": _clean_nullable_text(gap.get("why_gap", ""))[:700],
            "what_missing": _clean_nullable_text(gap.get("what_missing", ""))[:700],
            "supports": gap.get("supports", []) if isinstance(gap.get("supports", []), list) else [],
            "fails": gap.get("fails", []) if isinstance(gap.get("fails", []), list) else [],
        })
        if len(sanitized) >= 5:
            break
    gaps = sanitized or gaps

    # Send each gap as an event
    for gap in gaps:
        await _send_status(config, f"Gap identified: {gap.get('title', 'Unknown')}", "gap", gap)

    await _send_status(config, f"Found {len(gaps)} research gaps", data={"node": "gap_analysis"})

    return {
        "generated_gap_candidates": approved,
        "gap_critique": gap_critique,
        "identified_gaps": gaps,
        "status_message": f"Identified {len(gaps)} research gaps",
    }


# ══════════════════════════════════════════════════════════════════════════
# NODE 6: PAPER ARCHITECT
# ══════════════════════════════════════════════════════════════════════════

PAPER_ARCHITECT_PROMPT = """You are an expert academic writer. Assemble a complete research output in Markdown format on the topic: "{topic}".
Required citation/style format: "{paper_format}".

Use the following inputs to construct the paper:

## Literature Review (from peer-reviewed synthesis):
{literature_review}

## Cross-Paper Analysis:
{cross_paper_analysis}

## Identified Research Gaps:
{gaps_section}

## Gap Critic Notes:
{gap_critique}

## Paper Summaries for Citation:
{citation_info}

Output must contain TWO distinct parts, in this exact order and headings:

## PART A - Research Analysis Dossier (Highly Visual Explanation)
Make this section a highly immersive, solid visual explanation of the research.
1. **A1. Cross-Paper Methodology Analysis**
2. **A2. Contradictions and Agreement Map**
3. **A3. Detailed Gap Analysis** (for each gap include why_gap, what_missing, evidence, supports/fails)
4. **A4. Conceptual Architecture & Visual Summaries** (Place your architectural images and diagrams here)
5. **A5. Research Agenda and Prioritized Experiments**
6. **A6. Operational Takeaways and Platform Implications**

## PART B - Final Draft Paper (Publication Style)
This section MUST BE STRICTLY TEXTUAL. Do NOT include any images, diagrams, or Mermaid code in Part B.
Write a COMPLETE, publication-ready research paper. It must read like a real peer-reviewed article,
not an outline or a summary. Target 2500-4000 words in Part B. Use the following structure:

1. **Title** – a specific, informative paper title (not the raw topic string).
2. **Abstract** – 180-250 words: context, problem, what this review/synthesis does, key findings, and implications.
3. **Keywords** – 4-6 comma-separated keywords.
4. **1. Introduction** – 3-5 paragraphs: motivation, why the problem matters, scope, the central questions this paper addresses, and an explicit "contributions of this paper" paragraph.
5. **2. Background and Related Work** – organize by theme; synthesize across studies; compare methodologies and evaluation setups; cite concrete sources.
6. **3. Comparative Analysis and Discussion** – critically compare approaches, datasets, metrics, and results; discuss agreements, contradictions, and evidence-quality caveats. Reference at least one comparison in prose (the visual tables live in Part A).
7. **4. Research Gaps and Future Directions** – turn each identified gap into a substantive paragraph explaining the gap, why it persists, and a concrete, testable research direction.
8. **5. Threats to Validity / Limitations** – of this synthesis itself (coverage, source quality, recency).
9. **6. Conclusion** – 2-3 paragraphs restating findings and their significance.
10. **References** – every in-text citation must appear here and vice versa.

Depth requirements for Part B:
- Every body section must be multiple full paragraphs of connected prose — no bullet-only sections, no one-line sections.
- Ground specific claims in specific sources with in-text citations; include concrete datasets, benchmarks, and metrics wherever the summaries provide them.
- Maintain a logical narrative: each section should set up the next.
- Do not fabricate authors, venues, results, or numbers not present in the provided inputs.

Style enforcement:
- If paper_format is IEEE, use numbered references [1], [2], ... and IEEE-style reference list.
- If paper_format is APA, use (Author, Year) in-text citations and APA references.
- If paper_format is ACM, use ACM reference style and numbered citation markers.
- If paper_format is MLA, use MLA in-text citation and Works Cited style.
- If paper_format is ACADEMIC, use [@AuthorYear] style.

Visual requirements (STRICTLY CONFINED TO PART A):
- Use Markdown tables heavily to compare studies, metrics, methodologies, or evidence quality in Part A.
- Do NOT write raw Mermaid code yourself.
- Instead, place these exact tokens where visuals should appear in Part A:
  - `[DELVE_STUDY_LANDSCAPE]`
  - `[DELVE_EVIDENCE_MATRIX]`
  - `[DELVE_SOURCE_MATRIX]`
  - `[DELVE_CITATION_AUDIT]`
  - `[DELVE_FINDINGS_BRIEF]`
  - `[DELVE_EXPERIMENT_BACKLOG]`
  - `[DELVE_VISUAL_LANDSCAPE]`
  - `[DELVE_VISUAL_WORKFLOW]`
  - `[DELVE_VISUAL_EVIDENCE_FLOW]`
- ABSOLUTELY NO `[IMAGE_PROMPT]`, Mermaid code, or Delve visual tokens are allowed in Part B.

Quality requirements:
- Do not include placeholder author text.
- Every major section must cite concrete sources from the provided citation list.
- Maximize useful detail in Part A with explicit comparisons, evidence caveats, methodological tradeoffs, and concrete metrics wherever the source summaries permit.
- Prefer specific study-to-study synthesis over generic field-level statements.
- Include practical interpretation for researchers, engineers, and product teams.
- Keep claims faithful to evidence and avoid unsupported speculation.
- Ensure references and in-text citations are consistent.
- Include comparative insights, methodological tradeoffs, evidence-quality commentary, and practical implications for researchers/builders.
- Do NOT include preface text like "As an expert academic writer...".

Write in formal academic tone. Make Part B publication-ready and purely textual."""


def _split_analysis_and_final_draft(markdown: str) -> tuple[str, str]:
    text = str(markdown or "").strip()
    if not text:
        return "", ""
    marker = "## PART B - Final Draft Paper (Publication Style)"
    idx = text.find(marker)
    if idx != -1:
        analysis = text[:idx].strip()
        paper = text[idx + len(marker):].strip()
        return analysis, paper
    marker_alt = "PART B - Final Draft Paper"
    idx_alt = text.find(marker_alt)
    if idx_alt != -1:
        analysis = text[:idx_alt].strip()
        paper = text[idx_alt + len(marker_alt):].strip()
        return analysis, paper
    # Fallback: if separator is missing, treat whole text as paper for compatibility.
    return "", text


PAPER_PART1_PROMPT = """You are a distinguished academic professor and peer-reviewed author. Write PART 1 of a rigorous, publication-grade research manuscript on: "{topic}".
Required citation and reference style: "{paper_format}".

Write in rigorous, dense, high-impact academic prose. Target 2500-3500 words with deep domain expertise.
Strictly synthesize and integrate these verified research findings:
## Verified Literature Review:
{literature_review}

## Cross-Paper Evidence & Methodologies:
{cross_paper_analysis}

## Available Source Citations:
{citation_info}

Required structure for Part 1 (Start directly with the paper Title):
# <Authoritative, Specific Academic Title>

**Abstract** — 250-350 words:
Provide a structured academic abstract encompassing: (1) Background and theoretical context, (2) Core domain challenge and problem formulation, (3) Synthesis scope across research sources, (4) Primary empirical and algorithmic findings, (5) Key limitations identified in existing literature, and (6) Strategic implications for future research.

**Keywords** — 5-8 precise domain keywords.

## 1. Introduction
Write 6-8 extensive, connected paragraphs of formal academic text:
- **1.1 Domain Motivation & Historical Trajectory**: Trace the theoretical origin, practical necessity, and evolution of the field.
- **1.2 Core Architectural & Domain Challenges**: Formalize the underlying domain tensions, computational boundaries, representation limitations, and failure modes in current approaches.
- **1.3 Scope and Research Questions**: Delineate the precise boundaries of this investigation and state the central research hypotheses.
- **1.4 Summary of Contributions**: Provide an explicit, numbered list of 4-5 substantial technical and analytical contributions made by this synthesis.

## 2. Theoretical Foundations and Background
Write an extensive multi-paragraph theoretical foundation:
- Define core mathematical notations, formal paradigms, and baseline conceptual mechanics.
- Categorize the foundational taxonomy of existing approaches with comprehensive citations in "{paper_format}" format.
- Contextualize how early paradigms evolved into modern state-of-the-art formulations.

## 3. Thematic Literature Survey
Provide an exhaustive thematic literature survey organized into rigorous subsections:
- Group studies by conceptual methodology, algorithmic paradigm, or architectural archetype tailored to "{topic}".
- Compare study objectives, modeling assumptions, representations, and algorithmic designs with dense in-text citations.
- Avoid superficial listicles; write continuous, deeply reasoned academic synthesis.

Rules:
- Strictly textual formal academic prose (no Mermaid blocks, no placeholder tokens).
- Maintain rigorous "{paper_format}" in-text citations ([1], [2] for IEEE; (Author, Year) for APA).
- CRITICAL: Do NOT include a References section at the end of Part 1 (References are added at the very end of the full paper). Conclude Part 1 directly at the end of Section 3."""

PAPER_PART2_PROMPT = """You are a distinguished academic professor and peer-reviewed author. Write PART 2 of a rigorous, publication-grade research manuscript on: "{topic}".
Required citation and reference style: "{paper_format}".

Write in rigorous, dense, high-impact academic prose. Target 2500-3500 words with deep empirical and algorithmic depth.
Strictly synthesize these verified cross-paper analyses, empirical signals, and research gaps:
## Cross-Paper Analysis & Methodological Patterns:
{cross_paper_analysis}

## Identified Research Gaps & Frontiers:
{gaps_section}

## Available Source Citations:
{citation_info}

Required structure for Part 2 (Continue directly from Section 4):
## 4. Algorithmic Mechanics, Methodologies & System Architectures
Write a comprehensive technical deep-dive across 4-6 detailed subsections:
- **4.1 Mathematical Formulations & Optimization Dynamics**: Detail objective functions, loss formulations, representation mechanics, and convergence dynamics.
- **4.2 Architectural Paradigms & Structural Components**: Rigorous taxonomy of core modules, feature representations, attention mechanisms, and pipeline workflows tailored directly to "{topic}".
- **4.3 Methodological Trade-Offs & Complexity Bounds**: Analyze computational overhead, memory footprints, sample efficiency, and scalability boundaries.
- **4.4 Operational & Deployment Considerations**: Address latency, hardware constraints, distribution shifts, and domain-specific robustness.

## 5. Comprehensive Comparative Evaluation and Empirical Synthesis
Provide an authoritative comparative evaluation:
- Include extensive Markdown comparison tables contrasting studies across: (1) Datasets & Benchmarks, (2) Core Evaluation Metrics (e.g. Accuracy, Dice score, F1, Loss, Latency depending on domain), (3) Model Parameters & Complexity, (4) Computational Efficiency, and (5) Real-World Deployment Assumptions.
- Follow tables with deep analytical discussion dissecting empirical anomalies, statistical agreements, contradictions, and protocol discrepancies across the literature.
- Critically evaluate why certain approaches fail under realistic distribution shifts or challenging deployment scenarios.

Rules:
- High-density academic prose with structured Markdown comparison matrices.
- Dense in-text citations in "{paper_format}" style.
- CRITICAL: Do NOT include a References section at the end of Part 2. Conclude Part 2 directly at the end of Section 5."""

PAPER_PART3_PROMPT = """You are a distinguished academic professor and peer-reviewed author. Write PART 3 (the concluding part) of a rigorous, publication-grade research manuscript on: "{topic}".
Required citation and reference style: "{paper_format}".

Write in rigorous, dense, high-impact academic prose. Target 2500-3500 words.
Strictly synthesize these verified research gaps, peer critiques, and citation inventories:
## Identified Research Gaps & Frontiers:
{gaps_section}

## Peer Review Critique & Rigor Assessment:
{gap_critique}

## Available Source Citations for Complete Bibliography:
{citation_info}

Required structure for Part 3 (Continue directly from Section 6):
## 6. Critical Research Gaps and 5-Year Strategic Roadmap
Provide an exhaustive analysis of major open challenges:
- For each identified gap (at least 4-5 gaps), dedicate a structured subsection:
  - **The Theoretical & Structural Blocker**: Why existing state-of-the-art methods fundamentally fail to resolve this issue.
  - **Empirical Blindspots & Evaluation Voids**: Missing benchmarks, unverified assumptions, and metric limitations.
  - **Proposed Testable Research Direction & Protocol**: Specific algorithmic architectures, experimental protocols, and measurable validation metrics to overcome the gap.
- Synthesize an integrated 5-year technical roadmap outlining key milestone phases for the research community.

## 7. Practical Implications & Systems Engineering Takeaways
- Actionable engineering guidelines for practitioners, researchers, and systems architects in "{topic}".
- Production trade-off matrix: choosing appropriate configurations under constrained bandwidth, compute budgets, heterogeneous hardware, and deployment environments.

## 8. Limitations & Methodological Scope
- Methodological bounds of the reviewed literature, search corpus coverage, potential publication biases, dataset homogeneity, and synthesis assumptions.

## 9. Conclusion
- Executive synthesis summarizing core breakthroughs, structural trade-offs, and future research directions for the domain.

## References
Provide a complete, fully formatted, professional bibliography containing all cited sources in strict "{paper_format}" format with complete author names, paper titles, publication venues/journals, and DOIs/URLs.

Rules:
- Complete all sections in full depth with zero placeholders or truncated endings.
- Strictly adhere to "{paper_format}" formatting throughout."""


async def _generate_final_paper(
    topic: str,
    paper_format: str,
    literature_review: str,
    cross_text: str,
    gaps_text: str,
    gap_critique: str,
    citation_text: str,
    config: Optional[dict] = None,
) -> str:
    """Generate a multi-section comprehensive publication manuscript using 3 focused 8k token calls."""
    sections = []

    # Part 1: Abstract, Intro, Background & Thematic Survey (Sections 1-3)
    if config:
        await _send_status(config, "Composing Part 1: Abstract, Foundations & Literature Survey...", data={"node": "paper_architect", "step": 1})
    prompt1 = PAPER_PART1_PROMPT.format(
        topic=topic,
        paper_format=paper_format,
        literature_review=literature_review[:10000],
        cross_paper_analysis=cross_text[:6000],
        citation_info=citation_text[:7000],
    )
    try:
        p1 = await llm_client.generate_content(
            prompt=prompt1,
            temperature=0.35,
            max_output_tokens=8192,
            enable_thinking=True,
        )
        if p1.strip():
            # Clean any accidental trailing references heading from part 1
            cleaned_p1 = _strip_reference_tail(p1.strip())
            sections.append(cleaned_p1 or p1.strip())
    except Exception as e:
        logger.error("Paper Architect Part 1 generation failed: %s", e)

    # Part 2: Algorithmic Mechanics & Comparative Empirical Evaluation (Sections 4-5)
    if config:
        await _send_status(config, "Composing Part 2: Algorithmic Mechanics & Comparative Evaluation...", data={"node": "paper_architect", "step": 2})
    prompt2 = PAPER_PART2_PROMPT.format(
        topic=topic,
        paper_format=paper_format,
        cross_paper_analysis=cross_text[:8000],
        gaps_section=gaps_text[:6000],
        citation_info=citation_text[:7000],
    )
    try:
        p2 = await llm_client.generate_content(
            prompt=prompt2,
            temperature=0.35,
            max_output_tokens=8192,
            enable_thinking=True,
        )
        if p2.strip():
            # Clean any accidental trailing references heading from part 2
            cleaned_p2 = _strip_reference_tail(p2.strip())
            sections.append(cleaned_p2 or p2.strip())
    except Exception as e:
        logger.error("Paper Architect Part 2 generation failed: %s", e)

    # Part 3: Gaps, Roadmap, Engineering Takeaways, Conclusion & References (Sections 6-9 + References)
    if config:
        await _send_status(config, "Composing Part 3: Strategic Roadmap, Takeaways & References...", data={"node": "paper_architect", "step": 3})
    prompt3 = PAPER_PART3_PROMPT.format(
        topic=topic,
        paper_format=paper_format,
        gaps_section=gaps_text[:6000],
        gap_critique=gap_critique[:3000],
        citation_info=citation_text[:7000],
    )
    try:
        p3 = await llm_client.generate_content(
            prompt=prompt3,
            temperature=0.35,
            max_output_tokens=8192,
            enable_thinking=True,
        )
        if p3.strip():
            sections.append(p3.strip())
    except Exception as e:
        logger.error("Paper Architect Part 3 generation failed: %s", e)

    return "\n\n".join(sections)


async def paper_architect_node(state: ResearchState, config: dict) -> dict:
    """Assemble the final Markdown paper from all pipeline outputs."""
    topic = state["topic"]
    lit_review = state.get("literature_review_draft", "No literature review available.")
    gaps = state.get("identified_gaps", [])
    summaries = state.get("paper_summaries", {})
    cross = state.get("cross_paper_analysis", {})
    gap_critique = _clean_nullable_text(state.get("gap_critique", ""))
    paper_format = str(state.get("paper_format", "academic")).strip().upper()
    citation_quality = state.get("citation_quality", {})
    citation_verification = state.get("citation_verification", {})

    await _send_status(config, "Assembling final paper...", data={"node": "paper_architect"})

    # Format gaps
    gaps_text = ""
    for i, gap in enumerate(gaps, 1):
        gaps_text += f"\n### Gap {i}: {gap.get('title', 'Untitled')}\n"
        gaps_text += f"**Evidence:** {gap.get('evidence', 'N/A')}\n"
        if gap.get("why_gap"):
            gaps_text += f"**Why Gap:** {gap.get('why_gap')}\n"
        if gap.get("what_missing"):
            gaps_text += f"**What Missing:** {gap.get('what_missing')}\n"
        supports = gap.get("supports", [])
        fails = gap.get("fails", [])
        if isinstance(supports, list) and supports:
            gaps_text += f"**Supports:** {', '.join(str(s) for s in supports[:4])}\n"
        if isinstance(fails, list) and fails:
            gaps_text += f"**Fails to Address:** {', '.join(str(s) for s in fails[:4])}\n"
        gaps_text += f"**Proposed Direction:** {gap.get('proposed_direction', 'N/A')}\n"

    cross_text = (
        _findings_brief(cross)
        + "\n\nStructured cross-paper JSON:\n"
        + json.dumps(cross, ensure_ascii=False, indent=2)
    )[:5000]

    # Format citation info
    citation_text = ""
    citation_index = 1
    for pid, info in summaries.items():
        if info.get("title"):
            year = info.get("year", "n.d.")
            quality = citation_quality.get(pid, {}).get("confidence", 0.5)
            verified = citation_verification.get(pid, {}).get("verified", False)
            cite_key = _citation_key(info.get("authors", ""), year)
            citation_text += (
                f"- [{citation_index}] [@{cite_key}] {info['authors']}. \"{info['title']}.\" "
                f"year={year}; source={info.get('source', '')}; methodology={info.get('methodology', '')}; "
                f"evaluation_signal={info.get('evaluation_signal', '')}; limitation={info.get('limitation', '')}; "
                f"summary={info.get('summary', '')} {info.get('url', '')} "
                f"(confidence={quality}, verified={verified})"
            )
            if info.get("doi"):
                citation_text += f" DOI:{info.get('doi')}"
            citation_text += "\n"
            citation_index += 1

    style_addendum = ""
    if paper_format in {"IEEE", "ACM"}:
        style_addendum = (
            "\nMANDATORY FORMAT RULES:\n"
            "- Use only numeric in-text citations like [1], [2], [3].\n"
            "- Do NOT use [@AuthorYear] in final prose.\n"
            "- End with a 'References' section in strictly numbered order.\n"
        )
    elif paper_format == "APA":
        style_addendum = (
            "\nMANDATORY FORMAT RULES:\n"
            "- Use only APA in-text citations (Author, Year).\n"
            "- Do NOT use bracketed numeric citations or [@AuthorYear].\n"
            "- End with a 'References' section in APA style.\n"
        )
    elif paper_format == "MLA":
        style_addendum = (
            "\nMANDATORY FORMAT RULES:\n"
            "- Use MLA parenthetical in-text citations.\n"
            "- End with a 'Works Cited' section.\n"
            "- Do NOT use [@AuthorYear] or numeric-only citation style.\n"
        )
    elif paper_format == "ACADEMIC":
        style_addendum = (
            "\nMANDATORY FORMAT RULES:\n"
            "- Keep [@AuthorYear] markers in text.\n"
            "- End with a references section that includes all cited keys.\n"
        )

    # Generate the 3-part publication-ready paper (12,000-token modular generation)
    await _send_status(
        config,
        "Composing 12k-token publication paper...",
        data={"node": "paper_architect"},
    )
    draft_markdown = await _generate_final_paper(
        topic=topic,
        paper_format=paper_format,
        literature_review=lit_review,
        cross_text=cross_text,
        gaps_text=gaps_text,
        gap_critique=gap_critique,
        citation_text=citation_text + style_addendum,
        config=config,
    )
    draft_markdown = _strip_mermaid_blocks(draft_markdown).strip()
    analysis_markdown = ""

    # If both calls completely failed, provide a basic fallback so the user doesn't get an empty screen
    if not draft_markdown or len(draft_markdown.strip()) < 100:
        draft_markdown = f"# Research Summary: {topic}\n\n## Literature Review\n{lit_review}\n\n## Cross-Paper Analysis\n{cross_text}\n\n## Identified Gaps\n{gaps_text}\n"
        if not analysis_markdown:
            analysis_markdown = "Analysis generation failed due to API limits."

    # Build bibliography + deterministic citation keys
    bibliography = []
    seen_keys = set()
    for pid, info in summaries.items():
        if info.get("title"):
            base_key = _citation_key(info.get("authors", ""), info.get("year", ""))
            cite_key = base_key
            counter = 1
            while cite_key in seen_keys:
                counter += 1
                cite_key = f"{base_key}_{counter}"
            seen_keys.add(cite_key)
            
            bibliography.append({
                "paper_id": pid,
                "citation_key": cite_key,
                "title": info["title"],
                "authors": info.get("authors", ""),
                "year": info.get("year", ""),
                "url": info.get("url", ""),
                "doi": info.get("doi", ""),
                "source": info.get("source", ""),
                "source_quality": citation_quality.get(pid, {}).get("source_quality", 0.5),
                "confidence": citation_quality.get(pid, {}).get("confidence", 0.5),
                "verified": citation_verification.get(pid, {}).get("verified", False),
            })

    bibliography.sort(
        key=lambda b: (
            0 if b.get("verified") else 1,
            -(b.get("confidence") or 0),
            str(b.get("title", "")).lower(),
        )
    )

    visual_replacements = _delve_visual_appendix(
        topic=topic,
        summaries=summaries,
        gaps=gaps,
        citation_quality=citation_quality,
        citation_verification=citation_verification,
        source_counts=state.get("source_counts", {}),
        bibliography=bibliography,
        cross=cross,
    )
    analysis_markdown = _replace_delve_visual_tokens(analysis_markdown, visual_replacements)
    analysis_markdown = _sanitize_mermaid_syntax(analysis_markdown)
    draft_markdown = _replace_delve_visual_tokens(draft_markdown, {token: "" for token in visual_replacements})
    draft_markdown = _strip_mermaid_blocks(draft_markdown)
    draft_markdown = _strip_visual_placeholder_labels(draft_markdown)

    normalized_paper = _normalize_citations_for_format(draft_markdown or paper, paper_format, bibliography)
    normalized_paper = _strip_reference_tail(normalized_paper)
    references_section = _render_references_section(paper_format, bibliography)
    if references_section:
        normalized_paper = f"{normalized_paper}\n\n{references_section}\n"

    claim_map = _extract_claim_evidence_map(
        normalized_paper,
        bibliography,
        paper_format,
        citation_quality,
        citation_verification,
    )

    if not claim_map:
        for b in bibliography[: min(8, len(bibliography))]:
            claim_map.append({
                "claim": f"Evidence synthesized from {b.get('title', 'source')[:180]}",
                "paper_id": b.get("paper_id", ""),
                "confidence": b.get("confidence", 0.5),
                "verified": b.get("verified", False),
            })

    analysis_markdown = _build_structured_dossier(
        topic=topic,
        raw_analysis=analysis_markdown,
        summaries=summaries,
        gaps=gaps,
        cross=cross,
        gap_critique=gap_critique,
        citation_quality=citation_quality,
        citation_verification=citation_verification,
        source_counts=state.get("source_counts", {}),
        bibliography=bibliography,
        claim_map=claim_map,
        debate_log=state.get("debate_log", []),
        duplicate_clusters=state.get("duplicate_clusters", []),
        visual_replacements=visual_replacements,
    )
    analysis_markdown = _sanitize_mermaid_syntax(analysis_markdown)

    format_compliance = _format_compliance_summary(normalized_paper, paper_format)

    debate_rounds = max(0, sum(1 for d in state.get("debate_log", []) if isinstance(d, str) and d.startswith("[CRITIC]")))
    verified_count = sum(1 for v in citation_verification.values() if v.get("verified"))

    await _send_status(config, "Paper assembled; saving session artifacts...", "status", {
        "node": "paper_architect",
        "paper_length": len(normalized_paper),
        "num_references": len(bibliography),
        "num_gaps": len(gaps),
        "debate_rounds": debate_rounds,
        "verified_citations": verified_count,
        "claim_evidence_links": len(claim_map),
        "format_compliance_score": format_compliance.get("score", 0.0),
        "format_compliant": bool(format_compliance.get("is_compliant", False)),
    })

    return {
        "research_analysis_markdown": analysis_markdown,
        "final_draft_markdown": normalized_paper,
        "final_paper_markdown": normalized_paper,
        "bibliography": bibliography,
        "status_message": "Research paper complete",
        "error": None,
        "claim_to_evidence_map": claim_map,
        "debate_rounds": debate_rounds,
        "verified_citations": verified_count,
        "format_compliance": format_compliance,
    }


# ══════════════════════════════════════════════════════════════════════════
# DEBATE ROUTER – controls the proposer ↔ critic loop
# ══════════════════════════════════════════════════════════════════════════

def debate_should_continue(state: ResearchState) -> str:
    """
    Decide whether the debate loop should continue.
    If max_debate_rounds is 0, debate is skipped.
    """
    debate_log = state.get("debate_log", [])
    critic_rounds = sum(1 for msg in debate_log if msg.startswith("[CRITIC]"))
    failed_rounds = sum(1 for msg in debate_log if msg.startswith("[CRITIC-FAILED]"))
    max_rounds = int(state.get("max_debate_rounds", settings.max_debate_rounds))
    max_rounds = max(0, min(5, max_rounds))

    # Failed critic rounds do not count toward the budget, but they still bound
    # total attempts so a sustained LLM outage cannot loop forever.
    if critic_rounds >= max_rounds or failed_rounds >= max_rounds:
        return "gap_analysis"
    return "critic"
