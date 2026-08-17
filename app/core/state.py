"""
Delve Research State
────────────────────
Defines the ResearchState TypedDict used by LangGraph.
All agent nodes read from and write to this shared state.
"""

from __future__ import annotations

from typing import Annotated, Any, Optional
from typing_extensions import TypedDict


def _merge_lists(left: list, right: list) -> list:
    """Reducer: merge two lists (append new items)."""
    return left + right


def _merge_dicts(left: dict, right: dict) -> dict:
    """Reducer: merge two dicts (right overwrites left on key conflict)."""
    merged = {**left}
    merged.update(right)
    return merged


class ResearchState(TypedDict, total=False):
    """
    Shared state flowing through the LangGraph research pipeline.

    Fields are grouped by pipeline stage.  Annotated fields use reducers
    so that parallel nodes can safely append to shared collections.
    """

    # ── User Input ────────────────────────────────────────────────────────
    topic: str
    uploaded_paper_ids: list[str]

    # ── Planner Output ────────────────────────────────────────────────────
    refined_query: str
    search_queries: list[str]
    research_plan: list[str]
    planner_constraints: dict[str, Any]
    strict_mode: bool
    max_debate_rounds: int
    paper_format: str

    # ── Retrieval Output ──────────────────────────────────────────────────
    retrieved_papers: Annotated[list[dict[str, Any]], _merge_lists]
    vector_results: Annotated[list[dict[str, Any]], _merge_lists]
    duplicate_clusters: list[dict[str, Any]]
    source_counts: dict[str, int]

    # ── Summarizer Output ─────────────────────────────────────────────────
    paper_summaries: Annotated[dict[str, dict[str, str]], _merge_dicts]
    citation_quality: dict[str, dict[str, Any]]
    citation_verification: dict[str, dict[str, Any]]

    # ── Debate / Synthesis Output ─────────────────────────────────────────
    literature_review_draft: str
    debate_log: Annotated[list[str], _merge_lists]
    debate_round: int
    token_estimate: int

    # ── Gap Analysis Output ───────────────────────────────────────────────
    cross_paper_analysis: dict[str, Any]
    generated_gap_candidates: list[dict[str, Any]]
    gap_critique: str
    identified_gaps: list[dict[str, Any]]

    # ── Final Paper ───────────────────────────────────────────────────────
    research_analysis_markdown: str
    final_draft_markdown: str
    final_paper_markdown: str
    bibliography: list[dict[str, str]]
    claim_to_evidence_map: list[dict[str, Any]]
    debate_rounds: int
    verified_citations: int
    format_compliance: dict[str, Any]

    # ── Communication ─────────────────────────────────────────────────────
    status_message: str
    error: Optional[str]

    # ── Session ───────────────────────────────────────────────────────────
    session_id: str
    owner_id: str
