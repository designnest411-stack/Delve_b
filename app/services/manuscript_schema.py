"""
ResearchAgent Semantic Manuscript Schema
────────────────────────────────────────
Defines the strongly-typed document model for academic papers.
Separates content generation from typesetting and visual layout.
"""

from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


class AuthorAffiliation(BaseModel):
    name: str = "ResearchAgent Synthesis Engine"
    department: Optional[str] = "Autonomous Research Division"
    institution: Optional[str] = "Delve Academic Research"
    city: Optional[str] = None
    country: Optional[str] = None
    email: Optional[str] = None
    orcid: Optional[str] = None


class CitationSource(BaseModel):
    id: str
    citation_key: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | str = "2024"
    venue: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    entry_type: Literal["article", "inproceedings", "book", "misc"] = "article"
    verified: bool = False
    confidence: float = 0.85


class EquationBlock(BaseModel):
    id: str = "eq-1"
    latex: str
    number: Optional[int] = None
    label: Optional[str] = None
    description: Optional[str] = None


class FigureBlock(BaseModel):
    id: str = "fig-1"
    label: str = "fig:architecture"
    number: Optional[int] = 1
    caption: str
    image_path: str
    span_columns: bool = False  # False = single column (3.5"), True = page width span
    alt_text: Optional[str] = None


class TableBlock(BaseModel):
    id: str = "tbl-1"
    label: str = "tab:results"
    number: Optional[int] = 1
    caption: str
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    notes: Optional[str] = None
    span_columns: bool = False


class SemanticBlock(BaseModel):
    type: Literal[
        "paragraph",
        "heading",
        "equation",
        "figure",
        "table",
        "quote",
        "list_bullet",
        "list_numbered",
        "code",
        "algorithm",
        "callout",
    ] = "paragraph"
    content: Optional[str] = None
    items: list[str] = Field(default_factory=list)
    equation: Optional[EquationBlock] = None
    figure: Optional[FigureBlock] = None
    table: Optional[TableBlock] = None
    language: Optional[str] = None


class Section(BaseModel):
    id: str
    number: str = "1"
    title: str
    level: int = 1  # 1 = Main Section, 2 = Subsection, 3 = Subsubsection
    blocks: list[SemanticBlock] = Field(default_factory=list)


class ManuscriptDocument(BaseModel):
    """Semantic document representation of an academic paper."""
    title: str
    running_title: str
    authors: list[AuthorAffiliation] = Field(default_factory=list)
    abstract: str
    keywords: list[str] = Field(default_factory=list)
    ccs_concepts: list[str] = Field(default_factory=list)
    paper_type: Literal["experimental", "survey", "system", "position"] = "experimental"
    publication_style: Literal["ieee", "acm", "apa", "mla"] = "ieee"
    publication_date: Optional[str] = None
    doi: Optional[str] = None
    sections: list[Section] = Field(default_factory=list)
    figures: list[FigureBlock] = Field(default_factory=list)
    tables: list[TableBlock] = Field(default_factory=list)
    equations: list[EquationBlock] = Field(default_factory=list)
    references: list[CitationSource] = Field(default_factory=list)
    appendices: list[Section] = Field(default_factory=list)
    quality_score: Optional[dict[str, Any]] = None
