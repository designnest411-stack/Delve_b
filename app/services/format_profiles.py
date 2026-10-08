"""
ResearchAgent Publication Format Profiles
─────────────────────────────────────────
Authoritative style specifications for IEEE, ACM, APA 7th, and MLA 9th.
These profiles drive the typesetting engine and enforce publication conventions.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class TypographySpec:
    body_font: str
    heading_font: str
    mono_font: str
    body_size_pt: float
    line_spacing: float
    paragraph_indent_pt: float
    text_align: Literal["justify", "left"]


@dataclass(frozen=True)
class GeometrySpec:
    page_size: str  # "us-letter"
    margin_top_pt: float
    margin_bottom_pt: float
    margin_left_pt: float
    margin_right_pt: float
    columns: int
    column_gutter_pt: float


@dataclass(frozen=True)
class HeadingSpec:
    level1_prefix: Literal["roman", "arabic", "none"]
    level1_casing: Literal["uppercase", "titlecase"]
    level1_align: Literal["center", "left"]
    level1_bold: bool
    level1_small_caps: bool
    level2_prefix: Literal["letter", "arabic_dot", "none"]
    level2_italic: bool
    level2_bold: bool
    level3_prefix: Literal["number_paren", "arabic_dotdot", "none"]
    level3_run_in: bool


@dataclass(frozen=True)
class CitationSpec:
    csl_style: str
    in_text_format: Literal["bracket_numeric", "author_year", "author_page"]
    bibliography_heading: str
    bibliography_align: Literal["center", "left"]
    hanging_indent_pt: float


@dataclass(frozen=True)
class FormatProfile:
    name: str
    display_name: str
    geometry: GeometrySpec
    typography: TypographySpec
    heading: HeadingSpec
    citation: CitationSpec
    has_separate_title_page: bool = False
    has_running_head: bool = False
    author_block_layout: Literal["multi_column", "affiliation_cards", "title_page_byline", "first_page_header"] = "multi_column"
    figure_caption_prefix: str = "Fig. "
    figure_caption_separator: str = ". "
    table_caption_prefix: str = "TABLE "
    table_caption_separator: str = "\n"
    table_caption_casing: Literal["small_caps", "titlecase", "sentence"] = "small_caps"
    table_caption_above: bool = True
    abstract_prefix: str = "Abstract—"
    keywords_prefix: str = "Index Terms—"
    show_ccs_concepts: bool = False
    show_acm_ref_format: bool = False


# ── IEEE Conference Profile ──────────────────────────────────────────────────
IEEE_PROFILE = FormatProfile(
    name="ieee",
    display_name="IEEE Conference Proceedings",
    geometry=GeometrySpec(
        page_size="us-letter",
        margin_top_pt=54.0,       # 0.75 in
        margin_bottom_pt=72.0,    # 1.0 in
        margin_left_pt=45.0,      # 0.625 in
        margin_right_pt=45.0,     # 0.625 in
        columns=2,
        column_gutter_pt=18.0,    # 0.25 in (6.35 mm)
    ),
    typography=TypographySpec(
        body_font="Times New Roman",
        heading_font="Times New Roman",
        mono_font="Courier",
        body_size_pt=10.0,
        line_spacing=1.18,
        paragraph_indent_pt=12.0, # 1 pica / 3.5 mm
        text_align="justify",
    ),
    heading=HeadingSpec(
        level1_prefix="roman",
        level1_casing="uppercase",
        level1_align="center",
        level1_bold=False,
        level1_small_caps=True,
        level2_prefix="letter",
        level2_italic=True,
        level2_bold=False,
        level3_prefix="number_paren",
        level3_run_in=True,
    ),
    citation=CitationSpec(
        csl_style="ieee",
        in_text_format="bracket_numeric",
        bibliography_heading="REFERENCES",
        bibliography_align="center",
        hanging_indent_pt=12.0,
    ),
    has_separate_title_page=False,
    has_running_head=False,
    author_block_layout="multi_column",
    figure_caption_prefix="Fig. ",
    figure_caption_separator=". ",
    table_caption_prefix="TABLE ",
    table_caption_separator="\n",
    table_caption_casing="small_caps",
    table_caption_above=True,
    abstract_prefix="Abstract—",
    keywords_prefix="Index Terms—",
    show_ccs_concepts=False,
    show_acm_ref_format=False,
)

# ── ACM Conference (sigconf) Profile ─────────────────────────────────────────
ACM_PROFILE = FormatProfile(
    name="acm",
    display_name="ACM Conference (sigconf)",
    geometry=GeometrySpec(
        page_size="us-letter",
        margin_top_pt=54.0,       # 0.75 in
        margin_bottom_pt=54.0,    # 0.75 in
        margin_left_pt=54.0,      # 0.75 in
        margin_right_pt=54.0,     # 0.75 in
        columns=2,
        column_gutter_pt=23.76,   # 0.33 in
    ),
    typography=TypographySpec(
        body_font="Linux Libertine",
        heading_font="Linux Biolinum",
        mono_font="Inconsolata",
        body_size_pt=9.0,
        line_spacing=1.22,
        paragraph_indent_pt=10.8,
        text_align="justify",
    ),
    heading=HeadingSpec(
        level1_prefix="arabic",
        level1_casing="uppercase",
        level1_align="left",
        level1_bold=True,
        level1_small_caps=False,
        level2_prefix="arabic_dot",
        level2_italic=False,
        level2_bold=True,
        level3_prefix="arabic_dotdot",
        level3_run_in=False,
    ),
    citation=CitationSpec(
        csl_style="association-for-computing-machinery",
        in_text_format="bracket_numeric",
        bibliography_heading="REFERENCES",
        bibliography_align="left",
        hanging_indent_pt=14.0,
    ),
    has_separate_title_page=False,
    has_running_head=True,
    author_block_layout="affiliation_cards",
    figure_caption_prefix="Figure ",
    figure_caption_separator=": ",
    table_caption_prefix="Table ",
    table_caption_separator=": ",
    table_caption_casing="titlecase",
    table_caption_above=True,
    abstract_prefix="ABSTRACT",
    keywords_prefix="KEYWORDS",
    show_ccs_concepts=True,
    show_acm_ref_format=True,
)

# ── APA 7th Edition Professional Profile ─────────────────────────────────────
APA_PROFILE = FormatProfile(
    name="apa",
    display_name="APA 7th Edition Professional",
    geometry=GeometrySpec(
        page_size="us-letter",
        margin_top_pt=72.0,       # 1.0 in
        margin_bottom_pt=72.0,    # 1.0 in
        margin_left_pt=72.0,      # 1.0 in
        margin_right_pt=72.0,     # 1.0 in
        columns=1,
        column_gutter_pt=0.0,
    ),
    typography=TypographySpec(
        body_font="Times New Roman",
        heading_font="Times New Roman",
        mono_font="Courier",
        body_size_pt=11.5,
        line_spacing=1.65,        # 1.65x–2.0x clean line height
        paragraph_indent_pt=36.0, # 0.5 in (12.7 mm)
        text_align="left",        # Ragged right per APA 7
    ),
    heading=HeadingSpec(
        level1_prefix="none",
        level1_casing="titlecase",
        level1_align="center",
        level1_bold=True,
        level1_small_caps=False,
        level2_prefix="none",
        level2_italic=False,
        level2_bold=True,
        level3_prefix="none",
        level3_run_in=False,
    ),
    citation=CitationSpec(
        csl_style="american-psychological-association",
        in_text_format="author_year",
        bibliography_heading="References",
        bibliography_align="center",
        hanging_indent_pt=36.0,   # 0.5 in hanging indent
    ),
    has_separate_title_page=True,
    has_running_head=True,
    author_block_layout="title_page_byline",
    figure_caption_prefix="Figure ",
    figure_caption_separator="\n",
    table_caption_prefix="Table ",
    table_caption_separator="\n",
    table_caption_casing="titlecase",
    table_caption_above=True,
    abstract_prefix="Abstract",
    keywords_prefix="Keywords: ",
    show_ccs_concepts=False,
    show_acm_ref_format=False,
)

# ── MLA 9th Edition Profile ──────────────────────────────────────────────────
MLA_PROFILE = FormatProfile(
    name="mla",
    display_name="MLA 9th Edition",
    geometry=GeometrySpec(
        page_size="us-letter",
        margin_top_pt=72.0,       # 1.0 in
        margin_bottom_pt=72.0,    # 1.0 in
        margin_left_pt=72.0,      # 1.0 in
        margin_right_pt=72.0,     # 1.0 in
        columns=1,
        column_gutter_pt=0.0,
    ),
    typography=TypographySpec(
        body_font="Times New Roman",
        heading_font="Times New Roman",
        mono_font="Courier",
        body_size_pt=12.0,
        line_spacing=1.80,        # Double spaced
        paragraph_indent_pt=36.0, # 0.5 in (12.7 mm)
        text_align="left",        # Ragged right per MLA
    ),
    heading=HeadingSpec(
        level1_prefix="none",
        level1_casing="titlecase",
        level1_align="left",
        level1_bold=True,
        level1_small_caps=False,
        level2_prefix="none",
        level2_italic=True,
        level2_bold=False,
        level3_prefix="none",
        level3_run_in=False,
    ),
    citation=CitationSpec(
        csl_style="modern-language-association",
        in_text_format="author_page",
        bibliography_heading="Works Cited",
        bibliography_align="center",
        hanging_indent_pt=36.0,   # 0.5 in hanging indent
    ),
    has_separate_title_page=False,
    has_running_head=True,
    author_block_layout="first_page_header",
    figure_caption_prefix="Fig. ",
    figure_caption_separator=". ",
    table_caption_prefix="Table ",
    table_caption_separator=". ",
    table_caption_casing="sentence",
    table_caption_above=True,
    abstract_prefix="",
    keywords_prefix="",
    show_ccs_concepts=False,
    show_acm_ref_format=False,
)

PROFILES: dict[str, FormatProfile] = {
    "ieee": IEEE_PROFILE,
    "acm": ACM_PROFILE,
    "apa": APA_PROFILE,
    "mla": MLA_PROFILE,
    "academic": IEEE_PROFILE,
}


def get_profile(name: str | None) -> FormatProfile:
    key = str(name or "ieee").lower().strip()
    return PROFILES.get(key, IEEE_PROFILE)
