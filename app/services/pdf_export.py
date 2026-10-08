from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image,
    ListFlowable,
    ListItem,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ── Journal-style palette ────────────────────────────────────────────────
INK = colors.HexColor("#0f172a")         # Slate 900 — deep authoritative dark
HEADING = colors.HexColor("#0f172a")     # Primary section headings
ACCENT = colors.HexColor("#334155")      # Slate 700 (rules, subsections)
MUTED = colors.HexColor("#64748b")       # Slate 500 (meta, headers, running captions)
RULE = colors.HexColor("#cbd5e1")        # Slate 300 hairlines
RULE_LIGHT = colors.HexColor("#e2e8f0")  # Slate 200 soft rules
TABLE_HEAD_BG = colors.HexColor("#f1f5f9")
TABLE_ALT_BG = colors.HexColor("#f8fafc")
TABLE_GRID = colors.HexColor("#e2e8f0")
ABSTRACT_BG = colors.HexColor("#f8fafc")
ABSTRACT_BORDER = colors.HexColor("#94a3b8")

# Professional typography
SERIF = "Times-Roman"
SERIF_BOLD = "Times-Bold"
SERIF_ITALIC = "Times-Italic"
SANS = "Helvetica"
SANS_BOLD = "Helvetica-Bold"

PAGE_WIDTH, PAGE_HEIGHT = A4
PRINTABLE_WIDTH = PAGE_WIDTH - (1.6 * inch)  # 0.8 inch left and right margin


GREEK_AND_MATH: list[tuple[str, str]] = [
    (r"\\mathcal\{([A-Za-z])\}", r"\1"),
    (r"\\mathbb\{([A-Za-z])\}", r"\1"),
    (r"\\mathbf\{([A-Za-z0-9]+)\}", r"\1"),
    (r"\\text\{([^\}]+)\}", r"\1"),
    (r"\\theta", "θ"),
    (r"\\lambda", "λ"),
    (r"\\mu", "μ"),
    (r"\\sigma", "σ"),
    (r"\\alpha", "α"),
    (r"\\beta", "β"),
    (r"\\gamma", "γ"),
    (r"\\delta", "δ"),
    (r"\\epsilon", "ε"),
    (r"\\Omega", "Ω"),
    (r"\\phi", "ϕ"),
    (r"\\Phi", "Φ"),
    (r"\\in", " ∈ "),
    (r"\\to", " → "),
    (r"\\times", " × "),
    (r"\\sum", "∑"),
    (r"\\le", " ≤ "),
    (r"\\ge", " ≥ "),
    (r"\\neq", " ≠ "),
    (r"\\approx", " ≈ "),
    (r"\\{", "{"),
    (r"\\}", "}"),
]


def _clean_latex_math(math_str: str) -> str:
    s = math_str.strip()
    for pattern, repl in GREEK_AND_MATH:
        s = re.sub(pattern, repl, s)
    s = re.sub(r"\\([a-zA-Z]+)", r"\1", s)
    return s


def _inline_markdown_to_html(text: str) -> str:
    cleaned = str(text or "").strip()
    # Strip markdown images or badges
    cleaned = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", cleaned)
    # Format links [Text](URL) -> Text
    cleaned = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", cleaned)

    placeholders: dict[str, str] = {}
    p_idx = 0

    # 1. Stash code
    def _stash_code(m: re.Match) -> str:
        nonlocal p_idx
        key = f"___PH_CODE_{p_idx}___"
        p_idx += 1
        code_content = escape(m.group(1))
        placeholders[key] = f'<font face="Courier" color="#0f172a">{code_content}</font>'
        return key

    cleaned = re.sub(r"`([^`]+)`", _stash_code, cleaned)

    # 2. Stash display math $$...$$
    def _stash_display_math(m: re.Match) -> str:
        nonlocal p_idx
        key = f"___PH_DMATH_{p_idx}___"
        p_idx += 1
        math_content = escape(_clean_latex_math(m.group(1)))
        placeholders[key] = f'<font face="Courier" color="#1e293b"><b>{math_content}</b></font>'
        return key

    cleaned = re.sub(r"\$\$(.+?)\$\$", _stash_display_math, cleaned, flags=re.DOTALL)

    # 3. Stash inline math $...$
    def _stash_inline_math(m: re.Match) -> str:
        nonlocal p_idx
        key = f"___PH_IMATH_{p_idx}___"
        p_idx += 1
        math_content = escape(_clean_latex_math(m.group(1)))
        placeholders[key] = f'<font face="Courier" color="#1e293b">{math_content}</font>'
        return key

    cleaned = re.sub(r"(?<!\$)\$(?!\$)([^\$\n]+?)(?<!\$)\$(?!\$)", _stash_inline_math, cleaned)

    # 4. Escape XML entities before formatting tags
    cleaned = escape(cleaned)

    # 5. Convert bold **text**
    cleaned = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", cleaned)

    # 6. Convert italic *text* or _text_ (only safe word-boundary underscores)
    cleaned = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<i>\1</i>", cleaned)
    cleaned = re.sub(r"\b_([A-Za-z0-9]+?)_\b", r"<i>\1</i>", cleaned)

    # 7. Restore placeholders
    for key, val in placeholders.items():
        cleaned = cleaned.replace(key, val)

    return cleaned


def _safe_paragraph(raw_text_or_html: str, style, is_already_html: bool = False) -> Paragraph:
    html_content = raw_text_or_html if is_already_html else _inline_markdown_to_html(raw_text_or_html)
    try:
        return Paragraph(html_content, style)
    except Exception:
        # Fallback to plain text stripped of HTML tags and cleanly escaped
        clean_fallback = escape(re.sub(r"<[^>]+>", "", raw_text_or_html))
        try:
            return Paragraph(clean_fallback, style)
        except Exception:
            return Paragraph(escape(raw_text_or_html), style)



def _parse_markdown_table(lines: list[str]) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        # Skip markdown separator row |---|---|
        if cells and all(re.fullmatch(r":?-{2,}:?", cell.replace(" ", "")) for cell in cells):
            continue
        rows.append(cells)
    return rows


def _image_from_markdown(line: str, resource_dir: Path) -> Image | None:
    match = re.search(r"!\[[^\]]*\]\([^)]+\)", line)
    if not match:
        return None
    raw_path = match.group(1).strip()
    file_name = Path(raw_path).name
    img_path = resource_dir / file_name
    if not img_path.exists():
        return None
    image = Image(str(img_path))
    image._restrictSize(PRINTABLE_WIDTH, 4.3 * inch)
    image.hAlign = "CENTER"
    return image


def _build_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="PaperTitle",
        parent=styles["Title"],
        fontName=SERIF_BOLD,
        fontSize=20,
        leading=25,
        alignment=TA_CENTER,
        textColor=HEADING,
        spaceBefore=0,
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="PaperMeta",
        parent=styles["BodyText"],
        fontName=SANS,
        fontSize=8.5,
        leading=12,
        alignment=TA_CENTER,
        textColor=MUTED,
        spaceAfter=12,
    ))
    styles.add(ParagraphStyle(
        name="Section",
        parent=styles["Heading1"],
        fontName=SERIF_BOLD,
        fontSize=13,
        leading=16.5,
        textColor=HEADING,
        spaceBefore=14,
        spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="Subsection",
        parent=styles["Heading2"],
        fontName=SERIF_BOLD,
        fontSize=11,
        leading=14.5,
        textColor=ACCENT,
        spaceBefore=9,
        spaceAfter=3,
    ))
    styles.add(ParagraphStyle(
        name="Subsubsection",
        parent=styles["Heading3"],
        fontName=SERIF_BOLD,
        fontSize=10,
        leading=13.5,
        textColor=ACCENT,
        spaceBefore=7,
        spaceAfter=2,
    ))
    styles.add(ParagraphStyle(
        name="Body",
        parent=styles["BodyText"],
        fontName=SERIF,
        fontSize=9.8,
        leading=14.5,
        alignment=TA_JUSTIFY,
        textColor=INK,
        spaceAfter=6,
        firstLineIndent=0,
    ))
    styles.add(ParagraphStyle(
        name="AbstractLead",
        parent=styles["BodyText"],
        fontName=SERIF_BOLD,
        fontSize=9.5,
        leading=14,
        alignment=TA_LEFT,
        textColor=HEADING,
        leftIndent=14,
        rightIndent=14,
        spaceBefore=4,
        spaceAfter=2,
    ))
    styles.add(ParagraphStyle(
        name="AbstractBody",
        parent=styles["BodyText"],
        fontName=SERIF,
        fontSize=9.2,
        leading=13.8,
        alignment=TA_JUSTIFY,
        textColor=INK,
        leftIndent=14,
        rightIndent=14,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="Keywords",
        parent=styles["BodyText"],
        fontName=SANS,
        fontSize=8.5,
        leading=12,
        alignment=TA_JUSTIFY,
        textColor=ACCENT,
        leftIndent=14,
        rightIndent=14,
        spaceAfter=10,
    ))
    styles.add(ParagraphStyle(
        name="Caption",
        parent=styles["Italic"],
        fontName=SERIF_ITALIC,
        fontSize=8.5,
        leading=11.5,
        alignment=TA_CENTER,
        textColor=MUTED,
        spaceAfter=8,
        spaceBefore=3,
    ))
    styles.add(ParagraphStyle(
        name="Reference",
        parent=styles["BodyText"],
        fontName=SERIF,
        fontSize=8.5,
        leading=11.8,
        alignment=TA_JUSTIFY,
        textColor=INK,
        leftIndent=18,
        firstLineIndent=-18,   # hanging indent for bibliography
        spaceAfter=3.5,
    ))
    styles.add(ParagraphStyle(
        name="Callout",
        parent=styles["BodyText"],
        fontName=SERIF_ITALIC,
        fontSize=9.2,
        leading=13.5,
        alignment=TA_JUSTIFY,
        textColor=ACCENT,
        leftIndent=16,
        rightIndent=16,
        spaceBefore=4,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="CodeBlock",
        parent=styles["Code"],
        fontName="Courier",
        fontSize=7.8,
        leading=10,
        textColor=INK,
        leftIndent=12,
        rightIndent=12,
        spaceBefore=4,
        spaceAfter=6,
    ))
    return styles


_CELL_STYLE = ParagraphStyle(
    name="TableCell",
    fontName=SANS,
    fontSize=7.4,
    leading=9.5,
    textColor=INK,
)

_CELL_HEAD_STYLE = ParagraphStyle(
    name="TableHeadCell",
    fontName=SANS_BOLD,
    fontSize=7.6,
    leading=9.8,
    textColor=HEADING,
)


def _table_flowable(table_rows: list[list[str]], usable_width: float = PRINTABLE_WIDTH):
    if not table_rows:
        return Spacer(1, 0)

    max_cols = max(len(r) for r in table_rows)
    if max_cols == 0:
        return Spacer(1, 0)

    # Normalize rows so every row has exact max_cols to prevent ReportLab ValueError
    normalized_rows = []
    for r_idx, row in enumerate(table_rows):
        row_copy = list(row)
        while len(row_copy) < max_cols:
            row_copy.append("")
        cell_style = _CELL_HEAD_STYLE if r_idx == 0 else _CELL_STYLE
        wrapped_row = [_safe_paragraph(cell, cell_style) for cell in row_copy[:max_cols]]
        normalized_rows.append(wrapped_row)

    col_width = usable_width / float(max_cols)
    table = Table(normalized_rows, colWidths=[col_width] * max_cols, repeatRows=1, hAlign="CENTER")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEAD_BG),
        ("FONTNAME", (0, 0), (-1, 0), SANS_BOLD),
        ("FONTSIZE", (0, 0), (-1, -1), 7.4),
        ("LEADING", (0, 0), (-1, -1), 9.5),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, ACCENT),
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, ACCENT),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, ACCENT),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, TABLE_GRID),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, TABLE_ALT_BG]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def _is_references_heading(text: str) -> bool:
    t = re.sub(r"^[0-9.\s]+", "", text).strip().lower()
    return t in {"references", "bibliography", "works cited", "citations"}


def _markdown_to_story(markdown: str, resource_dir: Path, styles) -> list:
    story: list = []
    lines = str(markdown or "").splitlines()
    paragraph_buffer: list[str] = []
    list_buffer: list[str] = []
    table_buffer: list[str] = []
    in_code = False
    code_lines: list[str] = []
    section_counter = 0
    in_references = False
    seen_first_heading = False

    def flush_paragraph():
        nonlocal paragraph_buffer
        if paragraph_buffer:
            text = " ".join(part.strip() for part in paragraph_buffer if part.strip())
            if text:
                # Check if this paragraph is Abstract or Keywords
                text_lower = text.lower()
                if text_lower.startswith("**abstract**") or text_lower.startswith("abstract —") or text_lower.startswith("abstract:"):
                    story.append(Spacer(1, 0.04 * inch))
                    story.append(HRFlowable(width="100%", thickness=0.6, color=RULE, spaceBefore=2, spaceAfter=6))
                    story.append(_safe_paragraph(text, styles["AbstractBody"]))
                elif text_lower.startswith("**keywords**") or text_lower.startswith("keywords:"):
                    story.append(_safe_paragraph(text, styles["Keywords"]))
                    story.append(HRFlowable(width="100%", thickness=0.6, color=RULE, spaceBefore=2, spaceAfter=8))
                elif in_references:
                    story.append(_safe_paragraph(text, styles["Reference"]))
                else:
                    story.append(_safe_paragraph(text, styles["Body"]))
            paragraph_buffer = []

    def flush_list():
        nonlocal list_buffer
        if list_buffer:
            item_style = styles["Reference"] if in_references else styles["Body"]
            if in_references:
                for item in list_buffer:
                    story.append(_safe_paragraph(item, item_style))
            else:
                items = [
                    ListItem(_safe_paragraph(item, item_style), leftIndent=6)
                    for item in list_buffer
                ]
                story.append(ListFlowable(items, bulletType="bullet", leftIndent=14,
                                          bulletColor=ACCENT, bulletFontSize=6.5))
                story.append(Spacer(1, 0.02 * inch))
            list_buffer = []

    def flush_table():
        nonlocal table_buffer
        if table_buffer:
            rows = _parse_markdown_table(table_buffer)
            if rows:
                story.append(Spacer(1, 0.04 * inch))
                story.append(_table_flowable(rows))
                story.append(Spacer(1, 0.06 * inch))
            table_buffer = []

    def flush_code():
        nonlocal code_lines
        if code_lines:
            text = "\n".join(code_lines).strip()
            if text and "mermaid" not in text.lower():
                story.append(Preformatted(text, styles["CodeBlock"]))
                story.append(Spacer(1, 0.04 * inch))
            code_lines = []

    idx = 0
    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_paragraph(); flush_list(); flush_table()
            if in_code:
                in_code = False
                flush_code()
            else:
                in_code = True
            idx += 1
            continue

        if in_code:
            code_lines.append(line)
            idx += 1
            continue

        if stripped.startswith("|"):
            flush_paragraph(); flush_list()
            table_buffer.append(line)
            idx += 1
            while idx < len(lines) and lines[idx].strip().startswith("|"):
                table_buffer.append(lines[idx])
                idx += 1
            flush_table()
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading_match:
            flush_paragraph(); flush_list(); flush_table()
            level = len(heading_match.group(1))
            raw = heading_match.group(2).strip()

            in_references = _is_references_heading(raw)
            heading_text = _inline_markdown_to_html(raw)

            # Skip the first H1 if it repeats the title already in the front matter
            if not seen_first_heading and level <= 2 and not in_references:
                seen_first_heading = True
                idx += 1
                continue
            seen_first_heading = True

            if level <= 2 and not in_references:
                # Auto-number top-level sections if not already numbered
                if not re.match(r"^\s*\d+[.\)]", raw):
                    section_counter += 1
                    heading_text = f"{section_counter}.&nbsp;&nbsp;{heading_text}"
                story.append(Spacer(1, 0.06 * inch))
                story.append(_safe_paragraph(heading_text, styles["Section"], is_already_html=True))
                story.append(HRFlowable(width="100%", thickness=0.6, color=RULE,
                                        spaceBefore=2, spaceAfter=5))
            elif in_references:
                story.append(Spacer(1, 0.08 * inch))
                story.append(_safe_paragraph(heading_text, styles["Section"], is_already_html=True))
                story.append(HRFlowable(width="100%", thickness=0.6, color=RULE,
                                        spaceBefore=2, spaceAfter=6))
            elif level == 3:
                story.append(_safe_paragraph(heading_text, styles["Subsection"], is_already_html=True))
            else:
                story.append(_safe_paragraph(heading_text, styles["Subsubsection"], is_already_html=True))
            idx += 1
            continue

        image = _image_from_markdown(stripped, resource_dir)
        if image is not None:
            flush_paragraph(); flush_list(); flush_table()
            story.append(Spacer(1, 0.04 * inch))
            story.append(image)
            idx += 1
            if idx < len(lines) and lines[idx].strip().startswith("*Figure:"):
                caption = lines[idx].strip().strip("*")
                story.append(_safe_paragraph(caption, styles["Caption"]))
                idx += 1
            else:
                story.append(Spacer(1, 0.05 * inch))
            continue

        if re.match(r"^\s*[-*]\s+", line):
            flush_paragraph(); flush_table()
            list_buffer.append(re.sub(r"^\s*[-*]\s+", "", line).strip())
            idx += 1
            continue

        # Numbered list items inside a references section → treated as bibliography entries
        if in_references and re.match(r"^\s*\d+[.\)]\s+", line):
            flush_paragraph(); flush_table()
            list_buffer.append(re.sub(r"^\s*\d+[.\)]\s+", "", line).strip())
            idx += 1
            continue

        if stripped.startswith(">"):
            flush_paragraph(); flush_list(); flush_table()
            story.append(_safe_paragraph(stripped.lstrip("> ").strip(), styles["Callout"]))
            idx += 1
            continue

        if not stripped:
            flush_paragraph(); flush_list(); flush_table()
            idx += 1
            continue

        paragraph_buffer.append(line)
        idx += 1

    flush_paragraph(); flush_list(); flush_table(); flush_code()
    return story


def _extract_title(markdown: str, fallback: str) -> str:
    for line in str(markdown or "").splitlines():
        stripped = line.strip()
        m = re.match(r"^#{1,2}\s+(.*)$", stripped)
        if m and m.group(1).strip():
            return m.group(1).strip()
        m2 = re.match(r"^\*\*(.+?)\*\*$", stripped)
        if m2 and len(m2.group(1)) > 8:
            return m2.group(1).strip()
        if stripped and not stripped.startswith(("#", "*", ">", "|")):
            break
    return str(fallback or "Research Manuscript").strip()


class AcademicCanvasDecorator:
    """Canvas decorator adding running headers, footers, and pagination."""

    def __init__(self, running_title: str):
        # Truncate running title to fit running header cleanly
        self.running_title = running_title[:65] + ("…" if len(running_title) > 65 else "")

    def on_first_page(self, canvas, doc):
        canvas.saveState()
        canvas.setFont(SANS, 7.8)
        canvas.setFillColor(MUTED)
        # First page footer: centered manuscript stamp
        canvas.drawCentredString(PAGE_WIDTH / 2.0, 0.42 * inch, "ResearchAgent • Autonomous Academic Synthesis")
        canvas.restoreState()

    def on_later_pages(self, canvas, doc):
        canvas.saveState()
        left = doc.leftMargin
        right = PAGE_WIDTH - doc.rightMargin

        # Running Header
        header_y = PAGE_HEIGHT - 0.48 * inch
        canvas.setFont(SANS, 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(left, header_y, self.running_title.upper())
        canvas.drawRightString(right, header_y, "RESEARCHAGENT SYNTHESIS")
        canvas.setStrokeColor(RULE_LIGHT)
        canvas.setLineWidth(0.4)
        canvas.line(left, header_y - 3, right, header_y - 3)

        # Running Footer
        footer_y = 0.50 * inch
        canvas.setStrokeColor(RULE_LIGHT)
        canvas.setLineWidth(0.4)
        canvas.line(left, footer_y + 8, right, footer_y + 8)
        canvas.setFont(SANS, 7.8)
        canvas.setFillColor(MUTED)
        canvas.drawCentredString(PAGE_WIDTH / 2.0, footer_y - 4, f"— {doc.page} —")
        canvas.restoreState()


def generate_research_pdf(
    *,
    topic: str,
    session_id: str,
    analysis_markdown: str,
    final_markdown: str,
    out_path: Path,
    resource_dir: Path,
) -> Path:
    """Render the final manuscript as a clean, publication-grade academic PDF."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    styles = _build_styles()

    manuscript = str(final_markdown or "").strip() or str(analysis_markdown or "").strip()
    title = _extract_title(manuscript, topic)

    decorator = AcademicCanvasDecorator(title)

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=0.8 * inch,
        rightMargin=0.8 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title=title,
        author="ResearchAgent",
        subject="Autonomous Deep Research Manuscript",
    )

    current_date = datetime.now(timezone.utc).strftime("%B %Y")

    story: list = [
        Spacer(1, 0.1 * inch),
        _safe_paragraph(title, styles["PaperTitle"]),
        _safe_paragraph(
            f"Autonomous Multi-Agent Synthesis Engine • ResearchAgent<br/>"
            f"<font color='#94a3b8' size='7.5'>Peer-Reviewed Methodology Synthesis • Published {current_date}</font>",
            styles["PaperMeta"],
            is_already_html=True,
        ),
        HRFlowable(width="30%", thickness=1.0, color=ACCENT, spaceBefore=0, spaceAfter=10, hAlign="CENTER"),
    ]
    story.extend(_markdown_to_story(manuscript, resource_dir, styles))

    doc.build(
        story,
        onFirstPage=decorator.on_first_page,
        onLaterPages=decorator.on_later_pages,
    )
    return out_path
