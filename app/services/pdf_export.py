from __future__ import annotations

import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ── Journal-style palette ────────────────────────────────────────────────
INK = colors.HexColor("#1a1a1a")        # body text — near-black, warmer than pure black
HEADING = colors.HexColor("#111827")    # section headings
ACCENT = colors.HexColor("#334155")     # subtle slate accent (rules, captions)
MUTED = colors.HexColor("#6b7280")      # meta / captions
RULE = colors.HexColor("#d4d4d8")       # hairlines
TABLE_HEAD_BG = colors.HexColor("#f3f4f6")
TABLE_ALT_BG = colors.HexColor("#fafafa")
TABLE_GRID = colors.HexColor("#e5e7eb")

# Serif for the manuscript body (a real-journal feel); sans reserved for tables/meta.
SERIF = "Times-Roman"
SERIF_BOLD = "Times-Bold"
SERIF_ITALIC = "Times-Italic"
SANS = "Helvetica"
SANS_BOLD = "Helvetica-Bold"


def _inline_markdown_to_html(text: str) -> str:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", cleaned)
    cleaned = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", cleaned)
    cleaned = escape(cleaned)
    cleaned = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", cleaned)
    cleaned = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<i>\1</i>", cleaned)
    cleaned = re.sub(r"`([^`]+)`", r'<font face="Courier">\1</font>', cleaned)
    return cleaned


def _parse_markdown_table(lines: list[str]) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells):
            continue
        rows.append(cells)
    return rows


def _image_from_markdown(line: str, resource_dir: Path) -> Image | None:
    match = re.search(r"!\[[^\]]*\]\(([^)]+)\)", line)
    if not match:
        return None
    raw_path = match.group(1).strip()
    file_name = Path(raw_path).name
    img_path = resource_dir / file_name
    if not img_path.exists():
        return None
    image = Image(str(img_path))
    image._restrictSize(6.0 * inch, 4.3 * inch)
    image.hAlign = "CENTER"
    return image


def _build_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="PaperTitle",
        parent=styles["Title"],
        fontName=SERIF_BOLD,
        fontSize=21,
        leading=26,
        alignment=TA_CENTER,
        textColor=HEADING,
        spaceBefore=0,
        spaceAfter=10,
    ))
    styles.add(ParagraphStyle(
        name="PaperMeta",
        parent=styles["BodyText"],
        fontName=SANS,
        fontSize=8.5,
        leading=12,
        alignment=TA_CENTER,
        textColor=MUTED,
        spaceAfter=4,
    ))
    # Numbered top-level section headings (1, 2, 3 ...)
    styles.add(ParagraphStyle(
        name="Section",
        parent=styles["Heading1"],
        fontName=SERIF_BOLD,
        fontSize=13.5,
        leading=17,
        textColor=HEADING,
        spaceBefore=15,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="Subsection",
        parent=styles["Heading2"],
        fontName=SERIF_BOLD,
        fontSize=11.5,
        leading=15,
        textColor=ACCENT,
        spaceBefore=10,
        spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="Body",
        parent=styles["BodyText"],
        fontName=SERIF,
        fontSize=10.2,
        leading=15,
        alignment=TA_JUSTIFY,
        textColor=INK,
        spaceAfter=7,
        firstLineIndent=0,
    ))
    # Abstract: slightly inset, italicised lead — classic journal look.
    styles.add(ParagraphStyle(
        name="Abstract",
        parent=styles["BodyText"],
        fontName=SERIF,
        fontSize=9.6,
        leading=14,
        alignment=TA_JUSTIFY,
        textColor=INK,
        leftIndent=22,
        rightIndent=22,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="Caption",
        parent=styles["Italic"],
        fontName=SERIF_ITALIC,
        fontSize=8.6,
        leading=11,
        alignment=TA_CENTER,
        textColor=MUTED,
        spaceAfter=10,
        spaceBefore=3,
    ))
    styles.add(ParagraphStyle(
        name="Reference",
        parent=styles["BodyText"],
        fontName=SERIF,
        fontSize=8.8,
        leading=12,
        alignment=TA_JUSTIFY,
        textColor=INK,
        leftIndent=16,
        firstLineIndent=-16,   # hanging indent for reference lists
        spaceAfter=3,
    ))
    return styles


def _table_flowable(table_rows: list[list[str]]):
    wrapped = [
        [Paragraph(_inline_markdown_to_html(cell), _CELL_STYLE) for cell in row]
        for row in table_rows
    ]
    table = Table(wrapped, repeatRows=1, hAlign="CENTER")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), TABLE_HEAD_BG),
        ("FONTNAME", (0, 0), (-1, 0), SANS_BOLD),
        ("FONTSIZE", (0, 0), (-1, -1), 7.6),
        ("LEADING", (0, 0), (-1, -1), 9.5),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, ACCENT),
        ("LINEABOVE", (0, 0), (-1, 0), 0.6, ACCENT),
        ("LINEBELOW", (0, -1), (-1, -1), 0.6, ACCENT),
        ("INNERGRID", (0, 0), (-1, -1), 0.25, TABLE_GRID),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, TABLE_ALT_BG]),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
    ]))
    return table


# Cell paragraph style (sans, tight) — module-level so tables share one instance.
_CELL_STYLE = ParagraphStyle(
    name="TableCell",
    fontName=SANS,
    fontSize=7.6,
    leading=9.5,
    textColor=INK,
)


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
                style = styles["Reference"] if in_references else styles["Body"]
                story.append(Paragraph(_inline_markdown_to_html(text), style))
            paragraph_buffer = []

    def flush_list():
        nonlocal list_buffer
        if list_buffer:
            item_style = styles["Reference"] if in_references else styles["Body"]
            if in_references:
                # References render as a flat hanging-indent list, no bullets.
                for item in list_buffer:
                    story.append(Paragraph(_inline_markdown_to_html(item), item_style))
            else:
                items = [
                    ListItem(Paragraph(_inline_markdown_to_html(item), item_style), leftIndent=6)
                    for item in list_buffer
                ]
                story.append(ListFlowable(items, bulletType="bullet", leftIndent=16,
                                          bulletColor=ACCENT, bulletFontSize=7))
                story.append(Spacer(1, 0.03 * inch))
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
                story.append(Preformatted(text, styles["Code"]))
                story.append(Spacer(1, 0.06 * inch))
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

            # The first heading (any level) is the paper title — already
            # rendered in the front matter, so skip it to avoid duplication
            # and to keep section numbering starting at the real Section 1.
            if not seen_first_heading and level <= 2 and not in_references:
                seen_first_heading = True
                idx += 1
                continue
            seen_first_heading = True

            if level <= 2 and not in_references:
                # Auto-number top-level sections unless they're already numbered.
                if not re.match(r"^\s*\d+[.\)]", raw):
                    section_counter += 1
                    heading_text = f"{section_counter}.&nbsp;&nbsp;{heading_text}"
                story.append(Paragraph(heading_text, styles["Section"]))
                story.append(HRFlowable(width="100%", thickness=0.5, color=RULE,
                                        spaceBefore=2, spaceAfter=5))
            elif in_references:
                story.append(Spacer(1, 0.05 * inch))
                story.append(Paragraph(heading_text, styles["Section"]))
                story.append(HRFlowable(width="100%", thickness=0.5, color=RULE,
                                        spaceBefore=2, spaceAfter=5))
            else:
                story.append(Paragraph(heading_text, styles["Subsection"]))
            idx += 1
            continue

        image = _image_from_markdown(stripped, resource_dir)
        if image is not None:
            flush_paragraph(); flush_list(); flush_table()
            story.append(Spacer(1, 0.05 * inch))
            story.append(image)
            idx += 1
            if idx < len(lines) and lines[idx].strip().startswith("*Figure:"):
                caption = lines[idx].strip().strip("*")
                story.append(Paragraph(_inline_markdown_to_html(caption), styles["Caption"]))
                idx += 1
            else:
                story.append(Spacer(1, 0.06 * inch))
            continue

        if re.match(r"^\s*[-*]\s+", line):
            flush_paragraph(); flush_table()
            list_buffer.append(re.sub(r"^\s*[-*]\s+", "", line).strip())
            idx += 1
            continue

        # Numbered list items inside a references section → treat as references.
        if in_references and re.match(r"^\s*\d+[.\)]\s+", line):
            flush_paragraph(); flush_table()
            list_buffer.append(re.sub(r"^\s*\d+[.\)]\s+", "", line).strip())
            idx += 1
            continue

        if stripped.startswith(">"):
            flush_paragraph(); flush_list(); flush_table()
            story.append(Paragraph(_inline_markdown_to_html(stripped.lstrip("> ").strip()),
                                   styles["Caption"]))
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
        # First markdown heading (H1 or H2) is the paper title.
        m = re.match(r"^#{1,2}\s+(.*)$", stripped)
        if m and m.group(1).strip():
            return m.group(1).strip()
        # Some drafts lead with **Bold Title**
        m2 = re.match(r"^\*\*(.+?)\*\*$", stripped)
        if m2 and len(m2.group(1)) > 8:
            return m2.group(1).strip()
        # Stop scanning once real body content begins.
        if stripped and not stripped.startswith(("#", "*", ">", "|")):
            break
    return str(fallback or "Research Paper").strip()


def _page_decor(canvas, doc):
    canvas.saveState()
    canvas.setFont(SANS, 8)
    canvas.setFillColor(MUTED)
    # Minimal footer: centred page number only — like a real manuscript.
    canvas.drawCentredString(A4[0] / 2.0, 0.42 * inch, str(doc.page))
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
    """Render the final manuscript as a clean, journal-style PDF.

    Only the final paper is typeset — the internal analysis dossier and any
    session/tool metadata are intentionally omitted so the output reads as a
    standalone, publication-quality document.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    styles = _build_styles()

    manuscript = str(final_markdown or "").strip() or str(analysis_markdown or "").strip()
    title = _extract_title(manuscript, topic)

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
        topMargin=0.8 * inch,
        bottomMargin=0.75 * inch,
        title=title,
        author="Delve",
    )

    story: list = [
        Paragraph(_inline_markdown_to_html(title), styles["PaperTitle"]),
        HRFlowable(width="38%", thickness=1.1, color=ACCENT, spaceBefore=2, spaceAfter=14,
                   hAlign="CENTER"),
    ]
    story.extend(_markdown_to_story(manuscript, resource_dir, styles))
    doc.build(story, onFirstPage=_page_decor, onLaterPages=_page_decor)
    return out_path
