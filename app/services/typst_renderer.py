"""
ResearchAgent Typst Publication Renderer
─────────────────────────────────────────
High-performance academic document compiler supporting IEEE, ACM, APA 7th, and MLA 9th.
Produces publication-grade PDFs with authentic geometry, typography, and CSL bibliographies.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any, Optional

import typst

from app.services.citation_engine import generate_bibtex_database
from app.services.format_profiles import FormatProfile, get_profile
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


def _escape_typst_text(text: str, valid_labels: set[str] | None = None) -> str:
    """Escape Typst syntax characters in plain text prose."""
    s = str(text or "")
    s = s.replace("\\", "\\\\")
    s = s.replace("#", "\\#")
    s = s.replace("$", "\\$")

    if valid_labels is not None:
        def _check_at(m: re.Match) -> str:
            lbl = m.group(1)
            if lbl in valid_labels:
                return f"@{lbl}"
            return f"\\@{lbl}"
        s = re.sub(r"(?<!\\)@([a-zA-Z0-9_\-]+)", _check_at, s)
    else:
        # Default: escape single-character labels like @k, @1, or email symbols
        s = re.sub(r"(?<!\\)@(?=[0-9]|[a-zA-Z0-9_\-\.]+@[a-zA-Z0-9_\-]+|\b[a-zA-Z]\b)", r"\@", s)

    return s


def _clean_latex_to_typst_math(latex: str) -> str:
    """Convert LaTeX math syntax to Typst math syntax."""
    s = str(latex or "").strip()
    # Normalize unescaped tab characters in incoming math string
    s = s.replace("\t", " ")

    # Delimiters
    s = s.replace(r"\left(", "(").replace(r"\right)", ")")
    s = s.replace(r"\left[", "[").replace(r"\right]", "]")
    s = s.replace(r"\left\{", "{").replace(r"\right\}", "}")
    s = s.replace(r"\left|", "|").replace(r"\right|", "|")
    s = s.replace(r"\|", "||")

    # Functions and operators
    s = re.sub(r"\\operatorname\{([^{}]+)\}", r'"\1"', s)
    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1) / (\2)", s)
    s = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", s)
    s = re.sub(r"\\prod(?=[^a-zA-Z]|$)", "product", s)
    s = re.sub(r"\\sum(?=[^a-zA-Z]|$)", "sum", s)
    s = re.sub(r"\\int(?=[^a-zA-Z]|$)", "integral", s)
    s = re.sub(r"\\cdot(?=[^a-zA-Z]|$)", " dot ", s)
    s = re.sub(r"\\times(?=[^a-zA-Z]|$)", " times ", s)
    s = re.sub(r"\\in(?=[^a-zA-Z]|$)", " in ", s)
    s = re.sub(r"\\mid(?=[^a-zA-Z]|$)", " | ", s)
    s = re.sub(r"\\top(?=[^a-zA-Z]|$)", "top", s)
    s = re.sub(r"\\exp(?=[^a-zA-Z]|$)", "exp", s)
    s = re.sub(r"\\log(?=[^a-zA-Z]|$)", "log", s)
    s = re.sub(r"\\ln(?=[^a-zA-Z]|$)", "ln", s)
    s = re.sub(r"\\to(?=[^a-zA-Z]|$)", " arrow ", s)
    s = re.sub(r"\\rightarrow(?=[^a-zA-Z]|$)", " arrow ", s)
    s = re.sub(r"\\leq?(?=[^a-zA-Z]|$)", " <= ", s)
    s = re.sub(r"\\geq?(?=[^a-zA-Z]|$)", " >= ", s)
    s = re.sub(r"\\neq(?=[^a-zA-Z]|$)", " != ", s)
    s = re.sub(r"\\approx(?=[^a-zA-Z]|$)", " approx ", s)
    s = re.sub(r"\\infty(?=[^a-zA-Z]|$)", "oo", s)

    # Greek letters
    greek = [
        "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
        "iota", "kappa", "lambda", "mu", "nu", "xi", "pi", "rho", "sigma",
        "tau", "upsilon", "phi", "chi", "psi", "omega",
        "Alpha", "Beta", "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi",
        "Sigma", "Phi", "Psi", "Omega",
    ]
    for g in greek:
        s = re.sub(rf"\\{g}(?=[^a-zA-Z]|$)", g, s)

    # Clean fonts and tags
    s = re.sub(r"\\mathbf\{([^{}]+)\}", r"bold(\1)", s)
    s = re.sub(r"\\mathbb\{([^{}]+)\}", r"bb(\1)", s)
    s = re.sub(r"\\mathcal\{([^{}]+)\}", r"cal(\1)", s)
    s = re.sub(r"\\text\{([^{}]+)\}", r'"\1"', s)

    # Convert LaTeX brace scoping _{...} and ^{...} to Typst paren scoping _(...) and ^(...)
    for _ in range(3):
        s = re.sub(r"_\{([^{}]+)\}", r"_(\1)", s)
        s = re.sub(r"\^\{([^{}]+)\}", r"^(\1)", s)

    # Strip leftover backslashes
    s = re.sub(r"\\([a-zA-Z]+)", r"\1", s)
    return s.strip()


def _format_markdown_prose_for_typst(text: str, valid_labels: set[str] | None = None) -> str:
    """Format inline markdown (bold, italic, code, math) into Typst syntax."""
    s = str(text or "")

    # Convert [@key] or [@key1; @key2] to @key
    def _clean_cite_brackets(m: re.Match) -> str:
        inner = m.group(1).strip()
        keys = [k.strip().lstrip("@") for k in inner.split(";")]
        cites = [f"@{k}" if (valid_labels is None or k in valid_labels) else f"\\@{k}" for k in keys]
        return " ".join(cites)

    s = re.sub(r"\[(@[a-zA-Z0-9_\-]+(?:;\s*@[a-zA-Z0-9_\-]+)*)\]", _clean_cite_brackets, s)
    
    # 1. Protect display math $$...$$
    dmath: list[str] = []
    def _save_dmath(m: re.Match) -> str:
        idx = len(dmath)
        dmath.append(m.group(1))
        return f"___TYPST_DMATH_{idx}___"
    s = re.sub(r"\$\$(.+?)\$\$", _save_dmath, s, flags=re.DOTALL)

    # 2. Protect inline math $...$
    imath: list[str] = []
    def _save_imath(m: re.Match) -> str:
        idx = len(imath)
        imath.append(m.group(1))
        return f"___TYPST_IMATH_{idx}___"
    s = re.sub(r"(?<!\$)\$(?!\$)([^\$\n]+?)(?<!\$)\$(?!\$)", _save_imath, s)

    # 3. Protect code `...`
    codes: list[str] = []
    def _save_code(m: re.Match) -> str:
        idx = len(codes)
        codes.append(m.group(1))
        return f"___TYPST_CODE_{idx}___"
    s = re.sub(r"`([^`]+)`", _save_code, s)

    # 4. Escape remaining prose
    s = _escape_typst_text(s, valid_labels=valid_labels)

    # 5. Convert bold **text**
    s = re.sub(r"\*\*([^*]+)\*\*", r"#strong[\1]", s)

    # 6. Convert italic *text*
    s = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"#emph[\1]", s)

    # 7. Restore code
    for i, code in enumerate(codes):
        s = s.replace(f"___TYPST_CODE_{i}___", f"`{code}`")

    # 8. Restore inline math
    for i, m in enumerate(imath):
        typ_m = _clean_latex_to_typst_math(m)
        s = s.replace(f"___TYPST_IMATH_{i}___", f"$ {typ_m} $")

    # 9. Restore display math
    for i, m in enumerate(dmath):
        typ_m = _clean_latex_to_typst_math(m)
        s = s.replace(f"___TYPST_DMATH_{i}___", f"\n$ {typ_m} $\n")

    return s


class TypstDocumentBuilder:
    """Builds a complete, valid .typ document for a specific format profile."""

    def __init__(self, doc: ManuscriptDocument, profile: FormatProfile):
        self.doc = doc
        self.profile = profile
        self.lines: list[str] = []
        self.valid_labels = {src.citation_key for src in self.doc.references} | {
            "fig1", "fig2", "fig3", "tab1", "tab2", "tab3", "eq1", "eq2", "eq3"
        }

    def build_typst_source(self, bib_filename: str = "refs.bib") -> str:
        self.lines = []
        self._emit_page_geometry()
        self._emit_typography_rules()
        self._emit_title_and_authors()
        self._emit_body_content()
        self._emit_bibliography(bib_filename)
        return "\n".join(self.lines)

    def _emit_page_geometry(self) -> None:
        geom = self.profile.geometry
        top_in = geom.margin_top_pt / 72.0
        bot_in = geom.margin_bottom_pt / 72.0
        left_in = geom.margin_left_pt / 72.0
        right_in = geom.margin_right_pt / 72.0

        if self.profile.name == "apa":
            header_str = (
                f'#set page(paper: "{geom.page_size}", '
                f'margin: (top: {top_in}in, bottom: {bot_in}in, left: {left_in}in, right: {right_in}in), '
                f'header: [#text(size: 9pt)[{self.doc.running_title.upper()} #h(1fr) #context counter(page).display()]])'
            )
        elif self.profile.name == "mla":
            author_last = "Author"
            if self.doc.authors and self.doc.authors[0].name:
                author_last = self.doc.authors[0].name.split()[-1]
            header_str = (
                f'#set page(paper: "{geom.page_size}", '
                f'margin: (top: {top_in}in, bottom: {bot_in}in, left: {left_in}in, right: {right_in}in), '
                f'header: align(right)[#text(size: 10pt)[{author_last} #context counter(page).display()]])'
            )
        else:
            header_str = (
                f'#set page(paper: "{geom.page_size}", '
                f'margin: (top: {top_in}in, bottom: {bot_in}in, left: {left_in}in, right: {right_in}in))'
            )
        self.lines.append(header_str)

    def _emit_typography_rules(self) -> None:
        typo = self.profile.typography
        body_font = f'"{typo.body_font}"'
        if self.profile.name == "acm":
            body_font = '("Linux Libertine", "Times New Roman")'

        self.lines.append(f'#set text(font: {body_font}, size: {typo.body_size_pt}pt)')
        
        justify_bool = "true" if typo.text_align == "justify" else "false"
        self.lines.append(f'#set par(justify: {justify_bool}, leading: {typo.line_spacing * 0.45}em)')
        self.lines.append('#set math.equation(numbering: "(1)")')

        # Figure and Table supplements
        if self.profile.name == "ieee":
            self.lines.append('#show figure.where(kind: image): set figure(supplement: [Fig.])')
            self.lines.append('#show figure.where(kind: table): set figure(supplement: [TABLE])')
            self.lines.append('#show figure.where(kind: table): set figure.caption(position: top)')
        elif self.profile.name == "acm":
            self.lines.append('#show figure.where(kind: image): set figure(supplement: [Figure])')
            self.lines.append('#show figure.where(kind: table): set figure(supplement: [Table])')
            self.lines.append('#show figure.where(kind: table): set figure.caption(position: top)')

    def _emit_title_and_authors(self) -> None:
        p = self.profile
        doc = self.doc

        if p.name == "ieee":
            self.lines.append("#align(center)[")
            self.lines.append(f'  #text(size: 24pt, weight: "bold")[{doc.title}]')
            self.lines.append("  #v(1.2em)")
            self.lines.append("  #grid(")
            self.lines.append("    columns: (1fr, 1fr),")
            self.lines.append("    gutter: 1.5em,")
            self.lines.append("    align: center,")
            author1 = doc.authors[0] if doc.authors else AuthorAffiliation()
            author2 = doc.authors[1] if len(doc.authors) > 1 else AuthorAffiliation(
                name="ResearchAgent Synthesis Core",
                department="Autonomous Deliberation Group",
                institution="Delve Academic Research"
            )
            self.lines.append(f'    [{author1.name}\\ #text(size: 9pt, style: "italic")[{author1.department}\\ {author1.institution}]],')
            self.lines.append(f'    [{author2.name}\\ #text(size: 9pt, style: "italic")[{author2.department}\\ {author2.institution}]]')
            self.lines.append("  )")
            self.lines.append("]")
            self.lines.append("#v(1.2em)")

        elif p.name == "acm":
            self.lines.append("#align(left)[")
            self.lines.append(f'  #text(font: ("Linux Biolinum", "Helvetica", "Arial"), size: 18pt, weight: "bold")[{doc.title}]')
            self.lines.append("  #v(0.8em)")
            self.lines.append("  #grid(")
            self.lines.append("    columns: (1fr, 1fr),")
            self.lines.append("    gutter: 1.5em,")
            author1 = doc.authors[0] if doc.authors else AuthorAffiliation()
            author2 = doc.authors[1] if len(doc.authors) > 1 else AuthorAffiliation()
            self.lines.append(f'    [#strong[{author1.name}]\\ ACM Senior Member\\ {author1.institution}],')
            self.lines.append(f'    [#strong[{author2.name}]\\ {author2.department}\\ {author2.institution}]')
            self.lines.append("  )")
            self.lines.append("]")
            self.lines.append("#v(1.2em)")

        elif p.name == "apa":
            self.lines.append("#v(2.0in)")
            self.lines.append("#align(center)[")
            self.lines.append(f'  #text(weight: "bold", size: 14pt)[{doc.title}]')
            self.lines.append("  #v(1.8em)")
            authors_str = " and ".join(a.name for a in doc.authors) if doc.authors else "ResearchAgent Synthesis Engine"
            self.lines.append(f'  #text(size: 11.5pt)[{authors_str}]')
            self.lines.append("  #v(0.6em)")
            inst_str = doc.authors[0].institution if doc.authors else "Autonomous Deliberation Center"
            self.lines.append(f'  #text(size: 10.5pt, style: "italic")[{inst_str}]')
            self.lines.append("  #v(2.0em)")
            self.lines.append('  #text(size: 10pt)[*Author Note*\\ Delve Multi-Agent Research Platform]')
            self.lines.append("]")
            self.lines.append("#pagebreak()")

            # Page 2: Abstract
            self.lines.append("#align(center)[#text(weight: \"bold\")[Abstract]]")
            self.lines.append(f"#set par(first-line-indent: 0pt)")
            self.lines.append(_format_markdown_prose_for_typst(doc.abstract))
            if doc.keywords:
                kw_str = ", ".join(doc.keywords)
                self.lines.append(f'\n#v(1em)\n#h(0.5in) #text(style: "italic")[Keywords:] {kw_str}')
            self.lines.append("#pagebreak()")

            # Page 3: Body begins with title
            self.lines.append(f'#align(center)[#text(weight: "bold")[{doc.title}]]')
            self.lines.append('#set par(first-line-indent: 0.5in)')
            self.lines.append("#v(1em)")

        elif p.name == "mla":
            author_name = doc.authors[0].name if doc.authors else "ResearchAgent"
            self.lines.append(f"{author_name}\\")
            self.lines.append("Dr. Research Reviewer\\")
            self.lines.append("Department of Academic Intelligence\\")
            pub_date = doc.publication_date or "October 2026"
            self.lines.append(f"{pub_date}")
            self.lines.append("#v(1.2em)")
            self.lines.append(f"#align(center)[{doc.title}]")
            self.lines.append("#v(1.2em)")
            self.lines.append('#set par(first-line-indent: 0.5in)')

    def _emit_body_content(self) -> None:
        p = self.profile
        doc = self.doc

        # Two-column wrapper for IEEE and ACM
        if p.geometry.columns == 2:
            gutter_in = p.geometry.column_gutter_pt / 72.0
            self.lines.append(f"#columns(2, gutter: {gutter_in}in)[")

            # IEEE Abstract & Keywords at top of two columns
            if p.name == "ieee":
                abs_content = _format_markdown_prose_for_typst(doc.abstract)
                self.lines.append(f'#text(weight: "bold", style: "italic")[Abstract]---#text(weight: "bold")[{abs_content}]')
                if doc.keywords:
                    kw_str = ", ".join(doc.keywords)
                    self.lines.append(f'\n#v(0.6em)\n#text(weight: "bold", style: "italic")[Index Terms]---#text(style: "italic")[{kw_str}]')
                self.lines.append("\n#v(1em)\n")

            elif p.name == "acm":
                abs_content = _format_markdown_prose_for_typst(doc.abstract)
                self.lines.append('#text(font: ("Linux Biolinum", "Helvetica"), weight: "bold")[ABSTRACT]\\')
                self.lines.append(abs_content)
                if doc.ccs_concepts:
                    ccs_str = "; ".join(doc.ccs_concepts)
                    self.lines.append(f'\n#v(0.5em)\n#text(font: ("Linux Biolinum", "Helvetica"), weight: "bold")[CCS CONCEPTS]\\')
                    self.lines.append(f"• {ccs_str}")
                if doc.keywords:
                    kw_str = ", ".join(doc.keywords)
                    self.lines.append(f'\n#v(0.5em)\n#text(font: ("Linux Biolinum", "Helvetica"), weight: "bold")[KEYWORDS]\\')
                    self.lines.append(kw_str)
                # ACM Reference format
                self.lines.append('\n#v(0.6em)\n#rect(width: 100%, stroke: 0.4pt + luma(180), inset: 6pt)[')
                self.lines.append(f'  #text(size: 7.5pt)[*ACM Reference Format:*\\ {doc.authors[0].name if doc.authors else "Author"}. 2026. {doc.title}. In *Proceedings of the Autonomous Academic Synthesis Conference (Delve \'26)*.]')
                self.lines.append(']\n#v(0.8em)\n')

        # Sections
        for sec in doc.sections:
            self._emit_section(sec)

    def _emit_section(self, sec: Section) -> None:
        p = self.profile
        level = sec.level
        title = sec.title

        if p.name == "ieee":
            if level == 1:
                # Roman numeral heading
                roman_num = sec.number if sec.number and sec.number.isupper() else _to_roman(sec.id)
                self.lines.append(f'\n#align(center)[#text(size: 10pt)[#smallcaps[{roman_num}. {title}]]]\n#v(0.4em)')
            elif level == 2:
                self.lines.append(f'\n#text(size: 10pt, style: "italic")[_{title}_]\n#v(0.3em)')
            else:
                self.lines.append(f'\n#text(size: 9.5pt, style: "italic")[_{title}:_] ')

        elif p.name == "acm":
            if level == 1:
                self.lines.append(f'\n= {title.upper()}\n')
            elif level == 2:
                self.lines.append(f'\n== {title}\n')
            else:
                self.lines.append(f'\n=== {title}\n')

        elif p.name == "apa":
            if level == 1:
                self.lines.append(f'\n#align(center)[#text(weight: "bold")[{title}]]\n')
            elif level == 2:
                self.lines.append(f'\n#text(weight: "bold")[{title}]\n')
            else:
                self.lines.append(f'\n#text(weight: "bold", style: "italic")[{title}]\n')

        elif p.name == "mla":
            if level == 1:
                self.lines.append(f'\n#text(weight: "bold")[{title}]\n')
            else:
                self.lines.append(f'\n#text(style: "italic")[{title}]\n')

        # Emit blocks in section
        for blk in sec.blocks:
            self._emit_block(blk)

    def _emit_block(self, blk: SemanticBlock) -> None:
        if blk.type == "paragraph" and blk.content:
            text = _format_markdown_prose_for_typst(blk.content, valid_labels=self.valid_labels)
            self.lines.append(f"{text}\n")

        elif blk.type == "equation" and blk.equation:
            eq = blk.equation
            typ_math = _clean_latex_to_typst_math(eq.latex)
            label_tag = f" <{eq.label or eq.id}>" if (eq.label or eq.id) else ""
            self.lines.append(f"\n$ {typ_math} ${label_tag}\n")

        elif blk.type == "figure" and blk.figure:
            fig = blk.figure
            caption = _escape_typst_text(fig.caption, valid_labels=self.valid_labels)
            img_filename = Path(fig.image_path).name
            label_tag = f" <{fig.label or fig.id}>" if (fig.label or fig.id) else ""
            self.lines.append(f'\n#figure(\n  image("{img_filename}", width: 95%),\n  caption: [{caption}]\n){label_tag}\n')

        elif blk.type == "table" and blk.table:
            self._emit_table(blk.table)

        elif blk.type == "list_bullet":
            for it in blk.items:
                clean_it = _format_markdown_prose_for_typst(it, valid_labels=self.valid_labels)
                self.lines.append(f"- {clean_it}")
            self.lines.append("")

        elif blk.type == "list_numbered":
            for it in blk.items:
                clean_it = _format_markdown_prose_for_typst(it, valid_labels=self.valid_labels)
                self.lines.append(f"+ {clean_it}")
            self.lines.append("")

        elif blk.type == "quote" and blk.content:
            clean_q = _format_markdown_prose_for_typst(blk.content, valid_labels=self.valid_labels)
            self.lines.append(f"#quote(block: true)[{clean_q}]\n")

    def _emit_table(self, tbl: TableBlock) -> None:
        if not tbl.headers:
            return
        caption = _escape_typst_text(tbl.caption, valid_labels=self.valid_labels)
        num_cols = len(tbl.headers)
        col_spec = ", ".join(["1fr"] * num_cols)
        label_tag = f" <{tbl.label or tbl.id}>" if (tbl.label or tbl.id) else ""

        self.lines.append(f"#figure(\n  table(\n    columns: ({col_spec}),\n    stroke: (x, y) => if y == 0 {{ (bottom: 0.8pt + black, top: 1pt + black) }} else if y == 1 {{ (bottom: 0.5pt + black) }} else {{ none }},")
        
        # Headers
        header_cells = [f"[#strong[{_escape_typst_text(h, valid_labels=self.valid_labels)}]]" for h in tbl.headers]
        self.lines.append(f"    table.header({', '.join(header_cells)}),")
        
        # Rows
        for row in tbl.rows:
            row_cells = [f"[{_format_markdown_prose_for_typst(c, valid_labels=self.valid_labels)}]" for c in row]
            self.lines.append(f"    {', '.join(row_cells)},")
            
        self.lines.append(f"  ),\n  caption: [{caption}]\n){label_tag}\n")

    def _emit_bibliography(self, bib_filename: str) -> None:
        p = self.profile
        if p.name in {"apa", "mla"}:
            self.lines.append("#pagebreak()")
        
        csl = p.citation.csl_style
        heading_title = p.citation.bibliography_heading
        self.lines.append(f'#bibliography("{bib_filename}", title: [{heading_title}], style: "{csl}")')

        # Close two-column block if opened
        if p.geometry.columns == 2:
            self.lines.append("]")


def _to_roman(val: Any) -> str:
    """Helper to convert integer or string to Roman numerals."""
    try:
        n = int(str(val).split("-")[-1].replace("sec", ""))
    except Exception:
        n = 1
    num_map = [(10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    res = ""
    for v, r in num_map:
        while n >= v:
            res += r
            n -= v
    return res or "I"


def compile_manuscript_to_pdf(
    doc: ManuscriptDocument,
    out_path: Path,
    resource_dir: Path,
) -> Path:
    """
    Compiles a ManuscriptDocument into a publication-grade PDF using Typst.
    Generates accompanying .bib file and executes compilation in milliseconds.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = get_profile(doc.publication_style)
    
    # 1. Generate refs.bib
    bib_path = resource_dir / "refs.bib"
    bib_content = generate_bibtex_database(doc.references)
    bib_path.write_text(bib_content, encoding="utf-8")

    # 2. Build .typ source
    builder = TypstDocumentBuilder(doc, profile)
    typ_source = builder.build_typst_source(bib_filename="refs.bib")
    typ_path = resource_dir / f"manuscript_{profile.name}.typ"
    typ_path.write_text(typ_source, encoding="utf-8")

    # 3. Compile via Typst
    pdf_bytes = typst.compile(typ_path, root=resource_dir)
    out_path.write_bytes(pdf_bytes)
    return out_path
