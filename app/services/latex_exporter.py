"""
ResearchAgent LaTeX Publication Exporter & Compiler
───────────────────────────────────────────────────
Generates official IEEEtran, ACM acmart, APA7, and MLA LaTeX publication sources.
Produces compile-ready .tex and .bib bundles suitable for Overleaf and direct submission.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from app.services.citation_engine import generate_bibtex_database
from app.services.format_profiles import FormatProfile, get_profile
from app.services.manuscript_schema import ManuscriptDocument, Section, SemanticBlock


def _escape_latex(text: str) -> str:
    """Escape LaTeX special characters in plain text."""
    s = str(text or "")
    s = s.replace("&", r"\&").replace("%", r"\%").replace("$", r"\$")
    s = s.replace("#", r"\#").replace("_", r"\_").replace("~", r"\textasciitilde{}")
    s = s.replace("^", r"\textasciicircum{}")
    return s


def generate_latex_source(doc: ManuscriptDocument, profile: FormatProfile) -> str:
    """Generate complete, valid LaTeX source document."""
    lines: list[str] = []
    style = profile.name

    if style == "ieee":
        lines.append(r"\documentclass[conference]{IEEEtran}")
        lines.append(r"\usepackage{cite}")
        lines.append(r"\usepackage{amsmath,amssymb,amsfonts}")
        lines.append(r"\usepackage{algorithmic}")
        lines.append(r"\usepackage{graphicx}")
        lines.append(r"\usepackage{textcomp}")
        lines.append(r"\usepackage{xcolor}")
        lines.append(r"\usepackage{booktabs}")
        lines.append(r"\begin{document}")
        lines.append(f"\\title{{{_escape_latex(doc.title)}}}")
        author1 = doc.authors[0].name if doc.authors else "ResearchAgent Synthesis Engine"
        lines.append(f"\\author{{\\IEEEauthorblockN{{{_escape_latex(author1)}}}\\IEEEauthorblockA{{\\textit{{Autonomous Research Division}} \\\\ Delve Academic Research}}}}")
        lines.append(r"\maketitle")
        lines.append(r"\begin{abstract}")
        lines.append(_escape_latex(doc.abstract))
        lines.append(r"\end{abstract}")
        if doc.keywords:
            lines.append(f"\\begin{{IEEEkeywords}}\n{', '.join(doc.keywords)}\n\\end{{IEEEkeywords}}")

    elif style == "acm":
        lines.append(r"\documentclass[sigconf]{acmart}")
        lines.append(r"\usepackage{booktabs}")
        lines.append(f"\\title{{{_escape_latex(doc.title)}}}")
        author1 = doc.authors[0].name if doc.authors else "ResearchAgent"
        lines.append(f"\\author{{{_escape_latex(author1)}}}")
        lines.append(r"\affiliation{\institution{Delve Research Lab}\country{USA}}")
        lines.append(r"\begin{document}")
        lines.append(r"\begin{abstract}")
        lines.append(_escape_latex(doc.abstract))
        lines.append(r"\end{abstract}")
        if doc.keywords:
            lines.append(f"\\keywords{{{', '.join(doc.keywords)}}}")
        lines.append(r"\maketitle")

    elif style == "apa":
        lines.append(r"\documentclass[man,floatsintext]{apa7}")
        lines.append(r"\usepackage[american]{babel}")
        lines.append(r"\usepackage{csquotes}")
        lines.append(r"\usepackage{booktabs}")
        lines.append(f"\\title{{{_escape_latex(doc.title)}}}")
        lines.append(f"\\shorttitle{{{_escape_latex(doc.running_title[:50])}}}")
        author1 = doc.authors[0].name if doc.authors else "ResearchAgent"
        lines.append(f"\\author{{{_escape_latex(author1)}}}")
        lines.append(r"\affiliation{Delve Autonomous Research Institute}")
        lines.append(r"\begin{document}")
        lines.append(r"\maketitle")
        lines.append(r"\begin{abstract}")
        lines.append(_escape_latex(doc.abstract))
        lines.append(r"\end{abstract}")

    else:  # MLA
        lines.append(r"\documentclass[12pt]{article}")
        lines.append(r"\usepackage[letterpaper,margin=1in]{geometry}")
        lines.append(r"\usepackage{setspace}\doublespacing")
        lines.append(r"\usepackage{fancyhdr}")
        lines.append(r"\usepackage{booktabs}")
        author1 = doc.authors[0].name if doc.authors else "ResearchAgent"
        last_name = author1.split()[-1]
        lines.append(r"\pagestyle{fancy}")
        lines.append(r"\fancyhf{}")
        lines.append(f"\\rhead{{{last_name} \\thepage}}")
        lines.append(r"\renewcommand{\headrulewidth}{0pt}")
        lines.append(r"\begin{document}")
        lines.append(f"{_escape_latex(author1)}\\\\")
        lines.append(r"Dr. Research Reviewer\\")
        lines.append(r"Department of Computational Intelligence\\")
        lines.append(r"\today")
        lines.append(r"\begin{center}")
        lines.append(f"{_escape_latex(doc.title)}")
        lines.append(r"\end{center}")

    # Sections
    for sec in doc.sections:
        if sec.level == 1:
            lines.append(f"\n\\section{{{_escape_latex(sec.title)}}}")
        elif sec.level == 2:
            lines.append(f"\n\\subsection{{{_escape_latex(sec.title)}}}")
        else:
            lines.append(f"\n\\subsubsection{{{_escape_latex(sec.title)}}}")

        for blk in sec.blocks:
            if blk.type == "paragraph" and blk.content:
                # Convert [@key] or @key to \cite{key}
                p_text = re.sub(r"\[@([a-zA-Z0-9_\-]+)\]", r"\\cite{\1}", blk.content)
                p_text = re.sub(r"(?<!\\)@([a-zA-Z0-9_\-]+)", r"\\cite{\1}", p_text)
                lines.append(f"\n{p_text}\n")
            elif blk.type == "equation" and blk.equation:
                lines.append(f"\n\\begin{{equation}}\n{blk.equation.latex}\n\\end{{equation}}\n")

    # Bibliography
    if style == "ieee":
        lines.append(r"\bibliographystyle{IEEEtran}")
        lines.append(r"\bibliography{refs}")
    elif style == "acm":
        lines.append(r"\bibliographystyle{ACM-Reference-Format}")
        lines.append(r"\bibliography{refs}")
    else:
        lines.append(r"\bibliographystyle{plain}")
        lines.append(r"\bibliography{refs}")

    lines.append(r"\end{document}")
    return "\n".join(lines)


def export_latex_bundle(
    doc: ManuscriptDocument,
    out_dir: Path,
) -> Path:
    """Exports compile-ready .tex and refs.bib to the specified directory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    profile = get_profile(doc.publication_style)

    # 1. Write refs.bib
    bib_content = generate_bibtex_database(doc.references)
    (out_dir / "refs.bib").write_text(bib_content, encoding="utf-8")

    # 2. Write main.tex
    tex_content = generate_latex_source(doc, profile)
    tex_path = out_dir / "main.tex"
    tex_path.write_text(tex_content, encoding="utf-8")

    return tex_path
