"""Markdown -> .docx for the progress notes, no external binary needed.

  python scripts/tools/md_to_docx.py docs/results_feedback_20261105.md

Handles the subset these notes actually use: h1-h3, paragraphs, bullet and
numbered lists, pipe tables, fenced code, blockquotes, and images
(`![alt](path)` on its own line, path relative to the repo root). Tables become
real Word tables with a header row; images are scaled to the text width.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

REPO_ROOT = Path(__file__).resolve().parents[2]
ACCENT = RGBColor(0x1F, 0x4E, 0x79)
MUTED = RGBColor(0x5A, 0x5A, 0x5A)


def _set_code_font(run) -> None:
    run.font.name = "Consolas"
    run.font.size = Pt(9)
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Consolas")


def _shade(cell, hex_fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), hex_fill)
    tc_pr.append(shd)


def _inline(paragraph, text: str, bold: bool = False) -> None:
    """Render **bold**, `code` and plain runs, including code nested in bold."""
    for token in re.split(r"(\*\*.+?\*\*)", text):
        if not token:
            continue
        if token.startswith("**") and token.endswith("**") and len(token) > 4:
            _inline(paragraph, token[2:-2], bold=True)
            continue
        for part in re.split(r"(`[^`]+`|\*[^*]+\*)", token):
            if not part:
                continue
            if part.startswith("`") and part.endswith("`") and len(part) > 2:
                run = paragraph.add_run(part[1:-1])
                _set_code_font(run)
            elif part.startswith("*") and part.endswith("*") and len(part) > 2:
                run = paragraph.add_run(part[1:-1])
                run.italic = True
            else:
                run = paragraph.add_run(part)
            run.bold = bold


def _table(doc: Document, rows: list[list[str]]) -> None:
    header, body = rows[0], [r for r in rows[1:] if not set("".join(r)) <= {"-", " ", ":"}]
    table = doc.add_table(rows=1, cols=len(header))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, cell_text in enumerate(header):
        cell = table.rows[0].cells[i]
        cell.text = ""
        para = cell.paragraphs[0]
        _inline(para, cell_text)
        for run in para.runs:
            run.bold = True
        _shade(cell, "E8EEF5")
    for raw in body:
        cells = table.add_row().cells
        for i, cell_text in enumerate(raw[: len(header)]):
            cells[i].text = ""
            _inline(cells[i].paragraphs[0], cell_text)
            for run in cells[i].paragraphs[0].runs:
                run.font.size = Pt(9)
    for row in table.rows:
        for cell in row.cells:
            if not cell.paragraphs[0].runs:
                continue
    doc.add_paragraph()


def _image(doc: Document, alt: str, rel: str) -> None:
    path = (REPO_ROOT / rel).resolve()
    if not path.is_file():
        para = doc.add_paragraph()
        run = para.add_run(f"[missing image: {rel}]")
        run.italic = True
        run.font.color.rgb = MUTED
        return
    doc.add_picture(str(path), width=Inches(6.3))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    if alt and not alt.lower().startswith("image"):
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = cap.add_run(alt)
        run.italic = True
        run.font.size = Pt(9)
        run.font.color.rgb = MUTED


def convert(md_path: Path, out_path: Path | None = None) -> Path:
    lines = md_path.read_text(encoding="utf-8").splitlines()
    doc = Document()

    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10.5)
    style.paragraph_format.space_after = Pt(6)

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            i += 1
            block: list[str] = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            para = doc.add_paragraph()
            para.paragraph_format.left_indent = Inches(0.22)
            para.paragraph_format.space_before = Pt(4)
            para.paragraph_format.space_after = Pt(8)
            para.add_run("\n".join(block))
            for run in para.runs:
                _set_code_font(run)
            _shade_paragraph(para, "F4F6F8")
            i += 1
            continue

        if stripped.startswith("|"):
            rows: list[list[str]] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                rows.append(cells)
                i += 1
            if rows:
                _table(doc, rows)
            continue

        m = re.match(r"!\[(.*?)\]\((.*?)\)", stripped)
        if m:
            _image(doc, m.group(1), m.group(2))
            i += 1
            continue

        if stripped.startswith("# "):
            h = doc.add_heading(stripped[2:].strip(), level=1)
            for run in h.runs:
                run.font.color.rgb = ACCENT
        elif stripped.startswith("## "):
            h = doc.add_heading(stripped[3:].strip(), level=2)
            for run in h.runs:
                run.font.color.rgb = ACCENT
        elif stripped.startswith("### "):
            h = doc.add_heading(stripped[4:].strip(), level=3)
            for run in h.runs:
                run.font.color.rgb = ACCENT
        elif stripped.startswith("---") and set(stripped) <= {"-"}:
            doc.add_paragraph()
        elif re.match(r"^\s*[-*] ", line):
            para = doc.add_paragraph(style="List Bullet")
            _inline(para, re.sub(r"^\s*[-*] ", "", line).strip())
        elif re.match(r"^\s*\d+\.\s", line):
            para = doc.add_paragraph(style="List Number")
            _inline(para, re.sub(r"^\s*\d+\.\s+", "", line).strip())
        elif stripped.startswith("> "):
            para = doc.add_paragraph()
            para.paragraph_format.left_indent = Inches(0.3)
            _inline(para, stripped[2:])
            for run in para.runs:
                run.italic = True
                run.font.color.rgb = MUTED
        elif stripped:
            para = doc.add_paragraph()
            _inline(para, stripped)
        i += 1

    out = out_path or md_path.with_suffix(".docx")
    doc.save(str(out))
    return out


def _shade_paragraph(paragraph, hex_fill: str) -> None:
    p_pr = paragraph._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), hex_fill)
    p_pr.append(shd)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    print(convert(src, dst))
