"""Check a generated .docx for leftover markdown syntax and empty cells."""
from __future__ import annotations

import sys

from docx import Document

MARKERS = ("**", "`", "](", "![")
bad = 0
doc = Document(sys.argv[1] if len(sys.argv) > 1 else "docs/results_feedback_20261105.docx")

for i, p in enumerate(doc.paragraphs):
    if p.style.name.startswith("Heading") or not p.text:
        continue
    hits = [m for m in MARKERS if m in p.text or (m == "`" and "`" in p.text)]
    if hits:
        bad += 1
        print(f"[para {i}] leftover {hits}: {p.text[:90]}")

for ti, table in enumerate(doc.tables):
    for ri, row in enumerate(table.rows):
        for ci, cell in enumerate(row.cells):
            for m in MARKERS:
                if m in cell.text:
                    bad += 1
                    print(f"[table {ti} r{ri}c{ci}] leftover {m}: {cell.text[:70]}")
            if not cell.text.strip():
                print(f"[table {ti} r{ri}c{ci}] empty cell (may be intentional)")

print(f"images={len(doc.inline_shapes)} tables={len(doc.tables)} issues={bad}")
raise SystemExit(1 if bad else 0)
