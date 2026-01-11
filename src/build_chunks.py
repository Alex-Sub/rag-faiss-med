from __future__ import annotations

import os
import io
import json
from typing import Iterable

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from bs4 import BeautifulSoup
from docx import Document

from utils import chunk_text, clean_text

ROOT = os.path.dirname(os.path.dirname(__file__))
DOCS_DIR = os.path.join(ROOT, "documents")
OUT_PATH = os.path.join(ROOT, "processed", "chunks.jsonl")

# OCR settings
OCR_LANG = "rus"        # можно "rus+eng"
OCR_DPI = 250           # 200–300 обычно норм
MIN_TEXT_CHARS = 40     # если меньше — считаем страницей-сканом

ALLOWED_EXT = {".pdf", ".html", ".htm", ".txt", ".docx"}
SKIP_EXT = {".zip", ".7z", ".rar", ".exe", ".dll"}

def iter_files(root_dir: str) -> Iterable[str]:
    for root, _, files in os.walk(root_dir):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext in SKIP_EXT:
                continue
            if ext in ALLOWED_EXT:
                yield os.path.join(root, fn)

def html_to_text(path: str) -> str:
    with open(path, "rb") as f:
        soup = BeautifulSoup(f.read(), "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(separator="\n")

def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_docx(path: str) -> str:
    doc = Document(path)
    paras = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text and c.text.strip()]
            if cells:
                paras.append(" | ".join(cells))
    return "\n".join(paras)

def ocr_page_to_text(page: fitz.Page, dpi: int = OCR_DPI) -> str:
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img_bytes = pix.tobytes("png")
    img = Image.open(io.BytesIO(img_bytes))
    text = pytesseract.image_to_string(img, lang=OCR_LANG)
    return text or ""

def write_rec(out_f, rec: dict) -> None:
    out_f.write((json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8"))

def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    files = list(iter_files(DOCS_DIR))
    print(f"Documents root: {DOCS_DIR}")
    print(f"Found eligible files: {len(files)}")

    by_ext = {}
    for p in files:
        ext = os.path.splitext(p)[1].lower()
        by_ext[ext] = by_ext.get(ext, 0) + 1
    print("By type:", ", ".join([f"{k}={v}" for k, v in sorted(by_ext.items())]) or "none")

    count = 0
    ocr_pages = 0
    skipped_pages_total = 0

    with open(OUT_PATH, "wb") as out:
        for path in files:
            fname = os.path.basename(path)
            ext = os.path.splitext(path)[1].lower()

            # PDF: page -> text; if too short -> OCR fallback
            if ext == ".pdf":
                doc = fitz.open(path)
                for i in range(doc.page_count):
                    page = doc.load_page(i)
                    page_num = i + 1

                    txt = clean_text(page.get_text("text") or "")
                    extraction = "text"

                    if len(txt) < MIN_TEXT_CHARS:
                        extraction = "ocr"
                        ocr_pages += 1
                        txt = clean_text(ocr_page_to_text(page))

                    if len(txt) < MIN_TEXT_CHARS:
                        skipped_pages_total += 1
                        continue

                    chunks = chunk_text(txt)
                    for j, ch in enumerate(chunks, start=1):
                        ch = clean_text(ch)
                        if not ch:
                            continue
                        rec = {
                            "id": f"{fname}::p{page_num}::c{j}",
                            "text": ch,
                            "source_file": fname,
                            "page": page_num,
                            "chunk_in_page": j,
                            "type": "pdf",
                            "extraction": extraction,
                        }
                        write_rec(out, rec)
                        count += 1

            # HTML / HTM
            elif ext in (".html", ".htm"):
                txt = clean_text(html_to_text(path))
                if len(txt) < 20:
                    continue
                for j, ch in enumerate(chunk_text(txt), start=1):
                    ch = clean_text(ch)
                    if not ch:
                        continue
                    rec = {
                        "id": f"{fname}::c{j}",
                        "text": ch,
                        "source_file": fname,
                        "page": None,
                        "chunk_in_page": j,
                        "type": "html",
                    }
                    write_rec(out, rec)
                    count += 1

            # TXT
            elif ext == ".txt":
                txt = clean_text(read_txt(path))
                if len(txt) < 20:
                    continue
                for j, ch in enumerate(chunk_text(txt), start=1):
                    ch = clean_text(ch)
                    if not ch:
                        continue
                    rec = {
                        "id": f"{fname}::c{j}",
                        "text": ch,
                        "source_file": fname,
                        "page": None,
                        "chunk_in_page": j,
                        "type": "txt",
                    }
                    write_rec(out, rec)
                    count += 1

            # DOCX
            elif ext == ".docx":
                txt = clean_text(read_docx(path))
                if len(txt) < 20:
                    continue
                for j, ch in enumerate(chunk_text(txt), start=1):
                    ch = clean_text(ch)
                    if not ch:
                        continue
                    rec = {
                        "id": f"{fname}::c{j}",
                        "text": ch,
                        "source_file": fname,
                        "page": None,
                        "chunk_in_page": j,
                        "type": "docx",
                    }
                    write_rec(out, rec)
                    count += 1

    print(f"OK. chunks written: {count}")
    print(f"Output: {OUT_PATH}")
    print(f"OCR pages used: {ocr_pages}")
    if skipped_pages_total:
        print(f"PDF pages still skipped (no text even after OCR): {skipped_pages_total}")

if __name__ == "__main__":
    main()