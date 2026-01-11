from __future__ import annotations
import re

def clean_text(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def chunk_text(text: str, max_chars: int = 1000, overlap: int = 200) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    parts = re.split(r"\n\s*\n", text)  # грубо по абзацам
    chunks = []
    buf = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(buf) + len(p) + 2 <= max_chars:
            buf = (buf + "\n\n" + p).strip()
        else:
            if buf:
                chunks.append(buf)
            # новый буфер
            buf = p
    if buf:
        chunks.append(buf)

    # overlap на уровне строк (простая реализация)
    if overlap > 0 and len(chunks) > 1:
        out = [chunks[0]]
        for c in chunks[1:]:
            prev = out[-1]
            tail = prev[-overlap:] if len(prev) > overlap else prev
            out.append(tail + "\n" + c)
        return out
    return chunks
