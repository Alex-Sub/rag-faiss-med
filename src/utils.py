from __future__ import annotations
import re

def clean_text(s: str) -> str:
    if not s:
        return ""

    # базовая нормализация
    s = s.replace("\u00ad", "")  # soft hyphen
    s = s.replace("\xa0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()

    # --- NEW: чистим мусорные строки/подвалы ---
    lines = []
    for line in s.splitlines():
        t = line.strip()
        if not t:
            continue

        # номера страниц
        if re.search(r"^Страница\s+\d+\s+из\s+\d+", t, flags=re.I):
            continue

        # явные рекламные/веб-вставки
        if re.search(r"(http://|https://|www\.)", t, flags=re.I):
            continue

        # частый мусор из “улучшенной верстки”
        if re.search(r"улучшенн(ая|ой)\s+в[её]рстк", t, flags=re.I):
            continue

        lines.append(t)

    s = "\n".join(lines)

    # склейка переносов по дефису (часто в PDF)
    s = re.sub(r"(\w)-\n(\w)", r"\1\2", s)

    # снова нормализуем
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()

    return s

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
