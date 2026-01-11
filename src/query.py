from __future__ import annotations
import os, json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import ctypes
from ctypes import wintypes

ROOT = os.path.dirname(os.path.dirname(__file__))
VSTORE = r"D:\faiss_store\rag-faiss-med"
INDEX_PATH = os.path.join(VSTORE, "index.faiss")
META_PATH = os.path.join(VSTORE, "meta.json")

TOPK = 5
def get_short_path(path: str) -> str:
    """Return Windows 8.3 short path to avoid Unicode issues in some native libs."""
    GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
    GetShortPathNameW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
    GetShortPathNameW.restype = wintypes.DWORD

    buf = ctypes.create_unicode_buffer(4096)
    res = GetShortPathNameW(path, buf, 4096)
    if res == 0:
        # If short names are disabled / conversion failed, fall back to original
        return path
    return buf.value

def main():
    q = input("QUERY> ").strip()
    if not q:
        return

    with open(META_PATH, "r", encoding="utf-8") as f:
        store = json.load(f)

    model = SentenceTransformer(store["model"])
    index = faiss.read_index(get_short_path(INDEX_PATH))

    qv = model.encode([q], normalize_embeddings=True).astype("float32")
    D, I = index.search(qv, TOPK)  # scores, indices

    print("\n=== TOP RESULTS ===")
    for rank, (score, idx) in enumerate(zip(D[0], I[0]), start=1):
        rec_id = store["ids"][idx]
        m = store["meta"][idx]
        cite = f'{m["source_file"]}' + (f', стр. {m["page"]}' if m["page"] else "")
        print(f"\n#{rank}  score={score:.4f}")
        print(f"ID: {rec_id}")
        print(f"CITE: {cite}")
        # текст можно подтянуть по id из chunks.jsonl, но для PoC достаточно меты.
        # если хочешь — добавим быстрый map id->text, чтобы печатать фрагмент целиком.

if __name__ == "__main__":
    main()
