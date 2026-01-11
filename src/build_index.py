from __future__ import annotations
import os, json
import numpy as np
import orjson
import faiss
from sentence_transformers import SentenceTransformer
import ctypes
from ctypes import wintypes

ROOT = os.path.dirname(os.path.dirname(__file__))
CHUNKS = os.path.join(ROOT, "processed", "chunks.jsonl")
VSTORE = os.path.join(ROOT, "vector_store")
INDEX_PATH = os.path.join(VSTORE, "index.faiss")
META_PATH = os.path.join(VSTORE, "meta.json")

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # 384 dim :contentReference[oaicite:2]{index=2}
def get_short_path(path: str) -> str:
    """Return Windows 8.3 short path to avoid Unicode issues in some native libs."""
    GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
    GetShortPathNameW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
    GetShortPathNameW.restype = wintypes.DWORD

    buf = ctypes.create_unicode_buffer(4096)
    res = GetShortPathNameW(path, buf, 4096)
    if res == 0:
        # Если 8.3 имена отключены или не получилось — вернём исходный путь
        return path
    return buf.value

def main():
    os.makedirs(VSTORE, exist_ok=True)

    model = SentenceTransformer(MODEL_NAME)

    ids = []
    texts = []
    meta = []

    with open(CHUNKS, "rb") as f:
        for line in f:
            rec = orjson.loads(line)
            ids.append(rec["id"])
            texts.append(rec["text"])
            meta.append({
                "id": rec["id"],
                "source_file": rec["source_file"],
                "page": rec["page"],
                "type": rec["type"],
            })

    # embeddings
    emb = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,  # важно для cosine через IP
    ).astype("float32")

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    faiss.write_index(index, get_short_path(INDEX_PATH))
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({"model": MODEL_NAME, "ids": ids, "meta": meta}, f, ensure_ascii=False)

    print("OK.")
    print("Index:", INDEX_PATH)
    print("Meta :", META_PATH)
    print("Vectors:", index.ntotal, "Dim:", dim)

if __name__ == "__main__":
    main()
