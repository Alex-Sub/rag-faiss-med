from __future__ import annotations

import os
import json
import orjson
import faiss
from sentence_transformers import SentenceTransformer

ROOT = os.path.dirname(os.path.dirname(__file__))

# ВАЖНО: у тебя FAISS артефакты лежат в ASCII-пути (обход Unicode)
VSTORE = r"D:\faiss_store\rag-faiss-med"
INDEX_PATH = os.path.join(VSTORE, "index.faiss")
META_PATH = os.path.join(VSTORE, "meta.json")

CHUNKS_PATH = os.path.join(ROOT, "processed", "chunks.jsonl")

TOPK = 5
PREVIEW_CHARS = 900  # сколько символов текста показывать

def load_chunks_map() -> dict[str, str]:
    """id -> text"""
    m = {}
    with open(CHUNKS_PATH, "rb") as f:
        for line in f:
            rec = orjson.loads(line)
            m[rec["id"]] = rec["text"]
    return m

def main():
    # Предохранители, чтобы не падать "по-глупому"
    if not os.path.exists(META_PATH) or not os.path.exists(INDEX_PATH):
        print("No FAISS store found.")
        print("Expected:")
        print(" ", INDEX_PATH)
        print(" ", META_PATH)
        print("Run: python src/build_chunks.py then python src/build_index.py")
        return

    if not os.path.exists(CHUNKS_PATH):
        print("No chunks file found:", CHUNKS_PATH)
        print("Run: python src/build_chunks.py")
        return

    with open(META_PATH, "r", encoding="utf-8") as f:
        store = json.load(f)

    # chunks id->text
    chunks_map = load_chunks_map()

    model = SentenceTransformer(store["model"])
    index = faiss.read_index(INDEX_PATH)

    while True:
        q = input("QUERY> ").strip()
        if not q:
            break

        qv = model.encode([q], normalize_embeddings=True).astype("float32")
        D, I = index.search(qv, TOPK)

        print("\n=== TOP RESULTS ===")
        for rank, (score, idx) in enumerate(zip(D[0], I[0]), start=1):
            rec_id = store["ids"][idx]
            m = store["meta"][idx]

            cite = f'{m["source_file"]}' + (f', стр. {m["page"]}' if m["page"] else "")
            text = chunks_map.get(rec_id, "")
            preview = (text[:PREVIEW_CHARS] + "…") if len(text) > PREVIEW_CHARS else text

            print(f"\n#{rank}  score={score:.4f}")
            print(f"CITE: {cite}")
            print(preview)

if __name__ == "__main__":
    main()
