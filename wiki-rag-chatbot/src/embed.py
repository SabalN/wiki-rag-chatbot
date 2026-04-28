"""
Embedding and FAISS index construction for the RAG chatbot.

Loads chunks produced by ingest.py, embeds them with a sentence-transformer
model, and persists a FAISS index alongside the chunk metadata for fast
nearest-neighbour retrieval at query time.

Author: Sabal Nemkul
Module: COS6031-D Applied AI Professional (Element 3 portfolio)
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INDEX_DIR = DATA_DIR / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# 'all-MiniLM-L6-v2' — 384-dim, ~80MB, broadly used baseline.
# Trades a small amount of retrieval quality for fast local CPU inference.
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def build_index(chunks: list[dict], model_name: str = DEFAULT_MODEL,
                batch_size: int = 64) -> tuple[faiss.Index, np.ndarray]:
    """Embed all chunks and build a FAISS L2 index.

    Inner product on L2-normalised vectors is equivalent to cosine similarity,
    so we normalise and use IndexFlatIP for cosine retrieval semantics.
    """
    print(f"[model] Loading {model_name}…")
    model = SentenceTransformer(model_name)

    texts = [c["text"] for c in chunks]
    print(f"[embed] Embedding {len(texts)} chunks…")
    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # so inner product = cosine similarity
    ).astype(np.float32)
    elapsed = time.time() - t0
    print(f"[embed] Done in {elapsed:.1f}s — shape {embeddings.shape}")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # exact inner-product search
    index.add(embeddings)
    print(f"[faiss] Index size: {index.ntotal} vectors @ {dim} dims")
    return index, embeddings


def save_index(index: faiss.Index, embeddings: np.ndarray,
               chunks: list[dict], model_name: str,
               out_dir: Path = INDEX_DIR) -> None:
    """Persist the FAISS index, the embedding matrix, the chunk metadata,
    and a small manifest so we know which model produced this index."""
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "faiss.index"))
    np.save(out_dir / "embeddings.npy", embeddings)
    with open(out_dir / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    manifest = {
        "model": model_name,
        "dim": int(embeddings.shape[1]),
        "n_chunks": len(chunks),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n[done] Saved index, embeddings, chunks, and manifest → {out_dir}")


def load_index(index_dir: Path = INDEX_DIR
               ) -> tuple[faiss.Index, list[dict], dict]:
    """Reload a previously saved index for query-time use."""
    index = faiss.read_index(str(index_dir / "faiss.index"))
    with open(index_dir / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    manifest = json.loads((index_dir / "manifest.json").read_text())
    return index, chunks, manifest


def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from chunks.")
    parser.add_argument("--chunks", default=str(DATA_DIR / "chunks.json"),
                        help="Input chunks JSON (output of ingest.py)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Sentence-transformer model (default: {DEFAULT_MODEL})")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out-dir", default=str(INDEX_DIR))
    args = parser.parse_args()

    chunks = load_chunks(Path(args.chunks))
    if not chunks:
        print("[error] No chunks found. Run ingest.py first.")
        return
    print(f"[load] {len(chunks)} chunks from {args.chunks}")

    index, embeddings = build_index(chunks, args.model, args.batch_size)
    save_index(index, embeddings, chunks, args.model, Path(args.out_dir))


if __name__ == "__main__":
    main()
