"""
RAG pipeline: query → retrieve top-k chunks → build prompt → generate.

Wraps the FAISS index and the LLM call into a single class so the Streamlit
app and the notebook can use the same logic.

Author: Sabal Nemkul
Module: COS6031-D Applied AI Professional (Element 3 portfolio)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from embed import load_index, INDEX_DIR


# System prompt: instructs the LLM to answer ONLY from the retrieved context,
# cite the source titles, and refuse if the answer is not in the context.
# Refusing-when-unsure is the most important safety behaviour for a RAG system.
SYSTEM_PROMPT = """You are a helpful assistant that answers questions strictly using the supplied Wikipedia context.

Rules:
1. Use ONLY information present in the supplied context. Do not draw on outside knowledge.
2. If the context does not contain enough information to answer, say so clearly: "I don't have enough information in the provided sources to answer that."
3. Cite the source article titles in your answer where they support a claim, in the form (Source: Article Title).
4. Be concise. 3–6 sentences for most questions; bullet points for lists.
5. If the question is ambiguous, briefly state your interpretation before answering.
"""


@dataclass
class RetrievedChunk:
    chunk_id: str
    source: str
    url: str
    text: str
    score: float


@dataclass
class RAGResponse:
    question: str
    answer: str
    retrieved: list[RetrievedChunk]
    elapsed_seconds: float


# ─────────────────────────────────────────────────────────────────────────────
# Retriever
# ─────────────────────────────────────────────────────────────────────────────
class Retriever:
    """Loads the persisted FAISS index and embedding model, performs top-k
    cosine-similarity retrieval against the chunked corpus."""

    def __init__(self, index_dir: Path = INDEX_DIR):
        self.index, self.chunks, self.manifest = load_index(index_dir)
        self.model = SentenceTransformer(self.manifest["model"])
        print(f"[retriever] Loaded {self.manifest['n_chunks']} chunks · "
              f"model={self.manifest['model']}")

    def retrieve(self, query: str, k: int = 4,
                 min_score: float = 0.0) -> list[RetrievedChunk]:
        q_emb = self.model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)
        scores, indices = self.index.search(q_emb, k)

        out = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or score < min_score:
                continue
            c = self.chunks[idx]
            out.append(RetrievedChunk(
                chunk_id=c["chunk_id"], source=c["source"], url=c["url"],
                text=c["text"], score=float(score),
            ))
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Generator (LLM)
# ─────────────────────────────────────────────────────────────────────────────
class Generator:
    """Wraps the Anthropic API. Built so it's easy to swap for OpenAI or a
    local model — only `generate()` needs to change."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise RuntimeError(
                "anthropic SDK not installed. Run: pip install anthropic"
            ) from e
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Export it or use a .env file."
            )
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def generate(self, question: str,
                 retrieved: list[RetrievedChunk]) -> str:
        if not retrieved:
            return ("I don't have any relevant sources to answer that question. "
                    "The knowledge base may not cover this topic.")

        context = "\n\n".join(
            f"[Source {i+1}: {r.source}]\n{r.text}"
            for i, r in enumerate(retrieved)
        )
        user_message = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer using only the context above. Cite the sources you use."
        )
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        return msg.content[0].text


# ─────────────────────────────────────────────────────────────────────────────
# RAG facade
# ─────────────────────────────────────────────────────────────────────────────
class RAGChatbot:
    """End-to-end RAG: retrieve top-k, generate, return answer + sources."""

    def __init__(self, k: int = 4, min_score: float = 0.0,
                 llm_model: str = "claude-haiku-4-5-20251001"):
        self.retriever = Retriever()
        self.generator = Generator(model=llm_model)
        self.k = k
        self.min_score = min_score

    def ask(self, question: str) -> RAGResponse:
        import time
        t0 = time.time()
        retrieved = self.retriever.retrieve(question, k=self.k,
                                            min_score=self.min_score)
        answer = self.generator.generate(question, retrieved)
        elapsed = time.time() - t0
        return RAGResponse(
            question=question, answer=answer,
            retrieved=retrieved, elapsed_seconds=elapsed,
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI for quick testing
# ─────────────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Quick RAG CLI.")
    parser.add_argument("question", nargs="+", help="The question to ask")
    parser.add_argument("--k", type=int, default=4, help="Top-k chunks to retrieve")
    args = parser.parse_args()

    bot = RAGChatbot(k=args.k)
    response = bot.ask(" ".join(args.question))

    print("\n" + "=" * 60)
    print(f"Q: {response.question}")
    print("=" * 60)
    print(response.answer)
    print("\n" + "-" * 60)
    print(f"Sources used (top {len(response.retrieved)}):")
    for i, c in enumerate(response.retrieved, 1):
        print(f"  [{i}] {c.source}  (score={c.score:.3f})")
    print(f"\nElapsed: {response.elapsed_seconds:.2f}s")


if __name__ == "__main__":
    main()
