"""
Document ingestion for the Wikipedia RAG chatbot.

Fetches a curated list of Wikipedia articles, splits each into overlapping
chunks suitable for embedding, and persists the chunks as JSON for downstream
embedding/indexing.

Author: Sabal Nemkul
Module: COS6031-D Applied AI Professional (Element 3 portfolio)
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import wikipedia


# Default knowledge base — Renewable Energy.
# Chosen for broad coverage with technical depth, so the chatbot has
# something meaningful to retrieve on a wide range of plausible questions.
DEFAULT_TOPICS = [
    "Renewable energy",
    "Solar power",
    "Wind power",
    "Hydroelectricity",
    "Geothermal energy",
    "Biomass",
    "Tidal power",
    "Nuclear power",
    "Energy storage",
    "Smart grid",
    "Electric vehicle",
    "Battery (electricity)",
    "Photovoltaics",
    "Wind turbine",
    "Climate change mitigation",
]

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Chunk:
    """A single passage of text plus its provenance.

    `chunk_id` is unique per document; useful for citation in the chatbot UI.
    """
    chunk_id: str
    source: str          # Wikipedia article title
    url: str             # Article URL
    text: str            # Chunk text
    char_count: int      # For sanity checks


# ─────────────────────────────────────────────────────────────────────────────
# Fetching
# ─────────────────────────────────────────────────────────────────────────────
def fetch_article(title: str) -> tuple[str, str] | None:
    """Fetch full article text + URL for `title`. Returns None on disambiguation
    or page errors so the rest of the ingest can continue."""
    try:
        page = wikipedia.page(title, auto_suggest=False)
        return page.content, page.url
    except wikipedia.DisambiguationError as e:
        # Take the first option in case of ambiguity
        try:
            page = wikipedia.page(e.options[0], auto_suggest=False)
            print(f"[warn] '{title}' was ambiguous, using '{e.options[0]}'")
            return page.content, page.url
        except Exception as inner:
            print(f"[skip] '{title}' — disambiguation fallback failed: {inner}")
            return None
    except wikipedia.PageError:
        print(f"[skip] '{title}' — page not found")
        return None
    except Exception as e:
        print(f"[skip] '{title}' — unexpected error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────────────────
# Strategy: paragraph-aware splitting with target chunk size + overlap.
# This is the simplest defensible chunking strategy for general Wikipedia text;
# more sophisticated schemes (semantic chunking, recursive character splitting)
# would help on heterogeneous documents but add complexity for marginal gain
# at this scale.

def split_into_paragraphs(text: str) -> list[str]:
    """Split on blank lines, strip section headers, drop empty/very-short lines."""
    # Wikipedia plain text uses '== Section ==' for headers; strip them.
    cleaned = re.sub(r"^\s*=+\s*[^=]+?\s*=+\s*$", "", text, flags=re.MULTILINE)
    paragraphs = [p.strip() for p in cleaned.split("\n\n")]
    return [p for p in paragraphs if len(p) >= 50]


def chunk_text(text: str, target_chars: int = 800,
               overlap_chars: int = 100) -> list[str]:
    """Build chunks of approximately `target_chars` size by accumulating
    paragraphs. Add `overlap_chars` of trailing context from the previous
    chunk so that information at chunk boundaries is recoverable."""
    paragraphs = split_into_paragraphs(text)
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if not current:
            current = para
        elif len(current) + len(para) + 2 <= target_chars:
            current = f"{current}\n\n{para}"
        else:
            chunks.append(current)
            tail = current[-overlap_chars:] if overlap_chars else ""
            current = (tail + "\n\n" + para).strip() if tail else para

    if current:
        chunks.append(current)

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def ingest(topics: list[str], target_chars: int = 800,
           overlap_chars: int = 100) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for topic in topics:
        result = fetch_article(topic)
        if result is None:
            continue
        content, url = result
        chunks = chunk_text(content, target_chars, overlap_chars)
        for i, chunk_text_ in enumerate(chunks):
            all_chunks.append(Chunk(
                chunk_id=f"{topic.replace(' ', '_')}::{i:03d}",
                source=topic,
                url=url,
                text=chunk_text_,
                char_count=len(chunk_text_),
            ))
        print(f"[ok] {topic}: {len(chunks)} chunks")

    return all_chunks


def save_chunks(chunks: list[Chunk], out_path: Path) -> None:
    payload = [asdict(c) for c in chunks]
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\n[done] Wrote {len(chunks)} chunks → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Ingest Wikipedia articles for RAG.")
    parser.add_argument("--topics", nargs="+", default=DEFAULT_TOPICS,
                        help="Wikipedia article titles to fetch (default: renewable energy set)")
    parser.add_argument("--out", default=str(DATA_DIR / "chunks.json"),
                        help="Output JSON file for the chunks")
    parser.add_argument("--target-chars", type=int, default=800,
                        help="Target chunk size in characters (default: 800)")
    parser.add_argument("--overlap-chars", type=int, default=100,
                        help="Overlap between adjacent chunks (default: 100)")
    args = parser.parse_args()

    print(f"Ingesting {len(args.topics)} articles…\n")
    chunks = ingest(args.topics, args.target_chars, args.overlap_chars)

    if not chunks:
        print("[error] No chunks produced — check your topic list and network.")
        return

    avg = sum(c.char_count for c in chunks) / len(chunks)
    print(f"\nSummary: {len(chunks)} chunks · avg {avg:.0f} chars · "
          f"covers {len(set(c.source for c in chunks))} articles")

    save_chunks(chunks, Path(args.out))


if __name__ == "__main__":
    main()
