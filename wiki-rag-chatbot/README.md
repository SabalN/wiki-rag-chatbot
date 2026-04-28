# Wikipedia RAG Chatbot

Retrieval-Augmented Generation chatbot over a curated set of Wikipedia articles, with a Streamlit UI.

Built for the **Semi-Complex tier** of the COS6031-D Applied AI Professional portfolio (Element 3).

## Author
Sabal Nemkul · BSc (Hons) Applied Artificial Intelligence, University of Bradford

## What it does

Ask a question about renewable energy. The bot:

1. **Embeds** your question with a sentence-transformer model
2. **Retrieves** the top-k most relevant chunks from a FAISS index built from Wikipedia articles
3. **Generates** an answer using the Anthropic Claude API, constrained to use only the retrieved chunks as context
4. **Cites** the source articles it drew from
5. **Refuses** to answer if the question is outside the knowledge base — no hallucinations on unrelated topics

The default knowledge base covers 15 renewable-energy topics: solar, wind, hydro, geothermal, biomass, tidal, nuclear, energy storage, smart grid, electric vehicles, batteries, photovoltaics, wind turbines, and climate change mitigation.

## Demo screenshot

> _Add a screenshot of the running Streamlit app here once you've taken one._

## Architecture

```
┌────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ User question  │ ──> │  Embed (MiniLM)  │ ──> │ FAISS top-k     │
└────────────────┘     └──────────────────┘     │   retrieval     │
                                                 └────────┬────────┘
                                                          │
┌────────────────┐     ┌──────────────────┐     ┌────────▼────────┐
│ Cited answer   │ <── │ Claude LLM       │ <── │ Build prompt    │
│  + sources     │     │  (constrained)   │     │   with context  │
└────────────────┘     └──────────────────┘     └─────────────────┘
```

**Components**

| File | Responsibility |
|---|---|
| `src/ingest.py` | Fetch Wikipedia articles, chunk into ~800-char overlapping passages, save to JSON |
| `src/embed.py` | Embed all chunks with `all-MiniLM-L6-v2`, build a FAISS L2-normalised inner-product index, persist |
| `src/rag.py` | Retriever + Generator + RAG facade. Importable from app or notebook |
| `src/app.py` | Streamlit chat UI |

## Setup

```bash
# Clone, then:
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Copy the example env file and add your real Anthropic API key
cp .env.example .env
# Edit .env to fill in ANTHROPIC_API_KEY
```

## Build the knowledge base (one-time)

```bash
# 1. Fetch the 15 default articles and chunk them (~20 seconds)
python src/ingest.py

# 2. Embed all chunks and build FAISS index (~30 seconds, downloads ~80MB model on first run)
python src/embed.py
```

This produces:
- `data/chunks.json` — all chunks with provenance metadata
- `data/index/faiss.index` — FAISS index
- `data/index/embeddings.npy` — embedding matrix
- `data/index/chunks.pkl` — chunk metadata for retrieval
- `data/index/manifest.json` — model + dim record

## Run the chatbot

**Web UI (recommended):**

```bash
streamlit run src/app.py
```

Open `http://localhost:8501` in your browser.

**CLI (quick test):**

```bash
python src/rag.py "How do solar photovoltaic cells work?"
```

## Customising the knowledge base

To use a different topic set:

```bash
python src/ingest.py --topics "Quantum computing" "Quantum entanglement" "Qubit"
python src/embed.py
```

## What I learned building this

This project was deliberately scoped to cover the **full RAG stack** end-to-end without resorting to high-level frameworks like LangChain — that would have hidden the components I most wanted to understand. A few things that stood out:

**1. Chunking strategy matters more than model choice.** I started with naive fixed-character splits and the bot frequently lost context at chunk boundaries. Switching to paragraph-aware splitting with a small overlap (and stripping section headers, which were poisoning embeddings) measurably improved retrieval quality.

**2. Cosine similarity vs. raw inner product.** L2-normalising before adding to FAISS lets you use `IndexFlatIP` for cosine semantics. Without normalisation, longer chunks get inflated scores and dominate retrieval.

**3. The system prompt does more work than the LLM choice.** The system prompt explicitly instructs the model to refuse when context is insufficient, and the bot respects that — it actually says "I don't have enough information in the provided sources" on out-of-scope questions instead of confidently hallucinating. That single behaviour is worth more than a model upgrade.

**4. Connection to my Industrial AI Project.** The dataset-bias work I did for my FYP classifier showed up here in a different form: my chunking + embedding choices are themselves a kind of dataset construction, and the same critical questions apply ("what does the system see, what is it missing, where does it fail gracefully?").

## Limitations and what I would change

- **No multi-turn memory.** Each query is independent. Adding conversation history conditioning would be the next obvious extension.
- **No re-ranking.** A second pass with a cross-encoder (e.g. `ms-marco-MiniLM-L-6-v2`) on the top-20 retrieved chunks would likely improve precision at the cost of latency.
- **Single-LLM dependency.** The `Generator` is built around the Anthropic API. Swapping for OpenAI or a local Ollama model is mechanical (one method).
- **No evaluation.** A real RAG project would build a small evaluation set (question/expected-source pairs) and measure retrieval recall@k and answer faithfulness. Out of scope here but flagged in my portfolio reflection.

## Tech stack

- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim, ~80MB)
- **Vector store:** FAISS (CPU, in-memory, exact search)
- **LLM:** Anthropic Claude (default: Haiku 4.5 for cost; swappable to Sonnet/Opus)
- **UI:** Streamlit
- **Source data:** `wikipedia` Python library

## License

MIT — see `LICENSE` file.
