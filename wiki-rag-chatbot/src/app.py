"""
Streamlit UI for the Wikipedia RAG chatbot.

Run with:
    streamlit run src/app.py

Author: Sabal Nemkul
Module: COS6031-D Applied AI Professional (Element 3 portfolio)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

# Make src/ importable regardless of where streamlit is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag import RAGChatbot  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wikipedia RAG Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📚 Wikipedia RAG Chatbot")
st.caption(
    "Retrieval-Augmented Generation over a curated set of Wikipedia articles. "
    "Ask a question and the bot will retrieve the most relevant passages, "
    "then answer using only those sources."
)


# ─────────────────────────────────────────────────────────────────────────────
# Cached chatbot construction
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading retriever and embeddings...")
def get_chatbot(k: int = 4) -> RAGChatbot:
    return RAGChatbot(k=k)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — controls + index info
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    k = st.slider("Top-k chunks to retrieve", 1, 10, 4,
                  help="More chunks = more context for the LLM, but slower and "
                       "may dilute the most relevant passages.")
    show_chunks = st.checkbox("Show retrieved chunks", value=True)

    st.divider()
    st.subheader("About this knowledge base")
    st.markdown(
        """
        Default corpus: **Renewable Energy**.

        Articles include Solar Power, Wind Power, Hydroelectricity,
        Geothermal, Biomass, Tidal, Nuclear, Energy Storage, Smart Grid,
        Electric Vehicle, Battery, Photovoltaics, Wind Turbine,
        Climate Change Mitigation.

        **The bot will only answer based on these sources.**
        Questions outside this scope will be politely declined.
        """
    )

    st.divider()
    st.subheader("API key")
    if os.getenv("ANTHROPIC_API_KEY"):
        st.success("ANTHROPIC_API_KEY detected ✓")
    else:
        st.error("Set ANTHROPIC_API_KEY in your environment or `.env`")


# ─────────────────────────────────────────────────────────────────────────────
# Initialise the bot once
# ─────────────────────────────────────────────────────────────────────────────
try:
    bot = get_chatbot(k=k)
except Exception as e:
    st.error(f"Failed to load chatbot: {e}")
    st.info(
        "Make sure you have:\n"
        "1. Run `python src/ingest.py` to fetch articles\n"
        "2. Run `python src/embed.py` to build the index\n"
        "3. Set `ANTHROPIC_API_KEY` in your environment\n"
    )
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Conversation history (kept in session state; not used to condition the LLM
# in this version — each turn is independent. A future improvement would be
# multi-turn conversation memory.)
# ─────────────────────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []


# ─────────────────────────────────────────────────────────────────────────────
# Sample questions
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Try asking…")
cols = st.columns(3)
samples = [
    "How do solar photovoltaic cells work?",
    "What is the largest hydroelectric dam in the world?",
    "What are the main challenges with wind energy?",
]
for col, q in zip(cols, samples):
    if col.button(q, use_container_width=True):
        st.session_state.pending_question = q


# ─────────────────────────────────────────────────────────────────────────────
# Question input
# ─────────────────────────────────────────────────────────────────────────────
question = st.chat_input("Ask a question about renewable energy…")
if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")


# ─────────────────────────────────────────────────────────────────────────────
# Render conversation history (top-down)
# ─────────────────────────────────────────────────────────────────────────────
for entry in st.session_state.history:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        st.write(entry["answer"])
        if show_chunks and entry["retrieved"]:
            with st.expander(f"📄 Retrieved sources ({len(entry['retrieved'])})"):
                for i, c in enumerate(entry["retrieved"], 1):
                    st.markdown(
                        f"**[{i}] {c['source']}** "
                        f"· similarity = `{c['score']:.3f}` "
                        f"· [link]({c['url']})"
                    )
                    st.caption(c["text"][:400] + ("…" if len(c["text"]) > 400 else ""))
                    st.divider()
        st.caption(f"⏱ {entry['elapsed']:.2f}s")


# ─────────────────────────────────────────────────────────────────────────────
# Handle a new question
# ─────────────────────────────────────────────────────────────────────────────
if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating…"):
            try:
                response = bot.ask(question)
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()
        st.write(response.answer)
        if show_chunks and response.retrieved:
            with st.expander(f"📄 Retrieved sources ({len(response.retrieved)})"):
                for i, c in enumerate(response.retrieved, 1):
                    st.markdown(
                        f"**[{i}] {c.source}** "
                        f"· similarity = `{c.score:.3f}` "
                        f"· [link]({c.url})"
                    )
                    st.caption(c.text[:400] + ("…" if len(c.text) > 400 else ""))
                    st.divider()
        st.caption(f"⏱ {response.elapsed_seconds:.2f}s")

    st.session_state.history.append({
        "question": response.question,
        "answer": response.answer,
        "retrieved": [
            {"source": c.source, "url": c.url, "text": c.text, "score": c.score}
            for c in response.retrieved
        ],
        "elapsed": response.elapsed_seconds,
    })
