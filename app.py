# --- Warning filters ---
import warnings
warnings.filterwarnings("ignore", message="resource_tracker")
warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing")
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")



import streamlit as st

from src.ingestion.extractors import (
    extract_text_from_pdf,
    extract_text_from_txt,
    extract_text_from_docx,
)
from src.indexing.vector_store import VectorIndex
from src.summarization.baseline_summarizer import BaselineSummarizer
from src.summarization.llm_summarizer import LLMSummarizer
from src.config import TOP_K_DEFAULT


# ---------- CACHED RESOURCES ----------

@st.cache_resource
def get_baseline_summarizer():
    """Create and cache a single BaselineSummarizer (BART) instance."""
    return BaselineSummarizer()


@st.cache_resource
def get_llm_summarizer():
    """Create and cache a single LLMSummarizer (TinyLlama or other LLM) instance."""
    return LLMSummarizer()


# ---------- SESSION STATE INIT ----------

def init_session_state():
    """Initialize objects in Streamlit session_state if they don't exist yet."""
    if "index" not in st.session_state:
        st.session_state.index = None

    if "doc_stats" not in st.session_state:
        st.session_state.doc_stats = {"num_docs": 0, "num_chunks": 0}

    if "baseline_summarizer" not in st.session_state:
        st.session_state.baseline_summarizer = get_baseline_summarizer()

    # LLM summarizer is created lazily when needed


# ---------- MAIN APP ----------

def main():
    st.set_page_config(
        page_title="AI Document Search & Summarizer",
        page_icon="📄",
        layout="wide",
    )

    # ---------- CLEAN MODERN CSS ----------
    st.markdown(
        """
        <style>

        .main {
            background-color: #f5f6fa !important;
        }

        /* Step titles */
        .yc-step-title {
            font-size: 1.2rem;
            font-weight: 600;
            margin-top: 20px;
            margin-bottom: 5px;
            padding-left: 4px;
        }

        /* Thin light-grey separator */
        .yc-separator {
            border-bottom: 1px solid #d1d5db;
            margin-bottom: 16px;
            margin-top: 4px;
        }

        /* Subtitles for summary sections */
        .yc-subtitle {
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 6px;
        }

        /* Badge styles for BART and LLM labels */
        .yc-badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-bottom: 6px;
        }
        .yc-badge-blue {
            background-color: #e5edff;
            border: 1px solid #bcc8ff;
            color: #1f3b8f;
        }
        .yc-badge-green {
            background-color: #e4f6e8;
            border: 1px solid #b7e3c0;
            color: #166534;
        }

        /* Similarity score pill */
        .yc-score-pill {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 999px;
            font-size: 0.70rem;
            background-color: #eef2ff;
            color: #3730a3;
            margin-left: 6px;
        }

        /* Pretty <details> blocks for results */
        .yc-details {
            margin-bottom: 10px;
            border-radius: 8px;
            background-color: #ffffff;
            border: 1px solid #e5e7eb;
        }

        .yc-details summary {
            list-style: none;
        }

        .yc-details summary::-webkit-details-marker {
            display: none;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )

    # ---------- PAGE TITLE ----------
    st.markdown(
        """
        <h1>📄 AI Document Search & Summarizer</h1>
        <p style="color:#4b5563; font-size:0.95rem; margin-bottom:8px;">
          Step 1: Upload your documents → Step 2: Ask a question → Step 3: Compare summarizers
        </p>
        """,
        unsafe_allow_html=True,
    )

    # ---------- HOW TO USE BOX ----------
    st.markdown(
        """
        <div style="
            background-color:#e5f2ff;
            border:1px solid #bfdbfe;
            border-radius:10px;
            padding:10px 14px;
            margin-bottom:18px;
            font-size:0.9rem;
            color:#1f2933;">
          <b>How to use this app</b>
          <ol style="margin:6px 0 0 18px; padding:0;">
            <li>On the left, upload one or more PDF / TXT / DOCX files, then click <i>“Process documents and build index”</i>.</li>
            <li>Choose how many chunks to retrieve and whether to compare with the LLM summarizer.</li>
            <li>On the right, type a question about your documents and click <i>“Search & Summarize”</i>.</li>
            <li>Expand the retrieved passages displayed and compare the baseline BART summary with the LLM answer (if selected).</li>
          </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

    init_session_state()

    # Layout: Left column (inputs), Right column (search + results)
    left_col, right_col = st.columns([1, 2])

    # --------------------------------------------------------------
    # LEFT COLUMN — DOCUMENTS + SETTINGS
    # --------------------------------------------------------------
    with left_col:
        # ---------- SECTION 1: UPLOAD DOCUMENTS ----------
        st.markdown('<div class="yc-step-title">1️⃣ Upload & Index Documents</div>', unsafe_allow_html=True)
        st.markdown('<div class="yc-separator"></div>', unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "Upload one or more files (PDF, TXT, DOCX)",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
        )

        if st.button("📥 Process documents and build index"):
            if not uploaded_files:
                st.warning("Please upload at least one file.")
            else:
                docs = []

                for i, uploaded_file in enumerate(uploaded_files):
                    # Include filename in doc_id so the UI is clearer
                    doc_id = f"doc_{i} ({uploaded_file.name})"
                    filename = uploaded_file.name.lower()

                    if filename.endswith(".pdf"):
                        text = extract_text_from_pdf(uploaded_file)
                    elif filename.endswith(".txt"):
                        text = extract_text_from_txt(uploaded_file)
                    elif filename.endswith(".docx"):
                        text = extract_text_from_docx(uploaded_file)
                    else:
                        text = ""

                    if not text.strip():
                        st.warning(f"File '{uploaded_file.name}' contained no extractable text.")
                        continue

                    docs.append((doc_id, text))

                if not docs:
                    st.error("No valid documents found.")
                else:
                    with st.spinner("Building vector index..."):
                        index = VectorIndex()
                        index.add_documents(docs)

                        st.session_state.index = index
                        st.session_state.doc_stats["num_docs"] = len(docs)
                        st.session_state.doc_stats["num_chunks"] = len(index.text_chunks)

                    st.success("✅ Index built successfully!")

        # Status
        if st.session_state.index is None:
            st.info("No documents indexed yet.")
        else:
            st.write(
                f"- **Documents:** {st.session_state.doc_stats['num_docs']}\n"
                f"- **Chunks:** {st.session_state.doc_stats['num_chunks']}"
            )

        # ---------- SECTION 2: SEARCH SETTINGS ----------
        st.markdown('<div class="yc-step-title">2️⃣ Search & LLM Settings</div>', unsafe_allow_html=True)
        st.markdown('<div class="yc-separator"></div>', unsafe_allow_html=True)

        top_k = st.slider(
            "Number of retrieved chunks (top_k)",
            help="Default set on 3. Higher value will increase context but also noise for the LLM output.",
            min_value=1,
            max_value=5,
            value=TOP_K_DEFAULT,
        )

        compare_llm = st.checkbox(
            "Compare with advanced LLM summarizer",
            help="Shows both BART summary and LLM answer side by side."
        )

    # --------------------------------------------------------------
    # RIGHT COLUMN — SEARCH & RESULTS
    # --------------------------------------------------------------
    with right_col:
        # ---------- SECTION 3: ASK A QUESTION ----------
        st.markdown('<div class="yc-step-title">3️⃣ Ask a Question</div>', unsafe_allow_html=True)
        st.markdown('<div class="yc-separator"></div>', unsafe_allow_html=True)

        query = st.text_input(
            "Your question",
            placeholder="e.g. What are the key findings in these documents?"
        )

        run_search = st.button("🔎 Search & Summarize")

        if run_search:
            # Validation
            if st.session_state.index is None:
                st.error("Please upload and index documents first.")
            elif not query.strip():
                st.warning("Enter a question before searching.")
            else:
                # ---------- SEARCH ----------
                st.markdown('<div class="yc-step-title">Top Retrieved Passages</div>', unsafe_allow_html=True)
                st.markdown('<div class="yc-separator"></div>', unsafe_allow_html=True)

                with st.spinner("Searching..."):
                    results = st.session_state.index.search(query, top_k=top_k)

                retrieved_texts = []

                if not results:
                    st.info("No matching passages found.")
                else:
                    for r in results:
                        score_pct = int(round(r["score"] * 100))

                        # Escape HTML special characters in the chunk text
                        text = r["text"]
                        text = (
                            text.replace("&", "&amp;")
                                .replace("<", "&lt;")
                                .replace(">", "&gt;")
                        )

                        # Pretty <details> layout:
                        # ▸ icon + "Doc X • Chunk Y" + [24% match] all on the same line
                        st.markdown(
                            f"""
                            <details class="yc-details">
                              <summary style="
                                    display:flex;
                                    align-items:center;
                                    gap:8px;
                                    padding:6px 10px;
                                    cursor:pointer;
                                    font-size:0.95rem;
                                    font-weight:500;">
                                <span style="color:#6b7280; font-size:0.85rem;">▸</span>
                                <span>
                                  {r['doc_id']} • Chunk {r['chunk_id']}
                                  <span class="yc-score-pill">{score_pct}% match</span>
                                </span>
                              </summary>
                              <div style="padding:8px 12px 10px 12px; font-size:0.9rem; line-height:1.4;">
                                <pre style="white-space:pre-wrap; font-family: inherit; margin:0;">{text}</pre>
                              </div>
                            </details>
                            """,
                            unsafe_allow_html=True,
                        )

                        retrieved_texts.append(r["text"])

                # ---------- SUMMARIES ----------
                if retrieved_texts:
                    st.markdown('<div class="yc-step-title">Summaries</div>', unsafe_allow_html=True)
                    st.markdown('<div class="yc-separator"></div>', unsafe_allow_html=True)

                    # BART sees all retrieved chunks (up to top_k)
                    bart_chunks = retrieved_texts

                    # LLM only sees the top 2 chunks to keep context clean
                    llm_chunks = retrieved_texts[:2]

                    # Baseline BART summary
                    with st.spinner("Generating BART summary..."):
                        bart_summary = st.session_state.baseline_summarizer.summarize_chunks(
                            bart_chunks
                        )

                    if compare_llm:
                        # LLM summary (created lazily)
                        if "llm_summarizer" not in st.session_state:
                            st.session_state.llm_summarizer = get_llm_summarizer()

                        with st.spinner("Generating LLM answer..."):
                            llm_summary = st.session_state.llm_summarizer.summarize_with_llm(
                                query, llm_chunks
                            )

                        # Two-column comparison
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown(
                                '<span class="yc-badge yc-badge-blue">Baseline · BART summarizer (faster, generic, summarize with best chunk)</span>',
                                unsafe_allow_html=True,
                            )
                            st.markdown('<div class="yc-subtitle">BART Summary</div>', unsafe_allow_html=True)
                            st.write(bart_summary[:100] + "..." if len(bart_summary) > 100 else bart_summary)

                        with col2:
                            st.markdown(
                                '<span class="yc-badge yc-badge-green">Advanced · TinyLlama LLM (slower, higher quality, question-aware)</span>',
                                unsafe_allow_html=True,
                            )
                            st.markdown('<div class="yc-subtitle">LLM Answer</div>', unsafe_allow_html=True)
                            st.write(llm_summary)
                    else:
                        st.markdown(
                            '<span class="yc-badge yc-badge-blue">Baseline · BART</span>',
                            unsafe_allow_html=True,
                        )
                        st.markdown('<div class="yc-subtitle">BART Summary</div>', unsafe_allow_html=True)
                        st.write(bart_summary)


if __name__ == "__main__":
    main()