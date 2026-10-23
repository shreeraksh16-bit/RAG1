"""
Learn RAG — Chat with Your PDF
================================
Interactive Streamlit demonstration application for Retrieval-Augmented Generation (RAG).
Redesigned with a premium AI SaaS dashboard visual language (Linear/Raycast inspired).
Uses Qdrant Cloud as the vector database, Hugging Face sentence-transformers for embeddings,
and OpenRouter (recursive/flint-v1.0:free) for strictly grounded answer generation.
"""

import os
import sys

# Ensure current directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

from rag.config import get_config, validate_config
from rag.ingestion import (
    compute_document_id,
    load_pdf_from_bytes,
    split_document_pages,
    get_qdrant_client,
    ensure_collection_exists,
    is_document_indexed,
    delete_document_vectors,
    index_chunks_to_qdrant,
)
from rag.retrieval import (
    get_embedding_model,
    retrieve_relevant_chunks,
    format_citations,
)
from rag.generation import generate_grounded_answer


# Page configuration
st.set_page_config(
    page_title="Learn RAG — Personal PDF Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Comprehensive Dark SaaS Design System (Linear / Raycast inspired)
CUSTOM_CSS = """
<style>
/* Base theme & typography */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-primary: #080C14;
    --bg-secondary: #0D1322;
    --bg-surface: rgba(18, 25, 41, 0.75);
    --bg-surface-hover: rgba(26, 35, 56, 0.85);
    --border-subtle: rgba(255, 255, 255, 0.08);
    --border-accent: rgba(139, 92, 246, 0.35);
    --text-primary: #F8FAFC;
    --text-secondary: #94A3B8;
    --text-muted: #64748B;
    --accent-violet: #8B5CF6;
    --accent-indigo: #6366F1;
    --accent-cyan: #06B6D4;
    --accent-emerald: #10B981;
    --accent-rose: #F43F5E;
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: var(--text-primary);
}

/* Background canvas */
.stApp {
    background: radial-gradient(circle at 50% -20%, #171E32 0%, #080C14 65%, #05080E 100%);
    color: var(--text-primary);
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #0B101C !important;
    border-right: 1px solid var(--border-subtle) !important;
}

[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    gap: 0.85rem;
}

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: rgba(255, 255, 255, 0.12);
    border-radius: 999px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(255, 255, 255, 0.25);
}

/* Header & Typography Elements */
.header-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.15) 0%, rgba(99, 102, 241, 0.08) 100%);
    border: 1px solid rgba(139, 92, 246, 0.35);
    color: #C4B5FD;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}

.workspace-title {
    font-size: 2.35rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #FFFFFF 0%, #E2E8F0 50%, #94A3B8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem 0;
    line-height: 1.15;
}

.workspace-subtitle {
    font-size: 1rem;
    color: var(--text-secondary);
    line-height: 1.5;
    margin-bottom: 1.5rem;
}

/* Premium Card Surfaces */
.rag-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 1.25rem 1.4rem;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
}

.rag-card:hover {
    border-color: rgba(255, 255, 255, 0.14);
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
}

.rag-card-accent {
    border-color: var(--border-accent);
    background: linear-gradient(180deg, rgba(22, 28, 48, 0.8) 0%, rgba(14, 19, 34, 0.8) 100%);
}

/* Pulsing Status Indicators */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 0.82rem;
    padding: 5px 12px;
    border-radius: 8px;
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid var(--border-subtle);
    font-weight: 500;
}

.status-dot-green {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #10B981;
    box-shadow: 0 0 10px #10B981;
    animation: pulse-dot 2s infinite ease-in-out;
}

.status-dot-amber {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #F59E0B;
    box-shadow: 0 0 10px #F59E0B;
}

.status-dot-red {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #EF4444;
    box-shadow: 0 0 10px #EF4444;
}

@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.45; transform: scale(0.85); }
}

/* Animated Orb for Brand */
.brand-container {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 0.5rem 0 0.25rem 0;
}

.brand-orb {
    width: 32px;
    height: 32px;
    border-radius: 10px;
    background: linear-gradient(135deg, #8B5CF6 0%, #3B82F6 100%);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 0 20px rgba(139, 92, 246, 0.5);
    font-size: 1.05rem;
    color: white;
}

.brand-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: -0.02em;
    line-height: 1.2;
}

.brand-subtitle {
    font-size: 0.78rem;
    color: var(--text-muted);
}

/* Primary Button Styling */
.stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: -0.01em !important;
    padding: 0.55rem 1.15rem !important;
    transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    border: 1px solid var(--border-subtle) !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #7C3AED 0%, #6366F1 100%) !important;
    border: 1px solid rgba(167, 139, 250, 0.45) !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 18px rgba(124, 58, 237, 0.35) !important;
}

.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 24px rgba(124, 58, 237, 0.55) !important;
    border-color: rgba(196, 181, 253, 0.7) !important;
}

.stButton > button[kind="secondary"] {
    background: rgba(22, 30, 48, 0.75) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-subtle) !important;
}

.stButton > button[kind="secondary"]:hover {
    background: rgba(30, 42, 68, 0.9) !important;
    border-color: rgba(255, 255, 255, 0.18) !important;
}

/* File Uploader Custom Skin */
[data-testid="stFileUploader"] {
    background: rgba(14, 20, 34, 0.6);
    border: 1px dashed rgba(139, 92, 246, 0.35);
    border-radius: 14px;
    padding: 1.2rem;
    transition: all 0.25s ease;
}

[data-testid="stFileUploader"]:hover {
    border-color: #8B5CF6;
    background: rgba(20, 28, 48, 0.75);
    box-shadow: 0 0 25px rgba(139, 92, 246, 0.15);
}

/* Chat Messages */
[data-testid="stChatMessage"] {
    background: rgba(15, 22, 38, 0.7) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 14px !important;
    padding: 1.1rem 1.3rem !important;
    margin-bottom: 0.9rem !important;
    backdrop-filter: blur(8px) !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2) !important;
    transition: border-color 0.2s ease !important;
}

[data-testid="stChatMessage"]:hover {
    border-color: rgba(255, 255, 255, 0.14) !important;
}

/* Chat Input Bar */
[data-testid="stChatInput"] {
    background: #0E1526 !important;
    border: 1px solid rgba(139, 92, 246, 0.35) !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4) !important;
    transition: all 0.2s ease !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: #8B5CF6 !important;
    box-shadow: 0 0 20px rgba(139, 92, 246, 0.4) !important;
}

/* Citations Badge */
.citation-box {
    background: linear-gradient(90deg, rgba(6, 182, 212, 0.12) 0%, rgba(14, 21, 38, 0.4) 100%);
    border-left: 3px solid #06B6D4;
    border-radius: 8px;
    padding: 10px 14px;
    margin-top: 0.75rem;
    margin-bottom: 0.5rem;
    font-size: 0.88rem;
    color: #CBD5E1;
}

.citation-box strong {
    color: #38BDF8;
    font-weight: 600;
}

/* Retrieval Inspector Cards */
.retrieval-chunk-card {
    background: rgba(10, 15, 26, 0.75);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 10px;
    transition: all 0.2s ease;
}

.retrieval-chunk-card:hover {
    border-color: rgba(139, 92, 246, 0.4);
    background: rgba(14, 21, 36, 0.85);
}

.chunk-meta-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 0.8rem;
    color: #94A3B8;
    margin-bottom: 8px;
}

.chunk-score-pill {
    background: rgba(139, 92, 246, 0.15);
    color: #C4B5FD;
    border: 1px solid rgba(139, 92, 246, 0.3);
    padding: 2px 8px;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
}

.chunk-text-box {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    line-height: 1.5;
    color: #E2E8F0;
    background: rgba(6, 9, 16, 0.6);
    padding: 10px 12px;
    border-radius: 6px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    white-space: pre-wrap;
    word-break: break-word;
}

/* Empty State Box */
.empty-hero {
    text-align: center;
    padding: 2.5rem 1.5rem;
    background: rgba(14, 20, 34, 0.4);
    border: 1px dashed rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    margin: 1.5rem 0;
}

.empty-icon {
    font-size: 2.4rem;
    margin-bottom: 0.75rem;
    display: inline-block;
    filter: drop-shadow(0 0 16px rgba(139, 92, 246, 0.4));
}

/* Question Chips */
.chip-container {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 1rem;
}

/* Expander custom look */
[data-testid="stExpander"] {
    background: rgba(13, 19, 33, 0.6) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 12px !important;
}

/* Status Widget */
[data-testid="stStatusWidget"] {
    background: rgba(14, 20, 34, 0.8) !important;
    border: 1px solid rgba(139, 92, 246, 0.35) !important;
    border-radius: 12px !important;
}

@media (prefers-reduced-motion: reduce) {
    *, ::before, ::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "active_document" not in st.session_state:
    st.session_state.active_document = None

if "confirm_delete" not in st.session_state:
    st.session_state.confirm_delete = False

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# Retrieve application config and validate
config = get_config()
is_config_valid, config_issues = validate_config(config)

# ==========================================
# 1. SIDEBAR (Zone A)
# ==========================================
with st.sidebar:
    # Brand Header
    st.markdown(
        """
        <div class="brand-container">
            <div class="brand-orb">⚡</div>
            <div>
                <div class="brand-title">Learn RAG</div>
                <div class="brand-subtitle">Your personal PDF intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown("<hr style='border: 0; border-top: 1px solid rgba(255,255,255,0.08); margin: 0.8rem 0;'>", unsafe_allow_html=True)

    # Navigation Switcher
    active_view = st.radio(
        "Navigation",
        options=["⚡ Workspace", "📑 Document Info", "🔍 Retrieval Insights"],
        label_visibility="collapsed",
        key="main_nav_tab",
    )

    st.markdown("<hr style='border: 0; border-top: 1px solid rgba(255,255,255,0.08); margin: 0.8rem 0;'>", unsafe_allow_html=True)

    # Infrastructure Health Status
    st.markdown("<div style='font-size: 0.8rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;'>☁️ Cloud Infrastructure</div>", unsafe_allow_html=True)
    
    # Qdrant Status
    if config.qdrant_url and "localhost" not in config.qdrant_url and config.qdrant_api_key:
        cluster_host = config.qdrant_url.split("//")[-1].split(":")[0]
        st.markdown(
            f"""
            <div class="status-pill" style="width: 100%; margin-bottom: 6px;">
                <div class="status-dot-green"></div>
                <div>
                    <div style="font-size: 0.82rem; font-weight: 600; color: #F8FAFC;">Qdrant Cloud</div>
                    <div style="font-size: 0.72rem; color: #64748B; font-family: monospace;">{cluster_host[:26]}...</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="status-pill" style="width: 100%; margin-bottom: 6px;">
                <div class="status-dot-red"></div>
                <div>
                    <div style="font-size: 0.82rem; font-weight: 600; color: #FCA5A5;">Qdrant Disconnected</div>
                    <div style="font-size: 0.72rem; color: #94A3B8;">Check credentials in .env</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # OpenRouter LLM Status
    if config.openrouter_api_key:
        st.markdown(
            f"""
            <div class="status-pill" style="width: 100%; margin-bottom: 6px;">
                <div class="status-dot-green"></div>
                <div>
                    <div style="font-size: 0.82rem; font-weight: 600; color: #F8FAFC;">OpenRouter LLM</div>
                    <div style="font-size: 0.72rem; color: #A78BFA; font-family: monospace;">{config.openrouter_model}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="status-pill" style="width: 100%; margin-bottom: 6px;">
                <div class="status-dot-red"></div>
                <div>
                    <div style="font-size: 0.82rem; font-weight: 600; color: #FCA5A5;">OpenRouter Key Missing</div>
                    <div style="font-size: 0.72rem; color: #94A3B8;">Set key in .env</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(f"<div style='font-size: 0.75rem; color: #64748B; margin-top: 4px; padding-left: 4px;'>Collection: <span style='font-family: monospace; color: #94A3B8;'>{config.qdrant_collection_name}</span></div>", unsafe_allow_html=True)

    if not is_config_valid:
        with st.expander("⚠️ Configuration Checklist", expanded=True):
            for issue in config_issues:
                st.error(f"• {issue}")
            st.caption("Update `.env` in the workspace root to resolve these.")

    st.markdown("<hr style='border: 0; border-top: 1px solid rgba(255,255,255,0.08); margin: 0.8rem 0;'>", unsafe_allow_html=True)

    # Active Document Card
    st.markdown("<div style='font-size: 0.8rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px;'>📄 Active Document</div>", unsafe_allow_html=True)
    if st.session_state.active_document:
        doc_info = st.session_state.active_document
        st.markdown(
            f"""
            <div class="rag-card" style="padding: 10px 12px; margin-bottom: 8px;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
                    <span style="font-size: 1.1rem;">📑</span>
                    <strong style="font-size: 0.88rem; color: #F8FAFC; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{doc_info['filename']}</strong>
                </div>
                <div style="display: flex; gap: 6px; flex-wrap: wrap; margin-top: 6px;">
                    <span style="background: rgba(139, 92, 246, 0.2); color: #C4B5FD; font-size: 0.72rem; padding: 2px 6px; border-radius: 4px; font-weight: 600;">{doc_info.get('total_pages', 'N/A')} Pages</span>
                    <span style="background: rgba(6, 182, 212, 0.2); color: #67E8F9; font-size: 0.72rem; padding: 2px 6px; border-radius: 4px; font-weight: 600;">{doc_info.get('total_chunks', 'N/A')} Chunks</span>
                    <span style="background: rgba(255, 255, 255, 0.08); color: #94A3B8; font-size: 0.72rem; padding: 2px 6px; border-radius: 4px; font-family: monospace;">ID: {doc_info['doc_id'][:8]}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_switch, col_del = st.columns(2)
        with col_switch:
            if st.button("🔄 Change", use_container_width=True, help="Switch active PDF"):
                st.session_state.active_document = None
                st.session_state.confirm_delete = False
                st.rerun()

        with col_del:
            if not st.session_state.confirm_delete:
                if st.button("🗑️ Delete", use_container_width=True, help="Delete cloud vectors"):
                    st.session_state.confirm_delete = True
                    st.rerun()

        # Confirmation State for Deletion
        if st.session_state.confirm_delete:
            st.markdown(
                """
                <div style="background: rgba(239, 68, 68, 0.12); border: 1px solid rgba(239, 68, 68, 0.35); border-radius: 8px; padding: 8px 10px; margin-top: 6px; font-size: 0.78rem; color: #FCA5A5;">
                    ⚠️ Confirm deletion of all vector points for this PDF from Qdrant Cloud?
                </div>
                """,
                unsafe_allow_html=True,
            )
            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("Confirm Delete", type="primary", use_container_width=True):
                    with st.spinner("Deleting vectors from Qdrant Cloud..."):
                        try:
                            client = get_qdrant_client(config.qdrant_url, config.qdrant_api_key)
                            del_count = delete_document_vectors(
                                client, config.qdrant_collection_name, doc_info["doc_id"]
                            )
                            st.session_state.active_document = None
                            st.session_state.messages = []
                            st.session_state.confirm_delete = False
                            st.success(f"Removed {del_count} vectors!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Deletion failed: {e}")
            with col_no:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.confirm_delete = False
                    st.rerun()

    else:
        st.markdown(
            """
            <div style="font-size: 0.82rem; color: #64748B; font-style: italic; padding: 4px 0;">
                No PDF active. Upload and index a document to begin.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='border: 0; border-top: 1px solid rgba(255,255,255,0.08); margin: 0.8rem 0;'>", unsafe_allow_html=True)

    # Settings & Diagnostics
    show_diagnostics = st.checkbox(
        "🩺 Retrieval Inspector",
        value=True,
        help="Displays retrieved chunk text, source page numbers, and cosine similarity scores."
    )
    
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_query = None
        st.rerun()

    # Footer
    st.markdown(
        """
        <div style="margin-top: 1.5rem; text-align: center; font-size: 0.72rem; color: #475569;">
            Learn RAG v1.2 • Cloud Native RAG<br>
            Powered by Qdrant & OpenRouter
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================
# 2. MAIN WORKSPACE (Zone B & C)
# ==========================================

# Top Header Bar
st.markdown('<div class="header-tag">✦ AI Research Workspace</div>', unsafe_allow_html=True)
col_h_left, col_h_right = st.columns([3, 1])

with col_h_left:
    st.markdown('<div class="workspace-title">Chat with your documents.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="workspace-subtitle">Grounded semantic question-answering with exact page citations powered by managed Qdrant Cloud vectors and verified OpenRouter LLM synthesis.</div>',
        unsafe_allow_html=True,
    )

with col_h_right:
    if is_config_valid:
        st.markdown(
            """
            <div style="display: flex; justify-content: flex-end; align-items: center; height: 100%; padding-bottom: 1rem;">
                <div class="status-pill" style="border-color: rgba(16, 185, 129, 0.35);">
                    <div class="status-dot-green"></div>
                    <span style="font-size: 0.78rem; font-weight: 600; color: #A7F3D0;">CLUSTER READY</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="display: flex; justify-content: flex-end; align-items: center; height: 100%; padding-bottom: 1rem;">
                <div class="status-pill" style="border-color: rgba(239, 68, 68, 0.35);">
                    <div class="status-dot-red"></div>
                    <span style="font-size: 0.78rem; font-weight: 600; color: #FCA5A5;">SETUP REQUIRED</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Validate Configuration Banner
if not is_config_valid:
    st.warning("⚠️ **Application Setup Required:** Please configure your credentials in `.env` before proceeding.")
    with st.expander("📋 View Missing Configuration Details", expanded=True):
        for issue in config_issues:
            st.error(issue)
    st.stop()


# ------------------------------------------
# VIEW 2: Document Info Tab
# ------------------------------------------
if active_view == "📑 Document Info":
    st.markdown("### 📑 Document & Vector Store Details")
    if st.session_state.active_document:
        doc = st.session_state.active_document
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown(
                f"""
                <div class="rag-card">
                    <h4 style="margin: 0 0 10px 0; color: #F8FAFC;">Document Metadata</h4>
                    <p style="margin: 4px 0; font-size: 0.9rem;"><strong>Filename:</strong> <code style="color: #C4B5FD;">{doc['filename']}</code></p>
                    <p style="margin: 4px 0; font-size: 0.9rem;"><strong>Document Hash (SHA-256):</strong> <code style="color: #67E8F9;">{doc['doc_id']}</code></p>
                    <p style="margin: 4px 0; font-size: 0.9rem;"><strong>Total Pages Extracted:</strong> {doc.get('total_pages', 'N/A')}</p>
                    <p style="margin: 4px 0; font-size: 0.9rem;"><strong>Total Chunks Indexed:</strong> {doc.get('total_chunks', 'N/A')}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_d2:
            st.markdown(
                f"""
                <div class="rag-card">
                    <h4 style="margin: 0 0 10px 0; color: #F8FAFC;">Indexing Pipeline Parameters</h4>
                    <p style="margin: 4px 0; font-size: 0.9rem;"><strong>Chunk Size:</strong> {config.chunk_size} characters</p>
                    <p style="margin: 4px 0; font-size: 0.9rem;"><strong>Chunk Overlap:</strong> {config.chunk_overlap} characters</p>
                    <p style="margin: 4px 0; font-size: 0.9rem;"><strong>Embedding Model:</strong> <code>{config.embedding_model_name}</code> (384-dim)</p>
                    <p style="margin: 4px 0; font-size: 0.9rem;"><strong>Distance Metric:</strong> Cosine Similarity</p>
                    <p style="margin: 4px 0; font-size: 0.9rem;"><strong>Cloud Collection:</strong> <code>{config.qdrant_collection_name}</code></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("No active document currently selected. Upload and index a PDF in the **Workspace** tab.")

# ------------------------------------------
# VIEW 3: Retrieval Insights Tab
# ------------------------------------------
elif active_view == "🔍 Retrieval Insights":
    st.markdown("### 🔍 RAG Architecture & Retrieval Insights")
    st.markdown(
        """
        <div class="rag-card" style="margin-bottom: 1.2rem;">
            <h4 style="margin: 0 0 8px 0; color: #F8FAFC;">How Scoped Document Retrieval Works</h4>
            <p style="font-size: 0.9rem; color: #94A3B8; line-height: 1.6;">
                In this cloud-native RAG implementation, every document is assigned a unique deterministic 16-character SHA-256 hash (<code>doc_id</code>). 
                When chunk vectors are saved into <strong>Qdrant Cloud</strong>, each point includes <code>metadata.doc_id</code>, <code>metadata.page</code>, and <code>metadata.chunk_index</code>.
            </p>
            <p style="font-size: 0.9rem; color: #94A3B8; line-height: 1.6;">
                During similarity search, a Qdrant <strong>FieldCondition</strong> filter enforces <code>metadata.doc_id == active_doc_id</code>. 
                This mathematically guarantees that search results never mix content from different uploaded PDFs, even when multiple documents reside inside the same shared collection.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_ins1, col_ins2, col_ins3 = st.columns(3)
    with col_ins1:
        st.markdown(
            """
            <div class="rag-card">
                <div style="font-size: 1.5rem; margin-bottom: 6px;">📄</div>
                <strong style="color: #F8FAFC;">1-Indexed Page Citations</strong>
                <p style="font-size: 0.84rem; color: #94A3B8; margin-top: 4px;">PyPDFLoader 0-indexed pages are converted to human-friendly 1-indexed numbers (page + 1) for accurate academic citing.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_ins2:
        st.markdown(
            """
            <div class="rag-card">
                <div style="font-size: 1.5rem; margin-bottom: 6px;">📐</div>
                <strong style="color: #F8FAFC;">Normalized Dense Vectors</strong>
                <p style="font-size: 0.84rem; color: #94A3B8; margin-top: 4px;">384-dimensional dense vectors represent sentence semantics, where cosine similarity measures angular alignment.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_ins3:
        st.markdown(
            """
            <div class="rag-card">
                <div style="font-size: 1.5rem; margin-bottom: 6px;">🛡️</div>
                <strong style="color: #F8FAFC;">Anti-Hallucination Guardrails</strong>
                <p style="font-size: 0.84rem; color: #94A3B8; margin-top: 4px;">Strict grounding prompt instructs LLM to explicitly refuse answers outside the document context without guessing.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ------------------------------------------
# VIEW 1: Main Workspace Tab (Upload + Chat)
# ------------------------------------------
else:
    # DOCUMENT UPLOAD & INGESTION CARD
    st.markdown(
        """
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.6rem;">
            <div style="font-size: 1.05rem; font-weight: 700; color: #F8FAFC; display: flex; align-items: center; gap: 8px;">
                <span>📤</span> Document Ingestion & Cloud Indexing
            </div>
            <div style="font-size: 0.78rem; color: #64748B;">Supported: Text-based PDFs • up to 50MB</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose a PDF document to index into Qdrant Cloud:",
        type=["pdf"],
        help="Upload any text-based PDF (documentation, textbook, research paper, syllabus, or report).",
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        filename = uploaded_file.name
        doc_id = compute_document_id(file_bytes, filename)
        file_size_kb = len(file_bytes) / 1024
        
        try:
            qdrant_client = get_qdrant_client(config.qdrant_url, config.qdrant_api_key)
            already_indexed, existing_chunks = is_document_indexed(
                qdrant_client, config.qdrant_collection_name, doc_id
            )
        except Exception as e:
            st.error(f"❌ Could not connect to Qdrant Cloud: {e}. Check your `QDRANT_URL` and `QDRANT_API_KEY`.")
            already_indexed, existing_chunks = False, 0

        # Uploaded file details card
        col_info, col_action = st.columns([3, 1])
        
        with col_info:
            if already_indexed:
                st.markdown(
                    f"""
                    <div class="rag-card" style="border-color: rgba(16, 185, 129, 0.4); background: rgba(16, 185, 129, 0.08);">
                        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
                            <span style="color: #10B981; font-size: 1.1rem;">✓</span>
                            <strong style="color: #F8FAFC; font-size: 0.95rem;">{filename}</strong>
                            <span style="font-size: 0.78rem; color: #94A3B8;">({file_size_kb:.1f} KB)</span>
                        </div>
                        <div style="font-size: 0.82rem; color: #A7F3D0;">
                            Already indexed in Qdrant Cloud collection <code>{config.qdrant_collection_name}</code> with <strong>{existing_chunks} chunks</strong>. (ID: <code>{doc_id}</code>)
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="rag-card">
                        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 4px;">
                            <span style="color: #8B5CF6; font-size: 1.1rem;">📄</span>
                            <strong style="color: #F8FAFC; font-size: 0.95rem;">{filename}</strong>
                            <span style="font-size: 0.78rem; color: #94A3B8;">({file_size_kb:.1f} KB)</span>
                        </div>
                        <div style="font-size: 0.82rem; color: #94A3B8;">
                            Deterministic SHA-256 Document ID: <code style="color: #C4B5FD;">{doc_id}</code>. Ready for extraction, chunking, and embedding.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with col_action:
            button_label = "🔄 Re-Index" if already_indexed else "🚀 Index to Cloud"
            start_indexing = st.button(button_label, use_container_width=True, type="primary")

        # Reuse Indexed Document Action
        if already_indexed and (not st.session_state.active_document or st.session_state.active_document.get("doc_id") != doc_id):
            if st.button("👉 Use this indexed document for chat", use_container_width=False):
                st.session_state.active_document = {
                    "doc_id": doc_id,
                    "filename": filename,
                    "total_chunks": existing_chunks,
                }
                st.rerun()

        # Ingestion Execution Pipeline
        if start_indexing:
            with st.status("Processing PDF through RAG Pipeline...", expanded=True) as status:
                try:
                    status.write("📄 **Step 1/4:** Loading PDF and preserving 1-indexed page metadata...")
                    pages = load_pdf_from_bytes(file_bytes, filename)
                    status.write(f"✓ Extracted {len(pages)} page(s) with human-friendly page numbers.")

                    status.write(
                        f"✂️ **Step 2/4:** Splitting text into overlapping chunks ({config.chunk_size} chars, {config.chunk_overlap} overlap)..."
                    )
                    chunks = split_document_pages(
                        documents=pages,
                        doc_id=doc_id,
                        filename=filename,
                        chunk_size=config.chunk_size,
                        chunk_overlap=config.chunk_overlap,
                    )
                    status.write(f"✓ Created {len(chunks)} chunks with doc_id and page metadata attached.")

                    status.write(
                        f"🧠 **Step 3/4:** Generating dense embeddings via `{config.embedding_model_name}`..."
                    )
                    embeddings = get_embedding_model(config.embedding_model_name)
                    status.write("✓ Embedding model ready (384-dimensional normalized vectors).")

                    status.write(
                        f"☁️ **Step 4/4:** Indexing vectors and metadata into Qdrant Cloud `{config.qdrant_collection_name}`..."
                    )
                    indexed_count = index_chunks_to_qdrant(
                        client=qdrant_client,
                        embeddings_model=embeddings,
                        collection_name=config.qdrant_collection_name,
                        chunks=chunks,
                        doc_id=doc_id,
                        overwrite_existing=True,
                    )
                    
                    st.session_state.active_document = {
                        "doc_id": doc_id,
                        "filename": filename,
                        "total_pages": len(pages),
                        "total_chunks": indexed_count,
                    }
                    
                    status.update(
                        label=f"🎉 Successfully indexed '{filename}' ({indexed_count} chunks) in Qdrant Cloud!",
                        state="complete",
                        expanded=False,
                    )
                    st.success(
                        f"Ready! You can now ask questions about `{filename}` in the chat section below."
                    )
                    st.rerun()

                except Exception as exc:
                    status.update(label=f"❌ Ingestion failed: {exc}", state="error")
                    st.error(f"Error during ingestion: {exc}")

    else:
        # Pre-upload clean dropzone illustration card
        st.markdown(
            """
            <div class="empty-hero">
                <div class="empty-icon">📁</div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #F8FAFC; margin-bottom: 4px;">Drop your knowledge here</div>
                <div style="font-size: 0.88rem; color: #94A3B8; max-width: 520px; margin: 0 auto; line-height: 1.5;">
                    Upload any text-based PDF to build an interactive vector search index. Our pipeline handles 1-indexed page conversion, recursive chunking, and Qdrant Cloud storage.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='border: 0; border-top: 1px solid rgba(255,255,255,0.08); margin: 1.5rem 0;'>", unsafe_allow_html=True)

    # ==========================================
    # 3. AI CONVERSATION WORKSPACE (Zone C)
    # ==========================================
    st.markdown(
        """
        <div style="font-size: 1.15rem; font-weight: 700; color: #F8FAFC; display: flex; align-items: center; gap: 8px; margin-bottom: 0.5rem;">
            <span>💬</span> Grounded Conversation
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.active_document:
        st.markdown(
            """
            <div class="rag-card" style="text-align: center; padding: 2rem 1.5rem; color: #94A3B8;">
                <div style="font-size: 1.8rem; margin-bottom: 6px;">👆</div>
                <strong style="color: #E2E8F0; font-size: 0.95rem;">No Active Document Selected</strong><br>
                <span style="font-size: 0.84rem;">Please upload and index a PDF above to begin asking grounded questions.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        active_doc = st.session_state.active_document
        
        # Active document bar
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; justify-content: space-between; background: rgba(14, 21, 38, 0.7); border: 1px solid rgba(255,255,255,0.08); border-radius: 8px; padding: 6px 14px; margin-bottom: 1rem; font-size: 0.84rem;">
                <div>
                    <span style="color: #94A3B8;">Active Context:</span> <strong style="color: #F8FAFC;">{active_doc['filename']}</strong>
                </div>
                <div>
                    <span style="color: #64748B; font-family: monospace;">doc_id: {active_doc['doc_id']}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Empty conversation state with suggested question chips
        if len(st.session_state.messages) == 0:
            st.markdown(
                """
                <div class="empty-hero" style="padding: 1.8rem 1rem;">
                    <div style="font-size: 2rem; margin-bottom: 6px;">🤖</div>
                    <div style="font-size: 1.05rem; font-weight: 700; color: #F8FAFC; margin-bottom: 4px;">Ready for your questions</div>
                    <div style="font-size: 0.84rem; color: #94A3B8; margin-bottom: 1.2rem;">Answers will be synthesized strictly from the document excerpts with verified page citations.</div>
                    <div style="font-size: 0.78rem; font-weight: 700; color: #A78BFA; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">Example Questions to Explore:</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Interactive Example Question Chips
            col_q1, col_q2, col_q3 = st.columns(3)
            with col_q1:
                if st.button("📌 Summarize key concepts", use_container_width=True):
                    st.session_state.pending_query = "Summarize the primary topics and key concepts covered in this document."
                    st.rerun()
            with col_q2:
                if st.button("⚙️ Explain main methodology", use_container_width=True):
                    st.session_state.pending_query = "What methodology, features, or architecture does this document explain?"
                    st.rerun()
            with col_q3:
                if st.button("🛡️ Test out-of-scope refusal", use_container_width=True):
                    st.session_state.pending_query = "What is the speed of light in vacuum?"
                    st.rerun()

        # Render conversation history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
                if msg["role"] == "assistant":
                    citations = msg.get("citations", {})
                    citation_summary = citations.get("citation_summary", "")
                    
                    if citation_summary and "No citations" not in citation_summary:
                        st.markdown(
                            f"""
                            <div class="citation-box">
                                <strong>📄 Verified Source Citations:</strong> {citation_summary}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    
                    chunks = msg.get("chunks", [])
                    if chunks:
                        with st.expander(f"🔍 Retrieval Inspector ({len(chunks)} passages from Qdrant Cloud)", expanded=show_diagnostics):
                            for i, ch in enumerate(chunks, 1):
                                pg = ch.get("page", 1)
                                score = ch.get("score")
                                score_badge = f"{score:.4f}" if score is not None else "N/A"
                                char_len = len(ch.get("content", ""))
                                
                                st.markdown(
                                    f"""
                                    <div class="retrieval-chunk-card">
                                        <div class="chunk-meta-bar">
                                            <span><strong>Passage {i}</strong> &bull; Source Page: <strong>{pg}</strong> &bull; {char_len} chars</span>
                                            <span class="chunk-score-pill">Cosine Score: {score_badge}</span>
                                        </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                                st.code(ch.get("content", ""), language="text")

        # Handle Chat Input & Pending Chips
        user_input = st.chat_input("Ask a question about the uploaded PDF document...")
        
        # Check if an example question chip was clicked
        active_query = None
        if user_input:
            active_query = user_input
        elif st.session_state.pending_query:
            active_query = st.session_state.pending_query
            st.session_state.pending_query = None

        if active_query:
            # Append user message
            st.session_state.messages.append({"role": "user", "content": active_query})
            with st.chat_message("user"):
                st.markdown(active_query)

            # Assistant response generation
            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching Qdrant Cloud vectors & generating grounded answer..."):
                    try:
                        qdrant_client = get_qdrant_client(config.qdrant_url, config.qdrant_api_key)
                        embeddings = get_embedding_model(config.embedding_model_name)
                        
                        retrieved_docs = retrieve_relevant_chunks(
                            client=qdrant_client,
                            embeddings_model=embeddings,
                            collection_name=config.qdrant_collection_name,
                            query=active_query,
                            doc_id=active_doc["doc_id"],
                            top_k=config.top_k,
                        )
                        
                        result = generate_grounded_answer(
                            query=active_query,
                            retrieved_chunks=retrieved_docs,
                            openrouter_api_key=config.openrouter_api_key,
                            model_name=config.openrouter_model,
                        )
                        
                        answer_text = result["answer"]
                        citations = result["citations"]
                        citation_summary = citations.get("citation_summary", "")
                        
                        st.markdown(answer_text)
                        
                        if citation_summary and "No citations" not in citation_summary:
                            st.markdown(
                                f"""
                                <div class="citation-box">
                                    <strong>📄 Verified Source Citations:</strong> {citation_summary}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            
                        chunk_details = [
                            {
                                "page": doc.metadata.get("page", 1),
                                "score": doc.metadata.get("similarity_score"),
                                "content": doc.page_content,
                            }
                            for doc in retrieved_docs
                        ]
                        
                        if chunk_details:
                            with st.expander(f"🔍 Retrieval Inspector ({len(chunk_details)} passages from Qdrant Cloud)", expanded=show_diagnostics):
                                for i, ch in enumerate(chunk_details, 1):
                                    pg = ch["page"]
                                    sc = ch["score"]
                                    score_badge = f"{sc:.4f}" if sc is not None else "N/A"
                                    char_len = len(ch["content"])
                                    st.markdown(
                                        f"""
                                        <div class="retrieval-chunk-card">
                                            <div class="chunk-meta-bar">
                                                <span><strong>Passage {i}</strong> &bull; Source Page: <strong>{pg}</strong> &bull; {char_len} chars</span>
                                                <span class="chunk-score-pill">Cosine Score: {score_badge}</span>
                                            </div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )
                                    st.code(ch["content"], language="text")

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": answer_text,
                                "citations": citations,
                                "chunks": chunk_details,
                                "model": result.get("model_used"),
                            }
                        )

                    except Exception as query_exc:
                        err_msg = f"❌ Error retrieving answer: {query_exc}"
                        st.error(err_msg)
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": err_msg,
                                "citations": {},
                                "chunks": [],
                            }
                        )
