"""
Streamlit GUI — Fuzzy, GVSM, LSI Information Retrieval
Interactive web interface for comparing three IR methods with step-by-step computation display.
"""

import time
import re
import math
import numpy as np
import streamlit as st
import text_processor as tp
import ir_engine

# Page Config & Global Styles
st.set_page_config(
    page_title="IR Engine — Fuzzy, GVSM, LSI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global Reset */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .block-container { padding-top: 1rem; }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 2.2rem 2.8rem;
        border-radius: 16px;
        margin-bottom: 1.8rem;
        color: white;
        position: relative;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(48, 43, 99, 0.35);
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.15) 0%, transparent 70%);
        border-radius: 50%;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #fff 0%, #c7d2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .main-header p {
        margin: 0.5rem 0 0;
        opacity: 0.8;
        font-size: 1.05rem;
        font-weight: 300;
        color: #c7d2fe;
    }

    /* Stat Cards */
    .stat-card {
        background: linear-gradient(145deg, #1e1b4b 0%, #312e81 100%);
        padding: 1.3rem 1.5rem;
        border-radius: 14px;
        text-align: center;
        border: 1px solid rgba(99, 102, 241, 0.2);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(99, 102, 241, 0.25);
    }
    .stat-card h3 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-card p {
        margin: 0.3rem 0 0;
        color: #a5b4fc;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Section Headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin: 1.5rem 0 1rem;
    }
    .section-header h2 {
        margin: 0;
        font-size: 1.35rem;
        font-weight: 700;
        color: #e0e7ff;
    }
    .section-header .badge {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }

    /* Method Badge */
    .method-badge {
        display: inline-block;
        padding: 0.25rem 0.85rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .badge-fuzzy {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: #1a1a2e;
    }
    .badge-gvsm {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
    }
    .badge-lsi {
        background: linear-gradient(135deg, #8b5cf6, #7c3aed);
        color: white;
    }

    /* Result Card */
    .result-card {
        background: linear-gradient(145deg, #1e1b4b 0%, #1e1b3a 100%);
        border-left: 4px solid;
        border-image: linear-gradient(180deg, #818cf8, #c084fc) 1;
        padding: 1.2rem 1.5rem;
        border-radius: 0 12px 12px 0;
        margin-bottom: 0.85rem;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.15);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .result-card:hover {
        transform: translateX(4px);
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.2);
    }
    .result-card .doc-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.6rem;
    }
    .result-card .doc-name {
        font-weight: 700;
        font-size: 1.05rem;
        color: #c7d2fe;
    }
    .result-card .doc-score {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .result-card .doc-snippet {
        color: #94a3b8;
        font-size: 0.9rem;
        line-height: 1.65;
    }

    /* Highlight (stabilo effect) */
    .highlight-term {
        background: linear-gradient(135deg, rgba(250, 204, 21, 0.35), rgba(251, 191, 36, 0.25));
        color: #fef08a;
        font-weight: 700;
        padding: 0.1rem 0.35rem;
        border-radius: 4px;
        border-bottom: 2px solid #facc15;
        box-decoration-break: clone;
        -webkit-box-decoration-break: clone;
    }

    /* Search Summary */
    .search-summary {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12), rgba(52, 211, 153, 0.08));
        border: 1px solid rgba(16, 185, 129, 0.25);
        padding: 0.8rem 1.2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        color: #6ee7b7;
        font-weight: 500;
        font-size: 0.95rem;
    }

    /* No Result */
    .no-result {
        text-align: center;
        padding: 3rem 2rem;
        color: #64748b;
        font-size: 1.15rem;
        background: rgba(30, 27, 75, 0.3);
        border-radius: 12px;
        border: 1px dashed rgba(99, 102, 241, 0.2);
    }
    .no-result .emoji {
        font-size: 3rem;
        margin-bottom: 0.5rem;
        display: block;
    }

    /* Step Card */
    .step-card {
        background: linear-gradient(145deg, rgba(30, 27, 75, 0.6), rgba(49, 46, 129, 0.4));
        border: 1px solid rgba(99, 102, 241, 0.15);
        border-radius: 12px;
        padding: 1rem 1.3rem;
        margin-bottom: 0.8rem;
    }
    .step-card h4 {
        color: #a5b4fc;
        margin: 0 0 0.4rem;
        font-size: 1rem;
        font-weight: 700;
    }
    .step-card p {
        color: #94a3b8;
        font-size: 0.88rem;
        margin: 0;
        line-height: 1.6;
    }

    /* Pagination */
    .pagination-info {
        text-align: center;
        color: #a5b4fc;
        font-size: 0.85rem;
        font-weight: 500;
        padding: 0.5rem;
    }

    /* Sidebar tweaks */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1e1b4b 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #c7d2fe;
    }

    /* Divider */
    .custom-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.3), transparent);
        margin: 1.5rem 0;
    }

    /* Matrix table styling */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.markdown(
    """
    <div class="main-header">
        <h1>🧠 IR Engine — Fuzzy, GVSM, LSI</h1>
        <p>Mesin Pencari Dokumen dengan tiga metode: Fuzzy Retrieval, Generalized Vector Space Model, dan Latent Semantic Indexing</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Session State Initialization
if "loaded" not in st.session_state:
    st.session_state.loaded = False
if "matrix_page" not in st.session_state:
    st.session_state.matrix_page = 1

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Konfigurasi")
    st.markdown("---")

    dataset_path = st.text_input(
        "📁 Path Folder Dataset",
        value="./dataset",
        help="Path ke folder yang berisi file-file PDF dokumen.",
    )

    st.markdown("### 🔧 Opsi Pra-pemrosesan")
    use_stopword = st.checkbox(
        "Hapus Stopword", value=True,
        help="Buang kata-kata umum (dan, atau, yang, dll)",
    )
    use_stemming = st.checkbox(
        "Aktifkan Stemming", value=True,
        help="Ubah kata ke bentuk dasarnya (Sastrawi)",
    )

    st.markdown("---")
    st.markdown("### 🧠 Pengaturan LSI")
    lsi_k = st.slider(
        "Jumlah Dimensi (k)",
        min_value=1,
        max_value=15,
        value=5,
        help="Jumlah singular values yang dipertahankan untuk LSI.",
    )

    st.markdown("---")
    btn_load = st.button(
        "🚀 Load & Proses Dataset",
        use_container_width=True,
        type="primary",
    )

    # Info box
    if st.session_state.loaded:
        st.markdown("---")
        st.markdown("### 📊 Status")
        st.success(f"✅ {len(st.session_state.raw_docs)} dokumen terindeks")
        st.info(f"📝 {len(st.session_state.ir_index.vocabulary)} unique terms")

# Load & Process Dataset
if btn_load:
    t_start = time.time()

    # Load PDFs
    st.markdown(
        '<div class="section-header"><h2>📥 Memuat Dokumen</h2></div>',
        unsafe_allow_html=True,
    )
    progress_bar = st.progress(0, text="Memulai...")

    raw_docs: dict[str, str] = {}
    doc_loader = tp.load_documents_with_progress(dataset_path)

    for current, total, filename, docs in doc_loader:
        progress_bar.progress(
            current / total * 0.4,
            text=f"📄 Membaca ({current}/{total}): {filename}",
        )
        raw_docs = docs

    if not raw_docs:
        st.error("❌ Tidak ada file .pdf ditemukan di folder tersebut.")
        st.stop()

    # Preprocessing
    progress_bar.progress(0.45, text="🔤 Melakukan pra-pemrosesan teks...")

    processed_docs: dict[str, list[str]] = {}
    doc_list = list(raw_docs.items())
    for i, (name, content) in enumerate(doc_list):
        processed_docs[name] = tp.preprocess(
            content,
            use_stopword_removal=use_stopword,
            use_stemming=use_stemming,
        )
        progress_bar.progress(
            0.45 + (i + 1) / len(doc_list) * 0.35,
            text=f"🔤 Preprocessing ({i+1}/{len(doc_list)}): {name}",
        )

    # Build IR Index
    progress_bar.progress(0.85, text="📊 Membangun IR Index...")
    ir_index = ir_engine.build_index(processed_docs)

    progress_bar.progress(1.0, text="✅ Selesai!")
    t_end = time.time()

    # Save to session state
    st.session_state.raw_docs = raw_docs
    st.session_state.processed_docs = processed_docs
    st.session_state.ir_index = ir_index
    st.session_state.process_time = round(t_end - t_start, 2)
    st.session_state.use_stopword = use_stopword
    st.session_state.use_stemming = use_stemming
    st.session_state.loaded = True
    st.session_state.matrix_page = 1

    time.sleep(0.5)
    st.rerun()

# Helper Functions
def _generate_highlighted_snippet(
    raw_text: str,
    original_query: str,
    stemmed_query_terms: set[str],
    max_len: int = 350,
) -> str:
    """Generate a text snippet with highlighted query terms"""
    raw_query_words = set()
    for word in original_query.split():
        cleaned = re.sub(r"[^a-zA-Z]", "", word).lower()
        if cleaned:
            raw_query_words.add(cleaned)

    text_lower = raw_text.lower()
    best_pos = 0
    for word in raw_query_words:
        pos = text_lower.find(word)
        if pos != -1:
            best_pos = max(0, pos - 60)
            break

    snippet = raw_text[best_pos:best_pos + max_len]
    if best_pos > 0:
        snippet = "…" + snippet
    if best_pos + max_len < len(raw_text):
        snippet += "…"

    snippet = snippet.replace("<", "&lt;").replace(">", "&gt;")

    for word in raw_query_words:
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        snippet = pattern.sub(
            lambda m: f'<span class="highlight-term">{m.group()}</span>',
            snippet,
        )

    return snippet

def _render_step(step: dict, key_prefix: str = ""):
    """Render a single computation step inside an expander"""
    with st.expander(step["title"], expanded=False):
        st.markdown(
            f'<div class="step-card"><p>{step["desc"]}</p></div>',
            unsafe_allow_html=True,
        )

        if step["type"] == "dataframe" and step["data"] is not None:
            df = step["data"]
            if isinstance(df, pd.DataFrame):
                st.dataframe(
                    df.style.format(
                        {col: "{:.6f}" for col in df.select_dtypes(include="float").columns},
                        na_rep="—",
                    ),
                    use_container_width=True,
                    height=min(400, 35 * len(df) + 60),
                )
        elif step["type"] == "multi_dataframe" and step["data"] is not None:
            data = step["data"]
            for label, df in data.items():
                st.markdown(f"**{label}:**")
                if isinstance(df, pd.DataFrame):
                    st.dataframe(
                        df.style.format(
                            {col: "{:.6f}" for col in df.select_dtypes(include="float").columns},
                            na_rep="—",
                        ),
                        use_container_width=True,
                        height=min(300, 35 * len(df) + 60),
                    )
        elif step["type"] == "text":
            st.info(step.get("desc", "No data"))

def _render_paginated_matrix(
    index: ir_engine.IRIndex,
    matrix_type: str = "tfidf",
    key_prefix: str = "tfidf",
    per_page: int = 100,
):
    """Render a paginated matrix view."""
    page_key = f"{key_prefix}_page"
    if page_key not in st.session_state:
        st.session_state[page_key] = 1

    if matrix_type == "tfidf":
        matrix = index.tfidf_matrix
    else:
        matrix = index.tf_matrix

    df_page, total_pages = ir_engine.get_matrix_page(
        matrix, index.doc_names, index.vocabulary,
        page=st.session_state[page_key], per_page=per_page,
    )

    # Pagination controls
    nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns([1, 1, 2, 1, 1])

    with nav_col1:
        if st.button("⏮ Awal", key=f"{key_prefix}_first", use_container_width=True):
            st.session_state[page_key] = 1
            st.rerun()
    with nav_col2:
        if st.button("◀ Prev", key=f"{key_prefix}_prev", use_container_width=True):
            if st.session_state[page_key] > 1:
                st.session_state[page_key] -= 1
                st.rerun()
    with nav_col3:
        current = st.session_state[page_key]
        start_term = (current - 1) * per_page + 1
        end_term = min(current * per_page, len(index.vocabulary))
        st.markdown(
            f'<div class="pagination-info">'
            f'Halaman <b>{current}</b> / {total_pages} '
            f'(term {start_term}–{end_term} dari {len(index.vocabulary):,})</div>',
            unsafe_allow_html=True,
        )
    with nav_col4:
        if st.button("Next ▶", key=f"{key_prefix}_next", use_container_width=True):
            if st.session_state[page_key] < total_pages:
                st.session_state[page_key] += 1
                st.rerun()
    with nav_col5:
        if st.button("Akhir ⏭", key=f"{key_prefix}_last", use_container_width=True):
            st.session_state[page_key] = total_pages
            st.rerun()

    st.dataframe(
        df_page.style.format("{:.4f}"),
        use_container_width=True,
        height=420,
    )

# Main Content (after loading)
if st.session_state.loaded:
    raw_docs = st.session_state.raw_docs
    processed_docs = st.session_state.processed_docs
    ir_index: ir_engine.IRIndex = st.session_state.ir_index
    process_time = st.session_state.process_time

    # Stat Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f'<div class="stat-card"><h3>{len(raw_docs)}</h3><p>📄 Dokumen</p></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="stat-card"><h3>{len(ir_index.vocabulary):,}</h3><p>📝 Unique Terms</p></div>',
            unsafe_allow_html=True,
        )
    with col3:
        non_zero = int(np.count_nonzero(ir_index.tfidf_matrix))
        st.markdown(
            f'<div class="stat-card"><h3>{non_zero:,}</h3><p>🔢 Non-zero Values</p></div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f'<div class="stat-card"><h3>{process_time}s</h3><p>⏱️ Waktu Proses</p></div>',
            unsafe_allow_html=True,
        )

    # Preprocessing options display
    opts = []
    if st.session_state.get("use_stopword", True):
        opts.append("✅ Stopword Removal")
    else:
        opts.append("❌ Stopword Removal")
    if st.session_state.get("use_stemming", True):
        opts.append("✅ Stemming")
    else:
        opts.append("❌ Stemming")
    st.caption(f"Opsi pra-pemrosesan: {' • '.join(opts)}")

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    import pandas as pd

    # ───────────────────────────────────────────────────────
    #  Tabs
    # ───────────────────────────────────────────────────────
    tab_search, tab_fuzzy, tab_gvsm, tab_lsi, tab_tfidf, tab_idf = st.tabs([
        "🔍 Pencarian",
        "🔶 Step-by-Step Fuzzy",
        "🟢 Step-by-Step GVSM",
        "🟣 Step-by-Step LSI",
        "📊 Matriks TF-IDF",
        "📉 Nilai IDF",
    ])

    # ═══════════════════════════════════════════════════════
    #  TAB: Search
    # ═══════════════════════════════════════════════════════
    with tab_search:
        st.markdown(
            '<div class="section-header"><h2>Pencarian Dokumen</h2>'
            '<span class="badge">MULTI-METHOD</span></div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Masukkan kata kunci pencarian dan pilih metode IR. "
            "Dokumen akan diurutkan berdasarkan skor relevansi tertinggi."
        )

        query_col, method_col, btn_col = st.columns([4, 2, 1])
        with query_col:
            query = st.text_input(
                "Kata Kunci",
                placeholder="Contoh: sistem informasi teknologi",
                label_visibility="collapsed",
                key="search_query",
            )
        with method_col:
            method = st.selectbox(
                "Metode",
                ["Fuzzy (AND)", "Fuzzy (OR)", "GVSM", "LSI"],
                label_visibility="collapsed",
                key="search_method",
            )
        with btn_col:
            btn_search = st.button("🔍 Cari", use_container_width=True, type="primary")

        if btn_search:
            if not query.strip():
                st.warning("⚠️ Silakan masukkan kata kunci pencarian terlebih dahulu.")
            else:
                search_start = time.time()

                query_tokens = tp.preprocess(
                    query,
                    use_stopword_removal=st.session_state.get("use_stopword", True),
                    use_stemming=st.session_state.get("use_stemming", True),
                )

                st.info(f"🔤 Query setelah preprocessing: **{' '.join(query_tokens)}**")

                # Determine method and run search
                if method.startswith("Fuzzy"):
                    fuzzy_mode = "AND" if "AND" in method else "OR"
                    search_result = ir_engine.fuzzy_search(query_tokens, ir_index, mode=fuzzy_mode)
                    badge_class = "badge-fuzzy"
                    badge_text = f"FUZZY ({fuzzy_mode})"
                elif method == "GVSM":
                    search_result = ir_engine.gvsm_search(query_tokens, ir_index)
                    badge_class = "badge-gvsm"
                    badge_text = "GVSM"
                else:  # LSI
                    search_result = ir_engine.lsi_search(query_tokens, ir_index, k=lsi_k)
                    badge_class = "badge-lsi"
                    badge_text = f"LSI (k={lsi_k})"

                results = search_result["results"]
                search_time = round(time.time() - search_start, 4)

                st.markdown(
                    f'<span class="method-badge {badge_class}">{badge_text}</span>',
                    unsafe_allow_html=True,
                )

                if results:
                    st.markdown(
                        f'<div class="search-summary">'
                        f'✨ Ditemukan <b>{len(results)}</b> dokumen relevan dalam '
                        f'<b>{search_time}</b> detik</div>',
                        unsafe_allow_html=True,
                    )

                    query_terms_set = set(query_tokens)

                    for rank, (doc_name, score) in enumerate(results, 1):
                        raw_text = raw_docs[doc_name]
                        snippet = _generate_highlighted_snippet(
                            raw_text, query.strip(), query_terms_set, max_len=350
                        )

                        st.markdown(
                            f"""
                            <div class="result-card">
                                <div class="doc-header">
                                    <span class="doc-name">#{rank} 📄 {doc_name}</span>
                                    <span class="doc-score">Skor: {score:.6f}</span>
                                </div>
                                <div class="doc-snippet">{snippet}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.markdown(
                        '<div class="no-result">'
                        '<span class="emoji">🔍</span>'
                        'Tidak ada dokumen yang cocok dengan query tersebut'
                        '</div>',
                        unsafe_allow_html=True,
                    )

    # ═══════════════════════════════════════════════════════
    #  TAB: Step-by-Step Fuzzy
    # ═══════════════════════════════════════════════════════
    with tab_fuzzy:
        st.markdown(
            '<div class="section-header"><h2>Step-by-Step: Fuzzy Retrieval</h2>'
            '<span class="method-badge badge-fuzzy">FUZZY</span></div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Masukkan query untuk melihat langkah-langkah perhitungan metode Fuzzy IR. "
            "Setiap langkah dapat dibuka untuk melihat detail matriks dan nilai."
        )

        fcol1, fcol2, fcol3 = st.columns([4, 1.5, 1])
        with fcol1:
            fuzzy_query = st.text_input(
                "Query Fuzzy",
                placeholder="Contoh: sistem informasi",
                label_visibility="collapsed",
                key="fuzzy_query",
            )
        with fcol2:
            fuzzy_mode = st.selectbox(
                "Mode", ["AND (MIN)", "OR (MAX)"],
                label_visibility="collapsed",
                key="fuzzy_mode",
            )
        with fcol3:
            btn_fuzzy = st.button("▶ Hitung", key="btn_fuzzy", use_container_width=True, type="primary")

        if btn_fuzzy and fuzzy_query.strip():
            query_tokens = tp.preprocess(
                fuzzy_query,
                use_stopword_removal=st.session_state.get("use_stopword", True),
                use_stemming=st.session_state.get("use_stemming", True),
            )
            st.info(f"🔤 Query setelah preprocessing: **{' '.join(query_tokens)}**")

            mode = "AND" if "AND" in fuzzy_mode else "OR"
            result = ir_engine.fuzzy_search(query_tokens, ir_index, mode=mode)

            for step in result["steps"]:
                _render_step(step, key_prefix="fuzzy")

    # ═══════════════════════════════════════════════════════
    #  TAB: Step-by-Step GVSM
    # ═══════════════════════════════════════════════════════
    with tab_gvsm:
        st.markdown(
            '<div class="section-header"><h2>Step-by-Step: Generalized Vector Space Model</h2>'
            '<span class="method-badge badge-gvsm">GVSM</span></div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Masukkan query untuk melihat langkah-langkah perhitungan metode GVSM. "
            "GVSM memperluas VSM tradisional dengan memperhitungkan korelasi antar-term."
        )

        gcol1, gcol2 = st.columns([5, 1])
        with gcol1:
            gvsm_query = st.text_input(
                "Query GVSM",
                placeholder="Contoh: basis data relasional",
                label_visibility="collapsed",
                key="gvsm_query",
            )
        with gcol2:
            btn_gvsm = st.button("▶ Hitung", key="btn_gvsm", use_container_width=True, type="primary")

        if btn_gvsm and gvsm_query.strip():
            query_tokens = tp.preprocess(
                gvsm_query,
                use_stopword_removal=st.session_state.get("use_stopword", True),
                use_stemming=st.session_state.get("use_stemming", True),
            )
            st.info(f"🔤 Query setelah preprocessing: **{' '.join(query_tokens)}**")

            result = ir_engine.gvsm_search(query_tokens, ir_index)

            for step in result["steps"]:
                _render_step(step, key_prefix="gvsm")

    # ═══════════════════════════════════════════════════════
    #  TAB: Step-by-Step LSI
    # ═══════════════════════════════════════════════════════
    with tab_lsi:
        st.markdown(
            '<div class="section-header"><h2>Step-by-Step: Latent Semantic Indexing</h2>'
            '<span class="method-badge badge-lsi">LSI</span></div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Masukkan query untuk melihat langkah-langkah perhitungan metode LSI. "
            "LSI menggunakan SVD untuk menemukan 'konsep' tersembunyi dalam koleksi dokumen."
        )

        lcol1, lcol2, lcol3 = st.columns([4, 1.5, 1])
        with lcol1:
            lsi_query = st.text_input(
                "Query LSI",
                placeholder="Contoh: manajemen proyek perangkat lunak",
                label_visibility="collapsed",
                key="lsi_query",
            )
        with lcol2:
            lsi_k_step = st.number_input(
                "k (dimensi)",
                min_value=1,
                max_value=15,
                value=lsi_k,
                key="lsi_k_step",
            )
        with lcol3:
            btn_lsi = st.button("▶ Hitung", key="btn_lsi", use_container_width=True, type="primary")

        if btn_lsi and lsi_query.strip():
            query_tokens = tp.preprocess(
                lsi_query,
                use_stopword_removal=st.session_state.get("use_stopword", True),
                use_stemming=st.session_state.get("use_stemming", True),
            )
            st.info(f"🔤 Query setelah preprocessing: **{' '.join(query_tokens)}**")

            result = ir_engine.lsi_search(query_tokens, ir_index, k=int(lsi_k_step))

            for step in result["steps"]:
                _render_step(step, key_prefix="lsi")

    # ═══════════════════════════════════════════════════════
    #  TAB: TF-IDF Matrix
    # ═══════════════════════════════════════════════════════
    with tab_tfidf:
        st.markdown(
            '<div class="section-header"><h2>Matriks TF-IDF</h2>'
            '<span class="badge">FULL MATRIX</span></div>',
            unsafe_allow_html=True,
        )
        st.caption("Baris = Dokumen, Kolom = Term. Nilai = TF(t,d) × IDF(t)")

        _render_paginated_matrix(ir_index, matrix_type="tfidf", key_prefix="tfidf")

    # ═══════════════════════════════════════════════════════
    #  TAB: IDF Values
    # ═══════════════════════════════════════════════════════
    with tab_idf:
        st.markdown(
            '<div class="section-header"><h2>Nilai Inverse Document Frequency (IDF)</h2>'
            '<span class="badge">LOG SCALE</span></div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "IDF(t) = log(N / DF(t)), di mana N = total dokumen, "
            "DF(t) = jumlah dokumen mengandung term t"
        )

        idf_df = ir_engine.get_idf_dataframe(ir_index)

        idf_search = st.text_input(
            "🔎 Cari term di tabel IDF",
            placeholder="Ketik untuk filter...",
            key="idf_search",
        )
        if idf_search.strip():
            idf_df = idf_df[idf_df["Term"].str.contains(idf_search.strip(), case=False)]

        st.dataframe(
            idf_df,
            use_container_width=True,
            height=500,
            column_config={
                "Term": st.column_config.TextColumn("Term", width="medium"),
                "DF": st.column_config.NumberColumn("Document Frequency", format="%d"),
                "IDF": st.column_config.NumberColumn("IDF Value", format="%.6f"),
            },
        )
        st.caption(f"Total terms: {len(idf_df)}")

else:
    # ═══════════════════════════════════════════════════════
    #  Welcome Screen
    # ═══════════════════════════════════════════════════════
    st.markdown(
        """
        <div style="
            text-align: center;
            padding: 4rem 2rem;
            background: linear-gradient(145deg, rgba(30, 27, 75, 0.3), rgba(15, 12, 41, 0.3));
            border-radius: 16px;
            border: 1px dashed rgba(99, 102, 241, 0.25);
            margin-top: 1rem;
        ">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🧠</div>
            <h2 style="color: #c7d2fe; font-weight: 700; margin: 0 0 0.5rem;">
                Selamat Datang di IR Engine
            </h2>
            <p style="color: #94a3b8; font-size: 1.05rem; max-width: 600px; margin: 0 auto;">
                Mesin pencari dokumen dengan tiga metode Information Retrieval:<br>
                <b style="color: #f59e0b;">Fuzzy</b> •
                <b style="color: #10b981;">GVSM</b> •
                <b style="color: #8b5cf6;">LSI</b><br><br>
                Masukkan path folder dokumen PDF di sidebar, atur opsi pra-pemrosesan,<br>
                lalu klik <b style="color: #818cf8;">Load & Proses Dataset</b> untuk memulai.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Feature cards
    st.markdown("")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="stat-card">
                <h3 style="font-size: 2rem;">🔶</h3>
                <p style="font-size: 0.82rem; text-transform: none; letter-spacing: 0;">
                    <b>Fuzzy Retrieval</b> — Derajat keanggotaan term untuk perangkingan dokumen
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="stat-card">
                <h3 style="font-size: 2rem;">🟢</h3>
                <p style="font-size: 0.82rem; text-transform: none; letter-spacing: 0;">
                    <b>GVSM</b> — Korelasi antar-term memperluas representasi vektor tradisional
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="stat-card">
                <h3 style="font-size: 2rem;">🟣</h3>
                <p style="font-size: 0.82rem; text-transform: none; letter-spacing: 0;">
                    <b>LSI</b> — SVD menemukan konsep tersembunyi di balik koleksi dokumen
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
