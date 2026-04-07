import time
import re
import streamlit as st
import text_processor as tp
import tfidf_engine

# Page Config & Global Styles
st.set_page_config(
    page_title="TF-IDF Search Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /*  Global Reset  */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .block-container { padding-top: 1rem; }

    /*  Header  */
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

    /*  Stat Cards  */
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

    /*  Section Headers  */
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

    /*  Result Card  */
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

    /*  Highlight (stabilo effect)  */
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

    /*  Summary Label  */
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

    /*  No Result  */
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

    /*  Pagination  */
    .pagination-info {
        text-align: center;
        color: #a5b4fc;
        font-size: 0.85rem;
        font-weight: 500;
        padding: 0.5rem;
    }

    /*  Sidebar tweaks  */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1e1b4b 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #c7d2fe;
    }

    /*  Divider  */
    .custom-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.3), transparent);
        margin: 1.5rem 0;
    }

    /*  Matrix table styling  */
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
        <h1>🔍 TF-IDF Search Engine</h1>
        <p>Mesin Pencari Dokumen berbasis Term Frequency - Inverse Document Frequency</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Session State Initialization
if "loaded" not in st.session_state:
    st.session_state.loaded = False
if "matrix_page" not in st.session_state:
    st.session_state.matrix_page = 1
if "matrix_tab_page" not in st.session_state:
    st.session_state.matrix_tab_page = 1

# Sidebar - Configuration & Input
with st.sidebar:
    st.markdown("## ⚙️ Konfigurasi")
    st.markdown("---")

    dataset_path = st.text_input(
        "📁 Path Folder Dataset",
        value="./dataset",
        help="Path ke folder yang berisi file-file PDF dokumen.",
    )

    st.markdown("### 🔧 Opsi Pra-pemrosesan")
    use_stopword = st.checkbox("Hapus Stopword", value=True, help="Buang kata-kata umum (dan, atau, yang, dll)")
    use_stemming = st.checkbox("Aktifkan Stemming", value=True, help="Ubah kata ke bentuk dasarnya (Sastrawi)")

    st.markdown("---")
    btn_load = st.button("🚀 Load & Proses Dataset", use_container_width=True, type="primary")

    # Info box
    if st.session_state.loaded:
        st.markdown("---")
        st.markdown("### 📊 Status")
        st.success(f"✅ {len(st.session_state.raw_docs)} dokumen terindeks")
        st.info(f"📝 {len(st.session_state.tfidf_index.vocabulary)} unique terms")

# Load & Process Dataset
if btn_load:
    t_start = time.time()

    # Phase 1: Load PDFs with progress
    st.markdown('<div class="section-header"><h2>📥 Memuat Dokumen</h2></div>', unsafe_allow_html=True)
    progress_bar = st.progress(0, text="Memulai...")

    raw_docs: dict[str, str] = {}
    doc_loader = tp.load_documents_with_progress(dataset_path)

    for current, total, filename, docs in doc_loader:
        progress_bar.progress(
            current / total * 0.4,  # 40% for loading
            text=f"📄 Membaca ({current}/{total}): {filename}",
        )
        raw_docs = docs

    if not raw_docs:
        st.error("❌ Tidak ada file .pdf ditemukan di folder tersebut.")
        st.stop()

    # Phase 2: Preprocessing
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

    # Phase 3: Build TF-IDF Index
    progress_bar.progress(0.85, text="📊 Menghitung matriks TF-IDF...")
    tfidf_index = tfidf_engine.build_tfidf_index(processed_docs)

    progress_bar.progress(1.0, text="✅ Selesai!")
    t_end = time.time()

    # Save to session state
    st.session_state.raw_docs = raw_docs
    st.session_state.processed_docs = processed_docs
    st.session_state.tfidf_index = tfidf_index
    st.session_state.process_time = round(t_end - t_start, 2)
    st.session_state.use_stopword = use_stopword
    st.session_state.use_stemming = use_stemming
    st.session_state.loaded = True
    st.session_state.matrix_page = 1
    st.session_state.matrix_tab_page = 1

    time.sleep(0.5)
    st.rerun()

# Helper Functions (defined before main content so they're available at runtime)
def _generate_highlighted_snippet(
    raw_text: str,
    original_query: str,
    stemmed_query_terms: set[str],
    max_len: int = 350,
) -> str:
    """Generate a text snippet with highlighted (stabilo) query terms.
    
    Highlights both the original query words and their stemmed forms
    found in the raw text.
    """
    # Build list of raw query words to highlight (case-insensitive)
    raw_query_words = set()
    for word in original_query.split():
        cleaned = re.sub(r"[^a-zA-Z]", "", word).lower()
        if cleaned:
            raw_query_words.add(cleaned)
    
    # Try to find a relevant section of text that contains query words
    text_lower = raw_text.lower()
    best_pos = 0
    for word in raw_query_words:
        pos = text_lower.find(word)
        if pos != -1:
            best_pos = max(0, pos - 60)
            break

    # Extract snippet
    snippet = raw_text[best_pos:best_pos + max_len]
    if best_pos > 0:
        snippet = "…" + snippet
    if best_pos + max_len < len(raw_text):
        snippet += "…"

    # Clean snippet for HTML
    snippet = snippet.replace("<", "&lt;").replace(">", "&gt;")

    # Highlight original query words in snippet using stabilo effect
    for word in raw_query_words:
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        snippet = pattern.sub(
            lambda m: f'<span class="highlight-term">{m.group()}</span>',
            snippet,
        )

    return snippet


def _render_paginated_matrix(
    index: tfidf_engine.TFIDFIndex,
    matrix_type: str = "tfidf",
    key_prefix: str = "tfidf",
    per_page: int = 100,
):
    """Render a paginated matrix with navigation controls."""
    
    page_key = f"{key_prefix}_page"
    if page_key not in st.session_state:
        st.session_state[page_key] = 1

    # Get the page
    if matrix_type == "tfidf":
        df_page, total_pages = tfidf_engine.get_matrix_page(
            index, page=st.session_state[page_key], per_page=per_page
        )
    else:
        df_page, total_pages = tfidf_engine.get_tf_matrix_page(
            index, page=st.session_state[page_key], per_page=per_page
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

    # Display matrix
    st.dataframe(
        df_page.style.format("{:.4f}"),
        use_container_width=True,
        height=420,
    )

# Main Content (after loading)
if st.session_state.loaded:
    raw_docs = st.session_state.raw_docs
    processed_docs = st.session_state.processed_docs
    tfidf_index: tfidf_engine.TFIDFIndex = st.session_state.tfidf_index
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
            f'<div class="stat-card"><h3>{len(tfidf_index.vocabulary):,}</h3><p>📝 Unique Terms</p></div>',
            unsafe_allow_html=True,
        )
    with col3:
        non_zero = tfidf_index.tfidf_matrix.nnz
        st.markdown(
            f'<div class="stat-card"><h3>{non_zero:,}</h3><p>🔢 Non-zero Values</p></div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f'<div class="stat-card"><h3>{process_time}s</h3><p>⏱️ Waktu Proses</p></div>',
            unsafe_allow_html=True,
        )

    # Preprocessing Options Display 
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

    # Tabs: Search | TF-IDF Matrix | TF Matrix | IDF Values 
    tab_search, tab_tfidf, tab_tf, tab_idf = st.tabs([
        "🔍 Pencarian",
        "📊 Matriks TF-IDF",
        "📈 Matriks TF",
        "📉 Nilai IDF",
    ])

    # TAB: Search Engine
    with tab_search:
        st.markdown(
            '<div class="section-header"><h2>Pencarian Dokumen</h2>'
            '<span class="badge">TF-IDF RANKED</span></div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Masukkan kata kunci pencarian. Dokumen akan diurutkan berdasarkan "
            "skor TF-IDF tertinggi (paling relevan di atas)."
        )

        query_col, btn_col = st.columns([5, 1])
        with query_col:
            query = st.text_input(
                "Kata Kunci",
                placeholder="Contoh: sistem informasi teknologi",
                label_visibility="collapsed",
            )
        with btn_col:
            btn_search = st.button("🔍 Cari", use_container_width=True, type="primary")

        if btn_search:
            if not query.strip():
                st.warning("⚠️ Silakan masukkan kata kunci pencarian terlebih dahulu.")
            else:
                search_start = time.time()

                # Preprocess query with same settings as documents
                query_tokens = tp.preprocess(
                    query,
                    use_stopword_removal=st.session_state.get("use_stopword", True),
                    use_stemming=st.session_state.get("use_stemming", True),
                )

                st.info(f"🔤 Query setelah preprocessing: **{' '.join(query_tokens)}**")

                # Search
                results = tfidf_engine.search(query_tokens, tfidf_index)
                search_time = round(time.time() - search_start, 4)

                if results:
                    # Summary label
                    st.markdown(
                        f'<div class="search-summary">'
                        f'✨ Ditemukan <b>{len(results)}</b> dokumen relevan dalam '
                        f'<b>{search_time}</b> detik</div>',
                        unsafe_allow_html=True,
                    )

                    # Build set of query terms for highlighting
                    query_terms_set = set(query_tokens)

                    for rank, (doc_name, score) in enumerate(results, 1):
                        # Generate snippet with highlighting
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

    
    # TAB: TF-IDF Matrix
    with tab_tfidf:
        st.markdown(
            '<div class="section-header"><h2>Matriks TF-IDF</h2>'
            '<span class="badge">SPARSE MATRIX</span></div>',
            unsafe_allow_html=True,
        )
        st.caption("Baris = Dokumen, Kolom = Term. Nilai = TF(t,d) × IDF(t)")

        _render_paginated_matrix(tfidf_index, matrix_type="tfidf", key_prefix="tfidf")

    
    # TAB: TF Matrix
    with tab_tf:
        st.markdown(
            '<div class="section-header"><h2>Matriks Term Frequency (TF)</h2>'
            '<span class="badge">NORMALIZED</span></div>',
            unsafe_allow_html=True,
        )
        st.caption("Baris = Dokumen, Kolom = Term. Nilai = count(t,d) / total_words(d)")

        _render_paginated_matrix(tfidf_index, matrix_type="tf", key_prefix="tf")

    
    # TAB: IDF Values
    with tab_idf:
        st.markdown(
            '<div class="section-header"><h2>Nilai Inverse Document Frequency (IDF)</h2>'
            '<span class="badge">LOG SCALE</span></div>',
            unsafe_allow_html=True,
        )
        st.caption("IDF(t) = log(N / DF(t)), di mana N = total dokumen, DF(t) = jumlah dokumen mengandung term t")

        idf_df = tfidf_engine.get_idf_dataframe(tfidf_index)

        # Search/filter IDF
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
    #  Welcome Screen 
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
            <div style="font-size: 4rem; margin-bottom: 1rem;">📚</div>
            <h2 style="color: #c7d2fe; font-weight: 700; margin: 0 0 0.5rem;">
                Selamat Datang di TF-IDF Search Engine
            </h2>
            <p style="color: #94a3b8; font-size: 1.05rem; max-width: 600px; margin: 0 auto;">
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
                <h3 style="font-size: 2rem;">📊</h3>
                <p style="font-size: 0.82rem; text-transform: none; letter-spacing: 0;">
                    Visualisasi matriks TF, IDF, dan TF-IDF dengan pagination
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="stat-card">
                <h3 style="font-size: 2rem;">🔍</h3>
                <p style="font-size: 0.82rem; text-transform: none; letter-spacing: 0;">
                    Pencarian dokumen ranked berdasarkan skor TF-IDF
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="stat-card">
                <h3 style="font-size: 2rem;">⚡</h3>
                <p style="font-size: 0.82rem; text-transform: none; letter-spacing: 0;">
                    Dioptimasi dengan Sparse Matrix untuk dataset besar
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


