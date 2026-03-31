import streamlit as st
import text_processor as tp
import engine
 
st.set_page_config(
    page_title="Sistem Temu Kembali Informasi - Boolean Model",
    layout="wide",
)

st.markdown(
    """
    <style>
    /* Header gradient */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { margin: 0; font-size: 2rem; }
    .main-header p  { margin: 0.4rem 0 0; opacity: 0.85; font-size: 1.05rem; }

    /* Stat cards */
    .stat-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem 1.25rem;
        border-radius: 10px;
        text-align: center;
    }
    .stat-card h3 { margin: 0; font-size: 2rem; color: #4a3f9f; }
    .stat-card p  { margin: 0.25rem 0 0; color: #555; font-size: 0.9rem; }

    /* Result card */
    .result-card {
        background: #ffffff;
        border-left: 4px solid #667eea;
        padding: 1rem 1.25rem;
        border-radius: 6px;
        margin-bottom: 0.75rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }
    .result-card h4 { margin: 0 0 0.5rem; color: #4a3f9f; }
    .result-card p  { margin: 0; color: #444; font-size: 0.92rem; line-height: 1.55; }

    /* Tidak ada hasil */
    .no-result {
        text-align: center;
        padding: 2rem;
        color: #888;
        font-size: 1.1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="main-header">
        <h1>Sistem Temu Kembali Informasi</h1>
        <p>Boolean Retrieval Model - Pencarian dokumen teks dengan operator AND, OR, NOT</p>
    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("Konfigurasi")
    dataset_path = st.text_input(
        "Path Folder Dokumen",
        value="./dataset",
    )
    btn_load = st.button("Load Dataset", use_container_width=True)

# Session state
if "loaded" not in st.session_state:
    st.session_state.loaded = False

# Proses dokumen
if btn_load:
    with st.spinner("Memuat dan memproses dokumen…"):
        raw_docs = tp.load_documents(dataset_path)
        if not raw_docs:
            st.error("Tidak ada file .pdf ditemukan di folder tersebut.")
            st.stop()

        processed_docs: dict[str, list[str]] = {}
        for name, content in raw_docs.items():
            processed_docs[name] = tp.preprocess(content)

        inc_matrix = engine.build_incidence_matrix(processed_docs)
        inv_index = engine.build_inverted_index(processed_docs)

        # Simpan ke session_state
        st.session_state.raw_docs = raw_docs
        st.session_state.processed_docs = processed_docs
        st.session_state.inc_matrix = inc_matrix
        st.session_state.inv_index = inv_index
        st.session_state.all_docs = set(raw_docs.keys())
        st.session_state.loaded = True

    st.success(f"Berhasil memuat **{len(raw_docs)}** dokumen!")

# Tampilkan hasil pemrosesan
if st.session_state.loaded:
    raw_docs = st.session_state.raw_docs
    inc_matrix = st.session_state.inc_matrix
    inv_index = st.session_state.inv_index

    # Stat cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f'<div class="stat-card"><h3>{len(raw_docs)}</h3><p>Dokumen</p></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="stat-card"><h3>{len(inc_matrix)}</h3><p>Unique Terms</p></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="stat-card"><h3>{len(inv_index)}</h3><p>Inverted Index Entries</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Incidence Matrix & Inverted Index (tabs)
    tab_matrix, tab_index = st.tabs(["📊 Incidence Matrix", "📚 Inverted Index"])

    with tab_matrix:
        st.subheader("Incidence Matrix (Terms x Documents)")
        st.dataframe(inc_matrix, use_container_width=True, height=400)

    with tab_index:
        st.subheader("Inverted Index")
        st.json(inv_index)

    st.markdown("---")

    # Pencarian Boolean
    st.subheader("Pencarian Boolean")
    st.caption(
        "Gunakan operator **AND**, **OR**, **NOT** (huruf besar). "
        "Contoh: `sistem AND informasi NOT komputer`"
    )

    query = st.text_input("Masukkan Query", placeholder="teknologi AND informasi")

    if st.button("🔍 Cari", use_container_width=True):
        if not query.strip():
            st.warning("Silakan masukkan query terlebih dahulu.")
        else:
            q_tokens = engine.preprocess_query(query)
            st.info(f"Query setelah preprocessing: **{' '.join(q_tokens)}**")

            results = engine.evaluate_boolean_query(
                q_tokens, inv_index, st.session_state.all_docs
            )

            if results:
                st.success(f"Ditemukan **{len(results)}** dokumen yang relevan.")
                for doc_name in sorted(results):
                    snippet = raw_docs[doc_name][:300]
                    if len(raw_docs[doc_name]) > 300:
                        snippet += "…"
                    st.markdown(
                        f"""
                        <div class="result-card">
                            <h4>📄 {doc_name}</h4>
                            <p>{snippet}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    '<div class="no-result">Tidak ada dokumen yang cocok dengan query</div>',
                    unsafe_allow_html=True,
                )
else:
    st.info(
        "Masukkan path folder dokumen di sidebar, lalu klik **Load Dataset** untuk memulai"
    )
