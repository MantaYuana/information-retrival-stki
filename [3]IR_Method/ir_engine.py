"""
IR Engine - Fuzzy, GVSM, LSI
Core module for three Information Retrieval methods:
  1. Fuzzy Retrieval - membership-based document ranking
  2. Generalized Vector Space Model (GVSM) - term-term correlation
  3. Latent Semantic Indexing (LSI) - SVD dimensionality reduction

Each method provides step-by-step computation results for educational display in the Streamlit UI.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from dataclasses import dataclass, field
from numpy.linalg import norm

#  Shared Data Structures
@dataclass
class IRIndex:
    """Container for shared indexing artifacts.

    Attributes:
        tf_matrix: Dense matrix (n_docs × n_terms) - normalized TF.
        idf_values: 1-D array of IDF per term.
        tfidf_matrix: Dense matrix (n_docs × n_terms) - TF × IDF.
        vocabulary: Sorted list of unique terms.
        vocab_to_idx: Term → column index mapping.
        doc_names: Ordered document names.
        doc_term_counts: Total tokens per document.
    """
    tf_matrix: np.ndarray
    idf_values: np.ndarray
    tfidf_matrix: np.ndarray
    vocabulary: list[str]
    vocab_to_idx: dict[str, int]
    doc_names: list[str]
    doc_term_counts: dict[str, int]

#  Shared: Vocabulary, TF, IDF, TF-IDF
def build_vocabulary(processed_docs: dict[str, list[str]]) -> list[str]:
    """Build sorted vocabulary from all documents."""
    vocab: set[str] = set()
    for tokens in processed_docs.values():
        vocab.update(tokens)
    return sorted(vocab)

def compute_tf_matrix(
    processed_docs: dict[str, list[str]],
    vocabulary: list[str],
    vocab_to_idx: dict[str, int],
) -> tuple[np.ndarray, dict[str, int]]:
    """Compute normalized TF matrix (n_docs × n_terms).

    TF(t, d) = count(t in d) / total_words(d)
    """
    doc_names = list(processed_docs.keys())
    n_docs = len(doc_names)
    n_terms = len(vocabulary)

    tf = np.zeros((n_docs, n_terms))
    doc_term_counts: dict[str, int] = {}

    for doc_idx, doc_name in enumerate(doc_names):
        tokens = processed_docs[doc_name]
        total_words = len(tokens)
        doc_term_counts[doc_name] = total_words

        if total_words == 0:
            continue

        term_freq: dict[str, int] = {}
        for token in tokens:
            term_freq[token] = term_freq.get(token, 0) + 1

        for term, count in term_freq.items():
            if term in vocab_to_idx:
                tf[doc_idx, vocab_to_idx[term]] = count / total_words

    return tf, doc_term_counts


def compute_idf(
    processed_docs: dict[str, list[str]],
    vocabulary: list[str],
    vocab_to_idx: dict[str, int],
) -> np.ndarray:
    """Compute IDF array. IDF(t) = log(N / DF(t))."""
    n_docs = len(processed_docs)
    n_terms = len(vocabulary)

    df = np.zeros(n_terms)
    for tokens in processed_docs.values():
        unique_tokens = set(tokens)
        for token in unique_tokens:
            if token in vocab_to_idx:
                df[vocab_to_idx[token]] += 1

    idf = np.zeros(n_terms)
    for i in range(n_terms):
        if df[i] > 0:
            idf[i] = math.log(n_docs / df[i])
        else:
            idf[i] = 0.0

    return idf


def build_index(processed_docs: dict[str, list[str]]) -> IRIndex:
    """Build the complete shared IR index."""
    vocabulary = build_vocabulary(processed_docs)
    vocab_to_idx = {term: idx for idx, term in enumerate(vocabulary)}
    doc_names = list(processed_docs.keys())

    tf_matrix, doc_term_counts = compute_tf_matrix(
        processed_docs, vocabulary, vocab_to_idx
    )
    idf_values = compute_idf(processed_docs, vocabulary, vocab_to_idx)

    # TF-IDF = TF * IDF (broadcast row-wise)
    tfidf_matrix = tf_matrix * idf_values[np.newaxis, :]

    return IRIndex(
        tf_matrix=tf_matrix,
        idf_values=idf_values,
        tfidf_matrix=tfidf_matrix,
        vocabulary=vocabulary,
        vocab_to_idx=vocab_to_idx,
        doc_names=doc_names,
        doc_term_counts=doc_term_counts,
    )


def build_query_vector(
    query_tokens: list[str],
    index: IRIndex,
) -> np.ndarray:
    """Build a TF-IDF weighted query vector."""
    n_terms = len(index.vocabulary)
    q_tf = np.zeros(n_terms)

    if not query_tokens:
        return q_tf

    for token in query_tokens:
        if token in index.vocab_to_idx:
            q_tf[index.vocab_to_idx[token]] += 1

    total = len(query_tokens)
    if total > 0:
        q_tf /= total

    q_tfidf = q_tf * index.idf_values
    return q_tfidf

#  1. FUZZY RETRIEVAL
def fuzzy_search(
    query_tokens: list[str],
    index: IRIndex,
    mode: str = "AND",
) -> dict:
    """Perform Fuzzy IR search with step-by-step results.

    Steps:
      1. TF Matrix
      2. IDF Values
      3. TF-IDF Matrix
      4. Fuzzy Membership (min-max normalization per term)
      5. Query Membership Vector
      6. Fuzzy Score per document (AND=min, OR=max)
      7. Ranked Results

    Args:
        query_tokens: Preprocessed query tokens.
        index: Shared IR index.
        mode: "AND" (min) or "OR" (max).

    Returns:
        dict with keys: steps (list of step dicts), results (ranked list).
    """
    steps = []
    n_docs = len(index.doc_names)
    n_terms = len(index.vocabulary)

    # Step 1: TF Matrix
    tf_df = pd.DataFrame(
        index.tf_matrix,
        index=index.doc_names,
        columns=index.vocabulary,
    )
    steps.append({
        "title": "Step 1: Matriks Term Frequency (TF)",
        "desc": "TF(t, d) = jumlah_kemunculan(t, d) / total_kata(d). "
                "Menghitung frekuensi relatif setiap term di setiap dokumen.",
        "data": tf_df,
        "type": "dataframe",
    })

    # Step 2: IDF Values
    idf_data = []
    for i, term in enumerate(index.vocabulary):
        idf_val = index.idf_values[i]
        if idf_val > 0:
            df_val = round(n_docs / math.exp(idf_val))
        else:
            df_val = n_docs
        idf_data.append({"Term": term, "DF": df_val, "IDF": round(idf_val, 6)})
    idf_df = pd.DataFrame(idf_data)
    steps.append({
        "title": "Step 2: Inverse Document Frequency (IDF)",
        "desc": "IDF(t) = log(N / DF(t)). Term yang muncul di sedikit dokumen mendapat bobot lebih tinggi.",
        "data": idf_df,
        "type": "dataframe",
    })

    # Step 3: TF-IDF Matrix
    tfidf_df = pd.DataFrame(
        index.tfidf_matrix,
        index=index.doc_names,
        columns=index.vocabulary,
    )
    steps.append({
        "title": "Step 3: Matriks TF-IDF",
        "desc": "TF-IDF(t, d) = TF(t, d) × IDF(t). Menggabungkan frekuensi lokal dengan kepentingan global.",
        "data": tfidf_df,
        "type": "dataframe",
    })

    # Step 4: Fuzzy Membership - min-max normalization per term
    # μ(t, d) = TF-IDF(t, d) / max_d(TF-IDF(t, d))
    max_per_term = index.tfidf_matrix.max(axis=0)
    max_per_term[max_per_term == 0] = 1  # avoid division by zero

    membership_matrix = index.tfidf_matrix / max_per_term[np.newaxis, :]

    membership_df = pd.DataFrame(
        membership_matrix,
        index=index.doc_names,
        columns=index.vocabulary,
    )
    steps.append({
        "title": "Step 4: Matriks Fuzzy Membership (μ)",
        "desc": "μ(t, d) = TF-IDF(t, d) / max(TF-IDF(t, ·)). "
                "Normalisasi ke rentang [0, 1] - derajat keanggotaan term terhadap dokumen.",
        "data": membership_df,
        "type": "dataframe",
    })

    # Step 5: Query Membership
    query_term_indices = []
    query_terms_found = []
    for token in query_tokens:
        if token in index.vocab_to_idx:
            idx = index.vocab_to_idx[token]
            query_term_indices.append(idx)
            query_terms_found.append(token)

    if not query_term_indices:
        steps.append({
            "title": "Step 5: Query Membership",
            "desc": "Tidak ada term query yang ditemukan di vocabulary.",
            "data": None,
            "type": "text",
        })
        return {"steps": steps, "results": []}

    query_membership = membership_matrix[:, query_term_indices]
    query_membership_df = pd.DataFrame(
        query_membership,
        index=index.doc_names,
        columns=query_terms_found,
    )
    steps.append({
        "title": "Step 5: Fuzzy Membership untuk Query Terms",
        "desc": f"Mengambil kolom membership untuk term query: {query_terms_found}. "
                f"Setiap nilai menunjukkan derajat keanggotaan term di dokumen tersebut.",
        "data": query_membership_df,
        "type": "dataframe",
    })

    # Step 6: Fuzzy Score
    if mode == "AND":
        scores = query_membership.min(axis=1)
        mode_desc = "AND (MIN): score(d) = min(μ(t₁,d), μ(t₂,d), ...) - semua term harus relevan."
    else:
        scores = query_membership.max(axis=1)
        mode_desc = "OR (MAX): score(d) = max(μ(t₁,d), μ(t₂,d), ...) - cukup satu term yang relevan."

    score_df = pd.DataFrame({
        "Dokumen": index.doc_names,
        "Fuzzy Score": np.round(scores, 6),
    })
    steps.append({
        "title": f"Step 6: Fuzzy Similarity Score ({mode})",
        "desc": mode_desc,
        "data": score_df,
        "type": "dataframe",
    })

    # Step 7: Ranked Results
    results = []
    for i, doc_name in enumerate(index.doc_names):
        if scores[i] > 0:
            results.append((doc_name, float(scores[i])))
    results.sort(key=lambda x: x[1], reverse=True)

    result_df = pd.DataFrame(
        [(r[0], round(r[1], 6)) for r in results],
        columns=["Dokumen", "Skor"],
    ) if results else pd.DataFrame(columns=["Dokumen", "Skor"])

    steps.append({
        "title": "Step 7: Ranked Results",
        "desc": "Dokumen diurutkan berdasarkan skor tertinggi (paling relevan di atas). "
                "Hanya dokumen dengan skor > 0 yang ditampilkan.",
        "data": result_df,
        "type": "dataframe",
    })

    return {"steps": steps, "results": results}

#  2. GENERALIZED VECTOR SPACE MODEL (GVSM)
def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = norm(a)
    norm_b = norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def gvsm_search(
    query_tokens: list[str],
    index: IRIndex,
) -> dict:
    """Perform GVSM search with step-by-step results.

    Steps:
      1. TF-IDF Matrix
      2. Term-Term Correlation Matrix: M = A^T × A
      3. GVSM Document Vectors: D_gvsm = A × M
      4. GVSM Query Vector: q_gvsm = q × M
      5. Cosine Similarity
      6. Ranked Results

    Args:
        query_tokens: Preprocessed query tokens.
        index: Shared IR index.

    Returns:
        dict with keys: steps (list of step dicts), results (ranked list).
    """
    steps = []
    n_docs = len(index.doc_names)
    n_terms = len(index.vocabulary)

    # Step 1: TF-IDF Matrix
    tfidf_df = pd.DataFrame(
        index.tfidf_matrix,
        index=index.doc_names,
        columns=index.vocabulary,
    )
    steps.append({
        "title": "Step 1: Matriks TF-IDF (A)",
        "desc": "Matriks bobot TF-IDF - basis untuk menghitung korelasi antar term. "
                "Dimensi: (dokumen × term).",
        "data": tfidf_df,
        "type": "dataframe",
    })

    # Step 2: Term-Term Correlation Matrix
    # M = A^T × A (n_terms × n_terms)
    A = index.tfidf_matrix  # (n_docs × n_terms)
    correlation_matrix = A.T @ A  # (n_terms × n_terms)

    # Show limited view for display (max 50×50)
    display_limit = min(50, n_terms)
    corr_display = correlation_matrix[:display_limit, :display_limit]
    corr_df = pd.DataFrame(
        corr_display,
        index=index.vocabulary[:display_limit],
        columns=index.vocabulary[:display_limit],
    )
    steps.append({
        "title": "Step 2: Term-Term Correlation Matrix (M = Aᵀ × A)",
        "desc": f"Matriks korelasi antar-term. M(i,j) = Σ TF-IDF(tᵢ,dₖ) × TF-IDF(tⱼ,dₖ). "
                f"Dimensi: ({n_terms} × {n_terms}). "
                f"Ditampilkan: {display_limit} × {display_limit} term pertama.",
        "data": corr_df,
        "type": "dataframe",
    })

    # Step 3: GVSM Document Vectors
    # D_gvsm = A × M (n_docs × n_terms)
    doc_gvsm = A @ correlation_matrix

    doc_gvsm_display = doc_gvsm[:, :display_limit]
    doc_gvsm_df = pd.DataFrame(
        doc_gvsm_display,
        index=index.doc_names,
        columns=index.vocabulary[:display_limit],
    )
    steps.append({
        "title": "Step 3: Vektor Dokumen GVSM (D = A × M)",
        "desc": "Representasi dokumen yang diperluas berdasarkan korelasi term. "
                "Dokumen yang mengandung term berkorelasi mendapat bobot tambahan. "
                f"Ditampilkan: {display_limit} term pertama.",
        "data": doc_gvsm_df,
        "type": "dataframe",
    })

    # Step 4: GVSM Query Vector
    q_tfidf = build_query_vector(query_tokens, index)

    if np.all(q_tfidf == 0):
        steps.append({
            "title": "Step 4: Vektor Query GVSM",
            "desc": "Tidak ada term query yang ditemukan di vocabulary.",
            "data": None,
            "type": "text",
        })
        return {"steps": steps, "results": []}

    q_gvsm = q_tfidf @ correlation_matrix

    # Show query vectors
    query_terms_found = [t for t in query_tokens if t in index.vocab_to_idx]
    nonzero_indices = np.nonzero(q_gvsm)[0]
    if len(nonzero_indices) > 0:
        display_indices = nonzero_indices[:30]  # Show top 30 non-zero
        q_display_data = {
            "Term": [index.vocabulary[i] for i in display_indices],
            "Query TF-IDF": [round(q_tfidf[i], 6) for i in display_indices],
            "Query GVSM": [round(q_gvsm[i], 6) for i in display_indices],
        }
        q_df = pd.DataFrame(q_display_data)
    else:
        q_df = pd.DataFrame(columns=["Term", "Query TF-IDF", "Query GVSM"])

    steps.append({
        "title": "Step 4: Vektor Query GVSM (q_gvsm = q × M)",
        "desc": f"Query terms: {query_terms_found}. "
                f"Vektor query diperluas dengan korelasi term sehingga "
                f"memperhitungkan term-term yang berkorelasi dengan query.",
        "data": q_df,
        "type": "dataframe",
    })

    # Step 5: Cosine Similarity
    similarities = []
    for i in range(n_docs):
        sim = _cosine_similarity(q_gvsm, doc_gvsm[i])
        similarities.append(sim)

    sim_df = pd.DataFrame({
        "Dokumen": index.doc_names,
        "Cosine Similarity": [round(s, 6) for s in similarities],
    })
    steps.append({
        "title": "Step 5: Cosine Similarity",
        "desc": "cos(q, d) = (q · d) / (||q|| × ||d||). "
                "Mengukur kesamaan sudut antara vektor query dan vektor dokumen di ruang GVSM.",
        "data": sim_df,
        "type": "dataframe",
    })

    # Step 6: Ranked Results
    results = []
    for i, doc_name in enumerate(index.doc_names):
        if similarities[i] > 0:
            results.append((doc_name, similarities[i]))
    results.sort(key=lambda x: x[1], reverse=True)

    result_df = pd.DataFrame(
        [(r[0], round(r[1], 6)) for r in results],
        columns=["Dokumen", "Skor"],
    ) if results else pd.DataFrame(columns=["Dokumen", "Skor"])

    steps.append({
        "title": "Step 6: Ranked Results",
        "desc": "Dokumen diurutkan berdasarkan cosine similarity tertinggi.",
        "data": result_df,
        "type": "dataframe",
    })

    return {"steps": steps, "results": results}

#  3. LATENT SEMANTIC INDEXING (LSI)
def lsi_search(
    query_tokens: list[str],
    index: IRIndex,
    k: int = 5,
) -> dict:
    """Perform LSI search with step-by-step results.

    Steps:
      1. TF-IDF Matrix (A)
      2. SVD Decomposition: A^T = U × Σ × V^T
      3. Truncated SVD (rank-k): U_k, Σ_k, V_k^T
      4. Reduced Document Space: D_k = Σ_k × V_k^T
      5. Query Projection: q_k = q^T × U_k × Σ_k^{-1}
      6. Cosine Similarity in reduced space
      7. Ranked Results

    Args:
        query_tokens: Preprocessed query tokens.
        index: Shared IR index.
        k: Number of singular values to keep.

    Returns:
        dict with keys: steps (list of step dicts), results (ranked list).
    """
    steps = []
    n_docs = len(index.doc_names)
    n_terms = len(index.vocabulary)

    # Clamp k
    max_k = min(n_docs, n_terms) - 1
    if max_k < 1:
        max_k = 1
    k = min(k, max_k)

    # Step 1: TF-IDF Matrix
    tfidf_df = pd.DataFrame(
        index.tfidf_matrix,
        index=index.doc_names,
        columns=index.vocabulary,
    )
    steps.append({
        "title": "Step 1: Matriks TF-IDF (A)",
        "desc": f"Matriks TF-IDF asli. Dimensi: ({n_docs} dokumen × {n_terms} term). "
                f"Akan didekomposisi menggunakan SVD.",
        "data": tfidf_df,
        "type": "dataframe",
    })

    # Step 2: SVD Decomposition on A^T (terms × docs)
    # A^T = U × Σ × V^T
    # where U: (terms × terms), Σ: diagonal, V^T: (docs × docs)
    A_T = index.tfidf_matrix.T  # (n_terms × n_docs)

    U_full, S_full, Vt_full = np.linalg.svd(A_T, full_matrices=False)
    # U_full: (n_terms × min), S_full: (min,), Vt_full: (min × n_docs)

    min_dim = min(n_docs, n_terms)

    svd_info_df = pd.DataFrame({
        "Singular Value Index": list(range(1, min(20, len(S_full)) + 1)),
        "Singular Value (σ)": [round(s, 6) for s in S_full[:20]],
        "Explained Variance %": [
            round(s**2 / (S_full**2).sum() * 100, 4) for s in S_full[:20]
        ],
        "Cumulative %": [
            round((S_full[:i+1]**2).sum() / (S_full**2).sum() * 100, 4)
            for i in range(min(20, len(S_full)))
        ],
    })

    steps.append({
        "title": "Step 2: SVD Decomposition (Aᵀ = U × Σ × Vᵀ)",
        "desc": f"Singular Value Decomposition pada Aᵀ ({n_terms}×{n_docs}). "
                f"U: ({n_terms}×{min_dim}), Σ: ({min_dim}×{min_dim}) diagonal, "
                f"Vᵀ: ({min_dim}×{n_docs}). "
                f"Menampilkan 20 singular value teratas.",
        "data": svd_info_df,
        "type": "dataframe",
    })

    # Step 3: Truncated SVD (rank-k)
    U_k = U_full[:, :k]       # (n_terms × k)
    S_k = S_full[:k]          # (k,)
    Vt_k = Vt_full[:k, :]     # (k × n_docs)

    # Display U_k (limited rows)
    display_rows = min(30, n_terms)
    U_k_df = pd.DataFrame(
        U_k[:display_rows, :],
        index=index.vocabulary[:display_rows],
        columns=[f"Dim {i+1}" for i in range(k)],
    )

    sigma_df = pd.DataFrame({
        "Dimensi": [f"σ{i+1}" for i in range(k)],
        "Nilai": [round(s, 6) for s in S_k],
    })

    Vt_k_df = pd.DataFrame(
        Vt_k,
        index=[f"Dim {i+1}" for i in range(k)],
        columns=index.doc_names,
    )

    steps.append({
        "title": f"Step 3: Truncated SVD (rank-{k})",
        "desc": f"Memotong ke {k} dimensi terpenting. "
                f"U_k: ({n_terms}×{k}) - representasi term. "
                f"Σ_k: ({k}×{k}) - bobot dimensi. "
                f"V_kᵀ: ({k}×{n_docs}) - representasi dokumen.",
        "data": {"U_k": U_k_df, "Σ_k": sigma_df, "Vt_k": Vt_k_df},
        "type": "multi_dataframe",
    })

    # Step 4: Reduced Document Space
    # D_k = Σ_k × V_k^T → each column is a doc in k-dim space
    # Shape: (k × n_docs), transpose for display: (n_docs × k)
    Sigma_k_diag = np.diag(S_k)  # (k × k)
    D_k = Sigma_k_diag @ Vt_k    # (k × n_docs)

    D_k_df = pd.DataFrame(
        D_k.T,
        index=index.doc_names,
        columns=[f"Konsep {i+1}" for i in range(k)],
    )
    steps.append({
        "title": "Step 4: Representasi Dokumen di Reduced Space (Σ_k × V_kᵀ)",
        "desc": f"Setiap dokumen direpresentasikan dalam {k} dimensi 'konsep'. "
                f"Dimensi mewakili pola semantik tersembunyi dalam koleksi dokumen.",
        "data": D_k_df,
        "type": "dataframe",
    })

    # Step 5: Query Projection
    q_tfidf = build_query_vector(query_tokens, index)

    if np.all(q_tfidf == 0):
        steps.append({
            "title": "Step 5: Proyeksi Query ke Reduced Space",
            "desc": "Tidak ada term query yang ditemukan di vocabulary.",
            "data": None,
            "type": "text",
        })
        return {"steps": steps, "results": []}

    # q_k = q^T × U_k × Σ_k^{-1}
    Sigma_k_inv = np.diag(1.0 / S_k)  # (k × k)
    q_k = q_tfidf @ U_k @ Sigma_k_inv  # (k,)

    query_terms_found = [t for t in query_tokens if t in index.vocab_to_idx]

    q_projection_df = pd.DataFrame({
        "Konsep": [f"Konsep {i+1}" for i in range(k)],
        "Nilai Query": [round(v, 6) for v in q_k],
    })

    steps.append({
        "title": "Step 5: Proyeksi Query ke Reduced Space (q_k = qᵀ × U_k × Σ_k⁻¹)",
        "desc": f"Query terms: {query_terms_found}. "
                f"Query dipetakan ke ruang {k}-dimensi yang sama dengan dokumen.",
        "data": q_projection_df,
        "type": "dataframe",
    })

    # Step 6: Cosine Similarity in reduced space
    # D_k columns are document vectors, q_k is query vector
    similarities = []
    for i in range(n_docs):
        doc_vec = D_k[:, i]  # (k,)
        sim = _cosine_similarity(q_k, doc_vec)
        similarities.append(sim)

    sim_df = pd.DataFrame({
        "Dokumen": index.doc_names,
        "Cosine Similarity": [round(s, 6) for s in similarities],
    })
    steps.append({
        "title": "Step 6: Cosine Similarity di Reduced Space",
        "desc": "cos(q_k, d_k) - kesamaan antara proyeksi query dan proyeksi dokumen "
                "di ruang konsep berdimensi rendah.",
        "data": sim_df,
        "type": "dataframe",
    })

    # Step 7: Ranked Results
    results = []
    for i, doc_name in enumerate(index.doc_names):
        if similarities[i] > 0:
            results.append((doc_name, similarities[i]))
    results.sort(key=lambda x: x[1], reverse=True)

    result_df = pd.DataFrame(
        [(r[0], round(r[1], 6)) for r in results],
        columns=["Dokumen", "Skor"],
    ) if results else pd.DataFrame(columns=["Dokumen", "Skor"])

    steps.append({
        "title": "Step 7: Ranked Results",
        "desc": "Dokumen diurutkan berdasarkan cosine similarity tertinggi di reduced space.",
        "data": result_df,
        "type": "dataframe",
    })

    return {"steps": steps, "results": results}



#  Utility: Paginated Matrix View


def get_matrix_page(
    matrix: np.ndarray,
    doc_names: list[str],
    vocabulary: list[str],
    page: int = 1,
    per_page: int = 100,
) -> tuple[pd.DataFrame, int]:
    """Paginate a matrix for display (docs × terms)."""
    total_terms = len(vocabulary)
    total_pages = max(1, math.ceil(total_terms / per_page))
    page = max(1, min(page, total_pages))

    start_idx = (page - 1) * per_page
    end_idx = min(start_idx + per_page, total_terms)

    page_terms = vocabulary[start_idx:end_idx]
    sub_matrix = matrix[:, start_idx:end_idx]

    df = pd.DataFrame(sub_matrix, index=doc_names, columns=page_terms)
    return df, total_pages


def get_idf_dataframe(index: IRIndex) -> pd.DataFrame:
    """Return IDF values as a DataFrame."""
    n_docs = len(index.doc_names)
    rows = []
    for i, term in enumerate(index.vocabulary):
        idf_val = index.idf_values[i]
        if idf_val > 0:
            df_val = round(n_docs / math.exp(idf_val))
        else:
            df_val = n_docs
        rows.append({"Term": term, "DF": df_val, "IDF": round(idf_val, 6)})
    return pd.DataFrame(rows)
