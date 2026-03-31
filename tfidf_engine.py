"""
TF-IDF Search Engine
====================
Modul inti untuk perhitungan Term Frequency - Inverse Document Frequency.

Menggunakan scipy sparse matrix untuk efisiensi memori pada dataset besar.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from dataclasses import dataclass


@dataclass
class TFIDFIndex:
    """Container untuk menyimpan seluruh hasil indexing TF-IDF."""
    tfidf_matrix: csr_matrix       # sparse matrix (docs x terms)
    tf_matrix: csr_matrix          # sparse matrix TF (docs x terms)
    idf_values: np.ndarray         # IDF per term
    vocabulary: list[str]          # sorted list of terms
    vocab_to_idx: dict[str, int]   # term -> column index
    doc_names: list[str]           # ordered list of document names
    doc_term_counts: dict[str, int]  # jumlah total kata per dokumen


def build_vocabulary(processed_docs: dict[str, list[str]]) -> list[str]:
    """Kumpulkan semua unique terms dari seluruh dokumen, sorted.
    
    Args:
        processed_docs: {"doc_name": ["token1", "token2", ...], ...}
    
    Returns:
        Sorted list of unique terms
    """
    vocab: set[str] = set()
    for tokens in processed_docs.values():
        vocab.update(tokens)
    return sorted(vocab)


def compute_tf(
    processed_docs: dict[str, list[str]],
    vocabulary: list[str],
    vocab_to_idx: dict[str, int],
) -> tuple[csr_matrix, dict[str, int]]:
    """Hitung Term Frequency untuk setiap term di setiap dokumen.
    
    Formula: TF(t,d) = count(t in d) / total_words(d)
    
    Returns:
        (tf_matrix sebagai sparse CSR matrix [docs x terms], doc_term_counts)
    """
    doc_names = list(processed_docs.keys())
    n_docs = len(doc_names)
    n_terms = len(vocabulary)
    
    rows = []
    cols = []
    data = []
    doc_term_counts: dict[str, int] = {}
    
    for doc_idx, doc_name in enumerate(doc_names):
        tokens = processed_docs[doc_name]
        total_words = len(tokens)
        doc_term_counts[doc_name] = total_words
        
        if total_words == 0:
            continue
        
        # Count frequency of each term in this document
        term_freq: dict[str, int] = {}
        for token in tokens:
            term_freq[token] = term_freq.get(token, 0) + 1
        
        for term, count in term_freq.items():
            if term in vocab_to_idx:
                rows.append(doc_idx)
                cols.append(vocab_to_idx[term])
                data.append(count / total_words)
    
    tf_matrix = csr_matrix((data, (rows, cols)), shape=(n_docs, n_terms))
    return tf_matrix, doc_term_counts


def compute_idf(
    processed_docs: dict[str, list[str]],
    vocabulary: list[str],
    vocab_to_idx: dict[str, int],
) -> np.ndarray:
    """Hitung Inverse Document Frequency untuk setiap term.
    
    Formula: IDF(t) = log(N / DF(t))
    
    Returns:
        numpy array of IDF values (length = vocabulary size)
    """
    n_docs = len(processed_docs)
    n_terms = len(vocabulary)
    
    # Count document frequency for each term
    df = np.zeros(n_terms)
    for tokens in processed_docs.values():
        unique_tokens = set(tokens)
        for token in unique_tokens:
            if token in vocab_to_idx:
                df[vocab_to_idx[token]] += 1
    
    # Compute IDF, avoid division by zero
    idf = np.zeros(n_terms)
    for i in range(n_terms):
        if df[i] > 0:
            idf[i] = math.log(n_docs / df[i])
        else:
            idf[i] = 0.0
    
    return idf


def build_tfidf_index(processed_docs: dict[str, list[str]]) -> TFIDFIndex:
    """Bangun index TF-IDF lengkap dari dokumen yang sudah diproses.
    
    Args:
        processed_docs: {"doc_name": ["token1", "token2", ...], ...}
    
    Returns:
        TFIDFIndex containing all computed matrices and metadata
    """
    vocabulary = build_vocabulary(processed_docs)
    vocab_to_idx = {term: idx for idx, term in enumerate(vocabulary)}
    doc_names = list(processed_docs.keys())
    
    tf_matrix, doc_term_counts = compute_tf(processed_docs, vocabulary, vocab_to_idx)
    idf_values = compute_idf(processed_docs, vocabulary, vocab_to_idx)
    
    # TF-IDF = TF * IDF (element-wise multiply each column by its IDF)
    # Convert IDF to diagonal matrix for efficient multiplication
    idf_diag = csr_matrix(np.diag(idf_values))
    tfidf_matrix = tf_matrix.dot(idf_diag)
    
    return TFIDFIndex(
        tfidf_matrix=tfidf_matrix,
        tf_matrix=tf_matrix,
        idf_values=idf_values,
        vocabulary=vocabulary,
        vocab_to_idx=vocab_to_idx,
        doc_names=doc_names,
        doc_term_counts=doc_term_counts,
    )


def search(
    query_tokens: list[str],
    index: TFIDFIndex,
) -> list[tuple[str, float]]:
    """Cari dokumen yang relevan berdasarkan query menggunakan TF-IDF.
    
    Menghitung skor kumulatif TF-IDF dari kata-kata query terhadap 
    setiap dokumen, lalu mengurutkan descending.
    
    Args:
        query_tokens: List token query yang sudah dipreprocess
        index: TFIDFIndex hasil dari build_tfidf_index()
    
    Returns:
        List of (doc_name, score) sorted descending by score.
        Hanya dokumen dengan skor > 0 yang dikembalikan.
    """
    if not query_tokens:
        return []
    
    scores = np.zeros(len(index.doc_names))
    
    for token in query_tokens:
        if token in index.vocab_to_idx:
            col_idx = index.vocab_to_idx[token]
            # Ambil kolom TF-IDF untuk term ini (semua dokumen)
            col_data = index.tfidf_matrix.getcol(col_idx).toarray().flatten()
            scores += col_data
    
    # Buat list hasil, filter skor > 0
    results: list[tuple[str, float]] = []
    for doc_idx, score in enumerate(scores):
        if score > 0:
            results.append((index.doc_names[doc_idx], float(score)))
    
    # Sort descending by score
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def get_matrix_page(
    index: TFIDFIndex,
    page: int = 1,
    per_page: int = 100,
) -> tuple[pd.DataFrame, int]:
    """Ambil satu halaman matriks TF-IDF sebagai DataFrame.
    
    Args:
        index: TFIDFIndex
        page: Nomor halaman (1-indexed)
        per_page: Jumlah terms per halaman
    
    Returns:
        (DataFrame halaman matriks, total_pages)
        DataFrame: baris = dokumen, kolom = terms di halaman tersebut
    """
    total_terms = len(index.vocabulary)
    total_pages = max(1, math.ceil(total_terms / per_page))
    
    # Clamp page
    page = max(1, min(page, total_pages))
    
    start_idx = (page - 1) * per_page
    end_idx = min(start_idx + per_page, total_terms)
    
    # Ambil subset kolom
    page_terms = index.vocabulary[start_idx:end_idx]
    col_indices = list(range(start_idx, end_idx))
    
    # Extract subset dari sparse matrix -> dense
    sub_matrix = index.tfidf_matrix[:, col_indices].toarray()
    
    df = pd.DataFrame(
        sub_matrix,
        index=index.doc_names,
        columns=page_terms,
    )
    
    return df, total_pages


def get_tf_matrix_page(
    index: TFIDFIndex,
    page: int = 1,
    per_page: int = 100,
) -> tuple[pd.DataFrame, int]:
    """Ambil satu halaman matriks TF sebagai DataFrame.
    
    Returns:
        (DataFrame halaman matriks TF, total_pages)
    """
    total_terms = len(index.vocabulary)
    total_pages = max(1, math.ceil(total_terms / per_page))
    
    page = max(1, min(page, total_pages))
    
    start_idx = (page - 1) * per_page
    end_idx = min(start_idx + per_page, total_terms)
    
    page_terms = index.vocabulary[start_idx:end_idx]
    col_indices = list(range(start_idx, end_idx))
    
    sub_matrix = index.tf_matrix[:, col_indices].toarray()
    
    df = pd.DataFrame(
        sub_matrix,
        index=index.doc_names,
        columns=page_terms,
    )
    
    return df, total_pages


def get_idf_dataframe(index: TFIDFIndex) -> pd.DataFrame:
    """Return IDF values sebagai DataFrame untuk ditampilkan.
    
    Returns:
        DataFrame dengan kolom: Term, IDF, DF (document frequency)
    """
    n_docs = len(index.doc_names)
    rows = []
    for i, term in enumerate(index.vocabulary):
        idf_val = index.idf_values[i]
        # Hitung balik DF dari IDF: IDF = log(N/DF) -> DF = N / exp(IDF)
        if idf_val > 0:
            df_val = round(n_docs / math.exp(idf_val))
        else:
            df_val = n_docs
        rows.append({"Term": term, "DF": df_val, "IDF": round(idf_val, 6)})
    
    return pd.DataFrame(rows)
