"""
TF-IDF Search Engine
====================
Core module for Term Frequency - Inverse Document Frequency computation.

This module provides the complete TF-IDF pipeline:
- Building a sorted vocabulary from processed documents.
- Computing Term Frequency (TF) as a sparse CSR matrix.
- Computing Inverse Document Frequency (IDF) as a NumPy array.
- Combining TF and IDF into a TF-IDF weighted sparse matrix.
- Ranked document search by summing TF-IDF scores across query terms.
- Paginated matrix views for UI display.

Uses scipy sparse matrices (CSR format) for memory-efficient storage
and computation on large vocabularies.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from dataclasses import dataclass


@dataclass
class TFIDFIndex:
    """Container for storing all TF-IDF indexing results.

    This dataclass bundles every artifact produced by the indexing
    pipeline so that downstream consumers (search, display) can
    access them from a single object.

    Attributes:
        tfidf_matrix (csr_matrix): Sparse matrix of shape (n_docs, n_terms)
            where each cell holds the TF-IDF weight of a term in a document.
            Computed as ``TF(t,d) * IDF(t)``.
        tf_matrix (csr_matrix): Sparse matrix of shape (n_docs, n_terms)
            where each cell holds the normalized term frequency
            ``count(t in d) / total_words(d)``.
        idf_values (np.ndarray): 1-D array of length n_terms containing the
            IDF value for each term.  ``IDF(t) = log(N / DF(t))``.
        vocabulary (list[str]): Alphabetically sorted list of every unique
            term found across all documents.
        vocab_to_idx (dict[str, int]): Mapping from term string to its
            column index in the matrices, enabling O(1) lookup.
        doc_names (list[str]): Ordered list of document filenames matching
            the row order of the matrices.
        doc_term_counts (dict[str, int]): Total number of tokens in each
            document after preprocessing (used as the TF denominator).
    """

    tfidf_matrix: csr_matrix       # sparse matrix (docs x terms)
    tf_matrix: csr_matrix          # sparse matrix TF (docs x terms)
    idf_values: np.ndarray         # IDF per term
    vocabulary: list[str]          # sorted list of terms
    vocab_to_idx: dict[str, int]   # term -> column index
    doc_names: list[str]           # ordered list of document names
    doc_term_counts: dict[str, int]  # jumlah total kata per dokumen


def build_vocabulary(processed_docs: dict[str, list[str]]) -> list[str]:
    """Build a sorted vocabulary of unique terms from all documents.

    Iterates over every document's token list and collects all distinct
    tokens into a set, then returns them sorted alphabetically.  The
    sorted order ensures deterministic column indices in the matrices.

    Args:
        processed_docs: Dictionary mapping document names to their
            preprocessed token lists.
            Example: ``{"doc1.pdf": ["sistem", "informasi"], ...}``

    Returns:
        list[str]: Alphabetically sorted list of unique terms across
        the entire document collection.

    Example:
        >>> docs = {"a.pdf": ["cat", "dog"], "b.pdf": ["dog", "fish"]}
        >>> build_vocabulary(docs)
        ['cat', 'dog', 'fish']
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
    """Compute the Term Frequency (TF) sparse matrix for all documents.

    For each document, counts the occurrences of every term and normalizes
    by the total number of tokens in that document:

        TF(t, d) = count(t in d) / total_words(d)

    The result is stored in a scipy CSR (Compressed Sparse Row) matrix for
    efficient storage — only non-zero entries are kept in memory.

    Args:
        processed_docs: Dictionary mapping document names to their
            preprocessed token lists.
        vocabulary: Sorted list of all unique terms (defines column order).
        vocab_to_idx: Mapping from term string to column index in the matrix.

    Returns:
        tuple: A 2-element tuple containing:
            - **tf_matrix** (csr_matrix): Sparse matrix of shape
              ``(n_docs, n_terms)`` with normalized TF values.
            - **doc_term_counts** (dict[str, int]): Dictionary mapping each
              document name to its total token count (the denominator used
              for normalization).

    Note:
        Documents with zero tokens are skipped (no entries added to the
        sparse matrix), but they still appear as all-zero rows.
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
    """Compute the Inverse Document Frequency (IDF) for every term.

    IDF measures how rare or common a term is across the entire document
    collection.  Rare terms receive higher IDF values, making them more
    discriminative in search.

        IDF(t) = log(N / DF(t))

    Where:
        - N is the total number of documents.
        - DF(t) is the number of documents that contain term t (at least once).

    Args:
        processed_docs: Dictionary mapping document names to their
            preprocessed token lists.
        vocabulary: Sorted list of all unique terms (defines array order).
        vocab_to_idx: Mapping from term string to index in the IDF array.

    Returns:
        np.ndarray: 1-D NumPy array of length ``len(vocabulary)`` containing
        the IDF value for each term.  Terms that appear in zero documents
        (edge case) receive an IDF of 0.0.

    Note:
        Uses natural logarithm (``math.log``).  If a term appears in every
        document, ``IDF = log(N/N) = 0``, which effectively neutralizes
        that term's contribution to TF-IDF scoring.
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
    """Build the complete TF-IDF index from preprocessed documents.

    This is the main entry point for the indexing pipeline.  It orchestrates
    the full workflow:

    1. Build a sorted vocabulary from all documents.
    2. Create a term-to-index mapping for O(1) column lookups.
    3. Compute the TF sparse matrix.
    4. Compute the IDF array.
    5. Multiply TF x IDF (via diagonal matrix multiplication) to produce
       the final TF-IDF sparse matrix.
    6. Bundle everything into a ``TFIDFIndex`` dataclass.

    Args:
        processed_docs: Dictionary mapping document names to their
            preprocessed token lists.
            Example: ``{"doc1.pdf": ["sistem", "informasi", ...], ...}``

    Returns:
        TFIDFIndex: A dataclass containing the TF-IDF matrix, TF matrix,
        IDF values, vocabulary, vocab-to-index mapping, document names,
        and per-document token counts.

    Example:
        >>> import text_processor as tp
        >>> raw = tp.load_documents("./dataset")
        >>> processed = {name: tp.preprocess(text) for name, text in raw.items()}
        >>> index = build_tfidf_index(processed)
        >>> index.tfidf_matrix.shape
        (16, 4532)
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
    """Search for relevant documents using TF-IDF scored ranking.

    For each query token, retrieves its corresponding column from the
    TF-IDF matrix and accumulates scores across all documents.  Documents
    are then ranked in descending order of their total score.

    The scoring formula for a document d given query tokens Q is:

        score(d, Q) = Σ  TF-IDF(t, d)   for each t ∈ Q

    Args:
        query_tokens: List of preprocessed query tokens (should be
            processed through the same NLP pipeline as the documents).
        index: A ``TFIDFIndex`` object produced by ``build_tfidf_index()``.

    Returns:
        list[tuple[str, float]]: List of ``(document_name, score)`` tuples
        sorted in descending order by score.  Only documents with a
        score > 0 are included.  Returns an empty list if ``query_tokens``
        is empty or no tokens match any term in the vocabulary.

    Example:
        >>> results = search(["sistem", "informasi"], index)
        >>> for doc, score in results[:3]:
        ...     print(f"{doc}: {score:.4f}")
        doc1.pdf: 0.1523
        doc3.pdf: 0.0841
        doc7.pdf: 0.0312
    """
    if not query_tokens:
        return []
    
    scores = np.zeros(len(index.doc_names))
    
    for token in query_tokens:
        if token in index.vocab_to_idx:
            col_idx = index.vocab_to_idx[token]
            # Retrieve the TF-IDF column for this term (all documents)
            col_data = index.tfidf_matrix.getcol(col_idx).toarray().flatten()
            scores += col_data
    
    # Build result list, filter score > 0
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
    """Retrieve a single page of the TF-IDF matrix as a pandas DataFrame.

    Since the full TF-IDF matrix can have thousands of term columns, this
    function paginates the vocabulary axis so the UI can display manageable
    chunks.  Only the requested slice of columns is converted from sparse
    to dense format.

    Args:
        index: A ``TFIDFIndex`` object produced by ``build_tfidf_index()``.
        page: 1-indexed page number.  Clamped to ``[1, total_pages]``.
        per_page: Number of term columns to include per page.  Defaults
            to 100.

    Returns:
        tuple: A 2-element tuple containing:
            - **df** (pd.DataFrame): DataFrame where rows are documents
              (indexed by filename) and columns are the vocabulary terms
              for the requested page.  Values are TF-IDF weights.
            - **total_pages** (int): Total number of pages available.

    Example:
        >>> df, total = get_matrix_page(index, page=1, per_page=50)
        >>> df.shape
        (16, 50)
        >>> total
        91
    """
    total_terms = len(index.vocabulary)
    total_pages = max(1, math.ceil(total_terms / per_page))
    
    # Clamp page
    page = max(1, min(page, total_pages))
    
    start_idx = (page - 1) * per_page
    end_idx = min(start_idx + per_page, total_terms)
    
    # Extract subset of columns
    page_terms = index.vocabulary[start_idx:end_idx]
    col_indices = list(range(start_idx, end_idx))
    
    # Extract subset from sparse matrix -> dense
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
    """Retrieve a single page of the TF matrix as a pandas DataFrame.

    Behaves identically to ``get_matrix_page()`` but reads from
    ``index.tf_matrix`` (normalized term frequencies) instead of
    the TF-IDF matrix.

    Args:
        index: A ``TFIDFIndex`` object produced by ``build_tfidf_index()``.
        page: 1-indexed page number.  Clamped to ``[1, total_pages]``.
        per_page: Number of term columns to include per page.  Defaults
            to 100.

    Returns:
        tuple: A 2-element tuple containing:
            - **df** (pd.DataFrame): DataFrame where rows are documents
              and columns are vocabulary terms for the requested page.
              Values are normalized TF: ``count(t,d) / total_words(d)``.
            - **total_pages** (int): Total number of pages available.
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
    """Return all IDF values as a pandas DataFrame for UI display.

    Constructs a three-column DataFrame with one row per vocabulary term,
    showing the term name, its Document Frequency (DF), and its IDF value.

    The DF is reverse-computed from the IDF using:
        ``DF = N / exp(IDF)``  (rounded to nearest integer)

    This avoids storing DF separately since it can be derived from IDF.

    Args:
        index: A ``TFIDFIndex`` object produced by ``build_tfidf_index()``.

    Returns:
        pd.DataFrame: DataFrame with columns:
            - **Term** (str): The vocabulary term.
            - **DF** (int): Number of documents containing this term.
            - **IDF** (float): The IDF value, rounded to 6 decimal places.

    Example:
        >>> idf_df = get_idf_dataframe(index)
        >>> idf_df.head()
           Term  DF       IDF
        0  abad   2  2.079442
        1  acak   1  2.772589
        2  acara  3  1.673976
    """
    n_docs = len(index.doc_names)
    rows = []
    for i, term in enumerate(index.vocabulary):
        idf_val = index.idf_values[i]
        # Reverse-compute DF from IDF: IDF = log(N/DF) -> DF = N / exp(IDF)
        if idf_val > 0:
            df_val = round(n_docs / math.exp(idf_val))
        else:
            df_val = n_docs
        rows.append({"Term": term, "DF": df_val, "IDF": round(idf_val, 6)})
    
    return pd.DataFrame(rows)
