from __future__ import annotations

import pandas as pd
import text_processor as tp

# Struktur Data IR
def build_incidence_matrix(
    processed_docs: dict[str, list[str]],
) -> pd.DataFrame:
    """Buat Incidence Matrix (terms x documents, nilai 0/1)
    Args:
        processed_docs: ``{"doc1.pdf": ["kata", "dasar", ...], ...}``
    Returns:
        pandas DataFrame - baris = terms, kolom = nama dokumen
    """
    
    # Kumpulkan vocabulary
    vocab: set[str] = set()
    for tokens in processed_docs.values():
        vocab.update(tokens)
    vocab_sorted = sorted(vocab)

    doc_names = list(processed_docs.keys())

    # Bangun matriks
    matrix: dict[str, list[int]] = {}
    for term in vocab_sorted:
        row = []
        for doc in doc_names:
            row.append(1 if term in processed_docs[doc] else 0)
        matrix[term] = row

    return pd.DataFrame.from_dict(matrix, orient="index", columns=doc_names)


def build_inverted_index(
    processed_docs: dict[str, list[str]],
)-> dict[str, list[str]]:
    """Buat Inverted Index
    Returns:
        ``{"term": ["doc1.txt", "doc3.txt"], ...}``
    """
    
    index: dict[str, list[str]] = {}
    for doc_name, tokens in processed_docs.items():
        for token in set(tokens): # set agar tidak duplikat per doc
            index.setdefault(token, [])
            if doc_name not in index[token]:
                index[token].append(doc_name)
                
    # sorting posting-list
    for term in index:
        index[term] = sorted(index[term])
    return dict(sorted(index.items()))


# query Processing
_OPERATORS = {"AND", "OR", "NOT"}

def preprocess_query(query: str) -> list[str]:
    """Preprocess query: keep operator AND/OR/NOT lalu stem sisanya
    Returns:
        List token yang sudah di-stem, operator tetap uppercase
        e.g. ``["cari", "AND", "informasi", "NOT", "komputer"]``
    """
    
    raw_tokens = query.strip().split()
    result: list[str] = []
    for token in raw_tokens:
        upper = token.upper()
        if upper in _OPERATORS:
            result.append(upper)
        else:
            processed = tp.preprocess(token)
            result.extend(processed)
    return result


def evaluate_boolean_query(
    query_tokens: list[str],
    inverted_index: dict[str, list[str]],
    all_docs: set[str],
) -> set[str]:
    """Evaluate query boolean but simple (kiri-ke-kanan).
    Mendukung operator: AND, OR, NOT.
    NOT berlaku sebagai *unary* pada term berikutnya (ga sebagai
    operator biner). Jika tidak ada operator eksplisit antara dua term,
    default-nya adalah AND
    Args:
        query_tokens: output dari ``preprocess_query()``.
        inverted_index: output dari ``build_inverted_index()``.
        all_docs: array semua nama dokumen
    Returns:
        array nama dokumen yang sesuai sama query
    """
    
    if not query_tokens:
        return set()

    def _get_postings(term: str) -> set[str]:
        return set(inverted_index.get(term, []))

    result: set[str] | None = None
    current_op = "AND" # default operator
    negate_next = False

    for token in query_tokens:
        if token in ("AND", "OR"):
            current_op = token
            continue
        if token == "NOT":
            negate_next = True
            continue

        # Token biasa -> ambil posting list
        term_docs = _get_postings(token)
        if negate_next:
            term_docs = all_docs - term_docs
            negate_next = False

        if result is None:
            result = term_docs
        elif current_op == "AND":
            result = result & term_docs
        elif current_op == "OR":
            result = result | term_docs

        # Reset operator ke default
        current_op = "AND"

    return result if result is not None else set()
