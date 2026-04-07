import os
import re
import glob
from PyPDF2 import PdfReader
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Sastrawi singletons (expensive to create, reuse)
_stemmer_factory = StemmerFactory()
stemmer = _stemmer_factory.create_stemmer()

_sw_factory = StopWordRemoverFactory()
stopword_remover = _sw_factory.create_stop_word_remover()
stopwords = set(_sw_factory.get_stop_words())


# Public API
def load_documents(folder_path: str) -> dict[str, str]:
    """membaca semua file .pdf dari path folder
    Returns:
        dict  - ``{"nama_file.pdf": "isi teks", ...}``
    """
    
    docs: dict[str, str] = {}
    pattern = os.path.join(folder_path, "*.pdf")
    for filepath in sorted(glob.glob(pattern)):
        filename = os.path.basename(filepath)
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        docs[filename] = text.strip()
    return docs


def load_documents_with_progress(folder_path: str):
    """Membaca semua file .pdf dengan yield progress.
    Yields:
        (current_index, total_files, filename, docs_so_far)
    Returns via final yield the complete docs dict.
    """
    pattern = os.path.join(folder_path, "*.pdf")
    filepaths = sorted(glob.glob(pattern))
    total = len(filepaths)
    docs: dict[str, str] = {}
    
    for i, filepath in enumerate(filepaths):
        filename = os.path.basename(filepath)
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        docs[filename] = text.strip()
        yield (i + 1, total, filename, docs)


def case_fold(text: str) -> str:
    """ubah ke huruf kecil & hapus karakter non-alfabetik kecuali spasi"""
    
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

def tokenize(text: str) -> list[str]:
    """Pecah teks menjadi list token (kata)"""
    
    return text.split()

def remove_stopwords(tokens: list[str]) -> list[str]:
    """Hapus stopword bahasa Indonesia pake Sastrawi"""
    
    return [t for t in tokens if t not in stopwords]

def stem_tokens(tokens: list[str]) -> list[str]:
    """Stemming setiap token ke kata dasarnya pakai Sastrawi"""
    
    return [stemmer.stem(t) for t in tokens]

def preprocess(text: str, use_stopword_removal: bool = True, use_stemming: bool = True) -> list[str]:
    """Pipeline lengkap: case-fold -> tokenize -> [stopword removal] -> [stemming]
    
    Args:
        text: Teks mentah yang akan diproses
        use_stopword_removal: Aktifkan/nonaktifkan penghapusan stopword
        use_stemming: Aktifkan/nonaktifkan stemming
    
    Returns:
        List token yang sudah diproses
    """
    
    text = case_fold(text)
    tokens = tokenize(text)
    if use_stopword_removal:
        tokens = remove_stopwords(tokens)
    if use_stemming:
        tokens = stem_tokens(tokens)
    return tokens
