# 📚 Sistem Temu Kembali Informasi — TF-IDF Search Engine

A **TF-IDF (Term Frequency – Inverse Document Frequency)** search engine built with Python and Streamlit for querying Indonesian-language PDF documents with ranked relevance scoring.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)

---

## 📖 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [How It Works](#how-it-works)
  - [Architecture](#architecture)
  - [Text Preprocessing Pipeline](#text-preprocessing-pipeline)
  - [TF-IDF Computation](#tf-idf-computation)
  - [Search & Ranking](#search--ranking)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the App](#running-the-app)
  - [Loading Documents](#loading-documents)
  - [Performing Searches](#performing-searches)
  - [Exploring Matrices](#exploring-matrices)
- [Dependencies](#dependencies)

---

## Overview

This project implements a **TF-IDF Retrieval Model** — a widely used weighted scoring model in Information Retrieval (IR). It allows users to load a collection of PDF documents, automatically preprocess the text (designed for **Bahasa Indonesia**), build TF, IDF, and TF-IDF matrices using **scipy sparse matrices** for memory efficiency, and perform ranked keyword searches to find the most relevant documents.

The application provides an interactive web interface powered by **Streamlit** with live statistics, paginated matrix exploration, IDF value browsing, and formatted search results with highlighted query terms.

---

## Features

- 📄 **PDF Document Loading** — Reads and extracts text from all `.pdf` files in a specified folder with real-time progress tracking.
- 🔤 **Indonesian NLP Preprocessing** — Case folding, tokenization, stopword removal, and stemming using the [Sastrawi](https://github.com/har07/PySastrawi) library. Both stopword removal and stemming are toggleable from the UI.
- 📊 **TF-IDF Matrix** — Sparse matrix (documents × terms) showing the TF-IDF weight of each term per document, displayed with pagination.
- 📈 **TF Matrix** — Normalized term frequency matrix viewable with pagination.
- 📉 **IDF Values** — Browsable and searchable table of Inverse Document Frequency values for every term in the vocabulary.
- 🔍 **Ranked Search** — Keyword-based search that ranks documents by cumulative TF-IDF score (highest relevance first).
- ✨ **Query Term Highlighting** — Search results display text snippets with highlighted matching keywords.
- ⚡ **Sparse Matrix Optimization** — Uses `scipy.sparse.csr_matrix` for efficient storage and computation on large vocabularies.
- 🎨 **Styled UI** — Custom-styled Streamlit interface with gradient headers, stat cards, result cards, and micro-animations.

---

## How It Works

### Architecture

The application is split into three modules with clearly separated responsibilities:

```
┌─────────────────────────────────────────────────────────┐
│                     app.py (UI Layer)                    │
│  Streamlit interface: config, display, user interaction  │
├─────────────────────────────────────────────────────────┤
│              tfidf_engine.py (IR Engine)                 │
│  TF-IDF computation, sparse matrices, ranked search     │
├─────────────────────────────────────────────────────────┤
│             text_processor.py (NLP Pipeline)             │
│  PDF loading, Case Folding, Tokenization, Stopwords,    │
│  Stemming (Sastrawi)                                    │
└─────────────────────────────────────────────────────────┘
```

| Module              | Responsibility                                                                           |
|---------------------|------------------------------------------------------------------------------------------|
| `text_processor.py` | Loads PDF files and runs the full NLP preprocessing pipeline (case fold → tokenize → stopword removal → stemming). |
| `tfidf_engine.py`   | Builds vocabulary, computes TF/IDF/TF-IDF sparse matrices, performs ranked search queries, and paginates matrix views. |
| `app.py`            | Streamlit UI — handles user interaction, session state, displays matrices/IDF tables, and renders search results with highlighting. |

### Text Preprocessing Pipeline

Every document (and every query term) goes through the following pipeline:

```
Raw Text
  │
  ▼
Case Folding ──► Lowercasing + remove non-alphabetic characters
  │
  ▼
Tokenization ──► Split text into individual word tokens
  │
  ▼
Stopword Removal ──► Remove Indonesian stopwords (via Sastrawi)  [toggleable]
  │
  ▼
Stemming ──► Reduce words to root form (via Sastrawi stemmer)    [toggleable]
  │
  ▼
Processed Tokens (list of stemmed words)
```

**Example:**

| Step              | Output                                       |
|-------------------|----------------------------------------------|
| Raw Text          | `"Sistem Temu-Kembali Informasi 2024"`       |
| Case Folding      | `"sistem temukembali informasi "`             |
| Tokenization      | `["sistem", "temukembali", "informasi"]`      |
| Stopword Removal  | `["sistem", "temukembali", "informasi"]`      |
| Stemming          | `["sistem", "temukembali", "informasi"]`      |

### TF-IDF Computation

The engine computes three key values for each term-document pair:

#### Term Frequency (TF)

Measures how frequently a term appears in a document, normalized by document length:

```
TF(t, d) = count(t in d) / total_words(d)
```

#### Inverse Document Frequency (IDF)

Measures how rare or common a term is across the entire collection:

```
IDF(t) = log(N / DF(t))
```

Where `N` is the total number of documents and `DF(t)` is the number of documents containing term `t`.

#### TF-IDF Weight

The final weight is the product of TF and IDF:

```
TF-IDF(t, d) = TF(t, d) × IDF(t)
```

Terms that are frequent in a specific document but rare across the corpus receive higher TF-IDF scores, making them more useful for distinguishing relevant documents.

### Search & Ranking

When a user enters a search query:

1. The query is preprocessed using the **same pipeline** as the documents (case folding → tokenization → stopword removal → stemming).
2. For each query token, the corresponding column in the TF-IDF matrix is retrieved.
3. TF-IDF scores are **summed across all query tokens** for each document.
4. Documents are **ranked in descending order** by total score.
5. Only documents with a score > 0 are returned.

---

## Project Structure

```
stki/
├── app.py                # Streamlit UI and application entry point
├── tfidf_engine.py       # TF-IDF engine (vocabulary, matrices, search)
├── text_processor.py     # NLP pipeline (PDF loading, preprocessing)
├── requirements.txt      # Python dependencies
├── dataset/              # Default folder for PDF documents
│   ├── doc1.pdf
│   ├── doc2.pdf
│   ├── ...
│   └── (16 sample PDFs)
└── README.md             # This file
```

---

## Prerequisites

- **Python 3.10+** (uses `dict[str, list[str]]` type-hint syntax)
- **pip** (Python package manager)

---

## Installation

1. **Clone the repository** (or download the source code):

   ```bash
   git clone <repository-url>
   cd stki
   ```

2. **Create a virtual environment** (recommended):

   ```bash
   python -m venv venv
   ```

   Activate it:

   - **Windows:**
     ```bash
     venv\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Running the App

Start the Streamlit server with:

```bash
python -m streamlit run app.py
```

The application will open automatically in your default web browser (typically at `http://localhost:8501`).

### Loading Documents

1. In the **sidebar**, you will see a "Path Folder Dataset" input field. The default value is `./dataset`.
2. Place your `.pdf` documents inside that folder (the `dataset/` folder already contains 16 sample PDFs).
3. Optionally toggle **Stopword Removal** and **Stemming** switches in the sidebar.
4. Click the **"🚀 Load & Proses Dataset"** button to read, preprocess, and index all PDF files.
5. Once loaded, you will see:
   - **Stat cards** showing the number of documents, unique terms, non-zero values, and processing time.
   - Four tabs: **Pencarian**, **Matriks TF-IDF**, **Matriks TF**, and **Nilai IDF**.

### Performing Searches

1. Go to the **"🔍 Pencarian"** tab.
2. Type your keywords in the text input (e.g., `sistem informasi teknologi`).
3. Click the **"🔍 Cari"** button.
4. The app will display:
   - The preprocessed query tokens.
   - The number of matching documents and search time.
   - A ranked result card for each matching document showing its filename, TF-IDF score, and a text snippet with highlighted query terms.

### Exploring Matrices

- **Matriks TF-IDF** tab — Paginated view of the full TF-IDF sparse matrix (documents × terms). Navigate pages to browse all vocabulary terms.
- **Matriks TF** tab — Paginated view of the normalized Term Frequency matrix.
- **Nilai IDF** tab — Searchable table of all terms with their Document Frequency (DF) and IDF values.

---

## Dependencies

| Package    | Purpose                                                            |
|------------|--------------------------------------------------------------------|
| `streamlit`| Web UI framework for interactive dashboards                        |
| `pandas`   | DataFrame for matrix display and IDF tables                        |
| `Sastrawi` | Indonesian NLP — stemming and stopword removal                     |
| `PyPDF2`   | PDF text extraction                                                |
| `scipy`    | Sparse matrix implementation (`csr_matrix`) for efficient TF-IDF   |
| `numpy`    | Numerical computation for IDF and score arrays                     |

Install all dependencies at once:

```bash
pip install -r requirements.txt
```