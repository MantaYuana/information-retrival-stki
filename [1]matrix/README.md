# 📚 Sistem Temu Kembali Informasi — Boolean Retrieval Model

A **Boolean Retrieval** search engine built with Python and Streamlit for querying Indonesian-language PDF documents using **AND**, **OR**, and **NOT** operators.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)

---

## 📖 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [How It Works](#how-it-works)
  - [Architecture](#architecture)
  - [Text Preprocessing Pipeline](#text-preprocessing-pipeline)
  - [Data Structures](#data-structures)
  - [Boolean Query Evaluation](#boolean-query-evaluation)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the App](#running-the-app)
  - [Loading Documents](#loading-documents)
  - [Performing Queries](#performing-queries)
  - [Query Examples](#query-examples)
- [Dependencies](#dependencies)

---

## Overview

This project implements a classic **Boolean Retrieval Model** — one of the foundational models in Information Retrieval (IR). It allows users to load a collection of PDF documents, automatically preprocess the text (designed for **Bahasa Indonesia**), build an Incidence Matrix and Inverted Index, and perform Boolean queries to find relevant documents.

The application provides an interactive web interface powered by **Streamlit** with live statistics, visual data structure exploration, and formatted search results.

---

## Features

- 📄 **PDF Document Loading** — Reads and extracts text from all `.pdf` files in a specified folder.
- 🔤 **Indonesian NLP Preprocessing** — Case folding, tokenization, stopword removal, and stemming using the [Sastrawi](https://github.com/har07/PySastrawi) library.
- 📊 **Incidence Matrix** — Visual term × document binary matrix displayed as an interactive DataFrame.
- 📚 **Inverted Index** — Browsable JSON view of the inverted index mapping terms to their posting lists.
- 🔍 **Boolean Query Engine** — Supports `AND`, `OR`, and `NOT` operators with left-to-right evaluation.
- 🎨 **Styled UI** — Custom-styled Streamlit interface with gradient headers, stat cards, and result cards.

---

## How It Works

### Architecture

The application is split into three modules with clearly separated responsibilities:

```
┌─────────────────────────────────────────────────────────┐
│                     app.py (UI Layer)                    │
│  Streamlit interface: config, display, user interaction  │
├─────────────────────────────────────────────────────────┤
│                  engine.py (IR Engine)                   │
│  Incidence Matrix, Inverted Index, Query Evaluation      │
├─────────────────────────────────────────────────────────┤
│             text_processor.py (NLP Pipeline)             │
│  PDF loading, Case Folding, Tokenization, Stopwords,     │
│  Stemming (Sastrawi)                                     │
└─────────────────────────────────────────────────────────┘
```

| Module              | Responsibility                                                                 |
|---------------------|-------------------------------------------------------------------------------|
| `text_processor.py` | Loads PDF files and runs the full NLP preprocessing pipeline.                 |
| `engine.py`         | Builds the Incidence Matrix & Inverted Index; parses and evaluates Boolean queries. |
| `app.py`            | Streamlit UI — handles user interaction, displays data structures and results. |

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
Stopword Removal ──► Remove Indonesian stopwords (via Sastrawi)
  │
  ▼
Stemming ──► Reduce words to root form (via Sastrawi stemmer)
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

### Data Structures

#### Incidence Matrix

A binary matrix where rows are **terms** and columns are **documents**. A cell value of `1` indicates the term appears in the document, and `0` means it does not.

```
              doc1.pdf   doc2.pdf   doc3.pdf
sistem            1          0          1
informasi         1          1          0
komputer          0          1          1
```

#### Inverted Index

A dictionary mapping each term to a sorted **posting list** — the list of documents that contain that term.

```json
{
  "sistem": ["doc1.pdf", "doc3.pdf"],
  "informasi": ["doc1.pdf", "doc2.pdf"],
  "komputer": ["doc2.pdf", "doc3.pdf"]
}
```

### Boolean Query Evaluation

The query engine supports three operators:

| Operator | Description                                          | Example                        |
|----------|------------------------------------------------------|--------------------------------|
| `AND`    | Intersection — documents must contain **both** terms | `sistem AND informasi`         |
| `OR`     | Union — documents containing **either** term         | `sistem OR komputer`           |
| `NOT`    | Complement — excludes documents with the next term   | `sistem AND NOT komputer`      |

**Evaluation rules:**

1. **Left-to-right evaluation** — operators are applied sequentially from left to right (no precedence/parentheses).
2. **`NOT` is unary** — it negates the term immediately following it (e.g., `NOT komputer` = all documents that do NOT contain "komputer").
3. **Default operator is `AND`** — if no operator is specified between two terms, `AND` is assumed.
4. **Query terms are preprocessed** — each non-operator token goes through the same stemming pipeline as the documents, ensuring consistent matching.

---

## Project Structure

```
stki/
├── app.py                # Streamlit UI and application entry point
├── engine.py             # Boolean retrieval engine (matrix, index, query eval)
├── text_processor.py     # NLP pipeline (PDF loading, preprocessing)
├── requirements.txt      # Python dependencies
├── dataset/              # Default folder for PDF documents
│   ├── doc1.pdf
│   ├── doc2.pdf
│   ├── ...
│   └── doc7.pdf
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

1. In the **sidebar**, you will see a "Path Folder Dokumen" input field. The default value is `./dataset`.
2. Place your `.pdf` documents inside that folder (the `dataset/` folder already contains 7 sample PDFs).
3. Click the **"Load Dataset"** button to read, preprocess, and index all PDF files.
4. Once loaded, you will see:
   - **Stat cards** showing the number of documents, unique terms, and inverted index entries.
   - **Incidence Matrix** tab — an interactive table of the term × document binary matrix.
   - **Inverted Index** tab — a JSON view of the full inverted index.

### Performing Queries

1. Scroll down to the **"Pencarian Boolean"** section.
2. Type your Boolean query in the text input (e.g., `teknologi AND informasi`).
3. Click the **"🔍 Cari"** button.
4. The app will display:
   - The preprocessed query tokens.
   - The number of matching documents.
   - A result card for each matching document showing its filename and a text snippet.

### Query Examples

| Query                                | Description                                                 |
|--------------------------------------|-------------------------------------------------------------|
| `sistem`                             | Documents containing the term "sistem"                      |
| `sistem AND informasi`               | Documents containing both "sistem" AND "informasi"          |
| `teknologi OR komputer`              | Documents containing "teknologi" OR "komputer" (or both)    |
| `sistem AND NOT komputer`            | Documents with "sistem" but NOT "komputer"                  |
| `sistem AND informasi NOT komputer`  | Documents with "sistem" AND "informasi" but NOT "komputer"  |
| `data informasi`                     | Same as `data AND informasi` (AND is the default operator)  |

> **Note:** Operators must be written in **UPPERCASE** (`AND`, `OR`, `NOT`). Lowercase `and`, `or`, `not` will be treated as regular search terms.

---

## Dependencies

| Package    | Purpose                                                            |
|------------|--------------------------------------------------------------------|
| `streamlit`| Web UI framework for interactive dashboards                        |
| `pandas`   | DataFrame for the Incidence Matrix                                 |
| `Sastrawi` | Indonesian NLP — stemming and stopword removal                     |
| `PyPDF2`   | PDF text extraction                                                |

Install all dependencies at once:

```bash
pip install -r requirements.txt
```