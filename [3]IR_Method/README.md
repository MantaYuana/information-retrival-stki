# 🧠 Sistem Temu Kembali Informasi — Fuzzy, GVSM, LSI

Mesin pencari dokumen PDF berbahasa Indonesia menggunakan tiga metode **Information Retrieval**: **Fuzzy Retrieval**, **Generalized Vector Space Model (GVSM)**, dan **Latent Semantic Indexing (LSI)** — dilengkapi tampilan step-by-step perhitungan.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)

---

## 📖 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [How It Works](#how-it-works)
  - [Architecture](#architecture)
  - [Text Preprocessing Pipeline](#text-preprocessing-pipeline)
  - [1. Fuzzy Retrieval](#1-fuzzy-retrieval)
  - [2. Generalized Vector Space Model (GVSM)](#2-generalized-vector-space-model-gvsm)
  - [3. Latent Semantic Indexing (LSI)](#3-latent-semantic-indexing-lsi)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the App](#running-the-app)
  - [Loading Documents](#loading-documents)
  - [Performing Searches](#performing-searches)
  - [Step-by-Step Views](#step-by-step-views)
- [Dependencies](#dependencies)

---

## Overview

Project ini mengimplementasikan tiga metode **Information Retrieval (IR)** berbeda dalam satu aplikasi interaktif:

1. **Fuzzy Retrieval** — Menggunakan derajat keanggotaan (membership degree) berbasis normalisasi TF-IDF untuk perangkingan dokumen.
2. **Generalized Vector Space Model (GVSM)** — Memperluas Vector Space Model tradisional dengan korelasi antar-term sehingga term yang berkaitan saling memperkuat.
3. **Latent Semantic Indexing (LSI)** — Menggunakan Singular Value Decomposition (SVD) untuk mereduksi dimensi dan menemukan "konsep" semantik tersembunyi di balik koleksi dokumen.

Setiap metode dilengkapi **tampilan step-by-step** yang menunjukkan seluruh alur perhitungan — dari matriks TF-IDF hingga hasil perangkingan akhir — untuk tujuan edukasi.

Aplikasi web dibangun menggunakan **Streamlit** dengan antarmuka dark-themed yang modern.

---

## Features

- 📄 **PDF Document Loading** — Membaca dan mengekstrak teks dari semua file `.pdf` di folder tertentu dengan tracking progress real-time.
- 🔤 **Indonesian NLP Preprocessing** — Case folding, tokenization, stopword removal, dan stemming menggunakan library [Sastrawi](https://github.com/har07/PySastrawi). Keduanya dapat di-toggle dari UI.
- 🔶 **Fuzzy Retrieval** — Membership-based ranking dengan opsi AND (MIN) dan OR (MAX).
- 🟢 **GVSM** — Term-term correlation matrix memperluas representasi vektor dokumen.
- 🟣 **LSI** — SVD decomposition dengan jumlah dimensi (k) yang dapat diatur.
- 📊 **Step-by-Step Display** — Setiap metode menampilkan langkah-langkah perhitungan lengkap dalam format expandable.
- 🔍 **Multi-Method Search** — Pencarian dokumen dengan pilihan metode dalam satu tab.
- ✨ **Query Term Highlighting** — Hasil pencarian menampilkan snippet teks dengan highlighting pada keyword.
- 📈 **Matrix Views** — Matriks TF-IDF dengan pagination dan tabel IDF yang searchable.
- 🎨 **Styled UI** — Antarmuka dark gradient dengan stat cards, result cards, dan micro-animations.

---

## How It Works

### Architecture

Aplikasi dibagi menjadi tiga modul dengan tanggung jawab yang jelas:

```
┌──────────────────────────────────────────────────────────┐
│                     app.py (UI Layer)                     │
│  Streamlit interface: config, display, user interaction   │
├──────────────────────────────────────────────────────────┤
│               ir_engine.py (IR Engine)                    │
│  Fuzzy, GVSM, LSI — computation + step-by-step output    │
├──────────────────────────────────────────────────────────┤
│             text_processor.py (NLP Pipeline)              │
│  PDF loading, Case Folding, Tokenization, Stopwords,      │
│  Stemming (Sastrawi)                                      │
└──────────────────────────────────────────────────────────┘
```

| Module              | Responsibility                                                                 |
|---------------------|-------------------------------------------------------------------------------|
| `text_processor.py` | Memuat file PDF dan menjalankan pipeline NLP (case fold → tokenize → stopword removal → stemming). |
| `ir_engine.py`      | Membangun vocabulary, menghitung TF/IDF/TF-IDF, dan mengimplementasikan tiga metode IR (Fuzzy, GVSM, LSI) dengan output step-by-step. |
| `app.py`            | UI Streamlit — menangani interaksi pengguna, menampilkan matriks/step-by-step, dan merender hasil pencarian. |

### Text Preprocessing Pipeline

Setiap dokumen (dan query) melewati pipeline berikut:

```
Raw Text
  │
  ▼
Case Folding ──► Lowercasing + hapus karakter non-alfabetik
  │
  ▼
Tokenization ──► Pecah teks menjadi token kata
  │
  ▼
Stopword Removal ──► Hapus stopword Indonesia (Sastrawi)  [toggleable]
  │
  ▼
Stemming ──► Ubah kata ke bentuk dasar (Sastrawi)          [toggleable]
  │
  ▼
Processed Tokens (list of stemmed words)
```

**Contoh:**

| Step              | Output                                       |
|-------------------|----------------------------------------------|
| Raw Text          | `"Sistem Temu-Kembali Informasi 2024"`       |
| Case Folding      | `"sistem temukembali informasi "`             |
| Tokenization      | `["sistem", "temukembali", "informasi"]`      |
| Stopword Removal  | `["sistem", "temukembali", "informasi"]`      |
| Stemming          | `["sistem", "temukembali", "informasi"]`      |

---

### 1. Fuzzy Retrieval

Metode Fuzzy IR menggunakan derajat keanggotaan (membership degree) berbasis TF-IDF untuk menentukan seberapa kuat hubungan antara term dan dokumen.

**Langkah-langkah perhitungan:**

| Step | Proses | Formula |
|------|--------|---------|
| 1 | Hitung **TF Matrix** | `TF(t, d) = count(t in d) / total_words(d)` |
| 2 | Hitung **IDF** | `IDF(t) = log(N / DF(t))` |
| 3 | Hitung **TF-IDF Matrix** | `TF-IDF(t, d) = TF(t, d) × IDF(t)` |
| 4 | Hitung **Fuzzy Membership** | `μ(t, d) = TF-IDF(t, d) / max(TF-IDF(t, ·))` |
| 5 | Ambil **Query Membership** | Kolom membership untuk term-term query |
| 6 | Hitung **Fuzzy Score** | AND: `score(d) = min(μ(query_terms, d))` — OR: `score(d) = max(μ(query_terms, d))` |
| 7 | **Ranked Results** | Urutkan dokumen berdasarkan skor tertinggi |

**Fuzzy Membership** dinormalisasi ke rentang [0, 1] menggunakan min-max normalization per term — nilai 1 berarti term paling relevan di dokumen tersebut dibanding dokumen lain.

**Dua mode operasi:**
- **AND (MIN)** — Semua term query harus relevan. Skor = minimum dari semua membership values.
- **OR (MAX)** — Cukup satu term relevan. Skor = maximum dari semua membership values.

---

### 2. Generalized Vector Space Model (GVSM)

GVSM memperluas VSM tradisional dengan memperhitungkan **korelasi antar-term**. Jika dua term sering muncul bersama (berkorelasi), GVSM akan menangkap hubungan ini.

**Langkah-langkah perhitungan:**

| Step | Proses | Formula |
|------|--------|---------|
| 1 | Hitung **TF-IDF Matrix (A)** | Matriks bobot standar `(docs × terms)` |
| 2 | Hitung **Term-Term Correlation (M)** | `M = Aᵀ × A` — korelasi antar-term `(terms × terms)` |
| 3 | Hitung **GVSM Document Vectors** | `D_gvsm = A × M` — representasi dokumen yang diperluas |
| 4 | Hitung **GVSM Query Vector** | `q_gvsm = q × M` — query diperluas dengan korelasi |
| 5 | Hitung **Cosine Similarity** | `cos(q, d) = (q · d) / (‖q‖ × ‖d‖)` |
| 6 | **Ranked Results** | Urutkan berdasarkan cosine similarity tertinggi |

**Keunggulan GVSM:** Dapat menemukan dokumen relevan meskipun tidak mengandung term query secara langsung — asalkan mengandung term yang **berkorelasi** dengan query.

---

### 3. Latent Semantic Indexing (LSI)

LSI menggunakan **Singular Value Decomposition (SVD)** untuk mereduksi dimensi matriks TF-IDF dan menemukan pola semantik tersembunyi ("konsep latent").

**Langkah-langkah perhitungan:**

| Step | Proses | Formula |
|------|--------|---------|
| 1 | Hitung **TF-IDF Matrix (A)** | Matriks bobot standar `(docs × terms)` |
| 2 | **SVD Decomposition** | `Aᵀ = U × Σ × Vᵀ` — dekomposisi ke komponen singular |
| 3 | **Truncated SVD (rank-k)** | Potong ke k dimensi terpenting: `U_k`, `Σ_k`, `V_kᵀ` |
| 4 | **Reduced Document Space** | `D_k = Σ_k × V_kᵀ` — representasi dokumen di ruang k-dimensi |
| 5 | **Query Projection** | `q_k = qᵀ × U_k × Σ_k⁻¹` — petakan query ke ruang reduced |
| 6 | **Cosine Similarity** | Similarity di reduced space antara query dan dokumen |
| 7 | **Ranked Results** | Urutkan berdasarkan cosine similarity tertinggi |

**Keunggulan LSI:**
- Menangani **sinonimi** — term berbeda dengan makna serupa dapat dikelompokkan dalam konsep yang sama.
- Menangani **polisemi** — term ambigu direpresentasikan melalui kontribusinya di berbagai konsep.
- **Dimensi k** dapat disesuaikan — semakin kecil k, semakin tinggi level abstraksi.

---

## Project Structure

```
[3]IR_Fuzzy,GVSM,LSI/
├── app.py                # Streamlit UI dan entry point aplikasi
├── ir_engine.py          # IR Engine (Fuzzy, GVSM, LSI) + step-by-step output
├── text_processor.py     # NLP pipeline (PDF loading, preprocessing)
├── requirements.txt      # Python dependencies
├── dataset/              # Folder default untuk dokumen PDF
│   ├── doc1.pdf
│   ├── doc2.pdf
│   ├── ...
│   └── (16 sample PDFs)
└── README.md             # File ini
```

---

## Prerequisites

- **Python 3.10+** (menggunakan sintaks type-hint `dict[str, list[str]]`)
- **pip** (Python package manager)

---

## Installation

1. **Clone repository** (atau download source code):

   ```bash
   git clone <repository-url>
   cd stki/[3]IR_Fuzzy,GVSM,LSI
   ```

2. **Buat virtual environment** (recommended):

   ```bash
   python -m venv venv
   ```

   Aktivasi:

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

Jalankan server Streamlit:

```bash
python -m streamlit run app.py
```

Aplikasi akan terbuka otomatis di browser (biasanya di `http://localhost:8501`).

### Loading Documents

1. Di **sidebar**, terdapat input "Path Folder Dataset" dengan default `./dataset`.
2. Letakkan file `.pdf` di folder tersebut (folder `dataset/` sudah berisi 16 sample PDF).
3. Opsional: toggle **Stopword Removal** dan **Stemming** di sidebar.
4. Atur **jumlah dimensi (k)** untuk metode LSI di sidebar.
5. Klik tombol **"🚀 Load & Proses Dataset"** untuk memuat dan memproses semua file PDF.
6. Setelah dimuat, akan muncul:
   - **Stat cards** — jumlah dokumen, unique terms, non-zero values, dan waktu proses.
   - **6 tab** — Pencarian, Step-by-Step Fuzzy/GVSM/LSI, Matriks TF-IDF, dan Nilai IDF.

### Performing Searches

1. Buka tab **"🔍 Pencarian"**.
2. Ketik kata kunci di text input (contoh: `sistem informasi teknologi`).
3. Pilih metode IR: **Fuzzy (AND)**, **Fuzzy (OR)**, **GVSM**, atau **LSI**.
4. Klik tombol **"🔍 Cari"**.
5. Aplikasi menampilkan:
   - Token query setelah preprocessing.
   - Jumlah dokumen relevan dan waktu pencarian.
   - Result card untuk setiap dokumen dengan nama file, skor, dan snippet teks ber-highlight.

### Step-by-Step Views

Untuk melihat detail langkah perhitungan setiap metode:

1. Buka tab **"🔶 Step-by-Step Fuzzy"**, **"🟢 Step-by-Step GVSM"**, atau **"🟣 Step-by-Step LSI"**.
2. Masukkan query dan klik **"▶ Hitung"**.
3. Setiap langkah perhitungan ditampilkan sebagai **expander** yang bisa dibuka/tutup.
4. Di dalam setiap expander terdapat:
   - **Deskripsi** langkah dan formula yang digunakan.
   - **Matriks/tabel** hasil perhitungan pada langkah tersebut.

---

## Dependencies

| Package       | Purpose                                                     |
|---------------|-------------------------------------------------------------|
| `streamlit`   | Web UI framework untuk dashboard interaktif                 |
| `pandas`      | DataFrame untuk tampilan matriks dan tabel                  |
| `Sastrawi`    | Indonesian NLP — stemming dan stopword removal              |
| `PyPDF2`      | Ekstraksi teks dari file PDF                                |
| `scipy`       | Komputasi sparse matrix                                     |
| `numpy`       | Operasi numerik (matriks, SVD, cosine similarity)           |
| `scikit-learn`| Machine learning utilities (opsional untuk SVD alternatif)  |

Install semua dependencies sekaligus:

```bash
pip install -r requirements.txt
```
