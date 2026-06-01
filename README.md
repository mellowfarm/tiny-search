# tiny*search* 🩷

A tiny semantic search engine over Simple English Wikipedia — built to understand how production search works under the hood.

![tiny search UI](https://raw.githubusercontent.com/mellowfarm/tiny-search/main/assets/demo.png)

---

## How it works

Most real-world search engines use a **two-layer** retrieval pipeline to balance speed and accuracy:

```
query
  │
  ▼
[Bi-encoder]  →  top 10 candidates  (fast, approximate)
  │
  ▼
[Cross-encoder]  →  top 5 results   (slow, precise re-ranking)
```

**Layer 1 — Bi-encoder retrieval** (`all-MiniLM-L6-v2`)  
Encodes the query and all 368k Wikipedia passages into the same 384-dim vector space. A FAISS index finds the 10 nearest passages using cosine similarity in milliseconds.

**Layer 2 — Cross-encoder re-ranking** (`cross-encoder/ms-marco-MiniLM-L-6-v2`)  
Looks at each `(query, passage)` pair together to produce a precise relevance score. Slower than the bi-encoder, but much more accurate. The top 5 by this score are returned.

---

## Corpus

| | |
|---|---|
| Source | [Simple English Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) (20231101 snapshot) |
| Articles | 241,787 |
| Passages | 368,350 (200-word chunks) |

---

## Running locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

You'll also need `datasets` to build the index:

```bash
pip install datasets
```

### 2. Build the index

The FAISS index and passage corpus are too large for GitHub (~600 MB combined), so you generate them once locally. This takes about **20 minutes** on CPU:

```bash
python build_index.py
```

This creates two files in the project root:
- `index.faiss` — FAISS index with all passage embeddings
- `passages.pkl` — passage texts and titles

### 3. Start the server

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) and search.

---

## Project structure

```
tiny-search/
├── app.py            # Flask API — loads index, runs two-layer search
├── build_index.py    # Build index.faiss + passages.pkl
├── search.ipynb      # Exploratory notebook (same logic as build_index.py)
├── requirements.txt
└── templates/
    └── index.html    # Frontend
```

---

## Stack

- [sentence-transformers](https://www.sbert.net/) — bi-encoder & cross-encoder models
- [FAISS](https://github.com/facebookresearch/faiss) — fast approximate nearest-neighbor search
- [Flask](https://flask.palletsprojects.com/) — web server
- [Hugging Face Datasets](https://huggingface.co/docs/datasets) — Wikipedia corpus
