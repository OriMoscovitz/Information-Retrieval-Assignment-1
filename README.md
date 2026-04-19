# Information Retrieval System

This repository is an implementation of the first assignment in the Information Retrieval course (NPFL103). 
The system supports English and Czech, with two runs: a baseline (run-0) and an improved approach using BM25 + Pseudo-Relevance Feedback (run-1). 

---

## 🛠️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/OriMoscovitz/Information-Retrieval-Assignment-1.git
cd Information-Retrieval-Assignment-1
```
### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> The Czech pipeline requires **MorphoDiTa** (`ufal.morphodita`) and the model file placed at:
> `morfflex/czech-morfflex2.0-pdtc1.0-220710.tagger`

### 3. Directory structure

Ensure the following layout under `A1/`:

```
A1/
├── documents_en/          # English XML documents
├── documents_cs/          # Czech XML documents
├── documents_en.lst       # List of English document filenames
├── documents_cs.lst       # List of Czech document filenames
├── topics-train_en.xml
├── topics-test_en.xml
├── topics-train_cs.xml
├── topics-test_cs.xml
├── qrels-train_en.txt
├── qrels-train_cs.txt
└── trec_eval-9.0.7/
    └── trec_eval
outputs/                   # Results are written here
```

---

## ▶️ Running the System

### Run-0: Baseline (whitespace tokenization, no preprocessing)

**English**
```bash
py run.py -q topics-train_en.xml -d documents_en.lst -r 0 -o run-0_train_en.res
py run.py -q topics-test_en.xml  -d documents_en.lst -r 0 -o run-0_test_en.res
```

**Czech**
```bash
py run.py -q topics-train_cs.xml -d documents_cs.lst -r 0 -o run-0_train_cs.res
py run.py -q topics-test_cs.xml  -d documents_cs.lst -r 0 -o run-0_test_cs.res
```

---

### Run-1: Final Approach (BM25 + PRF + Lemmatization)

**English**
```bash
py run.py -q topics-train_en.xml -d documents_en.lst -r 1 -o run-1_train_en.res
py run.py -q topics-test_en.xml  -d documents_en.lst -r 1 -o run-1_test_en.res
```

**Czech**
```bash
py run.py -q topics-train_cs.xml -d documents_cs.lst -r 1 -o run-1_train_cs.res
py run.py -q topics-test_cs.xml  -d documents_cs.lst -r 1 -o run-1_test_cs.res
```

---

## 🧪 Reproducing Intermediate Experiments

The run-1 pipeline went through several iterations. To reproduce intermediate steps, manual code changes are needed as described below.

### Preprocessing steps (`file_handler.py → preprocess_text()`)

To disable individual preprocessing steps (stopword removal, lemmatization, stemming, equivalence classes), comment them out inside `preprocess_text()`.

### Czech lemmatization (`file_handler.py`)

Czech lemmatization via MorphoDiTa is applied in two places. To disable it, comment out the relevant calls in both:

- `parse_all_docs()`
- `query_constructor()`

### Document vector method (`utils.py → run()`)

Three approaches are available. Uncomment the one to test and comment out the others:

```python
# # baseline
# docs_vectors = dict_to_counter(parsed_docs)

# # 1st improvement
# docs_vectors = compute_tfidf_vectors(parsed_docs)

# final version
docs_vectors = compute_bm25_vectors(parsed_docs)
```

### Pseudo-Relevance Feedback (`utils.py → run()`)

PRF is only compatible with BM25. When using the baseline or TF-IDF vectors, comment out the PRF block:

```python
if run != 0:
    expanded_queries = {}
    for qid, query_vec in queries_vectors.items():
        expanded_queries[qid] = pseudo_relevance_feedback(
            query_vec,
            docs_vectors,
            top_k=10,
            top_terms=20
        )
    queries_vectors = expanded_queries
```

---

## 📊 Evaluation & Plots

To evaluate runs against training qrels and generate MAP / P@10 plots, run:

```bash
py plot_handler.py
```

This reads all `.res` files from `outputs/` and produces `results_english.png` and `results_czech.png`.

### Results (English)
![English](Information-Retrieval-Assignment-1/results_english.png)

* **step 0**: Baseline (run-0)
* **step 1**: Split on spaces, commas, punctuation, parentheses, hyphens
* **step 2**: Remove stop words
* **step 3**: Remove empty tokens
* **step 4**: Lemmatizing
* **step 5**: Stemming
* **step 6**: Normalize equivalent classes
* **step 7**: TF-IDF
* **step 8**: BM25
* **step 9**: BM25 + PRF

### Results (Czech)
![Czech](Information-Retrieval-Assignment-1/results_czech.png)

* **step 0**: Baseline (run-0)
* **step 1**: Split on spaces, commas, punctuation, parentheses, hyphens
* **step 2**: Remove stop words
* **step 3**: Remove empty tokens
* **step 4**: Lemmatizing
* **step 5**: Normalize equivalent classes
* **step 6**: TF-IDF
* **step 7**: BM25
* **step 8**: BM25 + PRF

### Results baseline (run-0) English vs. Czech
![None](Information-Retrieval-Assignment-1/11P_AP_results_0.png)

### Results restrained system (run-1) English vs. Czech
![None](Information-Retrieval-Assignment-1/11P_AP_results_1.png)
---

## 🗂️ System Overview

| Component | Description                                                                  |
|---|------------------------------------------------------------------------------|
| `run.py` | Entry point: parses CLI arguments and dispatches to `utils.run()`            |
| `utils.py` | Document loading, vectorization, query construction, PRF, and result writing |
| `file_handler.py` | Document parsing, tokenization, stopword removal, stemming, lemmatization    |
| `eval_handler.py` | BM25 / TF-IDF vector computation, cosine similarity, ranking, PRF            |
| `xml_handler.py` | XML cleaning and extraction utilities                                        |
| `plot_handler.py` | Calls `trec_eval` and plots MAP / P_10 across runs                           |
