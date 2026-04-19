import math
from collections import Counter


def pseudo_relevance_feedback(query_vec, bm25_dict, top_k=10, top_terms=20):
    # first pass: expand query with top terms from top-k docs
    initial_results = rank_documents_for_query(query_vec, bm25_dict)
    top_docs = [docno for docno, _ in initial_results[:top_k]]

    # collect term weights from top docs
    term_scores = Counter()
    for docno in top_docs:
        for term, weight in bm25_dict[docno].items():
            term_scores[term] += weight

    # add top new terms to query
    expansion_terms = [
        t for t, _ in term_scores.most_common(top_terms + len(query_vec))
        if t not in query_vec
    ][:top_terms]

    expanded = dict(query_vec)
    for term in expansion_terms:
        expanded[term] = expanded.get(term, 0) + 0.3

    return expanded

def compute_bm25_vectors(docs_dict, k1=1.2, b=0.75):
    # k1 = term saturation
    # b = length normalization (0=off, 1=full)
    print("Computing BM25 vectors...")
    N = len(docs_dict)

    tf_dict = {doc_id: Counter(tokens) for doc_id, tokens in docs_dict.items()}

    # average document length
    doc_lengths = {doc_id: sum(tf.values()) for doc_id, tf in tf_dict.items()}
    avgdl = sum(doc_lengths.values()) / N

    # document frequency per term
    df = Counter()
    for tokens in docs_dict.values():
        for term in set(tokens):
            df[term] += 1

    bm25_dict = {}
    for doc_id, tf in tf_dict.items():
        vec = {}
        doc_length = doc_lengths[doc_id]
        for term, count in tf.items():
            idf = math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1)
            tf_norm = (count * (k1 + 1)) / (count + k1 * (1 - b + b * doc_length / avgdl))
            vec[term] = idf * tf_norm
        bm25_dict[doc_id] = vec

    return bm25_dict

def compute_tfidf_vectors(docs_dict):
    # gets docs_dict: {doc_id: [token, token, ...]}
    # returns {doc_id: {term: tfidf_weight}}
    print("Computing TF-IDF vectors...")
    N = len(docs_dict)

    # raw term frequencies per doc
    tf_dict = {doc_id: Counter(tokens) for doc_id, tokens in docs_dict.items()}

    # document frequency for each term
    df = Counter()
    for tokens in docs_dict.values():
        for term in set(tokens):
            df[term] += 1

    # TF-IDF - log-normalized TF * IDF
    tfidf_dict = {}
    for doc_id, tf in tf_dict.items():
        vec = {}
        doc_len = sum(tf.values())
        for term, count in tf.items():
            # normalized tf
            tf_val = count / doc_len
            # smoothed idf
            idf_val = math.log((N + 1) / (df[term] + 1))
            vec[term] = tf_val * idf_val
        tfidf_dict[doc_id] = vec

    return tfidf_dict

def cosine_similarity(vec1, vec2):
    # dot product
    dot_product = sum(vec1[term] * vec2[term] for term in vec1 if term in vec2)

    # norms
    norm1 = math.sqrt(sum(value * value for value in vec1.values()))
    norm2 = math.sqrt(sum(value * value for value in vec2.values()))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def rank_all_queries(queries_vectors, docs_vectors, run):
    results = {}

    for qid, query_vec in queries_vectors.items():
        try:
            results[qid] = rank_documents_for_query(query_vec, docs_vectors)
        except Exception as e:
            print(f"failed for query {qid} and query vector {query_vec}: {e}")

    return results

def rank_documents(query_vectors, document_vectors, run, top_k=1000):
    print(f"Ranking all queries (top {top_k})...")
    results = {}

    for qid, query_vector in query_vectors.items():
        ranked = rank_documents_for_query(query_vector, document_vectors)
        results[qid] = ranked[:top_k]

    return results

def rank_documents_for_query(query_vector, document_vectors):
    scores = []

    for doc_id, doc_vector in document_vectors.items():
        score = cosine_similarity(query_vector, doc_vector)
        scores.append((doc_id, score))

    # sort by score descending
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores
