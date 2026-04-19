import os

from eval_handler import rank_all_queries, pseudo_relevance_feedback, compute_bm25_vectors, compute_tfidf_vectors
from file_handler import load_all_documents, dict_to_counter, query_constructor, write_results, ensure_a1_dataset, \
    download_and_extract


def run(lang, run, train, queries_path=None, documents_list=None, output_file=None):
    # downloads & extracts the data files from the link http://ufal.mff.cuni.cz/~pecina/courses/npfl103/data/A1.tgz
    download_and_extract()

    # load all documents and returns a dictionary of filename(str):list_of_tokens(list of strings)
    parsed_docs = load_all_documents(lang, run, documents_list)

    if run == 0:
        docs_vectors = dict_to_counter(parsed_docs)
    else:
        # # baseline
        # docs_vectors = dict_to_counter(parsed_docs)

        # # 1st improvement
        # docs_vectors = compute_tfidf_vectors(parsed_docs)

        # final version
        docs_vectors = compute_bm25_vectors(parsed_docs)

    queries = query_constructor(lang, run, train, queries_path)
    queries_vectors = dict_to_counter(queries)

    # Apply PRF only for BM25 runs
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

    results = rank_all_queries(queries_vectors, docs_vectors, run)

    train_str = "train" if train else "test"

    if output_file is not None:
        filename = os.path.join("outputs", output_file)
    else:
        filename = os.path.join("outputs", f"run-{run}_{train_str}_{lang}.res")

    write_results(results, filename, run)
