from rank_bm25 import BM25Okapi
import numpy as np
from CISILoader import CISILoader


class BM25Retriever:
    def __init__(self, doc_path, qry_path, rel_path):
        self.doc_set = CISILoader.load_documents(doc_path)
        self.qry_set = CISILoader.load_queries(qry_path)
        self.rel_set = CISILoader.load_relevance(rel_path)
        self.bm25 = self._initialize_bm25()

    def _initialize_bm25(self):
        corpus = list(self.doc_set.values())
        tokenized_corpus = [doc.split(" ") for doc in corpus]
        return BM25Okapi(tokenized_corpus)

    def retrieve_BM25(self, idx):
        query = self.qry_set[idx]

        tokenized_query = query.split(" ")
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1] + 1

        return top_indices
