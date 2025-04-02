import pandas as pd
from CISILoader import CISILoader
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer
from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher
from collections import defaultdict

class ColBERTRetriever:
    def __init__(self, doc_path, qry_path, rel_path,
                 index_root="colbert_indexes", index_name="cisi_colbert"):
        self.doc_path = doc_path
        self.qry_path = qry_path
        self.rel_path = rel_path
        self.index_root = index_root
        self.index_name = index_name

        # Load data
        self.doc_set = CISILoader.load_documents(doc_path)

        self.cpy_qry_set = CISILoader.load_queries(qry_path)

        self.rel_set = CISILoader.load_relevance(rel_path)
        self.documents_dict = {str(doc_id): text for doc_id, text in self.doc_set.items()}
        # Paths
        self.collection_path = "corpus.tsv"
        self.query_path = "queries.tsv"
        self.answers_path = "answers.tsv"

        # Index & search
        self._write_collection_tsv()
        self._write_queries_tsv()
        # self._build_index()
        self.retrieve_all()

    def _write_collection_tsv(self):
        """
        Rewrite documents to match ColBERT's required format.
        """
        with open(self.collection_path, "w", encoding="utf-8") as f:
            for doc_id, text in self.doc_set.items():
                f.write(f"{doc_id-1}\t{text.strip()}\n")
    def _write_queries_tsv(self):
        """
        Rewrite documents to match ColBERT's required format.
        """
        with open(self.query_path, "w", encoding="utf-8") as f:
            for q_id, text in self.cpy_qry_set.items():
                f.write(f"{q_id-1}\t{text.strip()}\n")

    def _build_index(self):
        """
        Index all documents using the ColBERT Collection Indexer.
        """

        config = ColBERTConfig(
            root=self.index_root,
            nbits=2
        )
        indexer = Indexer(checkpoint="../colbertv2.0", config=config)
        indexer.index(name=self.index_name, collection=self.collection_path, overwrite=True)

    def retrieve_all(self):
        config = ColBERTConfig(
            root=self.index_root,
            ncells=1024
        )
        searcher = Searcher(index=self.index_name, config=config)
        queries = Queries(self.query_path)
        ranking = searcher.search_all(queries, k=1460)
        ranking.save(self.answers_path)
        # with open(self.answers_path, "w", encoding="utf-8") as f:
        #     for q in range(112):
        #         val = searcher.search(text=self.cpy_qry_set[q+1],k=1460)
        #         for i in range(1460):
        #             f.write(f"{q}\t{val[0][i]}\t{val[1][i]}\t{val[2][i]}\n")


if __name__ == "__main__":
    doc_path = "../dataset/CISI.ALL"
    qry_path = "../dataset/CISI.QRY"
    rel_path = "../dataset/CISI.REL"

    colbert_retriever = ColBERTRetriever(doc_path, qry_path, rel_path)
    colbert_retriever.retrieve_all()