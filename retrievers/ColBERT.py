import pandas as pd
from retrievers.CISILoader import CISILoader
from colbert import ColBERT, ColBERTConfig
from colbert.indexing.collection_indexer import CollectionIndexer
from colbert.indexing.loaders import load_collection_from_tsv
from colbert.searcher import Searcher
from colbert.modeling.tokenization import QueryTokenizer
from colbert.infra import Run


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
        self.qry_set = pd.DataFrame(list(CISILoader.load_queries(qry_path).items()),
                                     columns=["query_id", "text"])
        self.cpy_qry_set = CISILoader.load_queries(qry_path)

        self.rel_set = CISILoader.load_relevance(rel_path)
        self.documents_dict = {str(doc_id): text for doc_id, text in self.doc_set.items()}

        # Paths
        self.collection_path = "corpus.tsv"

        # Index & search
        self._write_collection_tsv()
        self._build_index()
        self._load_searcher()

    def _write_collection_tsv(self):
        """
        Rewrite documents to match ColBERT's required format.
        """
        with open(self.collection_path, "w", encoding="utf-8") as f:
            for doc_id, text in self.doc_set.items():
                f.write(f"{doc_id}\t{text.strip()}\n")

    def _build_index(self):
        """
        Index all documents using the ColBERT Collection Indexer.
        """
        config = ColBERTConfig(
            root=self.index_root,
            index_name=self.index_name,
            doc_maxlen=512,
            max_query_length=32,
            faiss_depth=100,
            bsize=32,
            nbits=2
        )

        colbert = ColBERT(name="colbert-ir/colbertv2.0", config=config)
        collection = load_collection_from_tsv(self.collection_path)

        # Index the encoded documents
        indexer = CollectionIndexer(config)
        indexer.index(colbert, collection, overwrite=True)

    def _load_searcher(self):
        """
        Load the ColBERT search method.
        """
        self.config = ColBERTConfig(
            root=self.index_root,
            index_name=self.index_name
        )
        self.searcher = Searcher(index=self.index_name, config=self.config)
        self.run = Run().context()
        self.query_tokenizer = QueryTokenizer(self.config)

    def encode_query(self, query):
        """
        Encodes a query into ColBERT token embeddings.
        """
        return self.query_tokenizer.tokenize(query)

    def retrieve(self, query):
        """
        Retrieve all ranked documents.
        """
        total_docs = len(self.documents_dict)  # Get total number of indexed documents
        encoded_query = self.encode_query(query)

        with self.run:
            ranked = self.searcher.search(encoded_query, k=total_docs)

        results = []
        for rank, (doc_id, score) in enumerate(ranked.items(), start=1):
            doc_id_str = str(doc_id)
            results.append({
                "rank": rank,
                "doc_id": doc_id_str,
                "doc_text": self.documents_dict.get(doc_id_str, ""),
                "score": score
            })
        return results

    def retrieve_with_scores(self, idx):
        query = self.cpy_qry_set[idx]
        total_docs = len(self.documents_dict)  # Get total number of indexed documents
        encoded_query = self.encode_query(query)
        with self.run:
            ranked = self.searcher.search(encoded_query, k=total_docs)
        results = {}
        for rank, (doc_id, score) in enumerate(ranked.items(), start=1):
            results[doc_id] = score

        return results


    def search_all_queries(self):
        """
        Retrieve all documents for every query.
        """
        all_results = {}
        for _, row in self.qry_set.iterrows():
            query_id = str(row["query_id"])
            query_text = row["text"]
            all_results[query_id] = self.retrieve(query_text)
        return all_results
