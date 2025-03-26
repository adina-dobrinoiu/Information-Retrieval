from retrievers.CISILoader import CISILoader

class SequentialRetriever:
    def __init__(self, retrieval_model, rerank_model, doc_path, qry_path, rel_path, relevant_docs = 200):
        self.doc_set = CISILoader.load_documents(doc_path)
        self.qry_set = CISILoader.load_queries(qry_path)
        self.rel_set = CISILoader.load_relevance(rel_path)
        self.retrieval_model = retrieval_model
        self.rerank_model = rerank_model
        self.relevant_docs = relevant_docs

    def retrieve_docs(self, model, idx):
        model_result = model.retrieve_with_scores(idx)
        sorted_doc_ids = [str(doc_id) for doc_id in sorted(model_result, key=model_result.get, reverse=True)]
        return sorted_doc_ids

    def retrieve_score_sequential(self, idx):
        retrieval_model_docs = self.retrieve_docs(self.retrieval_model, idx)[:self.relevant_docs]
        rerank_model_docs = self.retrieve_docs(self.rerank_model, idx)
        sorted_docs_ids = [doc_id for doc_id in rerank_model_docs if doc_id in retrieval_model_docs]
        return sorted_docs_ids
