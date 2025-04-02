import pandas as pd
from retrievers.CISILoader import CISILoader
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
        self.qry_set = pd.DataFrame(list(CISILoader.load_queries(qry_path).items()),
                                     columns=["query_id", "text"])
        self.cpy_qry_set = CISILoader.load_queries(qry_path)

        self.rel_set = CISILoader.load_relevance(rel_path)
        self.documents_dict = {str(doc_id): text for doc_id, text in self.doc_set.items()}

        self.answers_path = "../retrievers/answers.tsv"

        self.results = None
        self.parse_answers()

    def parse_answers(self):
        self.results = defaultdict(list)

        with open(self.answers_path, 'r') as f:
            for line in f:
                qid, docid, rank, score = line.strip().split('\t')
                qid = int(qid)
                self.results[qid + 1].append({
                    'doc_id': int(docid) + 1,
                    'rank': int(rank),
                    'score': float(score)
                })

        for qid in self.results:
            self.results[qid].sort(key=lambda x: x['rank'])

    def retrieve(self, idx):
        """
        Retrieve all ranked documents.
        """

        return self.results[idx]

    def retrieve_with_scores(self, idx):
        ans = {}
        for dict in self.results[idx]:
            ans[dict['doc_id']] = dict['score']

        return ans
