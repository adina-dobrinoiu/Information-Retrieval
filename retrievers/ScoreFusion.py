from retrievers.CISILoader import CISILoader

class ScoreFusionRetriever:
    def __init__(self, models, weights, doc_path, qry_path, rel_path):
        self.doc_set = CISILoader.load_documents(doc_path)
        self.qry_set = CISILoader.load_queries(qry_path)
        self.rel_set = CISILoader.load_relevance(rel_path)
        self.models = models
        self.weights = weights

    def normalize_results(self, results):
        max_score = max(results.values())
        min_score = min(results.values())

        if max_score != min_score:
            for doc_id in results:
                results[doc_id] = (results[doc_id] - min_score) / (max_score - min_score)
        else:
            for doc_id in results:
                results[doc_id] = 1.0

        return results
    def retrieve_score_fusion(self, idx):
        aggregated_results = {}
        for i in range(len(self.models)):
            model = self.models[i]
            weight = self.weights[i]
            model_result = model.retrieve_with_scores(idx)
            normalized_model_result = self.normalize_results(model_result)
            for doc_id in normalized_model_result:
                aggregated_results[doc_id] = aggregated_results.get(doc_id, 0.0) + weight * normalized_model_result[doc_id]

        sorted_doc_ids = sorted(aggregated_results, key=aggregated_results.get, reverse=True)

        return sorted_doc_ids
