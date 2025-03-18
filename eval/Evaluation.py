"""
Evaluation class that defines methods to compute:
    - Precision, Recall (per query as well as averaged), F-measure
    - MAP (Mean Average Precision)
    - False Positives and False Negatives
"""
import numpy as np


class Evaluation:
    def __init__(self, retrieved_docs, relevant_docs, query_ids):
        self.retrieved_docs = retrieved_docs
        self.relevant_docs = relevant_docs
        self.query_ids = query_ids

    def save_evaluation_results(self, results_file):
        """Compute all evaluation metrics and save to file"""
        with open(results_file, "w") as f:
            header = f"{'Query ID':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AP':<10} {'FP':<5} {'FN':<5}\n"
            print(header)
            f.write(header)

            for qid in self.query_ids:
                precision = self.compute_precision_for_query(qid)
                recall = self.compute_recall_for_query(qid)
                ap = self.compute_average_precision_per_query(qid)
                fp = self.compute_fp_per_query(qid)
                fn = self.compute_fn_per_query(qid)

                result_line = f"{qid:<10} {precision:.3f}    {recall:.3f}        {ap:.3f}    {fp:.2f}%  {fn:.2f}%\n"
                print(result_line)
                f.write(result_line)

            map_score = self.compute_map()
            avg_precision = self.compute_average_precision()
            avg_recall = self.compute_average_recall()
            avg_fp = self.compute_avg_fp()
            avg_fn = self.compute_avg_fn()
            f1 = self.compute_f_measure()

            summary = f"""
            ===== Overall Evaluation Metrics =====
            Mean Average Precision (MAP): {map_score:.3f}
            Average Precision: {avg_precision:.3f}
            Average Recall: {avg_recall:.3f}
            Average False Positives (FP): {avg_fp:.3f}%
            Average False Negatives (FN): {avg_fn:.3f}%
            F1-Score: {f1:.3f}
            """
            print(summary)
            f.write(summary)

        print(f"Results saved to: {results_file}")

    def compute_avg_fp(self):
        """Compute Average FP - Retrieved but not relevant"""
        return np.mean([self.compute_fp_per_query(qid) for qid in self.query_ids])

    def compute_avg_fn(self):
        """Compute Average FN - Relevant but not retrieved"""
        return np.mean([self.compute_fn_per_query(qid) for qid in self.query_ids])

    def compute_fp_per_query(self, qid):
        """Compute FP - Retrieved but not relevant as a percentage of retrieved documents"""
        retrieved = self.retrieved_docs[qid]
        relevant = self.relevant_docs[qid]
        fp_count = len(set(retrieved) - set(relevant))
        return (fp_count / len(retrieved)) * 100 if retrieved else 0

    def compute_fn_per_query(self, qid):
        """Compute FN - Relevant but not retrieved as a percentage of relevant documents"""
        retrieved = self.retrieved_docs[qid]
        relevant = self.relevant_docs[qid]
        fn_count = len(set(relevant) - set(retrieved))
        return (fn_count / len(relevant)) * 100 if relevant else 0

    def compute_map(self):
        """"
        Compute MAP score
        """
        ap_scores = [self.compute_average_precision_per_query(qid) for qid in self.query_ids]
        return np.mean(ap_scores)

    def compute_average_precision_per_query(self, query_id):
        """"
        Compute avg precision for a query
        Used for computation of MAP
        """
        retrieved = self.retrieved_docs[query_id]
        relevant = self.relevant_docs[query_id]
        relevant_count = 0
        precision_sum = 0
        for i, doc in enumerate(retrieved):
            if doc in relevant:
                relevant_count += 1
                precision_sum += relevant_count / (i + 1)
        return precision_sum / len(relevant) if relevant else 0

    def compute_f_measure(self):
        """"
        F1 = 2 * P * R / (P + R)
        """
        precision = self.compute_average_precision()
        recall = self.compute_average_recall()
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    def compute_average_precision(self):
        """"
        Compute precision for all queries
        """
        return np.mean([self.compute_precision_for_query(qid) for qid in self.query_ids])

    def compute_average_recall(self):
        """
        Compute recall for all queries
        ! this might not exist tho (not in lectures)
        """
        return np.mean([self.compute_recall_for_query(qid) for qid in self.query_ids])

    def compute_precision_for_query(self, query_id):
        """"
        Precision = |Relevant Retrieved| / |Retrieved Set|
        RR = intersection(Relevant, Retrieved)
        """
        retrieved = self.retrieved_docs[query_id]
        relevant = self.relevant_docs[query_id]
        relevant_retrieved = set(retrieved) & set(relevant)
        return len(relevant_retrieved) / len(retrieved) if retrieved else 0

    def compute_recall_for_query(self, query_id):
        """"
        Recall = |Relevant Retrieved| / |Relevant Set|
        RR = intersection(Relevant, Retrieved)
        """
        retrieved = self.retrieved_docs[query_id]
        relevant = self.relevant_docs[query_id]
        relevant_retrieved = set(retrieved) & set(relevant)
        return len(relevant_retrieved) / len(relevant) if relevant else 0
