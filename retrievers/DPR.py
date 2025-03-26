from retrievers.CISILoader import CISILoader
import torch
import faiss
import pandas as pd

from transformers import (
    DPRQuestionEncoder,
    DPRContextEncoder,
    DPRQuestionEncoderTokenizer,
    DPRContextEncoderTokenizer,
)


class DPRRetriever:
    def __init__(self, doc_path, qry_path, rel_path):
        self.doc_set = CISILoader.load_documents(doc_path)
        self.qry_set = CISILoader.load_queries(qry_path)
        self.cpy_qry_set = CISILoader.load_queries(qry_path)
        self.rel_set = CISILoader.load_relevance(rel_path)
        self._initialize_dpr()
        var = self.index
        var = self.corpus
        var = self.doc_ids
        var = self.question_encoder
        var = self.question_tokenizer
        var = self.device

    def _initialize_dpr(self):
        # load data
        self.doc_set = pd.DataFrame(list(self.doc_set.items()), columns=["doc_id", "text"])
        self.qry_set = pd.DataFrame(list(self.qry_set.items()), columns=["query_id", "text"])

        self.corpus, self.doc_ids = self._corpus_creation(self.doc_set)

        # move to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # load DPR pretrained encoders and tokenizers
        self.question_encoder, context_encoder, self.question_tokenizer, context_tokenizer = self._load_dpr_encoders()
        self.question_encoder.to(self.device)
        context_encoder.to(self.device)

        # encode the corpus
        corpus_embeddings = self._encode_corpus(context_tokenizer, context_encoder, self.device, self.corpus)

        # build a FAISS index with the corpus embeddings
        dimension = corpus_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(corpus_embeddings)


    def _encode_query(self, query, question_tokenizer, question_encoder, device):
        inputs = question_tokenizer(query, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            q_emb = question_encoder(**inputs).pooler_output
        return q_emb.cpu().numpy()

    def _retrieve_dpr(self, query: str):
        q_emb = self._encode_query(query, self.question_tokenizer, self.question_encoder, self.device)
        top_k = self.index.ntotal
        distances, indices = self.index.search(q_emb, top_k)
        results = []
        for rank, idx in enumerate(indices[0], start=1):
            results.append({
                'rank': rank,
                'doc_id': self.doc_ids[idx],
                'doc_text': self.corpus[idx],
                'distance': float(distances[0][rank - 1])
            })
        return results

    def retrieve_with_scores(self, idx):
        query = self.cpy_qry_set[idx]
        q_emb = self._encode_query(query, self.question_tokenizer, self.question_encoder, self.device)
        top_k = self.index.ntotal
        distances, indices = self.index.search(q_emb, top_k)
        results = {}
        for rank, idx in enumerate(indices[0], start=1):
            results[self.doc_ids[idx]] = -float(distances[0][rank - 1])

        return results

    def _encode_corpus(self, context_tokenizer, context_encoder, device, corpus, batch_size=32):
        all_embs = []
        for i in range(0, len(corpus), batch_size):
            batch = corpus[i:i + batch_size]
            inputs = context_tokenizer(batch, padding=True, max_length=512, truncation=True, return_tensors="pt").to(
                device)
            with torch.no_grad():
                emb = context_encoder(**inputs).pooler_output
            all_embs.append(emb.cpu())
        return torch.cat(all_embs, dim=0).numpy()

    def _load_dpr_encoders(self):
        question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

        question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

        return question_encoder, context_encoder, question_tokenizer, context_tokenizer

    def _corpus_creation(self, docs):
        corpus = []
        doc_ids = []

        for _, d in docs.iterrows():
            doc_text = d.text
            corpus.append(doc_text)
            doc_ids.append(f"{d.doc_id}")

        return corpus, doc_ids

