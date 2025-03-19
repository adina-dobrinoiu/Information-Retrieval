from retrievers.CISILoader import CISILoader
import pandas as pd
import torch
import numpy as np
import faiss

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM

class SPLADERetriever:
    def __init__(self, doc_path, qry_path, rel_path):
        """
        Initialize the SPLADE retriever by loading documents and queries,
        setting up the SPLADE components, encoding the corpus, and building the FAISS index.
        """
        # Load documents and queries
        self.doc_set = CISILoader.load_documents(doc_path)
        self.qry_set = CISILoader.load_queries(qry_path)
        self.rel_set = CISILoader.load_relevance(rel_path)

        # Initialize SPLADE components
        self._initialize_splade()


    def _initialize_splade(self):
        # Convert to df
        self.doc_set = pd.DataFrame(list(self.doc_set.items()), columns=["doc_id", "text"])
        self.qry_set = pd.DataFrame(list(self.qry_set.items()), columns=["query_id", "text"])

        # Create a dictionary mapping doc_id to text
        self.documents_dict = {str(row.doc_id): row.text for _, row in self.doc_set.iterrows()}

        # Move to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load SPLADE tokenizer and model
        model_name = "naver/splade-cocondenser-ensembledistil"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)

        # Encode documents
        self.doc_embeddings = self.encode_documents_with_progress(self.documents_dict)

        # Build corpus and doc_ids lists in a consistent order
        self.doc_ids = list(self.doc_embeddings.keys())
        self.corpus = [self.documents_dict[doc_id] for doc_id in self.doc_ids]

        # Convert embeddings to numpy array and create the FAISS index
        doc_vectors = np.array(list(self.doc_embeddings.values()), dtype=np.float32)
        dimension = doc_vectors.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(doc_vectors)

    def splade_encode(self, text):
        """
        Encodes text into a sparse representation using SPLADE.
        """
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            outputs = self.model(**inputs).logits
            # Get the sparse representation (max pooling)
            return torch.max(outputs, dim=1).values.squeeze().cpu().numpy()

    def encode_documents_with_progress(self, documents):
        """
        Encodes documents using SPLADE with a tqdm progress bar.
        Documents should be provided as a dictionary {doc_id: text}.
        """
        doc_embeddings = {}
        for doc_id, text in tqdm(documents.items(), desc="Encoding Documents", leave=True):
            doc_embeddings[doc_id] = self.splade_encode(text)
        return doc_embeddings

    def _encode_query(self, query):
        """
        Encodes a single query.
        """
        return self.splade_encode(query)

    def retrieve(self, query):
        """
        Retrieves documents in order of similarity to the query using the FAISS index.
        Returns a list of dictionaries containing rank, doc_id, document text, and distance.
        """
        top_k = self.index.ntotal
        q_emb = self._encode_query(query).reshape(1, -1)
        distances, indices = self.index.search(q_emb, top_k)
        results = []
        for rank, idx in enumerate(indices[0], start=1):
            results.append({
                "rank": rank,
                "doc_id": self.doc_ids[idx],
                "doc_text": self.corpus[idx],
                "distance": float(distances[0][rank - 1])
            })
        return results

    def search_all_queries(self):
        """
        Searches through all queries and returns a dictionary mapping query_id to ranked results.
        """
        query_results = {}
        for _, row in tqdm(self.qry_set.iterrows(), total=self.qry_set.shape[0], desc="Searching Queries"):
            query_id = row["query_id"]
            query_text = row["text"]
            query_results[query_id] = self.retrieve(query_text)
        return query_results

