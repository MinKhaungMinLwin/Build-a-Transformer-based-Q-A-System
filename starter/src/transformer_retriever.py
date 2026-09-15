"""
Transformer-based Retrieval using Sentence Embeddings
"""
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class TransformerRetriever:
    """Semantic retrieval using transformer-based sentence embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize with a sentence transformer model.
        
        Args:
            model_name: HuggingFace model name for sentence embeddings
        """
        print(f"🤖 Loading transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.corpus_embeddings = None
        self.corpus = None
        print(f"✅ Model loaded: {model_name}")
        
    def build_index(self, corpus_texts: List[str]):
        """Build semantic index by encoding corpus texts."""
        if not corpus_texts:
            raise ValueError("corpus_texts must contain at least one document")

        print("🧠 Building semantic index...")
        self.corpus = list(corpus_texts)
        
        # YOUR CODE HERE: Encode corpus using sentence transformer
        self.corpus_embeddings = self.model.encode(
            corpus_texts,
            batch_size=32,
            show_progress_bar=len(corpus_texts) > 100,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        self.corpus_embeddings = np.asarray(self.corpus_embeddings)
        
        print(f"✅ Semantic index built for {len(corpus_texts):,} documents")
        if len(corpus_texts) > 0 and self.corpus_embeddings is not None:
            print(f"   Embedding dimension: {self.corpus_embeddings.shape[1]}")
        
    def retrieve(self, query_texts: List[str], k: int = 20) -> Dict[int, List[int]]:
        """Retrieve top-k documents using semantic similarity."""
        if self.corpus_embeddings is None:
            raise ValueError("Index not built. Call build_index() first.")
        if k < 0:
            raise ValueError("k must be non-negative")
            
        print(f"🔍 Running semantic retrieval for {len(query_texts)} queries...")
        
        # YOUR CODE HERE: Encode query texts using the transformer model
        if not query_texts:
            return {}

        query_embeddings = self.model.encode(
            query_texts,
            batch_size=32,
            show_progress_bar=len(query_texts) > 100,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        query_embeddings = np.asarray(query_embeddings)
        
        # YOUR CODE HERE: Calculate similarities and retrieve top-k documents
        similarities = cosine_similarity(query_embeddings, self.corpus_embeddings)
        limit = min(k, len(self.corpus))
        results = {
            query_idx: np.argsort(-scores, kind="stable")[:limit].tolist()
            for query_idx, scores in enumerate(similarities)
        }
        
        print(f"✅ Retrieved top-{k} documents using semantic similarity")
        return results
