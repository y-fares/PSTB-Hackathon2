from typing import List, Dict, Tuple, Optional

import numpy as np
import faiss

from src.ingestion.chunking import split_into_chunks
from src.indexing.embeddings import EmbeddingModel


class VectorIndex:
    """
    This class:
    - stores text chunks with their metadata
    - builds a FAISS index for fast similarity search
    - uses an embedding model to convert text to vectors

    We use:
    - normalized embeddings
    - FAISS IndexFlatIP (inner product)
    so that scores behave like cosine similarity
    (higher = more similar, roughly between 0 and 1).
    """

    def __init__(self):
        # List of dicts storing chunk metadata and text
        self.text_chunks: List[Dict] = []

        # FAISS index (we will create it when we add documents)
        self.index: Optional[faiss.IndexFlatIP] = None

        # Embedding model (SentenceTransformer wrapper)
        self.embedding_model = EmbeddingModel()

    def add_documents(self, docs: List[Tuple[str, str]]) -> None:
        """
        Add a list of documents to the index.

        docs is a list of (doc_id, text) tuples.

        Steps:
        - split each document into chunks
        - store the chunks with metadata
        - compute embeddings for all chunks
        - normalize embeddings
        - build a FAISS IndexFlatIP and add the embeddings
        """
        all_chunk_texts: List[str] = []

        # Clear previous data (optional, but simple for hackathon use)
        self.text_chunks = []
        self.index = None

        # 1) Split each document into chunks and collect them
        for doc_id, text in docs:
            chunks = split_into_chunks(text)
            for chunk_id, chunk_text in enumerate(chunks):
                # Save metadata for each chunk
                self.text_chunks.append(
                    {
                        "doc_id": doc_id,
                        "chunk_id": chunk_id,
                        "text": chunk_text,
                    }
                )
                all_chunk_texts.append(chunk_text)

        if not all_chunk_texts:
            # No chunks to index
            return

        # 2) Compute embeddings for all chunk texts
        embeddings = self.embedding_model.encode(all_chunk_texts)  # shape (N, dim)

        # 3) Normalize embeddings so that inner product ≈ cosine similarity
        #    This makes scores easier to interpret (0 to 1, higher = more similar).
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero just in case
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

        # 4) Create a FAISS index that uses inner product (dot product)
        dim = embeddings.shape[1]  # embedding dimension
        self.index = faiss.IndexFlatIP(dim)

        # 5) Add the normalized embeddings to the index
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for the most similar chunks to the query.

        Returns a list of dictionaries with:
        - score  -> cosine-like similarity (higher is better, ~0 to 1)
        - text   -> chunk text
        - doc_id
        - chunk_id
        """
        if self.index is None or len(self.text_chunks) == 0:
            return []

        # 1) Create embedding for the query
        query_embedding = self.embedding_model.encode([query])  # shape (1, dim)

        # 2) Normalize the query embedding to match the index embeddings
        norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        query_embedding = query_embedding / norm

        # 3) Search in FAISS using inner product on normalized vectors
        #    This behaves like cosine similarity.
        similarities, indices = self.index.search(query_embedding, top_k)

        results: List[Dict] = []
        for sim, idx in zip(similarities[0], indices[0]):
            # idx is the position of the chunk in self.text_chunks
            if idx < 0 or idx >= len(self.text_chunks):
                continue

            meta = self.text_chunks[idx]
            result = {
                "score": float(sim),      # cosine-like similarity score
                "text": meta["text"],
                "doc_id": meta["doc_id"],
                "chunk_id": meta["chunk_id"],
            }
            results.append(result)

        return results