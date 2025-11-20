from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL_NAME


class EmbeddingModel:
    """
    Simple wrapper around SentenceTransformer to create text embeddings.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        # Load the sentence-transformer model once when this class is created.
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str], batch_size: int = 4) -> np.ndarray:
        """
        Encode a list of texts into embedding vectors.

        We return a numpy array with shape (num_texts, embedding_dim).
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
        )
        # Make sure it's a float32 numpy array (required by FAISS)
        embeddings = np.array(embeddings).astype("float32")
        return embeddings