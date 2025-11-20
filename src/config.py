# Configuration values used across the project.

# Name of the embedding model from sentence-transformers
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Name of the summarization model from Hugging Face transformers (baseline)
SUMMARIZER_MODEL_NAME = "facebook/bart-base"

# Name of the LLM model for advanced summarization
#LLM_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct" ## Model initially implemented but generated a memory crash.
LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Chunking settings
CHUNK_SIZE = 1000        # number of characters per chunk (simple approach)
CHUNK_OVERLAP = 200      # overlap between chunks

# Default number of chunks to retrieve in search
TOP_K_DEFAULT = 3 # Recommendend for best output on CPU. Higher value will add context but also noise for the LLM output.