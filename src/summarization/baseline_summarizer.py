from typing import List

import torch
from transformers import pipeline

from src.config import SUMMARIZER_MODEL_NAME


# Limit the number of CPU threads used by PyTorch.
# This can help reduce multiprocessing noise and resource warnings.
torch.set_num_threads(1)


class BaselineSummarizer:
    """
    Simple summarizer using a Hugging Face summarization pipeline.

    This class takes a list of text chunks and returns a single summary string.
    """

    def __init__(self, model_name: str = SUMMARIZER_MODEL_NAME):
        """
        Load the summarization model.

        This may take a bit of time on first run, so we usually want to
        create this object only once (we do this with Streamlit caching).
        """
        self.pipe = pipeline("summarization", model=model_name)

    def summarize_chunks(self, chunks: List[str]) -> str:
        """
        Take a list of text chunks and return a single summary string.

        Steps:
        - Join all chunks into one long text.
        - Optionally cut the text if it is too long.
        - Call the summarization pipeline.
        - Return the "summary_text" field.
        """
        if not chunks:
            return "No content to summarize."

        # Join all chunks into one large string.
        full_text = "\n\n".join(chunks)

        # Many models have a maximum input size.
        # Here we simply cut the text to a safe length in characters.
        max_chars = 4000
        if len(full_text) > max_chars:
            full_text = full_text[:max_chars]

        # Call the summarization model.
        result = self.pipe(
            full_text,
            max_length=200,  # try to keep the summary short
            min_length=50,   # avoid extremely short summaries
            do_sample=False,  # deterministic output
        )

        # The pipeline returns a list with one dictionary.
        summary_text = result[0]["summary_text"]
        return summary_text