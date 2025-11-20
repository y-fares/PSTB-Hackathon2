from typing import List

from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split a long string into smaller overlapping chunks.

    This is a simple character-based approach:
    - We take "chunk_size" characters per chunk.
    - We move forward by (chunk_size - overlap) each time.
    - We ignore very short chunks.
    """
    chunks: List[str] = []
    text = text.strip()
    length = len(text)

    if length == 0:
        return chunks

    start = 0
    step = chunk_size - overlap

    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()

        # Only keep chunks that are not too small
        if len(chunk) > 50:
            chunks.append(chunk)

        start += step

    return chunks