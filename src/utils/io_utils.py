from pathlib import Path
from typing import IO


def save_uploaded_file(uploaded_file: IO[bytes], destination: Path) -> Path:
    """
    Save an uploaded file (from Streamlit) to a destination path.

    This function is not currently used in app.py, but it is kept here
    as a simple example of how to work with files on disk.
    """
    destination.write_bytes(uploaded_file.read())
    return destination