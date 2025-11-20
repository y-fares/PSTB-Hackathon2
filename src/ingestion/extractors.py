from typing import IO
from pathlib import Path

import PyPDF2
from docx import Document


def extract_text_from_pdf(file_obj: IO[bytes]) -> str:
    """
    Extract text from a PDF file-like object.

    We use PyPDF2 here. The file_obj is the uploaded file from Streamlit.
    """
    text_chunks = []
    # PyPDF2 expects a binary file object, so we use file_obj directly.
    reader = PyPDF2.PdfReader(file_obj)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_chunks.append(page_text)
    full_text = "\n".join(text_chunks)
    return full_text


def extract_text_from_txt(file_obj: IO[bytes]) -> str:
    """
    Extract text from a plain text file-like object.

    We read the bytes and decode them as UTF-8.
    """
    content_bytes = file_obj.read()
    try:
        text = content_bytes.decode("utf-8")
    except UnicodeDecodeError:
        # Fallback if utf-8 fails
        text = content_bytes.decode("latin-1", errors="ignore")
    return text


def extract_text_from_docx(file_obj: IO[bytes]) -> str:
    """
    Extract text from a DOCX file-like object using python-docx.

    We need to first save the file-like object to a temporary path,
    because python-docx works with file paths.
    """
    temp_path = Path("temp_docx_file.docx")
    # Write the uploaded bytes to a temporary file
    temp_path.write_bytes(file_obj.read())

    # Now open it with python-docx
    doc = Document(str(temp_path))
    paragraphs = [p.text for p in doc.paragraphs if p.text]
    full_text = "\n".join(paragraphs)

    # Clean up the temporary file
    temp_path.unlink(missing_ok=True)
    return full_text