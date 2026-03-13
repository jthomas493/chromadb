import os
from typing import List, Dict, Any
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging

logging.basicConfig(level=logging.INFO)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extracts text from each page of a PDF and returns a list of dicts with text, page number, and source file.
    """
    reader = PdfReader(pdf_path)
    chunks = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            chunks.append({
                "text": text,
                "page": i + 1,
                "source": os.path.basename(pdf_path)
            })
    return chunks

def chunk_texts(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunked = []
    for page in pages:
        splits = splitter.split_text(page["text"])
        for chunk in splits:
            chunked.append({
                "text": chunk,
                "page": page["page"],
                "source": page["source"]
            })
    logging.info(f"Chunked {len(pages)} pages into {len(chunked)} chunks.")
    return chunked

def load_and_chunk_pdfs(pdf_paths: List[str]) -> List[Dict[str, Any]]:
    all_chunks = []
    for path in pdf_paths:
        pages = extract_text_from_pdf(path)
        chunks = chunk_texts(pages)
        all_chunks.extend(chunks)
    return all_chunks
