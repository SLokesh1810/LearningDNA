import json
import re
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"D:\Lokesh files\Projects\LearningDNA")

RAW_DIR = PROJECT_ROOT / "content" / "raw"
EXTRACTED_DIR = PROJECT_ROOT / "content" / "extracted"
CHUNKS_DIR = PROJECT_ROOT / "content" / "chunks"
EMBEDDINGS_DIR = PROJECT_ROOT / "content" / "embeddings"


# ------------------------------------------------------------
# Embedding model
# ------------------------------------------------------------

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# ------------------------------------------------------------
# Chunk configuration
# ------------------------------------------------------------

# Approximate character limits.
# We use characters here instead of tokens because it keeps
# the first implementation simple and dependency-light.

CHUNK_SIZE = 2500
CHUNK_OVERLAP = 400


# ============================================================
# DIRECTORY SETUP
# ============================================================

EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:
    """
    Clean extracted PDF text while preserving paragraph structure.
    """

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove excessive spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove spaces around newlines
    text = re.sub(r" *\n *", "\n", text)

    return text.strip()


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf(pdf_path: Path):
    """
    Extract text page-by-page from a PDF.

    Returns:
        List of dictionaries containing page number and text.
    """

    pages = []

    print(f"\nReading PDF: {pdf_path.name}")

    document = fitz.open(pdf_path)

    for page_number, page in enumerate(document, start=1):

        text = page.get_text("text")
        text = clean_text(text)

        if not text:
            continue

        pages.append(
            {
                "page": page_number,
                "text": text,
            }
        )

    document.close()

    return pages


# ============================================================
# SAVE EXTRACTED TEXT
# ============================================================

def save_extracted_text(pdf_path: Path, pages):

    output_file = EXTRACTED_DIR / f"{pdf_path.stem}_extracted.json"

    data = {
        "document": pdf_path.name,
        "pages": pages,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Extracted text saved to:")
    print(output_file)

    return output_file


# ============================================================
# CHUNKING
# ============================================================

def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split text into overlapping chunks.

    Tries to split at paragraph/sentence boundaries rather
    than cutting text randomly.
    """

    paragraphs = text.split("\n\n")

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        # If adding this paragraph still fits
        if len(current_chunk) + len(paragraph) + 2 <= chunk_size:

            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph

        else:

            if current_chunk:
                chunks.append(current_chunk.strip())

            # If a single paragraph is too large,
            # split it further.
            if len(paragraph) > chunk_size:

                start = 0

                while start < len(paragraph):

                    end = start + chunk_size

                    chunk = paragraph[start:end].strip()

                    if chunk:
                        chunks.append(chunk)

                    start = end - overlap

                current_chunk = ""

            else:
                current_chunk = paragraph

    # Add remaining text
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# ============================================================
# CREATE STRUCTURED CHUNKS
# ============================================================

def create_chunks(pages, document_name):

    all_chunks = []

    chunk_id = 0

    for page_data in pages:

        page_number = page_data["page"]
        page_text = page_data["text"]

        page_chunks = split_text(page_text)

        for chunk in page_chunks:

            chunk_id += 1

            all_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "document": document_name,
                    "page": page_number,
                    "text": chunk,
                }
            )

    return all_chunks


# ============================================================
# SAVE CHUNKS
# ============================================================

def save_chunks(pdf_path: Path, chunks):

    output_file = CHUNKS_DIR / f"{pdf_path.stem}_chunks.json"

    data = {
        "document": pdf_path.name,
        "num_chunks": len(chunks),
        "chunks": chunks,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Created {len(chunks)} chunks.")

    print("Chunks saved to:")
    print(output_file)

    return output_file


# ============================================================
# EMBEDDING GENERATION
# ============================================================

def generate_embeddings(chunks):

    print("\nLoading embedding model...")
    print(EMBEDDING_MODEL_NAME)

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    texts = [chunk["text"] for chunk in chunks]

    print(f"\nGenerating embeddings for {len(texts)} chunks...")

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = embeddings.astype("float32")

    print("\nEmbedding shape:")
    print(embeddings.shape)

    return embeddings


# ============================================================
# BUILD FAISS INDEX
# ============================================================

def build_faiss_index(embeddings):

    dimension = embeddings.shape[1]

    # Inner product + normalized embeddings
    # = cosine similarity

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    print(f"\nFAISS index created.")
    print(f"Vectors stored: {index.ntotal}")
    print(f"Embedding dimension: {dimension}")

    return index


# ============================================================
# SAVE EMBEDDINGS + FAISS
# ============================================================

def save_embeddings(pdf_path: Path, embeddings, index):

    embedding_file = (
        EMBEDDINGS_DIR / f"{pdf_path.stem}_embeddings.npy"
    )

    index_file = (
        EMBEDDINGS_DIR / f"{pdf_path.stem}.faiss"
    )

    faiss.write_index(index, str(index_file))

    np.save(embedding_file, embeddings)

    print("\nSaved embedding matrix:")
    print(embedding_file)

    print("\nSaved FAISS index:")
    print(index_file)


# ============================================================
# MAIN PIPELINE
# ============================================================

def process_pdf(pdf_path: Path):

    print("=" * 70)
    print("LEARNINGDNA CONTENT PIPELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # Step 1: Extract
    # --------------------------------------------------------

    pages = extract_pdf(pdf_path)

    if not pages:
        raise ValueError(
            "No text could be extracted from this PDF."
        )

    print(f"\nPages containing text: {len(pages)}")

    save_extracted_text(pdf_path, pages)

    # --------------------------------------------------------
    # Step 2: Chunk
    # --------------------------------------------------------

    print("\nCreating chunks...")

    chunks = create_chunks(
        pages,
        pdf_path.name
    )

    save_chunks(pdf_path, chunks)

    # --------------------------------------------------------
    # Step 3: Embeddings
    # --------------------------------------------------------

    embeddings = generate_embeddings(chunks)

    # --------------------------------------------------------
    # Step 4: FAISS
    # --------------------------------------------------------

    index = build_faiss_index(embeddings)

    # --------------------------------------------------------
    # Step 5: Save
    # --------------------------------------------------------

    save_embeddings(
        pdf_path,
        embeddings,
        index
    )

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    pdf_files = list(RAW_DIR.glob("*.pdf"))

    if not pdf_files:

        raise FileNotFoundError(
            f"No PDF files found in:\n{RAW_DIR}"
        )

    # For the first test, process the first PDF.
    pdf_path = pdf_files[0]

    process_pdf(pdf_path)