import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    r"D:\Lokesh files\Projects\LearningDNA"
)

DOCUMENT_NAME = "OOPConcepts"

CHUNKS_FILE = (
    PROJECT_ROOT
    / "content"
    / "chunks"
    / f"{DOCUMENT_NAME}_chunks.json"
)

INDEX_FILE = (
    PROJECT_ROOT
    / "content"
    / "embeddings"
    / f"{DOCUMENT_NAME}.faiss"
)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 5


# ============================================================
# LOAD CHUNKS
# ============================================================

def load_chunks():

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data["chunks"]


# ============================================================
# LOAD FAISS INDEX
# ============================================================

def load_index():

    index = faiss.read_index(
        str(INDEX_FILE)
    )

    return index


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

def load_embedding_model():

    print("Loading embedding model...")

    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    return model


# ============================================================
# SEARCH
# ============================================================

def search(
    query,
    model,
    index,
    chunks,
    top_k=TOP_K
):

    # --------------------------------------------------------
    # Convert query to embedding
    # --------------------------------------------------------

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = query_embedding.astype(
        "float32"
    )

    # --------------------------------------------------------
    # Search FAISS
    # --------------------------------------------------------

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, index_id in zip(
        scores[0],
        indices[0]
    ):

        if index_id == -1:
            continue

        chunk = chunks[index_id]

        results.append(
            {
                "score": float(score),
                "chunk_id": chunk["chunk_id"],
                "page": chunk["page"],
                "text": chunk["text"],
            }
        )

    return results


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(
    query,
    results
):

    print("\n")
    print("=" * 80)
    print("QUERY")
    print("=" * 80)

    print(query)

    print("\n")
    print("=" * 80)
    print("RETRIEVED CHUNKS")
    print("=" * 80)

    for rank, result in enumerate(
        results,
        start=1
    ):

        print("\n" + "-" * 80)

        print(
            f"Rank       : {rank}"
        )

        print(
            f"Similarity : {result['score']:.4f}"
        )

        print(
            f"Chunk ID   : {result['chunk_id']}"
        )

        print(
            f"Page       : {result['page']}"
        )

        print("\nText:")
        print(result["text"])


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("LEARNINGDNA SEMANTIC RETRIEVAL")
    print("=" * 80)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    chunks = load_chunks()

    print(
        f"\nLoaded chunks: {len(chunks)}"
    )

    # --------------------------------------------------------
    # Load FAISS
    # --------------------------------------------------------

    index = load_index()

    print(
        f"Loaded FAISS vectors: {index.ntotal}"
    )

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    model = load_embedding_model()

    # --------------------------------------------------------
    # Interactive search
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("ENTER YOUR QUESTIONS")
    print("Type 'exit' to stop.")
    print("=" * 80)

    while True:

        query = input("\nQuery: ").strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        results = search(
            query=query,
            model=model,
            index=index,
            chunks=chunks,
            top_k=TOP_K
        )

        display_results(
            query,
            results
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()