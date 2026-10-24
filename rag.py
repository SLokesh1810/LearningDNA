import json
import os
from pathlib import Path

import faiss
import requests
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"D:\Lokesh files\Projects\LearningDNA")

DOCUMENT_NAME = "OOPConcepts"

CHUNKS_FILE = (
    PROJECT_ROOT / "content" / "chunks" / f"{DOCUMENT_NAME}_chunks.json"
)

INDEX_FILE = (
    PROJECT_ROOT / "content" / "embeddings" / f"{DOCUMENT_NAME}.faiss"
)

# Set these environment variables in Windows before running:
# LEARNINGDNA_LLM_URL = your Cloudflare trycloudflare.com URL
# LEARNINGDNA_API_KEY = the API key printed by your Colab FastAPI cell
LLM_API_URL = os.getenv("LEARNINGDNA_LLM_URL", "").rstrip("/")
LLM_API_KEY = os.getenv("LEARNINGDNA_API_KEY", "")

# ------------------------------------------------------------
# Models and retrieval
# ------------------------------------------------------------

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5
REQUEST_TIMEOUT = (10, 180)


# ============================================================
# LOAD RESOURCES
# ============================================================

def load_chunks():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data["chunks"]


def load_index():
    return faiss.read_index(str(INDEX_FILE))


def load_embedding_model():
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    return SentenceTransformer(EMBEDDING_MODEL)


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(query, embedding_model, index, chunks, top_k=TOP_K):
    if index.ntotal == 0 or not chunks:
        return []

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    effective_top_k = min(top_k, index.ntotal)
    scores, indices = index.search(query_embedding, effective_top_k)

    results = []
    for score, index_id in zip(scores[0], indices[0]):
        if index_id < 0 or index_id >= len(chunks):
            continue

        chunk = chunks[index_id]
        results.append({
            "score": float(score),
            "chunk_id": chunk["chunk_id"],
            "page": chunk["page"],
            "text": chunk["text"],
        })

    return results


def build_context(results):
    if not results:
        return "No relevant passages were retrieved from the uploaded learning material."

    context_parts = []
    for result in results:
        context_parts.append(
            f"SOURCE: Page {result['page']}\n"
            f"CHUNK ID: {result['chunk_id']}\n"
            f"SIMILARITY: {result['score']:.4f}\n\n"
            f"{result['text']}"
        )

    return "\n\n---\n\n".join(context_parts)


def display_sources(results):
    print("\n" + "=" * 80)
    print("RETRIEVED SOURCES")
    print("=" * 80)

    if not results:
        print("No chunks were retrieved.")
        return

    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. Page {result['page']} | "
            f"Chunk {result['chunk_id']} | "
            f"Similarity: {result['score']:.4f}"
        )


# ============================================================
# COLAB LLM API
# ============================================================

def check_api_health():
    if not LLM_API_URL:
        raise RuntimeError(
            "LEARNINGDNA_LLM_URL is not set. Set it to your active "
            "Cloudflare Tunnel URL, without a trailing slash."
        )

    try:
        response = requests.get(
            f"{LLM_API_URL}/health",
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        raise RuntimeError(
            "Could not reach the LearningDNA LLM API. Check that the "
            "Colab runtime, FastAPI server, and Cloudflare Tunnel are running. "
            f"Details: {exc}"
        ) from exc

    if not data.get("model_loaded", False):
        raise RuntimeError(f"The API reports that the model is not loaded: {data}")

    return data


def generate_answer(query, context, max_new_tokens=300, temperature=0.2):
    if not LLM_API_URL:
        raise RuntimeError(
            "LEARNINGDNA_LLM_URL is missing. Configure it in PowerShell."
        )

    if not LLM_API_KEY:
        raise RuntimeError(
            "LEARNINGDNA_API_KEY is missing. Configure it in PowerShell."
        )

    payload = {
        "prompt": query,
        "context": context,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
    }

    try:
        response = requests.post(
            f"{LLM_API_URL}/generate",
            headers={"X-API-Key": LLM_API_KEY},
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = ""
        if exc.response is not None:
            detail = f" Response: {exc.response.text[:1000]}"
        raise RuntimeError(
            f"LLM API returned HTTP {exc.response.status_code if exc.response else 'error'}."
            f"{detail}"
        ) from exc
    except requests.RequestException as exc:
        raise RuntimeError(
            "Could not complete the LLM API request. Check the tunnel and "
            f"Colab runtime. Details: {exc}"
        ) from exc

    try:
        result = response.json()
    except ValueError as exc:
        raise RuntimeError("LLM API returned a non-JSON response.") from exc

    answer = result.get("response")
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError(f"LLM API did not return a usable answer: {result}")

    generation_seconds = result.get("generation_seconds")
    if generation_seconds is not None:
        print(f"\nLLM generation time: {generation_seconds} seconds")

    return answer.strip()


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("LEARNINGDNA - LOCAL RAG + COLAB LLM")
    print("=" * 80)

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"Chunks file not found: {CHUNKS_FILE}")
    if not INDEX_FILE.exists():
        raise FileNotFoundError(f"FAISS index not found: {INDEX_FILE}")

    # Verify remote LLM endpoint before entering the question loop.
    health = check_api_health()
    print(f"Remote model: {health.get('model', 'unknown')}")
    print(f"Remote compute: {health.get('gpu', 'unknown')}")

    chunks = load_chunks()
    index = load_index()

    if index.ntotal != len(chunks):
        raise ValueError(
            f"FAISS/chunk mismatch: index contains {index.ntotal} vectors, "
            f"but the chunks file contains {len(chunks)} chunks. "
            "Rebuild the index or restore the matching chunks file."
        )

    print(f"Loaded chunks: {len(chunks)}")
    print(f"Loaded FAISS vectors: {index.ntotal}")

    embedding_model = load_embedding_model()

    print("\n" + "=" * 80)
    print("ASK QUESTIONS")
    print("Type 'exit' to stop.")
    print("=" * 80)

    while True:
        query = input("\nStudent: ").strip()

        if query.lower() == "exit":
            break
        if not query:
            continue

        results = retrieve(
            query=query,
            embedding_model=embedding_model,
            index=index,
            chunks=chunks,
            top_k=TOP_K,
        )

        display_sources(results)
        context = build_context(results)

        print("\n" + "=" * 80)
        print("LEARNINGDNA")
        print("=" * 80)

        try:
            answer = generate_answer(
                query=query,
                context=context,
                max_new_tokens=300,
                temperature=0.2,
            )
            print("\n" + answer)
        except RuntimeError as exc:
            print(f"\nERROR: {exc}")


if __name__ == "__main__":
    main()
