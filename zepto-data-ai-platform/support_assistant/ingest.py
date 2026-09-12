"""
Zepto Support Assistant — Document Ingestion & Vector Store
============================================================
Ingests 8 policy documents, chunks them, embeds using sentence-transformers
(all-MiniLM-L6-v2), and persists to ChromaDB at support_assistant/chroma_db/.
"""

import os
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "zepto_policies"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 300  # characters per chunk (approximate)
CHUNK_OVERLAP = 50


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks by character count."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
    return chunks


def ingest_documents():
    """Ingest all 8 policy documents into ChromaDB."""
    print("=" * 60)
    print("  Zepto Support Assistant — Document Ingestion")
    print("=" * 60)

    # Initialize embedding function
    print(f"\n  Loading embedding model: {EMBEDDING_MODEL}")
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

    # Initialize ChromaDB client with persistent storage
    print(f"  ChromaDB persist directory: {CHROMA_DIR}")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete existing collection if it exists (clean re-ingestion)
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"  Deleted existing collection: {COLLECTION_NAME}")
    except Exception:
        pass

    # Create collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )

    # Process each document
    all_chunks = []
    all_ids = []
    all_metadatas = []
    doc_ids_found = set()

    for doc_num in range(1, 9):
        doc_id = f"doc_{doc_num:02d}"
        doc_path = os.path.join(DOCS_DIR, f"{doc_id}.txt")

        if not os.path.exists(doc_path):
            print(f"  ⚠️  Missing: {doc_path}")
            continue

        with open(doc_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        chunks = chunk_text(text)
        doc_ids_found.add(doc_id)

        print(f"\n  {doc_id}.txt:")
        print(f"    Length: {len(text)} chars")
        print(f"    Chunks: {len(chunks)}")

        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i:03d}"
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_metadatas.append({
                "doc_id": doc_id,
                "chunk_index": i,
                "source_file": f"{doc_id}.txt"
            })

    # Add all chunks to collection
    if all_chunks:
        collection.add(
            documents=all_chunks,
            ids=all_ids,
            metadatas=all_metadatas
        )

    # Verification
    total_chunks = collection.count()
    print(f"\n{'=' * 60}")
    print(f"  INGESTION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total chunks ingested: {total_chunks}")
    print(f"  Distinct source documents: {len(doc_ids_found)}")
    print(f"  Document IDs: {sorted(doc_ids_found)}")

    # Verify all 8 docs are represented
    assert len(doc_ids_found) == 8, f"Expected 8 source docs, found {len(doc_ids_found)}"
    assert total_chunks > 0, "No chunks were ingested!"

    # Verify by querying metadata
    results = collection.get(include=["metadatas"])
    stored_doc_ids = set(m["doc_id"] for m in results["metadatas"])
    print(f"  Verified stored doc IDs: {sorted(stored_doc_ids)}")
    assert len(stored_doc_ids) == 8, f"Expected 8 doc IDs in store, found {len(stored_doc_ids)}"

    print(f"\n  ✅ All 8 documents successfully ingested and verified!")

    return total_chunks, doc_ids_found


if __name__ == "__main__":
    ingest_documents()
