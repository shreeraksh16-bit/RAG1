"""
Cloud Setup Validator for Learn RAG.
Tests live connectivity to Qdrant Cloud and OpenRouter LLM endpoints.
"""

import os
import sys

# Reconfigure stdout/stderr for Unicode support on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import uuid
from langchain_core.documents import Document

from rag.config import get_config, validate_config
from rag.ingestion import (
    get_qdrant_client,
    ensure_collection_exists,
    index_chunks_to_qdrant,
    delete_document_vectors,
)
from rag.retrieval import get_embedding_model, retrieve_relevant_chunks
from rag.generation import generate_grounded_answer


def main():
    print("=" * 60)
    print("[INFO] Learn RAG: Validating Cloud Infrastructure Setup")
    print("=" * 60)

    config = get_config()
    is_valid, issues = validate_config(config)

    if not is_valid:
        print("\n[!] Configuration validation failed:")
        for issue in issues:
            print(f"  * {issue}")
        print("\nPlease update your '.env' file with your actual cloud credentials.")
        sys.exit(1)

    print("\n[OK] Configuration loaded successfully.")
    print(f"  * Qdrant URL: {config.qdrant_url}")
    print(f"  * Qdrant Collection: {config.qdrant_collection_name}")
    print(f"  * OpenRouter Model: {config.openrouter_model}")

    # 1. Test Qdrant Cloud Connectivity & Vector Storage
    print("\n[1/3] Testing Qdrant Cloud connection & vector indexing...")
    try:
        client = get_qdrant_client(config.qdrant_url, config.qdrant_api_key)
        collections = client.get_collections()
        print(f"  [OK] Connected to Qdrant Cloud. Existing collections: {len(collections.collections)}")

        test_doc_id = f"test_{uuid.uuid4().hex[:8]}"
        test_chunks = [
            Document(
                page_content="Antigravity RAG is an interactive architecture for grounded document intelligence.",
                metadata={"filename": "cloud_test.pdf", "page": 1, "chunk_id": f"{test_doc_id}_0"}
            ),
            Document(
                page_content="Vector databases like Qdrant store high-dimensional embeddings for semantic search.",
                metadata={"filename": "cloud_test.pdf", "page": 2, "chunk_id": f"{test_doc_id}_1"}
            ),
        ]

        print("  * Loading local embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
        embeddings = get_embedding_model(config.embedding_model_name)

        print(f"  * Indexing {len(test_chunks)} test chunks into collection '{config.qdrant_collection_name}'...")
        indexed = index_chunks_to_qdrant(
            client=client,
            embeddings_model=embeddings,
            collection_name=config.qdrant_collection_name,
            chunks=test_chunks,
            doc_id=test_doc_id,
            overwrite_existing=True,
        )
        print(f"  [OK] Successfully indexed {indexed} chunks.")

        # 2. Test Scoped Similarity Search
        print("\n[2/3] Testing scoped similarity retrieval...")
        query = "What does Qdrant store?"
        results = retrieve_relevant_chunks(
            client=client,
            embeddings_model=embeddings,
            collection_name=config.qdrant_collection_name,
            query=query,
            doc_id=test_doc_id,
            top_k=2
        )
        print(f"  [OK] Retrieved {len(results)} chunks for query: '{query}'")
        for i, r in enumerate(results, 1):
            print(f"    - Chunk {i} (Page {r.metadata.get('page')}, Score: {r.metadata.get('similarity_score', 0):.4f}): {r.page_content[:60]}...")

        # 3. Test OpenRouter Generation
        print("\n[3/3] Testing OpenRouter grounded generation...")
        answer_result = generate_grounded_answer(
            query=query,
            retrieved_chunks=results,
            openrouter_api_key=config.openrouter_api_key,
            model_name=config.openrouter_model
        )
        print(f"  [OK] Model response: {answer_result['answer']}")
        print(f"  [OK] Citations: {answer_result['citations']['citation_summary']}")

        # Cleanup test vectors
        print("\n[*] Cleaning up test vectors from Qdrant Cloud...")
        deleted = delete_document_vectors(client, config.qdrant_collection_name, test_doc_id)
        print(f"  [OK] Removed {deleted} test points from Qdrant Cloud.")

        print("\n" + "=" * 60)
        print("[SUCCESS] Cloud setup is fully verified and functional!")
        print("=" * 60)

    except Exception as exc:
        print(f"\n[ERROR] Error during cloud validation: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
