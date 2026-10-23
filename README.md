# Learn RAG: Cloud-Native Document Intelligence with Qdrant & OpenRouter

An interactive, production-grade **Retrieval-Augmented Generation (RAG)** application built in Python using Streamlit, LangChain, Qdrant Cloud, Hugging Face embeddings, and OpenRouter.

Upload any text-based PDF, index its contents with page metadata into managed Qdrant Cloud, and ask questions. Answers are generated using OpenRouter (`recursive/flint-v1.0:free` or fallback models) and grounded **strictly** in the uploaded document with exact page citations and interactive diagnostics.

---

## 🏗️ Architecture & How It Works

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                             THE RAG WORKFLOW                                │
│                                                                             │
│ [PHASE A: DOCUMENT INGESTION & CLOUD INDEXING]                              │
│  1. Upload PDF ──► 2. PyPDFLoader ──► 3. RecursiveTextSplitter              │
│    (Streamlit)      (Extract Pages     (1,000 chars size,                   │
│                      + 1-idx Page)      400 chars overlap)                  │
│                                               │                             │
│                                               ▼                             │
│                      5. Qdrant Cloud   ◄── 4. Embeddings                    │
│                        (learning_rag        (all-MiniLM-L6                  │
│                         Collection)          384 dimensions)                │
│                                                                             │
│ [PHASE B: RETRIEVAL & GROUNDED ANSWER GENERATION]                           │
│  6. User Query ──► 7. Vectorize Query ──► 8. Cosine Similarity              │
│    ("What is...")   (all-MiniLM-L6)          (Scoped to doc_id)             │
│                                                       │                     │
│                                                       ▼                     │
│                     10. Answer + Citations ◄── 9. OpenRouter LLM            │
│                         (Page Numbers)          (recursive/flint-v1.0:free) │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Core Features:
- **Cloud-Native Vector Store:** Managed Qdrant Cloud cluster storage with metadata filtering by `doc_id` (no local Docker required).
- **Accurate Page Number Tracking:** Converts 0-indexed PDF pages to human-friendly 1-indexed numbers and preserves them through chunking and retrieval.
- **Overlap-Aware Chunking:** Splits text into 1,000-character segments with 400-character overlaps using `RecursiveCharacterTextSplitter`.
- **Strict Grounding:** System prompt forbids extrapolation and hallucinations; non-existent answers return: *"The provided document does not contain information to answer this question."*
- **Source Badging & Diagnostics:** Visual citation cards with expandable chunk inspection panels displaying retrieved text and cosine similarity scores.

---

## 📁 Project Structure

```text
.
├── app.py                      # Interactive Streamlit UI & chat workflow
├── rag/                        # Core modular RAG package
│   ├── __init__.py             # Public package exports
│   ├── config.py               # Environment configuration and validation
│   ├── ingestion.py            # PDF loading, SHA-256 hashing, chunking, Qdrant upload
│   ├── retrieval.py            # Embedding cache, scoped similarity search, citations
│   └── generation.py           # Grounded prompt engineering and OpenRouter LLM call
├── tests/                      # Automated test suite
│   ├── __init__.py             # Test package initializer
│   ├── test_rag_pipeline.py    # Unit tests for chunking, citations, and grounding
│   └── validate_cloud_setup.py # End-to-end live cloud infrastructure test
├── requirements.txt            # Python package dependencies
├── .env.example                # Template for environment configuration
├── .env                        # Local credentials file (gitignored)
├── .gitignore                  # Git ignore rules for secrets and virtualenvs
├── README.md                   # Project documentation
└── implementation.md           # Step-by-step implementation guide
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- **Python 3.10+** installed
- **Qdrant Cloud Account:** Free tier cluster at [cloud.qdrant.io](https://cloud.qdrant.io)
- **OpenRouter Account:** Free API key at [openrouter.ai](https://openrouter.ai)

### 2. Configure Credentials
Copy `.env.example` to `.env` and fill in your actual cloud credentials:

```bash
# Windows PowerShell
cp .env.example .env
```

Edit `.env`:
```env
QDRANT_URL=https://your-cluster-id.us-east-1-0.aws.cloud.qdrant.io:6333
QDRANT_API_KEY=your_actual_qdrant_api_key
QDRANT_COLLECTION_NAME=learning_rag
OPENROUTER_API_KEY=sk-or-v1-your_actual_openrouter_key
OPENROUTER_MODEL=recursive/flint-v1.0:free
```

### 3. Run Automated Tests
Execute the unit test suite without making live API calls:

```powershell
pytest tests/ -v
```

### 4. Validate Cloud Connectivity (Optional)
If you have configured your `.env` with live keys, run the live end-to-end validation:

```powershell
python tests/validate_cloud_setup.py
```

### 5. Launch the Streamlit App
Start the web application:

```powershell
streamlit run app.py
```

Or if port 8501 is busy:
```powershell
streamlit run app.py --server.port 8502
```

Navigate to `http://localhost:8501` in your browser.
