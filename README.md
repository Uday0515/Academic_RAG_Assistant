# Academic RAG Assistant

A retrieval-augmented generation system built for an engineering department to answer natural language queries over academic documents. The system retrieves semantically relevant content from syllabi, module notes, and question papers, and generates structured responses grounded strictly in source material.

## Architecture

```
Documents -> PyPDFLoader -> RecursiveCharacterTextSplitter
          -> HuggingFace Embeddings (all-MiniLM-L6-v2)
          -> FAISS Vector Store
          -> Top-K Retriever
          -> Gemini 2.5 Flash
          -> Streamlit UI
```

Chunking parameters: `chunk_size=2000`, `chunk_overlap=400`. Retrieval pool size is 30, filtered post-retrieval when a subject scope is selected.

## Tech Stack

- **Embedding Model** — sentence-transformers/all-MiniLM-L6-v2 (HuggingFace)
- **Vector Store** — FAISS (in-memory, CPU)
- **LLM** — Google Gemini 2.5 Flash via `google-generativeai`
- **Framework** — LangChain (document loading, splitting, retrieval)
- **UI** — Streamlit

## Features

- Subject-scoped retrieval via post-retrieval metadata filtering
- Runtime PDF ingestion without application restart
- Chat session export in plain text and HTML
- Source-level metadata tracking (filename, page number, subject folder)

## Setup

```bash
git clone https://github.com/Uday0515/Academic_RAG_Assistant.git
cd Academic_RAG_Assistant
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file:
```
GOOGLE_API_KEY=your_key_here
```

Place PDFs under `data/<subject-name>/` and run:
```bash
streamlit run ui.py
```

## Project Structure

```
.
├── app.py                  # RAG pipeline: loading, chunking, retrieval, generation
├── ui.py                   # Streamlit interface
├── data/                   # Source PDFs organised by subject
├── .streamlit/
│   └── config.toml
├── requirements.txt
└── Dockerfile
```

## Roadmap

- Query rewriting pre-retrieval to improve recall on vague queries
- Cross-encoder reranking post-retrieval to improve precision
- RAGAS-based evaluation pipeline (faithfulness, answer relevancy, context precision)
- Pinecone integration to replace in-memory FAISS for persistent storage
- Conversation memory for multi-turn query resolution
