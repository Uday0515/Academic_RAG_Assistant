from dotenv import load_dotenv
import os
import tempfile
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from google import genai

load_dotenv()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=400,
    separators=["\n\n", "\n", ".", " ", ""],
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


def load_vectorstore() -> FAISS:
    documents = []
    for file in Path("data").rglob("*.pdf"):
        docs = PyPDFLoader(str(file)).load()
        for doc in docs:
            doc.metadata["source_file"] = file.name
            doc.metadata["subject"] = file.parent.name if file.parent.name != "data" else "General"
        documents.extend(docs)
    chunks = text_splitter.split_documents(documents)
    return FAISS.from_documents(chunks, embeddings)


def add_pdfs_to_vectorstore(vectorstore: FAISS, files: list) -> FAISS:
    new_docs = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file["bytes"])
            tmp_path = tmp.name
        docs = PyPDFLoader(tmp_path).load()
        for doc in docs:
            doc.metadata["source_file"] = file["name"]
            doc.metadata["subject"] = "Uploaded"
        new_docs.extend(docs)
        os.unlink(tmp_path)
    if new_docs:
        vectorstore.add_documents(text_splitter.split_documents(new_docs))
    return vectorstore


def get_subjects_from_data() -> list[str]:
    subjects = set()
    for file in Path("data").rglob("*.pdf"):
        folder = file.parent.name
        subjects.add(folder if folder != "data" else "General")
    return sorted(subjects)


def get_answer(query: str, vectorstore: FAISS, subject_filter: str = "All") -> tuple[str, list[dict]]:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 50})
    docs = retriever.invoke(query)

    if subject_filter != "All":
        docs = [d for d in docs if d.metadata.get("subject") == subject_filter]
        docs = docs[:40]
    else:
        docs = docs[:40]

    if not docs:
        return "No relevant documents found for the selected subject filter.", []

    context = "\n".join(doc.page_content for doc in docs)

    prompt = f"""You are an academic assistant for the Robotics and Artificial Intelligence department.

Answer strictly using the provided document context.
Preserve original academic structure such as CO numbers, module titles, unit names, and ordering.
If the answer requires a list (modules, COs, objectives), ensure the list is complete and continuous.
Do not introduce assumptions or external information.

Context:
{context}

Question:
{query}

Answer:"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    seen = set()
    sources = []
    for doc in docs:
        page_raw = doc.metadata.get("page")
        page_display = (page_raw + 1) if isinstance(page_raw, int) else "?"
        key = (doc.metadata.get("source_file", "Unknown"), page_display)
        if key not in seen:
            seen.add(key)
            sources.append({
                "file": doc.metadata.get("source_file", "Unknown"),
                "page": page_display,
                "subject": doc.metadata.get("subject", ""),
            })

    return response.text, sources