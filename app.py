from dotenv import load_dotenv
import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from google import genai

load_dotenv()

documents = []
for file in Path("data").rglob("*.pdf"):
    documents.extend(PyPDFLoader(str(file)).load())

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=400,
    separators=["\n\n", "\n", ".", " ", ""],
)

chunks = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def get_answer(query: str) -> str:
    docs = retriever.invoke(query)
    context = "\n".join(doc.page_content for doc in docs)

    prompt = f"""
You are an academic assistant for the Robotics and Artificial Intelligence department.

Answer strictly using the provided document context.
Preserve original academic structure such as CO numbers, module titles, unit names, and ordering.
If the answer requires a list (modules, COs, objectives), ensure the list is complete and continuous.
Do not introduce assumptions or external information.

Context:
{context}

Question:
{query}

Answer:
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text
