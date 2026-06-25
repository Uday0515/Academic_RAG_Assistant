from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from backend.app import load_vectorstore, add_pdfs_to_vectorstore, get_answer, get_subjects_from_data

vectorstore = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vectorstore
    print("Loading vectorstore...")
    vectorstore = load_vectorstore()
    print("Vectorstore ready.")
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    subject_filter: str = "All"


@app.get("/subjects")
def get_subjects():
    return {"subjects": get_subjects_from_data()}


@app.post("/ask")
def ask(request: QueryRequest):
    answer, sources = get_answer(request.query, vectorstore, request.subject_filter)
    return {"answer": answer, "sources": sources}


@app.post("/upload")
async def upload_pdfs(files: list[UploadFile] = File(...)):
    global vectorstore
    file_data = []
    for file in files:
        content = await file.read()
        file_data.append({"name": file.filename, "bytes": content})
    vectorstore = add_pdfs_to_vectorstore(vectorstore, file_data)
    return {"uploaded": [f["name"] for f in file_data]}