import json
from app.retrieval.basic_retriever import retrieve_chunks
from app.services.llm_service import generate_answer
from app.retrieval.bm25_retriever import bm25_search
from app.retrieval.reranker import rerank_results
from app.retrieval.hybrid_retriever import hybrid_search
from app.rag.rag_service import answer_question
from app.services.text_processing import clean_text, create_chunks
from app.retrieval.semantic_retriever import (
    build_faiss_index,
    semantic_search,
)
from pathlib import Path

import fitz
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="Academic RAG QA System",
    description="Backend API for context-aware question answering over academic documents.",
    version="1.0.0",
)


# --------------------------------------------------
# CORS Configuration
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Upload Directory
# --------------------------------------------------

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# Root Endpoint
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Academic RAG QA System API is running",
        "status": "success",
    }


# --------------------------------------------------
# Health Check
# --------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "message": "Backend connection successful",
    }


# --------------------------------------------------
# PDF Upload Endpoint
# --------------------------------------------------

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    # Check file type
    if not file.filename.lower().endswith(".pdf"):
        return {
            "status": "error",
            "message": "Only PDF files are allowed.",
        }

    # Create file path
    file_path = UPLOAD_DIR / file.filename

    # Save uploaded PDF
    file_content = await file.read()

    with open(file_path, "wb") as buffer:
        buffer.write(file_content)

    # Open PDF using PyMuPDF
    document = fitz.open(file_path)

    # Extract text page by page
    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text()

        pages.append({
            "page_number": page_number,
            "text": text,
        })

    document.close()

    return {
        "status": "success",
        "filename": file.filename,
        "total_pages": len(pages),
        "pages": pages,
    }
    
@app.post("/process")
async def process_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        return {
            "status": "error",
            "message": "Only PDF files are allowed.",
        }

    file_content = await file.read()

    document = fitz.open(stream=file_content, filetype="pdf")

    all_chunks = []

    chunk_counter = 1

    for page_number, page in enumerate(document, start=1):
        raw_text = page.get_text()

        cleaned_text = clean_text(raw_text)

        page_chunks = create_chunks(cleaned_text)

        for chunk in page_chunks:
            all_chunks.append({
                "document_name": file.filename,
                "page_number": page_number,
                "chunk_id": chunk_counter,
                "chunk_text": chunk,
            })

            chunk_counter += 1

    document.close()
    
    data_directory = Path("app/data")
    data_directory.mkdir(parents=True, exist_ok=True)

    chunks_file = data_directory / "chunks.json"

    with open(chunks_file, "w", encoding="utf-8") as json_file:
        json.dump(all_chunks, json_file, ensure_ascii=False, indent=2)

    return {
        "status": "success",
        "filename": file.filename,
        "total_chunks": len(all_chunks),
        "chunks": all_chunks,
    }
    
@app.post("/retrieve")
async def retrieve(question: str):
    if not question.strip():
        return {
            "status": "error",
            "message": "Question cannot be empty.",
        }

    results = retrieve_chunks(question, top_k=3)

    return {
        "status": "success",
        "question": question,
        "total_results": len(results),
        "results": results,
    }
    
@app.post("/build-index")
async def build_index():
    try:
        total_chunks, dimension = build_faiss_index()

        return {
            "status": "success",
            "message": "FAISS index created successfully.",
            "total_chunks": total_chunks,
            "embedding_dimension": dimension,
        }

    except Exception as error:
        return {
            "status": "error",
            "message": str(error),
        }
        
@app.post("/semantic-search")
async def semantic_search_endpoint(question: str):
    if not question.strip():
        return {
            "status": "error",
            "message": "Question cannot be empty.",
        }

    results = semantic_search(question, top_k=3)

    return {
        "status": "success",
        "question": question,
        "total_results": len(results),
        "results": results,
    }
    
@app.post("/generate-answer")
async def generate_answer_endpoint(
    question: str,
    context: str
):
    if not question.strip():
        return {
            "status": "error",
            "message": "Question cannot be empty.",
        }

    if not context.strip():
        return {
            "status": "error",
            "message": "Context cannot be empty.",
        }

    try:
        answer = generate_answer(question, context)

        return {
            "status": "success",
            "question": question,
            "answer": answer,
        }

    except Exception as error:
        return {
            "status": "error",
            "message": str(error),
        }
        
@app.post("/ask")
async def ask_question(question: str):
    if not question.strip():
        return {
            "status": "error",
            "message": "Question cannot be empty.",
        }

    try:
        result = answer_question(question, top_k=5)

        return result

    except Exception as error:
        error_message = str(error)

        # Handle Gemini API rate-limit errors
        if "429" in error_message or "rate limit" in error_message.lower():
            return {
                "status": "error",
                "message": (
                    "Gemini API rate limit reached. "
                    "Please wait and try again later."
                ),
            }

        return {
            "status": "error",
            "message": error_message,
        }
        
@app.post("/bm25-search")
async def bm25_search_endpoint(question: str):
    if not question.strip():
        return {
            "status": "error",
            "message": "Question cannot be empty.",
        }

    try:
        results = bm25_search(question, top_k=5)

        return {
            "status": "success",
            "question": question,
            "total_results": len(results),
            "results": results,
        }

    except Exception as error:
        return {
            "status": "error",
            "message": str(error),
        }
        
@app.post("/hybrid-search")
async def hybrid_search_endpoint(question: str):
    if not question.strip():
        return {
            "status": "error",
            "message": "Question cannot be empty.",
        }

    try:
        results = hybrid_search(question, top_k=5)

        return {
            "status": "success",
            "question": question,
            "total_results": len(results),
            "results": results,
        }

    except Exception as error:
        return {
            "status": "error",
            "message": str(error),
        }
        
@app.post("/rerank")
async def rerank_endpoint(question: str):
    if not question.strip():
        return {
            "status": "error",
            "message": "Question cannot be empty.",
        }

    try:
        # First retrieve Hybrid candidates
        hybrid_results = hybrid_search(question, top_k=5)

        # Then rerank them
        reranked_results = rerank_results(
            question,
            hybrid_results,
            top_k=5
        )

        return {
            "status": "success",
            "question": question,
            "total_results": len(reranked_results),
            "results": reranked_results,
        }

    except Exception as error:
        return {
            "status": "error",
            "message": str(error),
        }