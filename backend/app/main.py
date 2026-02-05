from __future__ import annotations

import uuid
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .config import INDEX_DIR, UPLOAD_DIR
from .schemas import (
    AgentRequest,
    AgentResponse,
    AskRequest,
    AskResponse,
    Citation,
    HealthResponse,
    IndexRequest,
    IndexResponse,
    UploadResponse,
)
from .services.agent import ResearchAgent
from .services.chunker import chunk_text
from .services.document_loader import load_document
from .services.embeddings import TfidfEmbedder
from .services.rag import Retriever, SimpleAnswerGenerator
from .services.vector_store import FaissVectorStore, VectorStoreBundle

app = FastAPI(title="BioMed Research Agent Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _bundle_exists() -> bool:
    return (INDEX_DIR / "faiss.index").exists() and (INDEX_DIR / "vectorizer.pkl").exists()


def load_bundle() -> VectorStoreBundle:
    if not _bundle_exists():
        raise HTTPException(status_code=400, detail="Index not found. Please run /index first.")
    return VectorStoreBundle.load(INDEX_DIR)


@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)) -> UploadResponse:
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".md", ".markdown", ".txt"}:
        raise HTTPException(status_code=400, detail="Only PDF/Markdown/TXT files are supported.")
    file_id = f"{uuid.uuid4().hex}{suffix}"
    destination = UPLOAD_DIR / file_id
    content = await file.read()
    destination.write_bytes(content)
    return UploadResponse(file_id=file_id, filename=file.filename)


@app.post("/index", response_model=IndexResponse)
async def build_index(request: IndexRequest) -> IndexResponse:
    if request.file_ids:
        files = [UPLOAD_DIR / file_id for file_id in request.file_ids]
    else:
        files = list(UPLOAD_DIR.glob("*"))
    if not files:
        raise HTTPException(status_code=400, detail="No files available for indexing.")

    chunks: List[str] = []
    metadata: List[dict] = []
    for path in files:
        try:
            text, doc_type = load_document(path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        for idx, chunk in enumerate(chunk_text(text)):
            chunk_id = f"{path.name}-{idx}"
            chunks.append(chunk)
            metadata.append(
                {
                    "text": chunk,
                    "source": f"{path.name} ({doc_type})",
                    "chunk_id": chunk_id,
                }
            )

    if not chunks:
        raise HTTPException(status_code=400, detail="No text extracted from documents.")

    embedder = TfidfEmbedder.create()
    embeddings = embedder.fit_transform(chunks)
    store = FaissVectorStore.build(embeddings, metadata)
    bundle = VectorStoreBundle(store=store, embedder=embedder)
    bundle.save(INDEX_DIR)

    return IndexResponse(indexed_files=[path.name for path in files], chunks=len(chunks))


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest) -> AskResponse:
    bundle = load_bundle()
    retriever = Retriever(bundle)
    generator = SimpleAnswerGenerator()
    chunks = retriever.retrieve(request.question, request.top_k)
    answer = generator.generate(request.question, chunks)
    citations = [
        Citation(text=chunk.text, source=chunk.source, chunk_id=chunk.chunk_id)
        for chunk in chunks
    ]
    return AskResponse(answer=answer, citations=citations)


@app.post("/agent/run", response_model=AgentResponse)
async def run_agent(request: AgentRequest) -> AgentResponse:
    bundle = load_bundle()
    retriever = Retriever(bundle)
    agent = ResearchAgent(retriever)
    result = agent.run(request.question, request.action)
    citations = [
        Citation(text=chunk.text, source=chunk.source, chunk_id=chunk.chunk_id)
        for chunk in result.chunks
    ]
    return AgentResponse(
        answer=result.answer,
        citations=citations,
        report_markdown=result.report_markdown,
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")
