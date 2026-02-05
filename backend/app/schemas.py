from typing import List, Optional

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    file_id: str
    filename: str


class IndexRequest(BaseModel):
    file_ids: Optional[List[str]] = None


class IndexResponse(BaseModel):
    indexed_files: List[str]
    chunks: int


class AskRequest(BaseModel):
    question: str
    top_k: int = Field(default=4, ge=1, le=10)


class Citation(BaseModel):
    text: str
    source: str
    chunk_id: str


class AskResponse(BaseModel):
    answer: str
    citations: List[Citation]


class AgentRequest(BaseModel):
    question: str
    action: str = Field(default="answer", description="answer or poc_report")


class AgentResponse(BaseModel):
    answer: str
    citations: List[Citation]
    report_markdown: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
