from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .vector_store import VectorStoreBundle


@dataclass
class RetrievedChunk:
    text: str
    source: str
    chunk_id: str
    score: float


class Retriever:
    def __init__(self, bundle: VectorStoreBundle) -> None:
        self.bundle = bundle

    def retrieve(self, query: str, top_k: int) -> List[RetrievedChunk]:
        query_embedding = self.bundle.embedder.transform([query])
        matches = self.bundle.store.search(query_embedding, top_k)
        chunks = []
        for idx, score in matches:
            meta = self.bundle.store.metadata[idx]
            chunks.append(
                RetrievedChunk(
                    text=meta["text"],
                    source=meta["source"],
                    chunk_id=meta["chunk_id"],
                    score=score,
                )
            )
        return chunks


class SimpleAnswerGenerator:
    def generate(self, question: str, chunks: List[RetrievedChunk]) -> str:
        if not chunks:
            return "No relevant context found. Please upload and index documents first."
        context = "\n\n".join(f"Source: {chunk.source}\n{chunk.text}" for chunk in chunks)
        return (
            "Answer based on retrieved evidence:\n"
            f"Question: {question}\n\n"
            f"{context}\n\n"
            "(Tip: connect an LLM provider via OPENAI_API_KEY for richer synthesis.)"
        )
