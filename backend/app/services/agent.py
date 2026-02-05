from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .rag import RetrievedChunk, Retriever, SimpleAnswerGenerator
from .report import build_poc_report


@dataclass
class AgentResult:
    answer: str
    chunks: List[RetrievedChunk]
    report_markdown: str | None = None


class ResearchAgent:
    def __init__(self, retriever: Retriever) -> None:
        self.retriever = retriever
        self.answer_generator = SimpleAnswerGenerator()

    def run(self, question: str, action: str) -> AgentResult:
        chunks = self.retriever.retrieve(question, top_k=4)
        answer = self.answer_generator.generate(question, chunks)
        if action == "poc_report":
            report = build_poc_report(question, chunks, answer)
            return AgentResult(answer=answer, chunks=chunks, report_markdown=report)
        return AgentResult(answer=answer, chunks=chunks)
