from __future__ import annotations

from typing import List

from .rag import RetrievedChunk


def build_poc_report(question: str, chunks: List[RetrievedChunk], answer: str) -> str:
    sources = "\n".join(
        f"- `{chunk.source}` (chunk {chunk.chunk_id})" for chunk in chunks
    )
    evidence = "\n\n".join(
        f"> {chunk.text}\n\n_Source: {chunk.source} / chunk {chunk.chunk_id}_"
        for chunk in chunks
    )
    return (
        "# POC交付报告\n\n"
        "## 需求概述\n"
        f"- 用户问题：{question}\n\n"
        "## RAG答案摘要\n"
        f"{answer}\n\n"
        "## 关键引用\n"
        f"{evidence}\n\n"
        "## 使用的资料来源\n"
        f"{sources}\n\n"
        "## 建议的下一步\n"
        "- 连接领域专用大模型以提升总结质量。\n"
        "- 接入Milvus或pgvector进行规模化索引。\n"
        "- 补充结构化指标监控（响应时延/命中率/引用覆盖）。\n"
    )
