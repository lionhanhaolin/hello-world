from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Protocol, Tuple

import faiss
import numpy as np

from .embeddings import TfidfEmbedder


class VectorStore(Protocol):
    def add(self, embeddings: np.ndarray, metadata: List[Dict[str, str]]) -> None:
        ...

    def search(self, query_embeddings: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        ...

    def save(self, path: Path) -> None:
        ...

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        ...


@dataclass
class FaissVectorStore:
    index: faiss.Index
    metadata: List[Dict[str, str]]

    @classmethod
    def build(cls, embeddings: np.ndarray, metadata: List[Dict[str, str]]) -> "FaissVectorStore":
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return cls(index=index, metadata=metadata)

    def add(self, embeddings: np.ndarray, metadata: List[Dict[str, str]]) -> None:
        self.index.add(embeddings)
        self.metadata.extend(metadata)

    def search(self, query_embeddings: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        distances, indices = self.index.search(query_embeddings, top_k)
        return [(int(idx), float(dist)) for idx, dist in zip(indices[0], distances[0]) if idx != -1]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "faiss.index"))
        (path / "metadata.json").write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "FaissVectorStore":
        index = faiss.read_index(str(path / "faiss.index"))
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        return cls(index=index, metadata=metadata)


@dataclass
class VectorStoreBundle:
    store: FaissVectorStore
    embedder: TfidfEmbedder

    def save(self, path: Path) -> None:
        self.store.save(path)
        with (path / "vectorizer.pkl").open("wb") as handle:
            pickle.dump(self.embedder.vectorizer, handle)

    @classmethod
    def load(cls, path: Path) -> "VectorStoreBundle":
        store = FaissVectorStore.load(path)
        with (path / "vectorizer.pkl").open("rb") as handle:
            vectorizer = pickle.load(handle)
        embedder = TfidfEmbedder(vectorizer)
        return cls(store=store, embedder=embedder)


class MilvusVectorStore:
    """Placeholder for Milvus integration."""

    def __init__(self) -> None:
        raise NotImplementedError("Milvus integration is not implemented yet.")


class PgVectorStore:
    """Placeholder for pgvector integration."""

    def __init__(self) -> None:
        raise NotImplementedError("pgvector integration is not implemented yet.")
