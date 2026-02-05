from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class Embedder(Protocol):
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        ...

    def transform(self, texts: List[str]) -> np.ndarray:
        ...


@dataclass
class TfidfEmbedder:
    vectorizer: TfidfVectorizer

    @classmethod
    def create(cls) -> "TfidfEmbedder":
        return cls(TfidfVectorizer(stop_words="english"))

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        return self.vectorizer.fit_transform(texts).toarray().astype("float32")

    def transform(self, texts: List[str]) -> np.ndarray:
        return self.vectorizer.transform(texts).toarray().astype("float32")
