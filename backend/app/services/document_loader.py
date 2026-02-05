from pathlib import Path
from typing import Tuple

from pypdf import PdfReader


def load_document(path: Path) -> Tuple[str, str]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text, "pdf"
    if suffix in {".md", ".markdown", ".txt"}:
        return path.read_text(encoding="utf-8"), "markdown"
    raise ValueError(f"Unsupported file type: {suffix}")
