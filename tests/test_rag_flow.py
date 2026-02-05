import importlib
import os
from pathlib import Path

from fastapi.testclient import TestClient


def test_index_and_ask(tmp_path: Path) -> None:
    os.environ["DATA_DIR"] = str(tmp_path / "data")
    from backend.app import config

    importlib.reload(config)
    from backend.app import main

    importlib.reload(main)
    client = TestClient(main.app)

    sample_text = "BioMed testing document about oncology and biomarkers."
    upload = client.post(
        "/upload",
        files={"file": ("sample.md", sample_text.encode("utf-8"))},
    )
    assert upload.status_code == 200

    index = client.post("/index", json={})
    assert index.status_code == 200
    assert index.json()["chunks"] >= 1

    ask = client.post("/ask", json={"question": "What is this about?", "top_k": 2})
    assert ask.status_code == 200
    data = ask.json()
    assert "Answer" in data["answer"]
    assert data["citations"]
