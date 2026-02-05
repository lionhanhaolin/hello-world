import importlib
import os
from pathlib import Path

from fastapi.testclient import TestClient


def test_health(tmp_path: Path) -> None:
    os.environ["DATA_DIR"] = str(tmp_path / "data")
    from backend.app import config

    importlib.reload(config)
    from backend.app import main

    importlib.reload(main)
    client = TestClient(main.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
