from pathlib import Path


def get_data_dir() -> Path:
    base = Path.cwd() / "data"
    return Path(
        Path(
            __import__("os").environ.get("DATA_DIR", base)
        )
    ).resolve()


DATA_DIR = get_data_dir()
UPLOAD_DIR = DATA_DIR / "uploads"
INDEX_DIR = DATA_DIR / "index"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)
