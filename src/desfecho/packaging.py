"""Zip com o código e a lista de linguagens, bibliotecas e ferramentas."""

from __future__ import annotations

import zipfile
from pathlib import Path

INCLUDE_FILES = [
    "pyproject.toml",
    "uv.lock",
    "requirements.txt",
    "README.md",
    "LICENSE",
    "docs/ferramentas.md",
    "docs/abordagem.md",
    "docs/como-rodar.md",
    "docs/nota-de-pesquisa.md",
    "artifacts/modelo.joblib",
    "artifacts/limiar.json",
    "artifacts/metrics.json",
]

INCLUDE_DIRS = ["src"]


def build_zip(root: Path, destination: Path) -> Path:
    root = root.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    missing = [rel for rel in INCLUDE_FILES if not (root / rel).exists()]
    if missing:
        raise FileNotFoundError(
            "Ainda falta arquivo para o zip: " + ", ".join(missing) + ". Rode o treino e confira a documentação."
        )
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for rel in INCLUDE_FILES:
            archive.write(root / rel, arcname=rel)
        for folder in INCLUDE_DIRS:
            base = root / folder
            for path in sorted(base.rglob("*")):
                if path.is_dir() or "__pycache__" in path.parts or path.suffix == ".pyc":
                    continue
                archive.write(path, arcname=str(path.relative_to(root)))
    return destination
