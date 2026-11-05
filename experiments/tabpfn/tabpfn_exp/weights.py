"""Checksum do peso v2. Sem token na linha de comando."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

from tabpfn_exp.protocol import HF_FILE_URL, WEIGHT_BYTES, WEIGHT_FILENAME, WEIGHT_SHA256


def file_sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def verify_weight(path: Path) -> dict:
    if not path.is_file():
        return {"ok": False, "reason": "arquivo_ausente", "path": str(path)}
    digest, size = file_sha256(path)
    ok = digest == WEIGHT_SHA256 and size == WEIGHT_BYTES
    return {
        "ok": ok,
        "sha256": digest,
        "bytes": size,
        "expected_sha256": WEIGHT_SHA256,
        "expected_bytes": WEIGHT_BYTES,
        "reason": "ok" if ok else "checksum_diferente",
    }


def fetch_weight(dest_dir: Path) -> dict:
    """Baixa o checkpoint público do Hugging Face. Não abre login de licença."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / WEIGHT_FILENAME
    if target.is_file():
        checked = verify_weight(target)
        checked["downloaded"] = False
        return checked
    temporary = target.with_suffix(target.suffix + ".partial")
    try:
        with urllib.request.urlopen(HF_FILE_URL, timeout=180) as response:
            temporary.write_bytes(response.read())
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        return {"ok": False, "reason": "download_falhou", "error": f"{type(exc).__name__}: {exc}"}
    temporary.replace(target)
    checked = verify_weight(target)
    checked["downloaded"] = True
    if not checked["ok"]:
        target.unlink(missing_ok=True)
    return checked
