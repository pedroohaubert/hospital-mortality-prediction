"""Portões de disco e memória. Não instala nada e não apaga arquivo."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

# Soma das wheels macOS arm64 / universais do lock para CPython 3.12,
# arredondada para cima. Não é o tamanho medido numa instalação.
WHEEL_BYTES = 350 * 1024 * 1024
# Pico assume cache local do uv mais o venv descompactado, e o CPython gerenciado.
INSTALL_PEAK_BYTES = 2 * 1024 * 1024 * 1024
DISK_HEADROOM_BYTES = 5 * 1024 * 1024 * 1024
MIN_TOTAL_RAM_BYTES = 15 * 1024 * 1024 * 1024
MIN_AVAILABLE_RAM_BYTES = 6 * 1024 * 1024 * 1024


def decide(
    *,
    stage: str,
    free_bytes: int,
    total_ram_bytes: int | None,
    available_ram_bytes: int | None,
) -> dict:
    """`stage` é `install` ou `full_run`. Números em bytes."""
    if stage == "install":
        need = INSTALL_PEAK_BYTES + DISK_HEADROOM_BYTES
        ok = free_bytes >= need
        reason = "ok" if ok else "disco_abaixo_do_pico_mais_folga"
    elif stage == "full_run":
        disk_ok = free_bytes >= DISK_HEADROOM_BYTES
        ram_total_ok = total_ram_bytes is not None and total_ram_bytes >= MIN_TOTAL_RAM_BYTES
        ram_free_ok = available_ram_bytes is not None and available_ram_bytes >= MIN_AVAILABLE_RAM_BYTES
        ok = disk_ok and ram_total_ok and ram_free_ok
        if not disk_ok:
            reason = "disco_abaixo_da_folga"
        elif not ram_total_ok:
            reason = "ram_total_abaixo_de_15gib"
        elif not ram_free_ok:
            reason = "ram_disponivel_abaixo_de_6gib"
        else:
            reason = "ok"
        need = DISK_HEADROOM_BYTES
    else:
        raise ValueError(f"Etapa desconhecida: {stage}")
    return {
        "ok": ok,
        "stage": stage,
        "reason": reason,
        "free_bytes": free_bytes,
        "need_bytes": need,
        "install_peak_bytes": INSTALL_PEAK_BYTES,
        "headroom_bytes": DISK_HEADROOM_BYTES,
        "total_ram_bytes": total_ram_bytes,
        "available_ram_bytes": available_ram_bytes,
        "wheel_bytes_estimate": WHEEL_BYTES,
        "peak_is_measurement": False,
    }


def parse_vm_stat(text: str, page_size: int) -> int | None:
    """Soma páginas free, inactive e speculative. Devolve bytes ou None."""
    wanted = {"pages free", "pages inactive", "pages speculative"}
    pages = 0
    found = 0
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        if key.strip().lower() not in wanted:
            continue
        digits = "".join(ch for ch in value if ch.isdigit())
        if not digits:
            continue
        pages += int(digits)
        found += 1
    if found == 0:
        return None
    return pages * page_size


def snapshot(path: Path | None = None) -> dict:
    target = path or Path.cwd()
    free = shutil.disk_usage(target).free
    total_ram = None
    available = None
    if sys.platform == "darwin":
        try:
            total_ram = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
            page = int(subprocess.check_output(["sysctl", "-n", "hw.pagesize"], text=True).strip())
            vm = subprocess.check_output(["vm_stat"], text=True)
            available = parse_vm_stat(vm, page)
        except (OSError, subprocess.CalledProcessError, ValueError):
            total_ram = total_ram
            available = None
    return {
        "free_bytes": free,
        "total_ram_bytes": total_ram,
        "available_ram_bytes": available,
        "python": sys.version.split()[0],
        "platform": sys.platform,
    }


def enforce(stage: str, path: Path | None = None) -> dict:
    current = snapshot(path)
    result = decide(
        stage=stage,
        free_bytes=current["free_bytes"],
        total_ram_bytes=current["total_ram_bytes"],
        available_ram_bytes=current["available_ram_bytes"],
    )
    result["python"] = current["python"]
    result["platform"] = current["platform"]
    if current["python"].startswith("3.15") and stage == "install":
        result["python_note"] = "O python3 do sistema é 3.15 e não entra no venv. O uv baixa o 3.12."
    return result


def main() -> int:
    result = enforce("install")
    gib = 1024**3
    print(
        f"etapa install  livre {result['free_bytes'] / gib:.1f} GiB  "
        f"pico previsto {result['install_peak_bytes'] / gib:.1f} GiB  "
        f"folga {result['headroom_bytes'] / gib:.1f} GiB  "
        f"python {result['python']}"
    )
    print(f"motivo {result['reason']}")
    if result.get("python_note"):
        print(result["python_note"])
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
