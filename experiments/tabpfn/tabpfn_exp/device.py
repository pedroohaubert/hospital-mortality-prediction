"""MPS só entra se um fit minúsculo de verdade devolver probabilidade finita."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class SmokeResult:
    attempted: bool
    ok: bool
    device: str
    seconds: float | None = None
    peak_rss_mb: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class DeviceDecision:
    selected: str
    reason: str
    mps_built: bool | None
    mps_available: bool | None
    smoke_cpu: SmokeResult
    smoke_mps: SmokeResult


def torch_mps_probe() -> dict:
    try:
        import torch
    except ImportError:
        return {
            "torch_installed": False,
            "torch_version": None,
            "mps_built": None,
            "mps_available": None,
        }
    built = bool(torch.backends.mps.is_built())
    available = bool(torch.backends.mps.is_available()) if built else False
    return {
        "torch_installed": True,
        "torch_version": torch.__version__,
        "mps_built": built,
        "mps_available": available,
    }


def select_device(
    probe: dict,
    smoke: Callable[[str], SmokeResult],
) -> DeviceDecision:
    """Roda CPU sempre. MPS só é escolhido se o smoke desse device passar."""
    cpu = smoke("cpu")
    if not cpu.attempted or not cpu.ok:
        raise RuntimeError(f"O smoke em CPU falhou. {cpu.error or 'sem detalhe'}")
    mps_built = probe.get("mps_built")
    mps_available = probe.get("mps_available")
    if mps_built and mps_available:
        mps = smoke("mps")
        if mps.ok:
            return DeviceDecision(
                selected="mps",
                reason="smoke_mps_ok",
                mps_built=True,
                mps_available=True,
                smoke_cpu=cpu,
                smoke_mps=mps,
            )
        return DeviceDecision(
            selected="cpu",
            reason=f"smoke_mps_falhou: {mps.error or 'sem detalhe'}",
            mps_built=True,
            mps_available=True,
            smoke_cpu=cpu,
            smoke_mps=mps,
        )
    skipped = SmokeResult(attempted=False, ok=False, device="mps", error="mps_indisponivel")
    if not probe.get("torch_installed"):
        reason = "torch_ausente_cpu"
    elif not mps_built:
        reason = "mps_nao_compilado_cpu"
    else:
        reason = "mps_indisponivel_cpu"
    return DeviceDecision(
        selected="cpu",
        reason=reason,
        mps_built=mps_built,
        mps_available=bool(mps_available) if mps_available is not None else False,
        smoke_cpu=cpu,
        smoke_mps=skipped,
    )
