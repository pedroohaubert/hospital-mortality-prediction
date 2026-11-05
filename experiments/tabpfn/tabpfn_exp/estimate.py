"""Projeção do tempo só existe depois de um smoke medido."""

from __future__ import annotations

from tabpfn_exp.protocol import N_FITS, SMOKE_ROWS, TRAIN_ROWS_PER_FOLD

ASSUMPTION = "linear_em_linhas_de_treino_nao_testada"


def project_runtime(
    smoke_seconds: float | None,
    *,
    smoke_rows: int = SMOKE_ROWS,
    train_rows: int = TRAIN_ROWS_PER_FOLD,
    n_fits: int = N_FITS,
) -> dict:
    if smoke_seconds is None or smoke_seconds <= 0 or smoke_rows <= 0:
        return {
            "status": "unmeasured",
            "seconds": None,
            "assumption": None,
            "not_a_measurement": True,
        }
    seconds = float(smoke_seconds) * (train_rows / smoke_rows) * n_fits
    return {
        "status": "projected",
        "seconds": seconds,
        "assumption": ASSUMPTION,
        "smoke_rows": smoke_rows,
        "train_rows": train_rows,
        "n_fits": n_fits,
        "not_a_measurement": True,
    }


def project_memory(smoke_peak_rss_mb: float | None) -> dict:
    """O pico do smoke é medido. O pico do fold cheio não é."""
    return {
        "status": "unmeasured_full_fold",
        "smoke_peak_rss_mb": smoke_peak_rss_mb,
        "full_fold_mb": None,
        "not_a_measurement": True,
    }
