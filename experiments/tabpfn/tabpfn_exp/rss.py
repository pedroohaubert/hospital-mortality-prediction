"""Pico de RSS do processo. No Mac o valor vem em bytes. No Linux, em KiB."""

from __future__ import annotations

import resource
import sys


def peak_rss_mb() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return raw / (1024 * 1024)
    return raw / 1024
