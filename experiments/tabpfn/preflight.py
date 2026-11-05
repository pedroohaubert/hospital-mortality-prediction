#!/usr/bin/env python3
"""Roda antes do uv sync. Só a biblioteca padrão. Não mexe no ambiente."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tabpfn_exp.gates import main

if __name__ == "__main__":
    raise SystemExit(main())
