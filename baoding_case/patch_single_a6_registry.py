#!/usr/bin/env python3
"""Add compatibility metadata required by the established Baoding figure builder."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    path = Path(sys.argv[1])
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("gps_role", "offline calibration and scoring only; no GPS in held-out update")
    data.setdefault("typography", {"reference": "main-text Figure 4/5 rules"})
    data.setdefault("sources", {})
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
