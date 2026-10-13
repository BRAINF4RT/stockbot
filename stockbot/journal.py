"""Append-only decision log so forecasts can be scored for calibration later."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def log(state_dir: str, kind: str, **fields) -> None:
    path = Path(state_dir) / "journal.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "kind": kind, **fields}
    with path.open("a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
