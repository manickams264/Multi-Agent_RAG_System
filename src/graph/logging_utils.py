"""Task 4 — State transition logging helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def make_transition(
    node_name: str,
    summary: str,
    changed_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "node": node_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "changed_fields": changed_fields or {},
    }
