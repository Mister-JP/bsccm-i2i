"""Shared helpers for antibody metadata normalization."""

from __future__ import annotations

from typing import Any

MISSING_ANTIBODY_LABEL = "__missing_antibody__"


def normalize_antibody_label(value: Any) -> str:
    """Normalize raw antibody labels into stable non-empty strings."""
    if value is None:
        return MISSING_ANTIBODY_LABEL
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return MISSING_ANTIBODY_LABEL
    return text

