from __future__ import annotations

import re
from pathlib import Path

ALLOWED_EXTENSIONS = {".csv", ".xlsx"}
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9_.\-]")


class ValidationError(Exception):
    pass


def get_extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def validate_extension(filename: str) -> str:
    ext = get_extension(filename)
    if ext not in ALLOWED_EXTENSIONS:
        raise ValidationError(
            f"Unsupported file type '{ext or 'unknown'}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    return ext


def validate_size(size_bytes: int, max_bytes: int) -> None:
    if size_bytes <= 0:
        raise ValidationError("Uploaded file is empty.")
    if size_bytes > max_bytes:
        raise ValidationError(
            f"File too large ({size_bytes / 1024 / 1024:.1f} MB). Max allowed is {max_bytes / 1024 / 1024:.0f} MB."
        )


def sanitize_display_name(filename: str) -> str:
    # Path(...).name strips any directory components, which is what prevents
    # path traversal — the sanitized value is only ever used for display/metadata,
    # never as an actual filesystem path (storage paths use a generated uuid).
    name = Path(filename).name
    name = _UNSAFE_CHARS.sub("_", name)
    return name[:150] or "dataset"
