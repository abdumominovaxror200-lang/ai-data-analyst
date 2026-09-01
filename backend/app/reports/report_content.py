from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.data_quality_gate import evaluate_data_quality
from app.datasets.storage import DatasetRecord
from app.tools.report import generate_report

_INVALID_XML_CHARACTER = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")


@dataclass(frozen=True)
class ReportContent:
    report: dict[str, Any]
    caveats: Any
    quality_limitations: list[Any]
    analysis: Any | None
    ingestion_notices: list[Any]


def collect_report_content(record: DatasetRecord) -> ReportContent:
    """Collect the deterministic local facts shared by every document export."""
    report = generate_report(record.df, record.id, record.original_filename)
    caveats, quality_limitations = evaluate_data_quality(record.df)
    return ReportContent(
        report=report,
        caveats=caveats,
        quality_limitations=quality_limitations,
        analysis=record.latest_analysis,
        ingestion_notices=list(record.ingestion_notices or []),
    )


def report_filename(record: DatasetRecord, extension: str) -> str:
    stem = Path(record.original_filename).stem or "dataset"
    safe = "".join(
        character if character.isascii() and (character.isalnum() or character in "-_") else "_"
        for character in stem
    )
    return f"{safe[:100] or 'dataset'}-analysis-report.{extension}"


def combined_limitations(content: ReportContent) -> list[Any]:
    limitations = list(content.quality_limitations)
    if content.analysis is not None:
        limitations.extend(content.analysis.limitations)
    unique: dict[tuple[str, str, str], Any] = {}
    for limitation in limitations:
        unique[(limitation.category, limitation.severity, limitation.text)] = limitation
    return list(unique.values())


def safe_text(value: Any, limit: int = 8_000) -> str:
    if value is None:
        return ""
    return _INVALID_XML_CHARACTER.sub("", str(value))[:limit]
