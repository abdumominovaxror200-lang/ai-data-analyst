"""Conservative typed definitions for dataset metrics and their denominators."""
from __future__ import annotations

import re
from typing import Literal
import pandas as pd
from pydantic import BaseModel, Field

MetricKind = Literal["measure", "count", "rate", "ratio"]
ResolutionStatus = Literal["resolved", "needs_definition"]
_DERIVED_NAME = re.compile(r"(?:^|_)(rate|ratio|pct|percent|percentage|share)(?:$|_)", re.I)


class MetricDefinition(BaseModel):
    name: str
    kind: MetricKind
    numerator_column: str | None = None
    denominator_column: str | None = None
    numerator_aggregation: Literal["sum", "count", "mean"] | None = None
    denominator_aggregation: Literal["sum", "count", "count_distinct", "row_count"] | None = None
    unit: str | None = None
    status: ResolutionStatus = "resolved"
    reason: str


class MetricRegistry(BaseModel):
    metrics: dict[str, MetricDefinition] = Field(default_factory=dict)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "MetricRegistry":
        definitions: dict[str, MetricDefinition] = {}
        for raw_name in df.columns:
            name, series = str(raw_name), df[raw_name]
            if not pd.api.types.is_numeric_dtype(series):
                continue
            lowered = name.lower()
            if _DERIVED_NAME.search(name):
                definitions[name] = MetricDefinition(
                    name=name, kind="rate" if any(token in lowered for token in ("rate", "pct", "percent")) else "ratio",
                    numerator_column=name, unit="percent" if "pct" in lowered or "percent" in lowered else None,
                    status="needs_definition",
                    reason="Stored derived metric has no verified denominator definition in the dataset schema.",
                )
            else:
                definitions[name] = MetricDefinition(name=name, kind="measure", numerator_column=name, numerator_aggregation="sum", reason="Direct numeric dataset column.")
            values = set(series.dropna().unique().tolist())
            if values and values.issubset({0, 1, False, True}):
                rate_name = f"{name}_rate"
                definitions[rate_name] = MetricDefinition(
                    name=rate_name, kind="rate", numerator_column=name, numerator_aggregation="sum",
                    denominator_aggregation="row_count", unit="proportion",
                    reason=f"Binary indicator '{name}' divided by the eligible row population.",
                )
        return cls(metrics=definitions)

    def definition_for(self, name: str) -> MetricDefinition | None:
        return self.metrics.get(name)

    def register(self, definition: MetricDefinition) -> None:
        """Register an explicit reviewed definition; callers must supply all semantics."""
        if definition.kind in ("rate", "ratio") and definition.status == "resolved":
            if definition.denominator_column is None and definition.denominator_aggregation != "row_count":
                raise ValueError("A resolved rate/ratio requires a denominator column or row_count denominator.")
            if definition.numerator_column is None:
                raise ValueError("A resolved rate/ratio requires a numerator column.")
        self.metrics[definition.name] = definition

    def require_resolved(self, name: str) -> MetricDefinition:
        definition = self.definition_for(name)
        if definition is None or definition.status != "resolved":
            raise ValueError(f"Metric '{name}' needs an explicit numerator and denominator definition before analysis.")
        return definition

    def public_definitions(self) -> list[MetricDefinition]:
        return [self.metrics[name] for name in sorted(self.metrics)]
