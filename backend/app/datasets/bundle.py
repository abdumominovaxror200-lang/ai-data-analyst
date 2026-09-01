from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import duckdb
import pandas as pd
from pydantic import BaseModel, Field, model_validator

from app.datasets.storage import DatasetRecord, DatasetStore
from app.datasets.validation import ValidationError

_ALIAS_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,31}$")


class JoinCardinality(str, Enum):
    ONE_TO_ONE = "one_to_one"
    ONE_TO_MANY = "one_to_many"
    MANY_TO_ONE = "many_to_one"
    MANY_TO_MANY = "many_to_many"


class DatasetBundleMember(BaseModel):
    dataset_id: str
    alias: str

    @model_validator(mode="after")
    def validate_alias(self) -> "DatasetBundleMember":
        if not _ALIAS_RE.fullmatch(self.alias):
            raise ValueError("alias must start with a letter and contain only letters, digits, or underscores")
        return self


class JoinKey(BaseModel):
    left: str
    right: str


class DatasetJoin(BaseModel):
    left_alias: str
    right_alias: str
    keys: list[JoinKey] = Field(min_length=1, max_length=8)
    join_type: Literal["inner", "left", "right", "full"] = "left"
    allow_many_to_many: bool = False


class DatasetBundle(BaseModel):
    members: list[DatasetBundleMember] = Field(min_length=2, max_length=8)
    joins: list[DatasetJoin] = Field(min_length=1, max_length=7)
    name: str = Field(default="joined_dataset.csv", max_length=160)

    @model_validator(mode="after")
    def validate_shape(self) -> "DatasetBundle":
        aliases = [member.alias for member in self.members]
        if len(set(aliases)) != len(aliases):
            raise ValueError("Dataset aliases must be unique.")
        if len(set(member.dataset_id for member in self.members)) != len(self.members):
            raise ValueError("A dataset may appear only once in a bundle.")
        if len(self.joins) != len(self.members) - 1:
            raise ValueError("A bundle must contain exactly one fewer joins than members.")
        introduced = {aliases[0]}
        for join in self.joins:
            if join.left_alias not in introduced:
                raise ValueError(f"Join left alias '{join.left_alias}' has not been introduced yet.")
            if join.right_alias in introduced or join.right_alias not in aliases:
                raise ValueError(f"Join right alias '{join.right_alias}' must introduce one bundle member.")
            introduced.add(join.right_alias)
        if introduced != set(aliases):
            raise ValueError("Every bundle member must be connected by the declared joins.")
        if not self.name.lower().endswith(".csv"):
            raise ValueError("Joined dataset name must end in .csv.")
        return self


class JoinDiagnostics(BaseModel):
    left_alias: str
    right_alias: str
    join_type: str
    cardinality: JoinCardinality
    left_rows: int
    right_rows: int
    left_null_key_rows: int
    right_null_key_rows: int
    left_duplicate_key_rows: int
    right_duplicate_key_rows: int
    unmatched_left_rows: int
    unmatched_right_rows: int


@dataclass
class BundleJoinResult:
    record: DatasetRecord
    diagnostics: list[JoinDiagnostics]
    source_rows: int
    output_rows: int
    row_amplification: float

def join_bundle(bundle: DatasetBundle, store: DatasetStore, *, max_output_rows: int) -> BundleJoinResult:
    records = {member.alias: store.get(member.dataset_id) for member in bundle.members}
    diagnostics: list[JoinDiagnostics] = []
    for join in bundle.joins:
        left = records[join.left_alias].df
        right = records[join.right_alias].df
        _validate_keys(left, right, join)
        report = _diagnose_join(left, right, join)
        if report.cardinality == JoinCardinality.MANY_TO_MANY and not join.allow_many_to_many:
            raise ValidationError(
                f"Join {join.left_alias} -> {join.right_alias} is many-to-many and may multiply rows. "
                "Set allow_many_to_many=true only after reviewing duplicate-key diagnostics."
            )
        diagnostics.append(report)

    result = _execute_generated_join(bundle, records, max_output_rows)
    source_rows = sum(len(record.df) for record in records.values())
    largest_source = max(len(record.df) for record in records.values())
    amplification = len(result) / largest_source if largest_source else 0.0
    content = result.to_csv(index=False).encode("utf-8")
    record = store.save(bundle.name, content)
    return BundleJoinResult(
        record=record,
        diagnostics=diagnostics,
        source_rows=source_rows,
        output_rows=len(result),
        row_amplification=round(amplification, 6),
    )


def _validate_keys(left: pd.DataFrame, right: pd.DataFrame, join: DatasetJoin) -> None:
    for key in join.keys:
        if key.left not in left.columns:
            raise ValidationError(f"Column '{key.left}' was not found in dataset alias '{join.left_alias}'.")
        if key.right not in right.columns:
            raise ValidationError(f"Column '{key.right}' was not found in dataset alias '{join.right_alias}'.")
        left_kind = pd.api.types.infer_dtype(left[key.left].dropna(), skipna=True)
        right_kind = pd.api.types.infer_dtype(right[key.right].dropna(), skipna=True)
        numeric = {"integer", "floating", "mixed-integer-float", "decimal"}
        if left_kind != right_kind and not ({left_kind, right_kind} <= numeric):
            raise ValidationError(
                f"Join key types are incompatible: {join.left_alias}.{key.left} ({left_kind}) and "
                f"{join.right_alias}.{key.right} ({right_kind})."
            )


def _diagnose_join(left: pd.DataFrame, right: pd.DataFrame, join: DatasetJoin) -> JoinDiagnostics:
    left_keys = [key.left for key in join.keys]
    right_keys = [key.right for key in join.keys]
    left_null = left[left_keys].isna().any(axis=1)
    right_null = right[right_keys].isna().any(axis=1)
    left_valid = left.loc[~left_null, left_keys]
    right_valid = right.loc[~right_null, right_keys]
    left_dup = left_valid.duplicated(keep=False)
    right_dup = right_valid.duplicated(keep=False)
    cardinality = _cardinality(bool(left_dup.any()), bool(right_dup.any()))

    left_index = pd.MultiIndex.from_frame(left_valid)
    right_index = pd.MultiIndex.from_frame(right_valid.set_axis(left_keys, axis=1))
    unmatched_left = int((~left_index.isin(right_index)).sum())
    unmatched_right = int((~right_index.isin(left_index)).sum())
    return JoinDiagnostics(
        left_alias=join.left_alias,
        right_alias=join.right_alias,
        join_type=join.join_type,
        cardinality=cardinality,
        left_rows=len(left),
        right_rows=len(right),
        left_null_key_rows=int(left_null.sum()),
        right_null_key_rows=int(right_null.sum()),
        left_duplicate_key_rows=int(left_dup.sum()),
        right_duplicate_key_rows=int(right_dup.sum()),
        unmatched_left_rows=unmatched_left,
        unmatched_right_rows=unmatched_right,
    )


def _cardinality(left_many: bool, right_many: bool) -> JoinCardinality:
    if left_many and right_many:
        return JoinCardinality.MANY_TO_MANY
    if left_many:
        return JoinCardinality.MANY_TO_ONE
    if right_many:
        return JoinCardinality.ONE_TO_MANY
    return JoinCardinality.ONE_TO_ONE


def _execute_generated_join(
    bundle: DatasetBundle, records: dict[str, DatasetRecord], max_output_rows: int
) -> pd.DataFrame:
    conn = duckdb.connect(":memory:")
    try:
        aliases = [member.alias for member in bundle.members]
        for alias, record in records.items():
            conn.register(alias, record.df)
        select_columns = [
            f'{_q(alias)}.{_q(str(column))} AS {_q(f"{alias}__{column}")}'
            for alias in aliases
            for column in records[alias].df.columns
        ]
        sql = f"SELECT {', '.join(select_columns)} FROM {_q(aliases[0])}"
        join_words = {"inner": "INNER", "left": "LEFT", "right": "RIGHT", "full": "FULL OUTER"}
        for join in bundle.joins:
            predicates = [
                f'{_q(join.left_alias)}.{_q(key.left)} = {_q(join.right_alias)}.{_q(key.right)}'
                for key in join.keys
            ]
            sql += f" {join_words[join.join_type]} JOIN {_q(join.right_alias)} ON {' AND '.join(predicates)}"
        result = conn.execute(f"SELECT * FROM ({sql}) joined LIMIT ?", [max_output_rows + 1]).fetchdf()
    except duckdb.Error as exc:
        raise ValidationError(f"Dataset join could not be completed: {str(exc).splitlines()[0]}") from exc
    finally:
        conn.close()
    if len(result) > max_output_rows:
        raise ValidationError(
            f"Joined dataset exceeds the safe {max_output_rows:,}-row limit. "
            "Use more selective keys or filters before joining."
        )
    return result


def _q(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'
