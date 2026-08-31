"""Fail-closed dataset egress policy enforced at the provider boundary."""

from __future__ import annotations

import ipaddress
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from urllib.parse import urlparse

import pandas as pd

from app.agent.providers import LLMProvider, LLMProviderError, ProviderResponse, ToolCall

egress_logger = logging.getLogger("app.security.llm_egress")


class EgressMode(StrEnum):
    LOCAL_ONLY = "local_only"
    EXTERNAL_REDACTED = "external_redacted"
    LLM_DISABLED = "llm_disabled"


class PIIKind(StrEnum):
    EMAIL = "email"
    PHONE = "phone"
    NAME = "name"
    ADDRESS = "address"
    GOVERNMENT_ID = "government_id"
    PAYMENT_CARD = "payment_card"
    SENSITIVE_IDENTIFIER = "sensitive_identifier"


_NAME_HINTS: dict[PIIKind, tuple[str, ...]] = {
    PIIKind.EMAIL: ("email", "e_mail"),
    PIIKind.PHONE: ("phone", "mobile", "telephone", "whatsapp"),
    PIIKind.NAME: ("name", "full_name", "first_name", "last_name", "customer_name", "contact_name"),
    PIIKind.ADDRESS: ("address", "street", "postal", "zipcode", "zip_code"),
    PIIKind.GOVERNMENT_ID: ("passport", "ssn", "tax_id", "national_id", "government_id"),
    PIIKind.PAYMENT_CARD: ("card_number", "credit_card", "pan", "iban", "bank_account"),
    PIIKind.SENSITIVE_IDENTIFIER: ("customer_id", "user_id", "account_id", "merchant_id", "employee_id", "device_id", "ip_address"),
}
_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
# Deliberately conservative: a bare run of digits (order IDs, SKUs, timestamps,
# large integers) is NOT a phone number. Require an international "+CC" prefix or
# an explicitly grouped/parenthesised layout, then confirm the digit count in
# _classify_value. This avoids flagging whole numeric analysis columns as PII.
_PHONE_RE = re.compile(
    r"(?<![\w])"
    r"(?:\+\d{1,3}[\s.\-]?)?"
    r"(?:\(\d{2,4}\)[\s.\-]?)?"
    r"\d{2,6}(?:[\s.\-]\d{2,6}){1,5}"
    r"(?![\w])"
)
# Uppercase-only: real passport / national-ID prefixes are upper case; dropping
# re.I stops lower-case product codes like "ab1234567" from matching.
_GOV_RE = re.compile(r"\b(?:\d{3}-\d{2}-\d{4}|[A-Z]{1,2}\d{6,9})\b")
_ADDRESS_RE = re.compile(r"\b\d{1,6}\s+[\w.'-]+(?:\s+[\w.'-]+){0,5}\s(?:street|st|road|rd|avenue|ave|lane|ln|drive|dr)\b", re.I)
_PATH_RE = re.compile(r"(?:[A-Za-z]:\\|/)(?:[^\s<>:'\"|?*]+[\\/])+[^\s<>:'\"|?*]*")
_SECRET_RE = re.compile(r"(?i)\b(?:bearer\s+|sk-|api[_-]?key\s*[:=]\s*)[A-Za-z0-9._-]{8,}")
_SAFE_PAYLOAD_VALUES = {
    "sum", "mean", "median", "count", "min", "max", "row", "entity", "true", "false",
    "iqr", "zscore", "pearson", "spearman", "kendall", "not_supported", "estimated", "known",
}
_APPROVED_AGGREGATE_TOOLS = {
    "compare_periods_inference", "localized_period_change", "period_outlier_sensitivity",
    "profile_dataset", "describe_data", "group_and_aggregate", "compare_periods",
    "correlation_analysis", "generate_business_insights", "t_test", "chi_square_test",
    "anova_test", "confidence_interval", "effect_size", "linear_regression",
    "regression_diagnostics", "train_test_split_timeseries", "decompose_timeseries",
    "forecast", "backtest_forecast", "kmeans_cluster", "pca_reduce", "cohort_analysis",
    "analyze_cardinality", "analyze_distributions", "contribution_analysis",
    "mix_decomposition", "executive_summary", "duplicate_analysis", "data_quality_report",
    "correlation_heatmap_data", "boxplot_data", "pareto_chart_data",
}


@dataclass(frozen=True)
class PrivacyProfile:
    aliases: dict[str, str]
    pii_columns: dict[str, tuple[PIIKind, ...]]
    quarantined_values: tuple[str, ...]
    sensitive_value_hashes: frozenset[str]

    @property
    def reverse_aliases(self) -> dict[str, str]:
        return {alias: source for source, alias in self.aliases.items()}


# Free-text PII kinds have no reliable value-level regex, so for a column whose
# *name* signals one of these we still quarantine name-like sampled values.
_FREE_TEXT_PII: frozenset[PIIKind] = frozenset({PIIKind.NAME, PIIKind.ADDRESS})


def _is_quarantinable(value: str) -> bool:
    """`quarantined_values` drives a blind substring replace over the whole
    prompt, so only accept values specific enough that replacing them cannot
    corrupt unrelated text: at least 6 chars, not a short bare number."""
    stripped = value.strip()
    if not (6 <= len(stripped) <= 256):
        return False
    if stripped.isdigit() and len(stripped) < 9:
        return False
    return stripped.lower() not in _SAFE_PAYLOAD_VALUES


def _looks_like_free_text_pii(value: str) -> bool:
    stripped = value.strip()
    if not (6 <= len(stripped) <= 256):
        return False
    if stripped.replace(" ", "").isdigit():
        return False
    return " " in stripped or len(stripped) >= 10


def classify_dataset(df: pd.DataFrame) -> PrivacyProfile:
    aliases = {str(column): f"column_{index}" for index, column in enumerate(df.columns, 1)}
    pii_columns: dict[str, tuple[PIIKind, ...]] = {}
    quarantined: set[str] = set()
    sensitive_hashes: set[str] = set()
    for column in df.columns:
        name = str(column)
        name_kinds = _classify_name(name)
        kinds = set(name_kinds)
        sample = df[column].dropna().astype(str).head(200)
        for value in sample:
            value_kinds = _classify_value(value)
            kinds.update(value_kinds)
            if value_kinds and _is_quarantinable(value):
                quarantined.add(value)
            elif (name_kinds & _FREE_TEXT_PII) and _looks_like_free_text_pii(value):
                quarantined.add(value)
        if kinds:
            pii_columns[name] = tuple(sorted(kinds, key=str))
            for value in df[column].dropna().astype(str):
                if len(value.strip()) >= 3:
                    sensitive_hashes.add(_value_hash(value))
    return PrivacyProfile(
        aliases=aliases,
        pii_columns=pii_columns,
        quarantined_values=tuple(sorted(quarantined, key=len, reverse=True)),
        sensitive_value_hashes=frozenset(sensitive_hashes),
    )


def _value_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def _classify_name(name: str) -> set[PIIKind]:
    normalized = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return {kind for kind, hints in _NAME_HINTS.items() if any(hint == normalized or hint in normalized for hint in hints)}


def _classify_value(value: str) -> set[PIIKind]:
    kinds: set[PIIKind] = set()
    if _EMAIL_RE.search(value): kinds.add(PIIKind.EMAIL)
    phone_match = _PHONE_RE.search(value)
    if phone_match:
        # A grouped/prefixed layout alone is not enough — confirm a realistic
        # phone digit count so "12 3456" or "2024 900" style non-phones drop out.
        phone_digits = re.sub(r"\D", "", phone_match.group())
        if 9 <= len(phone_digits) <= 15 or (value.strip().startswith("+") and 8 <= len(phone_digits) <= 15):
            kinds.add(PIIKind.PHONE)
    if _GOV_RE.search(value): kinds.add(PIIKind.GOVERNMENT_ID)
    if _ADDRESS_RE.search(value): kinds.add(PIIKind.ADDRESS)
    digits = re.sub(r"\D", "", value)
    if 13 <= len(digits) <= 19 and _luhn_valid(digits): kinds.add(PIIKind.PAYMENT_CARD)
    return kinds


def _luhn_valid(digits: str) -> bool:
    total = 0
    parity = len(digits) % 2
    for index, char in enumerate(digits):
        number = int(char)
        if index % 2 == parity:
            number *= 2
            if number > 9: number -= 9
        total += number
    return total % 10 == 0


def validate_local_endpoint(base_url: str) -> None:
    parsed = urlparse(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ValueError("local_only requires an explicit HTTP(S) endpoint.")
    host = parsed.hostname.lower()
    if host == "localhost":
        return
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise ValueError("local_only rejects non-literal external hostnames; use localhost or a private/loopback IP.") from exc
    if not (address.is_loopback or address.is_private):
        raise ValueError("local_only rejects external provider endpoints.")


class DisabledProvider(LLMProvider):
    def __init__(self, model: str = "") -> None:
        self._model = model

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ProviderResponse:
        _emit_egress_audit(
            mode=EgressMode.LLM_DISABLED,
            model=self._model,
            messages=messages,
            tools=tools,
            profile=None,
            call_made=False,
        )
        raise LLMProviderError("LLM functionality is disabled by the server privacy policy.")


class PrivacyEnforcingProvider(LLMProvider):
    """The sole egress boundary: sanitize every call and unalias tool calls locally."""

    def __init__(
        self,
        delegate: LLMProvider,
        mode: EgressMode,
        profile: PrivacyProfile | None = None,
        model: str = "",
    ) -> None:
        self._delegate, self._mode, self._profile, self._model = delegate, mode, profile, model

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ProviderResponse:
        if self._mode == EgressMode.LLM_DISABLED:
            _emit_egress_audit(
                mode=self._mode,
                model=self._model,
                messages=messages,
                tools=tools,
                profile=self._profile,
                call_made=False,
            )
            raise LLMProviderError("LLM functionality is disabled by the server privacy policy.")
        if self._mode == EgressMode.LOCAL_ONLY:
            _emit_egress_audit(
                mode=self._mode,
                model=self._model,
                messages=messages,
                tools=tools,
                profile=self._profile,
                call_made=True,
            )
            return self._delegate.complete(messages, tools)
        if self._profile is None:
            raise LLMProviderError("External LLM egress is blocked until a dataset privacy profile is available.")
        safe_messages = [_sanitize_message(message, self._profile) for message in messages]
        safe_tools = _sanitize_generic(tools, self._profile, payload=False)
        _emit_egress_audit(
            mode=self._mode,
            model=self._model,
            messages=safe_messages,
            tools=safe_tools,
            profile=self._profile,
            original_messages=messages,
            call_made=True,
        )
        response = self._delegate.complete(safe_messages, safe_tools)
        calls = [ToolCall(id=call.id, name=call.name, arguments=_unalias(call.arguments, self._profile)) for call in response.tool_calls]
        content = _restore_aliases(response.content, self._profile) if response.content else response.content
        return ProviderResponse(content=content, tool_calls=calls)


def _emit_egress_audit(
    *,
    mode: EgressMode,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    profile: PrivacyProfile | None,
    call_made: bool,
    original_messages: list[dict[str, Any]] | None = None,
) -> None:
    """Emit metadata about one boundary decision without logging payload content."""
    serialized = json.dumps(
        {"messages": messages, "tools": tools},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    original_text = json.dumps(original_messages or messages, ensure_ascii=False, default=str)
    aliased_hits = 0
    quarantined_hits = 0
    if profile is not None:
        aliased_hits = sum(original_text.count(source) for source in profile.aliases if len(source) >= 2)
        quarantined_hits = sum(original_text.count(value) for value in profile.quarantined_values)
    withheld = sum(
        1
        for message in messages
        if message.get("role") == "tool" and "\"privacy_status\": \"payload_withheld\"" in str(message.get("content", ""))
    )
    event = {
        "event": "llm_egress",
        "mode": mode.value,
        "model": model,
        "message_count": len(messages),
        "tool_names": _tool_names(tools),
        "redactions": {
            "aliased_columns": aliased_hits,
            "quarantined_hits": quarantined_hits,
            "withheld_tool_payloads": withheld,
        },
        "payload_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "call_made": call_made,
    }
    egress_logger.info("%s", json.dumps(event, sort_keys=True, separators=(",", ":")))


def _tool_names(tools: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        if isinstance(name, str):
            names.append(name)
    return names


def _sanitize_message(message: dict[str, Any], profile: PrivacyProfile) -> dict[str, Any]:
    safe = dict(message)
    content = safe.get("content")
    is_tool = safe.get("role") == "tool"
    if isinstance(content, str) and is_tool:
        marker, _, raw = content.partition("\n")
        tool_name = str(safe.get("name", ""))
        if tool_name not in _APPROVED_AGGREGATE_TOOLS:
            safe["content"] = marker + "\n" + json.dumps(
                {"privacy_status": "payload_withheld", "reason": "tool output is not approved for external egress"}
            )
            return safe
        try:
            payload = json.loads(raw)
            safe["content"] = marker + "\n" + json.dumps(_sanitize_generic(payload, profile, payload=True))
        except (ValueError, TypeError):
            safe["content"] = redact_text(content, profile)
    elif isinstance(content, str):
        safe["content"] = redact_text(content, profile)
    if "tool_calls" in safe:
        safe["tool_calls"] = _sanitize_generic(safe["tool_calls"], profile, payload=False)
    return safe


def redact_text(text: str, profile: PrivacyProfile | None = None) -> str:
    result = _SECRET_RE.sub("[REDACTED_SECRET]", _PATH_RE.sub("[REDACTED_PATH]", text))
    result = _EMAIL_RE.sub("[REDACTED_EMAIL]", result)
    result = _PHONE_RE.sub("[REDACTED_PHONE]", result)
    result = _GOV_RE.sub("[REDACTED_ID]", result)
    result = _ADDRESS_RE.sub("[REDACTED_ADDRESS]", result)
    if profile:
        for value in profile.quarantined_values:
            result = result.replace(value, "[REDACTED]")
        for source, alias in sorted(profile.aliases.items(), key=lambda item: len(item[0]), reverse=True):
            # A 1-char column name aliased across free text turns every stray
            # "a"/"x" in prose into "column_N"; skip those in the text pass. The
            # structured tool-arg path (_unalias) still maps them exactly.
            if len(source) < 2:
                continue
            result = re.sub(rf"(?<![\w]){re.escape(source)}(?![\w])", alias, result)
    return result


def _sanitize_generic(value: Any, profile: PrivacyProfile, *, payload: bool) -> Any:
    if isinstance(value, dict):
        return {redact_text(str(key), profile): _sanitize_generic(item, profile, payload=payload) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_generic(item, profile, payload=payload) for item in value]
    if isinstance(value, str):
        if _value_hash(value) in profile.sensitive_value_hashes:
            return "[REDACTED]"
        replaced = redact_text(value, profile)
        if payload and replaced == value and value.lower() not in _SAFE_PAYLOAD_VALUES and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            return "[REDACTED]"
        return replaced
    if payload and isinstance(value, (int, float)) and _value_hash(str(value)) in profile.sensitive_value_hashes:
        return "[REDACTED]"
    return value


def _unalias(value: Any, profile: PrivacyProfile) -> Any:
    if isinstance(value, dict): return {profile.reverse_aliases.get(str(key), key): _unalias(item, profile) for key, item in value.items()}
    if isinstance(value, list): return [_unalias(item, profile) for item in value]
    if isinstance(value, str): return profile.reverse_aliases.get(value, value)
    return value


def _restore_aliases(text: str, profile: PrivacyProfile) -> str:
    result = text
    for alias, source in profile.reverse_aliases.items():
        result = re.sub(rf"\b{re.escape(alias)}\b", source, result)
    return result
