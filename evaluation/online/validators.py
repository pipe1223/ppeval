from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Sequence


@dataclass
class ValidationResult:
    validator: str
    passed: bool
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BaseValidator:
    name = "base"

    def validate(self, sample: Dict[str, Any]) -> ValidationResult:
        raise NotImplementedError


class MaxLengthValidator(BaseValidator):
    name = "max_length"

    def __init__(self, field: str = "answer", max_chars: int = 4000):
        self.field = field
        self.max_chars = max_chars

    def validate(self, sample: Dict[str, Any]) -> ValidationResult:
        value = str(sample.get(self.field, "") or "")
        passed = len(value) <= self.max_chars
        return ValidationResult(self.name, passed, f"{self.field} length={len(value)} max={self.max_chars}")


class RegexMatchValidator(BaseValidator):
    name = "regex_match"

    def __init__(self, field: str, pattern: str):
        self.field = field
        self.pattern = re.compile(pattern)

    def validate(self, sample: Dict[str, Any]) -> ValidationResult:
        value = str(sample.get(self.field, "") or "")
        passed = bool(self.pattern.search(value))
        return ValidationResult(self.name, passed, f"pattern={self.pattern.pattern}")


class BannedPhraseValidator(BaseValidator):
    name = "banned_phrase"

    def __init__(self, field: str = "answer", phrases: Sequence[str] = ()): 
        self.field = field
        self.phrases = [p.lower() for p in phrases]

    def validate(self, sample: Dict[str, Any]) -> ValidationResult:
        value = str(sample.get(self.field, "") or "").lower()
        hit = next((p for p in self.phrases if p in value), None)
        passed = hit is None
        return ValidationResult(self.name, passed, f"banned_phrase={hit}" if hit else "no banned phrase")


class ContainsCitationValidator(BaseValidator):
    name = "contains_citation"

    def __init__(self, field: str = "answer"):
        self.field = field

    def validate(self, sample: Dict[str, Any]) -> ValidationResult:
        value = str(sample.get(self.field, "") or "")
        passed = ("[" in value and "]" in value) or bool(sample.get("cited_ids"))
        return ValidationResult(self.name, passed, "citation markers or cited_ids required")


class JsonFieldValidator(BaseValidator):
    name = "required_field"

    def __init__(self, field: str):
        self.field = field

    def validate(self, sample: Dict[str, Any]) -> ValidationResult:
        passed = self.field in sample and sample.get(self.field) is not None
        return ValidationResult(self.name, passed, f"field={self.field}")


def run_validators(sample: Dict[str, Any], validators: Iterable[BaseValidator]) -> Dict[str, Any]:
    results = [validator.validate(sample).to_dict() for validator in validators]
    return {
        "passed": all(r["passed"] for r in results),
        "results": results,
    }
