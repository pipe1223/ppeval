from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence


@dataclass
class JudgeScore:
    score: float
    rationale: str
    label: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class JudgeAdapter(Protocol):
    def score(self, sample: Dict[str, Any], rubric: Optional[Sequence[Dict[str, Any]]] = None) -> JudgeScore:
        ...


class CallableJudgeAdapter:
    def __init__(self, fn: Callable[[Dict[str, Any], Optional[Sequence[Dict[str, Any]]]], JudgeScore]):
        self.fn = fn

    def score(self, sample: Dict[str, Any], rubric: Optional[Sequence[Dict[str, Any]]] = None) -> JudgeScore:
        return self.fn(sample, rubric)


class RuleBasedJudgeAdapter:
    """A minimal fallback judge for local testing when no LLM judge is configured."""

    def score(self, sample: Dict[str, Any], rubric: Optional[Sequence[Dict[str, Any]]] = None) -> JudgeScore:
        answer = str(sample.get("answer", "") or "")
        ref = str(sample.get("gold_answer", "") or "")
        score = 1.0 if answer.strip().lower() == ref.strip().lower() and ref else 0.0
        rationale = "exact match" if score == 1.0 else "fallback rule-based mismatch"
        return JudgeScore(score=score, rationale=rationale, label="rule_based")


def evaluate_with_judge(samples: Iterable[Dict[str, Any]], judge: JudgeAdapter, rubric: Optional[Sequence[Dict[str, Any]]] = None) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    scores: List[float] = []
    for sample in samples:
        out = judge.score(sample, rubric)
        rows.append({"sample_id": sample.get("sample_id"), **out.to_dict()})
        scores.append(float(out.score))
    mean_score = sum(scores) / float(len(scores)) if scores else 0.0
    return {"results": {"JudgeMeanScore": mean_score}, "details": rows}
