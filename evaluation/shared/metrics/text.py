from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Dict, Iterable, List

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")


def normalized_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = _PUNCT.sub(" ", text)
    text = _ARTICLES.sub(" ", text)
    text = _WS.sub(" ", text)
    return text.strip()


def exact_match(prediction: str, reference: str) -> float:
    return 1.0 if normalized_text(prediction) == normalized_text(reference) else 0.0


def _tokens(text: str) -> List[str]:
    return [tok for tok in normalized_text(text).split(" ") if tok]


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = _tokens(prediction)
    ref_tokens = _tokens(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    common = sum(min(pred_counts[t], ref_counts[t]) for t in pred_counts)
    if common == 0:
        return 0.0
    precision = common / float(len(pred_tokens))
    recall = common / float(len(ref_tokens))
    return 2.0 * precision * recall / (precision + recall)


def sequence_similarity(prediction: str, reference: str) -> float:
    return SequenceMatcher(None, normalized_text(prediction), normalized_text(reference)).ratio()


def answer_correctness_score(prediction: str, reference: str) -> Dict[str, float]:
    em = exact_match(prediction, reference)
    f1 = token_f1(prediction, reference)
    sim = sequence_similarity(prediction, reference)
    combined = 0.5 * f1 + 0.3 * sim + 0.2 * em
    return {
        "ExactMatch": em,
        "TokenF1": f1,
        "SequenceSimilarity": sim,
        "AnswerCorrectness": combined,
    }
