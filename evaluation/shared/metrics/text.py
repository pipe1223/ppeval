from __future__ import annotations

import re
import string
from collections import Counter


def normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def exact_match(prediction: str, reference: str) -> float:
    return 1.0 if normalize_text(prediction) == normalize_text(reference) else 0.0


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
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
