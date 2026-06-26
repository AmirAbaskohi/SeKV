from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    """Lowercase, remove punctuation/articles/extra spaces (SQuAD-style)."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _f1_single(prediction: str, reference: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return (2 * precision * recall) / (precision + recall)


def qa_f1(prediction: str, references: list[str]) -> float:
    if not references:
        return 0.0
    return max(_f1_single(prediction, ref) for ref in references)


def exact_match(prediction: str, references: list[str]) -> float:
    pred = normalize_answer(prediction)
    return 1.0 if any(pred == normalize_answer(r) for r in references) else 0.0


def rouge_l(prediction: str, reference: str) -> float:
    pred = prediction.split()
    ref = reference.split()
    if not pred or not ref:
        return 0.0

    # Classic LCS DP on tokens.
    dp = [[0] * (len(ref) + 1) for _ in range(len(pred) + 1)]
    for i in range(1, len(pred) + 1):
        for j in range(1, len(ref) + 1):
            if pred[i - 1] == ref[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[-1][-1]
    prec = lcs / len(pred)
    rec = lcs / len(ref)
    if prec + rec == 0:
        return 0.0
    beta = 1.2
    return ((1 + beta**2) * prec * rec) / (rec + beta**2 * prec)


def substring_match(prediction: str, references: list[str]) -> float:
    pred = prediction.lower()
    for ref in references:
        if ref.lower() in pred:
            return 1.0
    return 0.0


def classification_score(prediction: str, references: list[str]) -> float:
    pred = normalize_answer(prediction)
    for ref in references:
        if pred == normalize_answer(ref):
            return 1.0
    return 0.0


def _extract_last_number(text: str) -> str | None:
    # Prefer GSM8K-style final answer marker.
    marker = re.findall(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
    if marker:
        return marker[-1].replace(",", "")

    all_nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if not all_nums:
        return None
    return all_nums[-1].replace(",", "")


def gsm8k_answer_match(prediction: str, reference: str) -> float:
    p = _extract_last_number(prediction)
    r = _extract_last_number(reference)
    if p is None or r is None:
        return 0.0
    return 1.0 if p == r else 0.0


LONGBENCH_TASK_METRIC: dict[str, str] = {
    "narrativeqa": "qa_f1",
    "qasper": "qa_f1",
    "multifieldqa_en": "qa_f1",
    "hotpotqa": "qa_f1",
    "2wikimqa": "qa_f1",
    "musique": "qa_f1",
    "gov_report": "rouge_l",
    "qmsum": "rouge_l",
    "multi_news": "rouge_l",
    "trec": "classification",
    "triviaqa": "qa_f1",
    "samsum": "rouge_l",
    "passage_count": "exact_match",
    "passage_retrieval_en": "exact_match",
    "lcc": "exact_match",
    "repobench-p": "exact_match",
}


def score_with(metric_key: str, pred: str, refs: list[str]) -> float:
    if metric_key == "qa_f1":
        return qa_f1(pred, refs)
    if metric_key == "exact_match":
        return exact_match(pred, refs)
    if metric_key == "rouge_l":
        return max((rouge_l(pred, r) for r in refs), default=0.0)
    if metric_key == "substring_match":
        return substring_match(pred, refs)
    if metric_key == "classification":
        return classification_score(pred, refs)
    if metric_key == "gsm8k":
        return gsm8k_answer_match(pred, refs[0] if refs else "")

    raise KeyError(f"Unknown metric key: {metric_key}")
