"""Conservative, passage-grounded verification of reader answers."""

from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Sequence, Tuple


def _normalized(value: Any) -> str:
    return " ".join(re.findall(r"[\w]+", str(value or "").lower(), flags=re.UNICODE))


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return default


def build_grounded_answer_verifier_messages(
    question: str,
    original_answer: str,
    passages: Sequence[str],
    *,
    max_passage_chars: int = 1800,
) -> list[dict[str, str]]:
    evidence = [
        f"[Passage {rank}] {str(passage)[:max_passage_chars]}"
        for rank, passage in enumerate(passages, start=1)
    ]
    prompt = f"""Question: {question}
Original answer: {original_answer}

Retrieved passages:
{chr(10).join(evidence)}

Verify whether the original answer satisfies the exact relation and answer granularity requested by the question. Change it only when one retrieved passage explicitly supports a clearly better exact answer span. Do not use outside knowledge and do not infer an unsupported bridge. Return one JSON object with exactly these fields:
{{"keep_original": true, "original_supported": true, "answer": "", "support_passage": 0, "support_span": "", "confidence": 0.0, "reason": ""}}
If there is any ambiguity, keep the original answer. When changing it, answer must be a minimal substring of support_span and support_span must be copied verbatim from exactly one passage."""
    return [
        {
            "role": "system",
            "content": (
                "You are a conservative evidence-grounded answer verifier. "
                "Never replace an answer without explicit textual support. Output JSON only."
            ),
        },
        {"role": "user", "content": prompt},
    ]


def validate_grounded_answer_decision(
    decision: Mapping[str, Any],
    original_answer: str,
    passages: Sequence[str],
    *,
    min_confidence: float = 0.93,
    max_answer_tokens: int = 16,
) -> Tuple[str, Dict[str, Any]]:
    log: Dict[str, Any] = {
        "attempted": True,
        "committed": False,
        "reason": "keep_original",
        "confidence": float(decision.get("confidence", 0.0) or 0.0),
    }
    if _as_bool(decision.get("keep_original"), True):
        return original_answer, log
    if _as_bool(decision.get("original_supported"), True):
        log["reason"] = "original_marked_supported"
        return original_answer, log
    if log["confidence"] < float(min_confidence):
        log["reason"] = "confidence_below_threshold"
        return original_answer, log

    answer = str(decision.get("answer", "") or "").strip()
    support_span = str(decision.get("support_span", "") or "").strip()
    try:
        support_rank = int(decision.get("support_passage", 0) or 0)
    except (TypeError, ValueError):
        support_rank = 0
    if not answer or not support_span or not (1 <= support_rank <= len(passages)):
        log["reason"] = "missing_answer_or_support"
        return original_answer, log
    if len(answer.split()) > int(max_answer_tokens):
        log["reason"] = "answer_too_long"
        return original_answer, log

    passage_key = _normalized(passages[support_rank - 1])
    span_key = _normalized(support_span)
    answer_key = _normalized(answer)
    original_key = _normalized(original_answer)
    if not span_key or span_key not in passage_key:
        log["reason"] = "support_span_not_verbatim"
        return original_answer, log
    if not answer_key or answer_key not in span_key:
        log["reason"] = "answer_not_in_support_span"
        return original_answer, log
    if answer_key == original_key:
        log["reason"] = "unchanged_answer"
        return original_answer, log

    log.update({
        "committed": True,
        "reason": "grounded_replacement",
        "answer": answer,
        "support_passage": support_rank,
        "support_span": support_span,
    })
    return answer, log
