"""Query-adaptive selection over complete PPR, dense, and sparse rankings.

The module deliberately keeps access-path selection separate from ontology
execution.  Every path produces a complete ranking first; a gold-free router
then selects one ranking as the input to the downstream evidence planner.
Training-only replay labels are added by ``CompassRAG.record_*`` and are never
consumed by this runtime module.
"""

from __future__ import annotations

import heapq
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


ACCESS_PATHS = ("ppr", "dense", "sparse")
ALTERNATIVES = ACCESS_PATHS[1:]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


class SparseBM25Index:
    """Deterministic in-memory BM25 index for the frozen evidence corpus."""

    def __init__(
        self,
        documents: Sequence[str],
        *,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        self.documents = [str(value) for value in documents]
        self.k1 = float(k1)
        self.b = float(b)
        self.lengths: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        document_frequency: Counter[str] = Counter()
        for index, document in enumerate(self.documents):
            counts = Counter(_tokens(document))
            self.lengths.append(sum(counts.values()))
            for term, frequency in counts.items():
                self.postings[term].append((index, int(frequency)))
                document_frequency[term] += 1
        self.average_length = sum(self.lengths) / max(len(self.lengths), 1)
        count = len(self.documents)
        self.idf = {
            term: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def search(
        self,
        query: str,
        *,
        top_k: int,
        fallback_ids: Sequence[int] = (),
    ) -> Tuple[np.ndarray, np.ndarray]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(_tokens(query)):
            inverse_frequency = self.idf.get(term)
            if inverse_frequency is None:
                continue
            for index, term_frequency in self.postings[term]:
                normalization = term_frequency + self.k1 * (
                    1.0
                    - self.b
                    + self.b
                    * self.lengths[index]
                    / max(self.average_length, 1.0)
                )
                scores[index] += (
                    inverse_frequency
                    * term_frequency
                    * (self.k1 + 1.0)
                    / max(normalization, 1e-12)
                )

        limit = max(int(top_k), 1)
        ranked = heapq.nlargest(
            limit,
            scores.items(),
            key=lambda item: (item[1], -item[0]),
        )
        selected_ids = [int(index) for index, _ in ranked]
        selected_scores = [float(score) for _, score in ranked]
        selected = set(selected_ids)
        floor = min(selected_scores, default=0.0)
        for index in fallback_ids:
            index = int(index)
            if index in selected:
                continue
            selected.add(index)
            selected_ids.append(index)
            floor -= 1e-9
            selected_scores.append(floor)
            if len(selected_ids) >= limit:
                break
        if not selected_ids:
            selected_ids = list(range(min(limit, len(self.documents))))
            selected_scores = [0.0] * len(selected_ids)
        values = np.asarray(selected_scores, dtype=float)
        if values.size:
            low = float(values.min())
            high = float(values.max())
            values = (values - low) / (high - low) if high > low else np.ones_like(values)
        return np.asarray(selected_ids, dtype=int), values


def _score_statistics(scores: Sequence[float], prefix: str) -> Dict[str, float]:
    values = np.asarray(list(scores)[:20], dtype=float)
    if not values.size:
        return {
            f"{prefix}_top1": 0.0,
            f"{prefix}_margin12": 0.0,
            f"{prefix}_margin25": 0.0,
            f"{prefix}_entropy5": 0.0,
        }
    top1 = float(values[0])
    top2 = float(values[1]) if values.size > 1 else top1
    top5 = float(values[min(4, values.size - 1)])
    head = values[:5] - float(values[:5].max())
    probabilities = np.exp(head)
    probabilities /= max(float(probabilities.sum()), 1e-12)
    entropy = -float(np.sum(probabilities * np.log(probabilities + 1e-12)))
    return {
        f"{prefix}_top1": top1,
        f"{prefix}_margin12": top1 - top2,
        f"{prefix}_margin25": top2 - top5,
        f"{prefix}_entropy5": entropy,
    }


def _overlap(left: Sequence[int], right: Sequence[int], top_k: int) -> float:
    left_set = {int(value) for value in list(left)[:top_k]}
    right_set = {int(value) for value in list(right)[:top_k]}
    if not left_set:
        return 0.0
    return len(left_set & right_set) / len(left_set)


def build_router_features(
    *,
    query: str,
    query_type: str,
    fact_scores: Sequence[float],
    top_facts: Sequence[Sequence[Any]],
    rankings: Mapping[str, Sequence[int]],
    ranking_scores: Mapping[str, Sequence[float]],
) -> Dict[str, float]:
    """Return only execution-time features; dataset identity is excluded."""
    query_tokens = _tokens(query)
    fact_values = sorted((float(value) for value in fact_scores), reverse=True)
    unique_entities = {
        str(entity).strip().lower()
        for fact in top_facts
        if len(fact) >= 3
        for entity in (fact[0], fact[2])
        if str(entity).strip()
    }
    features: Dict[str, float] = {
        "query_token_count_log": math.log1p(len(query_tokens)),
        "query_wh_count": float(sum(token in {"who", "what", "where", "when", "which", "how"} for token in query_tokens)),
        "query_composition_count": float(sum(token in {"whose", "where", "which", "that", "between", "containing", "born", "died", "member", "director", "author", "spouse"} for token in query_tokens)),
        "query_comparative": float(any(token in {"first", "earlier", "later", "older", "younger", "same", "both", "before", "after"} for token in query_tokens)),
        "fact_count_log": math.log1p(len(top_facts)),
        "fact_entity_count_log": math.log1p(len(unique_entities)),
        "fact_top1": fact_values[0] if fact_values else 0.0,
        "fact_margin12": (fact_values[0] - fact_values[1]) if len(fact_values) > 1 else 0.0,
    }
    known_types = (
        "direct_slot_factoid",
        "extractive_factoid",
        "nested_bridge_factoid",
        "long_compositional_factoid",
        "comparative",
        "category_or_alias",
    )
    for value in known_types:
        features[f"query_type_{value}"] = float(query_type == value)
    for path in ACCESS_PATHS:
        features.update(_score_statistics(ranking_scores.get(path, ()), path))
    for left, right in (("ppr", "dense"), ("ppr", "sparse"), ("dense", "sparse")):
        for top_k in (2, 5, 20):
            features[f"overlap_{left}_{right}_{top_k}"] = _overlap(
                rankings.get(left, ()), rankings.get(right, ()), top_k
            )
    return features


def _sigmoid(value: float) -> float:
    if value >= 0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


class AdaptiveBaseAccessRouter:
    """Runtime evaluator for a JSON-serialized gain/risk access-path model."""

    def __init__(self, model_path: str = "") -> None:
        self.model_path = str(model_path or "")
        self.model: Dict[str, Any] = {}
        if self.model_path:
            path = Path(self.model_path)
            if path.is_file():
                self.model = json.loads(path.read_text(encoding="utf-8"))

    @property
    def ready(self) -> bool:
        return bool(
            self.model.get("schema_version") == "adaptive_base_access_router_v1"
            and self.model.get("feature_names")
            and self.model.get("paths")
        )

    @property
    def enabled_paths(self) -> tuple[str, ...]:
        configured = self.model.get("enabled_paths")
        if not isinstance(configured, list):
            configured = list((self.model.get("paths", {}) or {}).keys())
        return tuple(
            path for path in configured
            if path in ALTERNATIVES and path in (self.model.get("paths", {}) or {})
        )

    def _vector(self, features: Mapping[str, float]) -> np.ndarray:
        names = list(self.model.get("feature_names", []) or [])
        values = np.asarray([float(features.get(name, 0.0)) for name in names], dtype=float)
        scaler = self.model.get("scaler", {}) or {}
        mean = np.asarray(scaler.get("mean", [0.0] * len(names)), dtype=float)
        scale = np.asarray(scaler.get("scale", [1.0] * len(names)), dtype=float)
        scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)
        return (values - mean) / scale

    @staticmethod
    def _linear(head: Mapping[str, Any], vector: np.ndarray) -> float:
        if "constant" in head:
            return float(head.get("constant", 0.0) or 0.0)
        coefficients = np.asarray(head.get("coef", []), dtype=float)
        if coefficients.size != vector.size:
            return 0.0
        return float(np.dot(coefficients, vector) + float(head.get("intercept", 0.0) or 0.0))

    def decide(self, features: Mapping[str, float]) -> Dict[str, Any]:
        if not self.ready:
            return {
                "selected_path": "ppr",
                "reason": "model_unavailable",
                "model_ready": False,
                "predictions": {},
            }
        vector = self._vector(features)
        policy = self.model.get("policy", {}) or {}
        maximum_risk = float(policy.get("maximum_risk", 0.25) or 0.25)
        minimum_net_gain = float(policy.get("minimum_net_gain", 0.0) or 0.0)
        risk_weight = float(policy.get("risk_weight", 0.5) or 0.5)
        predictions: Dict[str, Dict[str, float]] = {
            "ppr": {"gain": 0.0, "risk": 0.0, "net_gain": 0.0}
        }
        selected = "ppr"
        selected_net = 0.0
        for path in self.enabled_paths:
            heads = (self.model.get("paths", {}) or {}).get(path, {}) or {}
            gain = self._linear(heads.get("gain", {}) or {}, vector)
            risk_logit = self._linear(heads.get("risk", {}) or {}, vector)
            risk = (
                float((heads.get("risk", {}) or {}).get("constant"))
                if "constant" in (heads.get("risk", {}) or {})
                else _sigmoid(risk_logit)
            )
            risk = min(max(risk, 0.0), 1.0)
            net = gain - risk_weight * risk
            predictions[path] = {
                "gain": round(gain, 8),
                "risk": round(risk, 8),
                "net_gain": round(net, 8),
            }
            if risk <= maximum_risk and net >= minimum_net_gain and net > selected_net:
                selected = path
                selected_net = net
        return {
            "selected_path": selected,
            "reason": "predicted_safe_gain" if selected != "ppr" else "ppr_safe_noop",
            "model_ready": True,
            "predictions": predictions,
            "maximum_risk": maximum_risk,
            "minimum_net_gain": minimum_net_gain,
            "risk_weight": risk_weight,
        }
