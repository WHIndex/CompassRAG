"""Typed evidence plans and cost-based physical access-path selection.

The module deliberately separates three database concerns that were
previously mixed into candidate-ranking heuristics:

* a logical evidence plan compiled from an ontology match;
* a train-split catalog of observed operator statistics; and
* a physical optimizer that chooses an access strategy or a no-op.

No gold labels are consumed at inference time. Gold outcomes may be used by
the offline catalog builder, in the same way validation workloads are used to
calibrate a conventional cost model.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from statistics import NormalDist
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


CATALOG_TYPE = "ontology_evidence_catalog"
CATALOG_VERSION = 1


def _clamp(value: Any, low: float = 0.0, high: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    return min(max(number, low), high)


def _unique(values: Iterable[Any]) -> Tuple[str, ...]:
    output: List[str] = []
    seen = set()
    for value in values or []:
        text = " ".join(str(value or "").strip().lower().split())
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return tuple(output)


def physical_feature_signature(
    logical_plan: Any,
    runtime_statistics: Optional[Mapping[str, Any]] = None,
) -> str:
    """Return the bounded query-level key shared by training and inference."""

    def field_value(name: str, default: Any) -> Any:
        if isinstance(logical_plan, Mapping):
            return logical_plan.get(name, default)
        return getattr(logical_plan, name, default)

    anchors = list(field_value("anchors", ()) or ())
    relations = list(field_value("relations", ()) or ())
    predicate_coverage = _clamp(field_value("predicate_coverage", 0.0))
    contract_confidence = _clamp(field_value("contract_confidence", 0.0))
    runtime = dict(runtime_statistics or {})
    entity_keys = max(float(runtime.get("entity_key_count", 0.0) or 0.0), 1.0)
    materialized_rows = max(
        float(runtime.get("materialized_row_count", 0.0) or 0.0), 0.0
    )
    rows_per_key = materialized_rows / entity_keys
    if rows_per_key < 2.0:
        selectivity_bucket = "s0"
    elif rows_per_key < 8.0:
        selectivity_bucket = "s1"
    elif rows_per_key < 32.0:
        selectivity_bucket = "s2"
    else:
        selectivity_bucket = "s3"
    return ":".join((
        f"a{min(len(anchors), 3)}",
        f"r{min(len(relations), 4)}",
        f"p{min(int(predicate_coverage * 4.0), 3)}",
        f"c{min(int(contract_confidence * 4.0), 3)}",
        selectivity_bucket,
    ))


@dataclass(frozen=True)
class LogicalEvidencePlan:
    plan_id: str
    source_schema_ids: Tuple[str, ...]
    dataset: str
    query_shape: str
    anchors: Tuple[str, ...]
    relations: Tuple[str, ...]
    answer_type: str
    operators: Tuple[Dict[str, Any], ...]
    required_branch_count: int
    required_hop_count: int
    predicate_coverage: float
    contract_confidence: float
    proof_obligations: Tuple[str, ...]
    admissible: bool
    rejection_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PhysicalEvidencePlan:
    plan_id: str
    operator: str
    access_strategy: str
    max_index_rows: int
    dense_top_k: int
    beam_width: int
    estimated_cardinality: float
    estimated_quality: float
    estimated_recall_delta: float
    estimated_recall_at2_delta: float
    estimated_recall_at5_delta: float
    recall_at2_lower_bound: float
    recall_at5_lower_bound: float
    estimated_proof_coverage: float
    estimated_cost: float
    estimated_loss_probability: float
    loss_probability_upper_bound: float
    objective: float
    catalog_match_level: str
    catalog_sample_count: int
    estimate_source: str
    admissible: bool
    rejection_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OntologyEvidenceQueryCompiler:
    """Compile an ontology match into a small typed evidence algebra."""

    @staticmethod
    def _shape(comparison: bool, relations: Sequence[str]) -> str:
        if comparison:
            return "comparison"
        if len(relations) >= 2:
            return "multi_hop"
        return "single_hop"

    def compile(
        self,
        query: str,
        dataset: str,
        anchors: Sequence[str],
        relations: Sequence[str],
        answer_type: str,
        comparison: bool,
        typed_contract: Optional[Mapping[str, Any]] = None,
        source_schema_ids: Sequence[str] = (),
    ) -> LogicalEvidencePlan:
        anchors_tuple = _unique(anchors)
        relations_tuple = _unique(relations)
        contract = dict(typed_contract or {})
        shape = self._shape(comparison, relations_tuple)
        required_branches = 2 if shape == "comparison" else 1
        predicate_coverage = _clamp(
            contract.get("query_predicate_coverage", 1.0 if relations_tuple else 0.0)
        )
        contract_confidence = _clamp(
            contract.get("contract_confidence", predicate_coverage)
        )
        branch_contract = dict(contract.get("comparison_branches", {}) or {})
        branch_specs = [
            dict(item) for item in branch_contract.get("branches", []) or []
            if isinstance(item, Mapping)
        ]
        realization_contract = dict(contract.get("relation_realization", {}) or {})
        realization_by_relation = dict(realization_contract.get("relations", {}) or {})

        def physical_predicates(relation: str) -> List[str]:
            values = [relation]
            for item in realization_by_relation.get(relation, []) or []:
                if isinstance(item, Mapping):
                    values.append(str(item.get("predicate", "") or ""))
            return list(_unique(values))[:13]

        operators: List[Dict[str, Any]] = []
        if shape == "comparison":
            for branch_index, anchor in enumerate(anchors_tuple[:2]):
                branch = branch_specs[branch_index] if branch_index < len(branch_specs) else {}
                operators.append({
                    "op": "Anchor",
                    "branch": branch_index,
                    "key": anchor,
                    "qualifiers": list(branch.get("qualifiers", []) or []),
                    "output": f"b{branch_index}_0",
                })
                for hop_index, relation in enumerate(relations_tuple):
                    operators.append({
                        "op": "Traverse",
                        "branch": branch_index,
                        "hop": hop_index,
                        "relation": relation,
                        "physical_predicates": physical_predicates(relation),
                        "input": f"b{branch_index}_{hop_index}",
                        "output": f"b{branch_index}_{hop_index + 1}",
                    })
            operators.append({
                "op": "BranchJoin",
                "inputs": [f"b0_{len(relations_tuple)}", f"b1_{len(relations_tuple)}"],
                "output": "joined_evidence",
            })
            current = "joined_evidence"
        else:
            operators.append({
                "op": "Anchor",
                "key": anchors_tuple[0] if anchors_tuple else "",
                "output": "b0",
            })
            current = "b0"
            for hop_index, relation in enumerate(relations_tuple):
                next_binding = f"b{hop_index + 1}"
                operators.append({
                    "op": "Traverse",
                    "hop": hop_index,
                    "relation": relation,
                    "physical_predicates": physical_predicates(relation),
                    "input": current,
                    "output": next_binding,
                })
                if hop_index:
                    operators.append({
                        "op": "BindJoin",
                        "join_key": current,
                        "input": next_binding,
                        "output": next_binding,
                    })
                current = next_binding

        operators.append({
            "op": "Verify",
            "input": current,
            "constraints": ["exact_binding", "source_provenance", "predicate_coverage"],
            "output": "verified_evidence",
        })
        operators.append({
            "op": "Project",
            "input": "verified_evidence",
            "answer_type": str(answer_type or "entity").strip().lower(),
            "output": "answer_evidence",
        })

        obligations = ["source_provenance", "predicate_coverage"]
        if len(relations_tuple) >= 2:
            obligations.append("join_connectivity")
        if shape == "comparison":
            obligations.extend(("branch_completeness", "branch_identity"))

        rejection_reason = ""
        if not anchors_tuple:
            rejection_reason = "logical_plan_missing_anchor"
        elif not relations_tuple:
            rejection_reason = "logical_plan_missing_relation"
        elif shape == "comparison" and len(anchors_tuple) < 2:
            rejection_reason = "logical_plan_incomplete_branches"
        elif contract and not bool(contract.get("admissible", True)):
            rejection_reason = str(
                contract.get("rejection_reason", "typed_contract_rejected")
                or "typed_contract_rejected"
            )

        fingerprint = json.dumps(
            {
                "dataset": dataset,
                "shape": shape,
                "anchors": anchors_tuple,
                "relations": relations_tuple,
                "answer_type": answer_type,
                "query": " ".join(str(query or "").lower().split()),
            },
            ensure_ascii=True,
            sort_keys=True,
        )
        plan_id = "lep_" + hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()[:16]
        return LogicalEvidencePlan(
            plan_id=plan_id,
            source_schema_ids=_unique(source_schema_ids),
            dataset=str(dataset or "").strip().lower(),
            query_shape=shape,
            anchors=anchors_tuple,
            relations=relations_tuple,
            answer_type=str(answer_type or "entity").strip().lower(),
            operators=tuple(operators),
            required_branch_count=required_branches,
            required_hop_count=len(relations_tuple),
            predicate_coverage=predicate_coverage,
            contract_confidence=contract_confidence,
            proof_obligations=tuple(obligations),
            admissible=not rejection_reason,
            rejection_reason=rejection_reason,
        )


class EvidenceCatalog:
    """Read-only catalog learned from isolated training traces."""

    def __init__(self, payload: Mapping[str, Any], source_path: str = "") -> None:
        if payload.get("type") != CATALOG_TYPE:
            raise ValueError(
                f"incompatible evidence catalog type: {payload.get('type')!r}"
            )
        if int(payload.get("version", 0) or 0) != CATALOG_VERSION:
            raise ValueError(
                f"unsupported evidence catalog version: {payload.get('version')!r}"
            )
        metadata = dict(payload.get("metadata", {}) or {})
        if metadata.get("source_split") != "train":
            raise ValueError("evidence catalog must be built from source_split=train")
        if metadata.get("inference_uses_gold") is not False:
            raise ValueError("evidence catalog violates the gold-free inference contract")
        self.metadata = metadata
        self.source_path = source_path
        self.prior_strength = max(float(payload.get("prior_strength", 20.0) or 20.0), 1.0)
        self._profiles: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
        self._feature_profiles: Dict[Tuple[str, str, str, str, str], Dict[str, Any]] = {}
        self._schema_profiles: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
        for raw_profile in payload.get("profiles", []) or []:
            if not isinstance(raw_profile, Mapping):
                continue
            profile = dict(raw_profile)
            key = (
                str(profile.get("dataset", "*") or "*").lower(),
                str(profile.get("operator", "") or "").lower(),
                str(profile.get("query_shape", "*") or "*").lower(),
                str(profile.get("access_strategy", "adaptive") or "adaptive").lower(),
            )
            feature_signature = str(profile.get("feature_signature", "") or "")
            if key[1] and feature_signature:
                self._feature_profiles[key + (feature_signature,)] = profile
            elif key[1]:
                self._profiles[key] = profile
        for raw_profile in payload.get("schema_profiles", []) or []:
            if not isinstance(raw_profile, Mapping):
                continue
            profile = dict(raw_profile)
            key = (
                str(profile.get("dataset", "*") or "*").lower(),
                str(profile.get("schema_id", "") or ""),
                str(profile.get("operator", "") or "").lower(),
                str(profile.get("access_strategy", "adaptive") or "adaptive").lower(),
            )
            if key[1] and key[2]:
                self._schema_profiles[key] = profile
        if not self._profiles and not self._feature_profiles:
            raise ValueError("evidence catalog contains no operator profiles")

    @classmethod
    def from_path(cls, path: str) -> "EvidenceCatalog":
        return _load_catalog(str(Path(path).expanduser().resolve()))

    def lookup(
        self,
        dataset: str,
        operator: str,
        query_shape: str,
        access_strategy: str,
        query_features: Optional[Mapping[str, Any]] = None,
        source_schema_ids: Sequence[str] = (),
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        dataset = str(dataset or "").lower()
        operator = str(operator or "").lower()
        query_shape = str(query_shape or "*").lower()
        strategy = str(access_strategy or "adaptive").lower()
        schema_candidates = []
        for schema_id in source_schema_ids or ():
            for schema_dataset in (dataset, "*"):
                profile = self._schema_profiles.get(
                    (schema_dataset, str(schema_id), operator, strategy)
                )
                if profile is not None:
                    schema_candidates.append(profile)
        if schema_candidates:
            profile = max(
                schema_candidates,
                key=lambda item: int(item.get("sample_count", 0) or 0),
            )
            return dict(profile), f"schema_observed:{profile.get('schema_id', '')}"
        feature_signature = str(
            (query_features or {}).get("feature_signature", "") or ""
        )
        if feature_signature:
            feature_keys = (
                ((dataset, operator, query_shape, strategy, feature_signature), "query_feature_exact"),
                (("*", operator, query_shape, strategy, feature_signature), "query_feature_operator_shape"),
            )
            for key, level in feature_keys:
                profile = self._feature_profiles.get(key)
                if profile is not None:
                    return dict(profile), level
        keys = (
            ((dataset, operator, query_shape, strategy), "dataset_operator_shape_strategy"),
            ((dataset, operator, "*", strategy), "dataset_operator_strategy"),
            (("*", operator, query_shape, strategy), "operator_shape_strategy"),
            (("*", operator, "*", strategy), "operator_strategy"),
            ((dataset, operator, query_shape, "adaptive"), "dataset_operator_shape"),
            ((dataset, operator, "*", "adaptive"), "dataset_operator"),
            (("*", operator, query_shape, "adaptive"), "operator_shape"),
            (("*", operator, "*", "adaptive"), "operator"),
        )
        for key, level in keys:
            profile = self._profiles.get(key)
            if profile is not None:
                return dict(profile), level
        return None, "missing"


@lru_cache(maxsize=16)
def _load_catalog(path: str) -> EvidenceCatalog:
    catalog_path = Path(path)
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    return EvidenceCatalog(payload, source_path=str(catalog_path))


class CostBasedEvidenceQueryOptimizer:
    """Enumerate semantically valid physical plans and choose by utility."""

    _STRATEGY_FACTORS = {
        "index_narrow": {
            "access_strategy": "index_only",
            "quality": 0.88,
            "recall": 0.82,
            "proof": 0.92,
            "cost": 0.42,
            "risk": 1.12,
        },
        "index_wide": {
            "access_strategy": "index_only",
            "quality": 0.94,
            "recall": 0.90,
            "proof": 0.96,
            "cost": 0.62,
            "risk": 1.06,
        },
        "adaptive_hybrid": {
            "access_strategy": "adaptive",
            "quality": 1.00,
            "recall": 1.00,
            "proof": 1.00,
            "cost": 1.00,
            "risk": 1.00,
        },
        "dense_first": {
            "access_strategy": "dense_first",
            "quality": 1.04,
            "recall": 1.10,
            "proof": 1.02,
            "cost": 1.45,
            "risk": 1.04,
        },
    }

    def __init__(
        self,
        catalog: EvidenceCatalog,
        quality_weight: float = 0.45,
        recall_weight: float = 0.25,
        proof_weight: float = 0.30,
        cost_weight: float = 0.15,
        risk_weight: float = 0.35,
        min_proof_coverage: float = 0.20,
        max_estimated_cost: float = 8.0,
        no_op_utility: float = 0.0,
        selection_mode: str = "weighted_utility",
        max_loss_probability: float = 0.25,
        confidence_delta: float = 0.05,
        min_catalog_samples: int = 5,
    ) -> None:
        self.catalog = catalog
        self.quality_weight = max(float(quality_weight), 0.0)
        self.recall_weight = max(float(recall_weight), 0.0)
        self.proof_weight = max(float(proof_weight), 0.0)
        self.cost_weight = max(float(cost_weight), 0.0)
        self.risk_weight = max(float(risk_weight), 0.0)
        self.min_proof_coverage = _clamp(min_proof_coverage)
        self.max_estimated_cost = max(float(max_estimated_cost), 0.0)
        self.no_op_utility = float(no_op_utility)
        normalized_mode = str(selection_mode or "weighted_utility").strip().lower()
        if normalized_mode not in {"weighted_utility", "constrained_lexicographic"}:
            raise ValueError(f"unsupported physical planner selection mode: {selection_mode}")
        self.selection_mode = normalized_mode
        self.max_loss_probability = _clamp(max_loss_probability)
        self.confidence_delta = min(max(float(confidence_delta), 1e-6), 0.50)
        self.min_catalog_samples = max(int(min_catalog_samples), 1)

    def _confidence_radius(self, profile: Mapping[str, Any], metric: str) -> float:
        samples = max(int(profile.get("sample_count", 0) or 0), 1)
        standard_error = max(
            float(profile.get(f"{metric}_standard_error", 0.0) or 0.0),
            0.0,
        )
        if standard_error > 0.0:
            z = NormalDist().inv_cdf(1.0 - self.confidence_delta / 2.0)
            return z * standard_error
        return math.sqrt(math.log(2.0 / self.confidence_delta) / (2.0 * samples))

    def _loss_upper_bound(self, losses: float, samples: int) -> float:
        if samples <= 0:
            return 1.0
        probability = _clamp(losses / samples)
        z = NormalDist().inv_cdf(1.0 - self.confidence_delta / 2.0)
        denominator = 1.0 + z * z / samples
        center = (probability + z * z / (2.0 * samples)) / denominator
        radius = z * math.sqrt(
            probability * (1.0 - probability) / samples
            + z * z / (4.0 * samples * samples)
        ) / denominator
        return _clamp(center + radius)

    @staticmethod
    def _operator_for_shape(shape: str) -> str:
        return {
            "comparison": "dual_branch_comparison_join",
            "multi_hop": "cascaded_bind_join",
            "single_hop": "entity_relation_index_scan",
        }.get(shape, "")

    @staticmethod
    def _budgets(
        strategy_name: str,
        max_index_rows: int,
        dense_top_k: int,
        beam_width: int,
    ) -> Tuple[int, int, int]:
        if strategy_name == "index_narrow":
            return min(max_index_rows, 24), min(dense_top_k, 16), min(beam_width, 2)
        if strategy_name == "index_wide":
            return max_index_rows, min(dense_top_k, 24), beam_width
        if strategy_name == "dense_first":
            return min(max_index_rows, 24), dense_top_k, beam_width
        return max_index_rows, dense_top_k, beam_width

    def enumerate(
        self,
        logical_plan: LogicalEvidencePlan,
        enabled_operators: Mapping[str, bool],
        runtime_statistics: Optional[Mapping[str, Any]] = None,
        max_index_rows: int = 64,
        dense_top_k: int = 48,
        beam_width: int = 4,
    ) -> List[PhysicalEvidencePlan]:
        operator = self._operator_for_shape(logical_plan.query_shape)
        if not logical_plan.admissible or not operator or not enabled_operators.get(operator, False):
            return []
        runtime = dict(runtime_statistics or {})
        entity_keys = max(float(runtime.get("entity_key_count", 0.0) or 0.0), 1.0)
        materialized_rows = max(float(runtime.get("materialized_row_count", 0.0) or 0.0), 0.0)
        rows_per_key = materialized_rows / entity_keys
        cardinality_factor = 1.0 + min(rows_per_key / max(max_index_rows, 1), 1.0) * 0.25
        query_features = {
            "feature_signature": physical_feature_signature(logical_plan, runtime),
        }

        plans: List[PhysicalEvidencePlan] = []
        strategy_items = list(self._STRATEGY_FACTORS.items())
        if self.selection_mode == "constrained_lexicographic":
            strategy_items = [
                ("index_wide", {"access_strategy": "index_only"}),
                ("adaptive_hybrid", {"access_strategy": "adaptive"}),
                ("dense_first", {"access_strategy": "dense_first"}),
            ]
        for strategy_name, factors in strategy_items:
            access_strategy = str(factors["access_strategy"])
            profile, match_level = self.catalog.lookup(
                logical_plan.dataset,
                operator,
                logical_plan.query_shape,
                access_strategy,
                query_features=query_features,
                source_schema_ids=logical_plan.source_schema_ids,
            )
            if profile is None:
                continue
            samples = max(int(profile.get("sample_count", 0) or 0), 0)
            if match_level.startswith("schema_observed:") and samples < self.min_catalog_samples:
                profile, match_level = self.catalog.lookup(
                    logical_plan.dataset,
                    operator,
                    logical_plan.query_shape,
                    access_strategy,
                    query_features=query_features,
                    source_schema_ids=(),
                )
                if profile is None:
                    continue
                samples = max(int(profile.get("sample_count", 0) or 0), 0)
            if (
                self.selection_mode == "constrained_lexicographic"
                and str(profile.get("access_strategy", "") or "") != access_strategy
            ):
                continue
            mean_r2 = float(profile.get("mean_delta_r2", 0.0) or 0.0)
            mean_r5 = float(profile.get("mean_delta_r5", 0.0) or 0.0)
            r2_lower = mean_r2 - self._confidence_radius(profile, "delta_r2")
            r5_lower = mean_r5 - self._confidence_radius(profile, "delta_r5")
            loss_count = float(profile.get("loss_count", 0.0) or 0.0)
            if not loss_count and samples:
                loss_count = float(profile.get("loss_rate", 0.0) or 0.0) * samples
            loss_upper = self._loss_upper_bound(loss_count, samples)
            uncertainty = self.catalog.prior_strength / (
                samples + self.catalog.prior_strength
            )
            if self.selection_mode == "constrained_lexicographic":
                precision = _clamp(profile.get("candidate_precision", 0.0))
                hit_rate = _clamp(profile.get("candidate_hit_rate", 0.0))
                quality = math.sqrt(precision * hit_rate)
                recall_delta = mean_r2
                estimated_cardinality = max(float(profile.get("candidate_yield", 0.0) or 0.0), 0.0)
                proof_coverage = min(
                    _clamp(profile.get("proof_coverage", 0.0)),
                    logical_plan.predicate_coverage,
                )
                estimated_cost = max(float(profile.get("mean_cost", 0.0) or 0.0), 0.0)
                loss_probability = _clamp(profile.get("loss_rate", 0.0))
                objective = r2_lower
            else:
                quality = _clamp(
                    (
                        0.62 * float(profile.get("candidate_precision", 0.0) or 0.0)
                        + 0.38 * float(profile.get("candidate_hit_rate", 0.0) or 0.0)
                    )
                    * float(factors["quality"])
                    * (1.0 - 0.20 * uncertainty)
                )
                recall_delta = (0.60 * mean_r2 + 0.40 * mean_r5) * float(factors["recall"])
                estimated_cardinality = max(
                    float(profile.get("candidate_yield", 0.0) or 0.0)
                    * float(factors["recall"]),
                    0.0,
                )
                proof_coverage = _clamp(
                    float(profile.get("proof_coverage", 0.0) or 0.0)
                    * float(factors["proof"])
                    * (0.75 + 0.25 * logical_plan.predicate_coverage)
                )
                estimated_cost = max(
                    float(profile.get("mean_cost", 0.0) or 0.0)
                    * float(factors["cost"])
                    * cardinality_factor,
                    0.0,
                )
                loss_probability = _clamp(
                    float(profile.get("loss_rate", 0.0) or 0.0)
                    * float(factors["risk"])
                    + 0.10 * uncertainty
                )
                normalized_cost = math.log1p(estimated_cost) / math.log(1.0 + 8.0)
                objective = (
                    self.quality_weight * quality
                    + self.recall_weight * recall_delta
                    + self.proof_weight * proof_coverage
                    - self.cost_weight * normalized_cost
                    - self.risk_weight * loss_probability
                )
            rejection_reason = ""
            if samples < self.min_catalog_samples:
                rejection_reason = "catalog_support_below_constraint"
            elif proof_coverage < self.min_proof_coverage:
                rejection_reason = "estimated_proof_coverage_below_constraint"
            elif estimated_cost > self.max_estimated_cost:
                rejection_reason = "estimated_cost_above_budget"
            elif (
                self.selection_mode == "constrained_lexicographic"
                and loss_upper > self.max_loss_probability
            ):
                rejection_reason = "loss_probability_above_constraint"
            rows, dense_k, beam = self._budgets(
                strategy_name,
                max(int(max_index_rows), 1),
                max(int(dense_top_k), 1),
                max(int(beam_width), 1),
            )
            estimate_source = (
                "observed"
                if str(profile.get("access_strategy", "adaptive")) == access_strategy
                else "catalog_backoff_with_strategy_adjustment"
            )
            plans.append(PhysicalEvidencePlan(
                plan_id=f"{logical_plan.plan_id}:{strategy_name}",
                operator=operator,
                access_strategy=access_strategy,
                max_index_rows=rows,
                dense_top_k=dense_k,
                beam_width=beam,
                estimated_cardinality=round(estimated_cardinality, 6),
                estimated_quality=round(quality, 6),
                estimated_recall_delta=round(recall_delta, 6),
                estimated_recall_at2_delta=round(mean_r2, 6),
                estimated_recall_at5_delta=round(mean_r5, 6),
                recall_at2_lower_bound=round(r2_lower, 6),
                recall_at5_lower_bound=round(r5_lower, 6),
                estimated_proof_coverage=round(proof_coverage, 6),
                estimated_cost=round(estimated_cost, 6),
                estimated_loss_probability=round(loss_probability, 6),
                loss_probability_upper_bound=round(loss_upper, 6),
                objective=round(objective, 6),
                catalog_match_level=match_level,
                catalog_sample_count=samples,
                estimate_source=estimate_source,
                admissible=not rejection_reason,
                rejection_reason=rejection_reason,
            ))
        if self.selection_mode == "constrained_lexicographic":
            return sorted(
                plans,
                key=lambda item: (
                    item.recall_at2_lower_bound,
                    item.recall_at5_lower_bound,
                    item.estimated_quality,
                    item.estimated_proof_coverage,
                    -item.estimated_cost,
                ),
                reverse=True,
            )
        return sorted(plans, key=lambda item: item.objective, reverse=True)

    def optimize(
        self,
        logical_plan: LogicalEvidencePlan,
        enabled_operators: Mapping[str, bool],
        runtime_statistics: Optional[Mapping[str, Any]] = None,
        max_index_rows: int = 64,
        dense_top_k: int = 48,
        beam_width: int = 4,
    ) -> Dict[str, Any]:
        alternatives = self.enumerate(
            logical_plan=logical_plan,
            enabled_operators=enabled_operators,
            runtime_statistics=runtime_statistics,
            max_index_rows=max_index_rows,
            dense_top_k=dense_top_k,
            beam_width=beam_width,
        )
        admissible = [item for item in alternatives if item.admissible]
        selected = admissible[0] if admissible else None
        reason = "selected"
        if selected is None:
            reason = "no_admissible_physical_plan"
        elif (
            self.selection_mode == "constrained_lexicographic"
            and selected.estimated_recall_at2_delta <= 0.0
            and selected.estimated_recall_at5_delta <= 0.0
        ):
            selected = None
            reason = "incumbent_dominates_expected_recall"
        elif self.selection_mode == "weighted_utility" and selected.objective <= self.no_op_utility:
            selected = None
            reason = "no_op_dominates"
        return {
            "enabled": True,
            "catalog_path": self.catalog.source_path,
            "catalog_metadata": dict(self.catalog.metadata),
            "logical_plan_id": logical_plan.plan_id,
            "selected": selected.to_dict() if selected else {},
            "selected_operator": selected.operator if selected else "",
            "selected_access_strategy": selected.access_strategy if selected else "",
            "decision_reason": reason,
            "selection_mode": self.selection_mode,
            "decision_rule": (
                "maximize_recall_at2_lcb_then_recall_at5_lcb_subject_to_proof_cost_loss_constraints"
                if self.selection_mode == "constrained_lexicographic"
                else "weighted_utility"
            ),
            "constraints": {
                "min_proof_coverage": self.min_proof_coverage,
                "max_estimated_cost": self.max_estimated_cost,
                "max_loss_probability_upper_bound": self.max_loss_probability,
                "min_catalog_samples": self.min_catalog_samples,
                "confidence_delta": self.confidence_delta,
            },
            "no_op_utility": self.no_op_utility,
            "alternative_count": len(alternatives),
            "admissible_alternative_count": len(admissible),
            "alternatives": [item.to_dict() for item in alternatives],
        }
