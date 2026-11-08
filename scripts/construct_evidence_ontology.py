#!/usr/bin/env python
import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import sqlite3
import sys
import threading
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
COMPASSRAG_SRC = ROOT / "src" / "compassrag"
for value in (str(SRC), str(COMPASSRAG_SRC)):
    if value not in sys.path:
        sys.path.insert(0, value)

from compassrag.remote import RemoteEndpoint
from query_ontology_prior import extract_json_object, normalize_text, tokenize
from query_plan_ontology import relation_signature


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def retryable_api_error(exc: BaseException) -> bool:
    """Return whether an API failure is safe to retry without changing input."""
    error_name = type(exc).__name__
    if error_name in {
        "APIConnectionError",
        "APITimeoutError",
        "ConnectError",
        "ConnectTimeout",
        "InternalServerError",
        "RateLimitError",
        "ReadError",
        "ReadTimeout",
        "RemoteProtocolError",
    }:
        return True
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    return status_code in {408, 409, 429} or (
        isinstance(status_code, int) and status_code >= 500
    )


class CachedChatClient:
    def __init__(
        self,
        model: str,
        base_url: str,
        cache_path: Path,
        temperature: float,
        max_tokens: int,
        cache_only: bool = False,
        max_attempts: int | None = None,
        initial_backoff_seconds: float | None = None,
        max_backoff_seconds: float | None = None,
        timeout_seconds: float | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache_only = bool(cache_only)
        self.max_attempts = max(
            int(max_attempts or os.environ.get("MULTIAGENT_API_MAX_ATTEMPTS", "8")),
            1,
        )
        self.initial_backoff_seconds = max(
            float(
                initial_backoff_seconds
                if initial_backoff_seconds is not None
                else os.environ.get("MULTIAGENT_API_BACKOFF_SECONDS", "2")
            ),
            0.0,
        )
        self.max_backoff_seconds = max(
            float(
                max_backoff_seconds
                if max_backoff_seconds is not None
                else os.environ.get("MULTIAGENT_API_MAX_BACKOFF_SECONDS", "60")
            ),
            self.initial_backoff_seconds,
        )
        self.timeout_seconds = max(
            float(
                timeout_seconds
                if timeout_seconds is not None
                else os.environ.get("MULTIAGENT_API_TIMEOUT_SECONDS", "120")
            ),
            1.0,
        )
        self.cache_path = cache_path
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.client = RemoteEndpoint(
            base_url=base_url,
            api_key=os.environ.get("MODEL_API_KEY", ""),
            timeout=self.timeout_seconds,
        )
        self._stats_lock = threading.Lock()
        with sqlite3.connect(self.cache_path, timeout=30.0) as conn:
            conn.execute("PRAGMA busy_timeout = 30000")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, response TEXT, metadata TEXT)"
            )
            conn.commit()
        self.calls = 0
        self.cache_hits = 0
        self.network_attempts = 0
        self.api_retries = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def ask_json(self, messages: List[Dict[str, str]], agent_name: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        key_payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "agent": agent_name,
            "messages": messages,
        }
        key = hashlib.sha256(stable_json(key_payload).encode("utf-8")).hexdigest()
        with sqlite3.connect(self.cache_path, timeout=30.0) as conn:
            conn.execute("PRAGMA busy_timeout = 30000")
            row = conn.execute("SELECT response, metadata FROM cache WHERE key = ?", (key,)).fetchone()
        if row:
            with self._stats_lock:
                self.cache_hits += 1
            return json.loads(row[0]), {**json.loads(row[1]), "cache_hit": True}

        if self.cache_only:
            raise RuntimeError(
                f"multi-agent cache miss in --cache-only mode: agent={agent_name} key={key}"
            )

        with self._stats_lock:
            self.calls += 1
        response = None
        for attempt in range(1, self.max_attempts + 1):
            with self._stats_lock:
                self.network_attempts += 1
            try:
                response = self.client.chat(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                break
            except Exception as exc:
                if not retryable_api_error(exc) or attempt >= self.max_attempts:
                    raise
                delay = min(
                    self.initial_backoff_seconds * (2 ** (attempt - 1)),
                    self.max_backoff_seconds,
                )
                with self._stats_lock:
                    self.api_retries += 1
                print(
                    f"WARNING: {agent_name} API attempt {attempt}/{self.max_attempts} "
                    f"failed with {type(exc).__name__}: {exc}; retrying in {delay:.1f}s",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)
        if response is None:
            raise RuntimeError(f"{agent_name} API request produced no response")
        choice = response["choices"][0]
        content = choice.get("message", {}).get("content", "{}") or "{}"
        parsed = extract_json_object(content)
        usage = response.get("usage", {}) or {}
        metadata = {
            "cache_hit": False,
            "finish_reason": choice.get("finish_reason"),
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "raw_response": content,
        }
        with self._stats_lock:
            self.prompt_tokens += int(metadata["prompt_tokens"] or 0)
            self.completion_tokens += int(metadata["completion_tokens"] or 0)
        with sqlite3.connect(self.cache_path, timeout=30.0) as conn:
            conn.execute("PRAGMA busy_timeout = 30000")
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, response, metadata) VALUES (?, ?, ?)",
                (key, json.dumps(parsed, ensure_ascii=False), json.dumps(metadata, ensure_ascii=False)),
            )
            conn.commit()
        return parsed, metadata


def compact_plan(plan: Dict[str, Any], max_examples: int) -> Dict[str, Any]:
    return {
        "id": plan.get("id", ""),
        "intent": plan.get("intent", ""),
        "question_pattern": plan.get("question_pattern", ""),
        "triggers": plan.get("triggers", [])[:20],
        "answer_type": plan.get("answer_type", ""),
        "support": plan.get("support", 0),
        "confidence": plan.get("confidence", 0.0),
        "precision": plan.get("precision", 0.0),
        "path": plan.get("path", [])[:4],
        "answer_constraints": plan.get("answer_constraints", {}),
        "examples": plan.get("examples", [])[:max_examples],
    }


def evidence_path_shape(example: Dict[str, Any]) -> Tuple[str, ...]:
    """Return the executable relation shape carried by one induction example."""
    return tuple(
        signature
        for triple in example.get("evidence_path", []) or []
        if isinstance(triple, (list, tuple)) and len(triple) >= 3
        for signature in [relation_signature(str(triple[1]))]
        if signature
    )


def split_plan_by_path_shape(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Split heterogeneous schemas before agents attempt a single repair.

    The split uses only induction evidence and is independent of dataset names.
    It prevents a repair for one relation chain from being evaluated against
    examples belonging to a different chain.
    """
    groups: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}
    unshaped: List[Dict[str, Any]] = []
    for example in plan.get("examples", []) or []:
        if not isinstance(example, dict):
            continue
        shape = evidence_path_shape(example)
        if shape:
            groups.setdefault(shape, []).append(example)
        else:
            unshaped.append(example)
    if len(groups) <= 1:
        return [plan]

    children = []
    for shape, examples in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        child = deepcopy(plan)
        shape_id = hashlib.sha256("|".join(shape).encode("utf-8")).hexdigest()[:10]
        child["id"] = f"{plan.get('id', 'plan')}__shape_{shape_id}"
        child["parent_plan_id"] = str(plan.get("id", "") or "")
        child["parent_support"] = int(plan.get("support", 0) or 0)
        child["examples"] = deepcopy(examples)
        child["support"] = len(examples)
        source_path = [step for step in plan.get("path", []) or [] if isinstance(step, dict)]
        rebuilt_path = []
        representative = examples[0].get("evidence_path", []) or []
        for step_idx, signature in enumerate(shape):
            source_step = source_path[step_idx] if step_idx < len(source_path) else {}
            source_signature = relation_signature(
                str(source_step.get("relation_signature") or source_step.get("relation_family") or "")
            )
            raw_relation = (
                str(representative[step_idx][1])
                if step_idx < len(representative)
                and isinstance(representative[step_idx], (list, tuple))
                and len(representative[step_idx]) >= 3
                else signature.replace("_", " ")
            )
            if source_signature == signature:
                step = deepcopy(source_step)
            else:
                step = {
                    "relation_family": signature,
                    "relation_signature": signature,
                    "relation_aliases": [raw_relation],
                }
            step.update({
                "step": step_idx,
                "from": "anchor" if step_idx == 0 else f"node_{step_idx}",
                "to": "answer" if step_idx == len(shape) - 1 else f"node_{step_idx + 1}",
            })
            rebuilt_path.append(step)
        child["path"] = rebuilt_path
        child["path_shape_split"] = {
            "parent_plan_id": str(plan.get("id", "") or ""),
            "relation_signatures": list(shape),
            "unshaped_parent_example_count": len(unshaped),
        }
        children.append(child)
    if unshaped:
        child = deepcopy(plan)
        child["id"] = f"{plan.get('id', 'plan')}__shape_unresolved"
        child["parent_plan_id"] = str(plan.get("id", "") or "")
        child["parent_support"] = int(plan.get("support", 0) or 0)
        child["examples"] = deepcopy(unshaped)
        child["support"] = len(unshaped)
        child["path_shape_split"] = {
            "parent_plan_id": str(plan.get("id", "") or ""),
            "relation_signatures": [],
            "unresolved_shape": True,
        }
        children.append(child)
    return children


def split_plan_by_cq_cluster(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Assign each schema proposal to one local CQ-specialist team."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    unresolved: List[Dict[str, Any]] = []
    for example in plan.get("examples", []) or []:
        if not isinstance(example, dict):
            continue
        cluster_id = str(
            (example.get("contrastive_cq", {}) or {}).get("cluster_id", "")
        )
        if cluster_id:
            grouped.setdefault(cluster_id, []).append(example)
        else:
            unresolved.append(example)
    if len(grouped) <= 1:
        local = deepcopy(plan)
        local["local_cq_cluster_id"] = next(iter(grouped), "unresolved")
        local["local_team_contract"] = "entity_relation_bridge_verify_critic_repair"
        return [local]
    children: List[Dict[str, Any]] = []
    for cluster_id, examples in sorted(grouped.items()):
        child = deepcopy(plan)
        child["id"] = f"{plan.get('id', 'plan')}__cluster_{cluster_id}"
        child["parent_plan_id"] = str(plan.get("id", "") or "")
        child["parent_support"] = int(plan.get("support", 0) or 0)
        child["examples"] = deepcopy(examples)
        child["support"] = len(examples)
        child["local_cq_cluster_id"] = cluster_id
        child["local_team_contract"] = "entity_relation_bridge_verify_critic_repair"
        children.append(child)
    if unresolved:
        child = deepcopy(plan)
        child["id"] = f"{plan.get('id', 'plan')}__cluster_unresolved"
        child["parent_plan_id"] = str(plan.get("id", "") or "")
        child["parent_support"] = int(plan.get("support", 0) or 0)
        child["examples"] = deepcopy(unresolved)
        child["support"] = len(unresolved)
        child["local_cq_cluster_id"] = "unresolved"
        child["local_team_contract"] = "entity_relation_bridge_verify_critic_repair"
        children.append(child)
    return children


def schema_fingerprint(plan: Dict[str, Any]) -> str:
    payload = {
        "intent": normalize_text(plan.get("intent", "")),
        "answer_type": normalize_text(plan.get("answer_type", "")),
        "path": [
            {
                "relation": relation_signature(
                    str(step.get("relation_signature") or step.get("relation_family") or "")
                ),
                "direction": str(step.get("direction", "either") or "either"),
                "from": str(step.get("from", "") or ""),
                "to": str(step.get("to", "") or ""),
            }
            for step in plan.get("path", []) or []
            if isinstance(step, dict)
        ],
    }
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()[:20]


def retain_parent_incumbents(
    parent_plans: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Keep one Q_ind-derived incumbent per memo group.

    Local specialist teams propose narrower alternatives.  Their rejection of a
    proposal must not delete the broader schema that was deterministically
    induced from Q_ind; only the held-out Q_val deployment mask may do that.
    """
    incumbents: List[Dict[str, Any]] = []
    for parent in parent_plans:
        incumbent = deepcopy(parent)
        plan_id = str(incumbent.get("id", "") or "")
        contrastive_report = deterministic_contrastive_report(incumbent)
        tested = int(
            contrastive_report.get("testable_hard_negative_count", 0) or 0
        )
        rejection_rate = float(
            contrastive_report.get("hard_negative_rejection_rate", 0.0) or 0.0
        )
        path = [
            step for step in incumbent.get("path", []) or []
            if isinstance(step, dict)
        ]
        signed_path = bool(path) and all(
            normalize_text(step.get("direction", "")) in {"forward", "reverse"}
            and relation_signature(str(
                step.get("relation_signature")
                or step.get("relation_family")
                or ""
            )) not in {"", "attribute", "is", "was", "has"}
            for step in path
        )
        execution_eligible = bool(
            signed_path and tested >= 1 and rejection_rate >= 0.75
        )
        incumbent["builder_stage"] = "global_supervisor_parent_incumbent"
        incumbent["memo_role"] = "q_ind_parent_incumbent"
        incumbent["execution_eligible"] = execution_eligible
        incumbent["execution_eligibility_basis"] = (
            "signed_contract_and_q_ind_contrastive_gate"
            if execution_eligible
            else "pending_multiagent_or_q_ind_contrastive_repair"
        )
        quality = dict(incumbent.get("multiagent_quality", {}) or {})
        quality.update({
            "hard_negative_test_count": tested,
            "hard_negative_rejection_rate": rejection_rate,
        })
        incumbent["multiagent_quality"] = quality
        incumbent["contrastive_verification"] = contrastive_report
        incumbent["global_supervisor_retention"] = {
            "operator": "RETAIN_INCUMBENT",
            "source_schema_id": plan_id,
            "decision_basis": "preserve_q_ind_coverage_until_q_val_certification",
            "deployment_authority": "fixed_q_val_schema_mask",
        }
        incumbents.append(incumbent)
    return incumbents


def hierarchical_aggregate(
    plans: List[Dict[str, Any]], audits: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Merge local proposals deterministically and retain supervisor memory."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for plan in plans:
        if plan.get("memo_role") == "q_ind_parent_incumbent":



            group_key = f"incumbent:{plan.get('id', '')}"
        else:
            group_key = f"proposal:{schema_fingerprint(plan)}"
        grouped.setdefault(group_key, []).append(plan)
    accepted: List[Dict[str, Any]] = []
    duplicate_decisions = []
    for group_key, members in sorted(grouped.items()):
        fingerprint = schema_fingerprint(members[0])
        members.sort(
            key=lambda value: (
                float(value.get("precision", 0.0) or 0.0),
                float(value.get("confidence", 0.0) or 0.0),
                int(value.get("support", 0) or 0),
            ),
            reverse=True,
        )
        winner = deepcopy(members[0])
        merged_examples = []
        seen_examples = set()
        for member in members:
            for example in member.get("examples", []) or []:
                example_id = str(example.get("id", "") or "")
                if example_id and example_id in seen_examples:
                    continue
                if example_id:
                    seen_examples.add(example_id)
                merged_examples.append(deepcopy(example))
        winner["examples"] = merged_examples
        winner["support"] = max(
            [int(item.get("parent_support", 0) or 0) for item in members]
            + [sum(int(item.get("support", 0) or 0) for item in members)]
        )
        winner_path = list(winner.get("path", []) or [])
        for step_idx, step in enumerate(winner_path):
            peer_steps = [
                item.get("path", [])[step_idx]
                for item in members
                if step_idx < len(item.get("path", []) or [])
            ]
            step["relation_family_alternatives"] = merge_string_lists(
                *[
                    [peer.get("relation_family", "")]
                    + list(peer.get("relation_family_alternatives", []) or [])
                    for peer in peer_steps
                ],
                limit=8,
            )
            step["relation_signature_alternatives"] = merge_string_lists(
                *[
                    [peer.get("relation_signature", "")]
                    + list(peer.get("relation_signature_alternatives", []) or [])
                    for peer in peer_steps
                ],
                limit=12,
            )
            step["relation_aliases"] = merge_string_lists(
                *[peer.get("relation_aliases", []) or [] for peer in peer_steps],
                limit=16,
            )
        winner["path"] = winner_path
        source_clusters = sorted({
            str(item.get("local_cq_cluster_id", "") or "")
            for item in members
            if item.get("local_cq_cluster_id") is not None
        })
        source_schema_ids = [str(item.get("id", "") or "") for item in members]
        winner["hierarchical_aggregation"] = {
            "memo_group_key": group_key,
            "schema_fingerprint": fingerprint,
            "operator": "MERGE_SCHEMA" if len(members) > 1 else "ADD_SCHEMA",
            "source_schema_ids": source_schema_ids,
            "source_cq_clusters": source_clusters,
            "duplicate_count": max(len(members) - 1, 0),
            "decision": "accepted_by_global_supervisor",
        }
        accepted.append(winner)
        if len(members) > 1:
            duplicate_decisions.append({
                "fingerprint": fingerprint,
                "winner": str(winner.get("id", "") or ""),
                "merged": source_schema_ids[1:],
                "reason": "semantic_and_executable_schema_duplicate",
            })
    memory = {
        "accepted_proposals": [str(item.get("id", "") or "") for item in accepted],
        "rejected_proposals": [
            str(item.get("plan_id", "") or "")
            for item in audits
            if item.get("decision") != "keep"
        ],
        "recurring_conflicts": sorted({
            str(violation.get("code", "") or "")
            for audit in audits
            for round_payload in audit.get("rounds", []) or []
            for violation in (
                (round_payload.get("typed_post_repair_critic", {}) or {}).get(
                    "blocking_violations", []
                ) or []
            )
            if isinstance(violation, dict) and violation.get("code")
        }),
        "merged_duplicates": duplicate_decisions,
        "proposal_count": len(plans),
        "accepted_schema_count": len(accepted),
        "aggregation": "parent_fingerprint_then_global_supervisor",
    }
    return accepted, memory


SYSTEM_PROMPT = """You are a careful ontology-construction agent for retrieval-augmented QA.
Return one valid JSON object only. Do not include markdown.
The ontology must be useful for retrieval: it should identify reusable query intents and graph path schemas that help find supporting evidence, not just restate examples.
You are a local specialist working on one CQ cluster. Contrastive CQ records
contain corpus-retrieved hard negatives. Preserve constraints that separate the
ground-truth proof from those near misses; never treat a hard negative as positive
support and never invent a negative that is absent from the input.
Distinguish reproducible blocking violations from non-blocking uncertainty. Do not
reject a schema merely because a hypothetical risk can be imagined.
Pay special attention to generic OpenIE relations such as "is", "was", "is a", and "was a": these must have answer/object constraints before they can be useful."""


def agent_messages(agent_name: str, task: str, payload: Dict[str, Any]) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Agent: {agent_name}\n"
                f"Task:\n{task}\n\n"
                f"Input JSON:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
            ),
        },
    ]


ENTITY_TASK = """Analyze the candidate query-plan ontology.
Identify:
1. query anchor slots,
2. answer slot type,
3. intermediate entity slot types,
4. query terms that should be required,
5. query terms that would make this plan unsafe,
6. answer-object constraints that describe what the terminal answer entity/value should look like.
Return JSON:
{
  "anchor_slots": [{"name": "...", "type": "...", "source": "query_mention", "constraints": ["..."]}],
  "intermediate_slots": [{"name": "node_1", "type": "...", "constraints": ["..."]}],
  "answer_type": "...",
  "answer_constraints": {
    "allowed_terms": ["terms or short phrases expected in answer/terminal objects"],
    "reject_terms": ["terms or short phrases that indicate a wrong terminal object"],
    "object_type_rules": ["natural-language constraints for the answer object"]
  },
  "required_query_terms": ["..."],
  "unsafe_query_terms": ["..."],
  "quality_score": 0.0-1.0,
  "issues": ["..."]
}"""

RELATION_TASK = """Review the relation/path schema.
Normalize relation semantics for each step. Detect if signatures are too broad, too narrow, or wrong.
For each terminal step, decide whether generic relation surfaces need an answer-object gate.
Return JSON:
{
  "steps": [
    {
      "step": 0,
      "relation_family": "...",
      "relation_signature": "...",
      "relation_aliases": ["..."],
      "required_relation_terms": ["..."],
      "reject_relation_terms": ["..."],
      "generic_relation_policy": "allow|allow_only_with_answer_object_gate|reject_generic",
      "terminal_object_allow_terms": ["..."],
      "terminal_object_reject_terms": ["..."],
      "direction": "either|forward|reverse",
      "issue": ""
    }
  ],
  "path_quality_score": 0.0-1.0,
  "issues": ["..."]
}"""

BRIDGE_TASK = """Construct the best executable graph path schema from the agent analyses.
Keep the schema general enough to transfer to new queries, but specific enough to avoid noisy paths.
For every proposed path step, cite at least one short exact span from any supplied
supporting_passage that grounds that step. Different steps may be grounded by
different examples because multi-hop evidence is distributed across documents.
Never invent or paraphrase a citation.
Return JSON:
{
  "path": [
    {
      "step": 0,
      "from": "anchor",
      "to": "answer|node_1",
      "relation_family": "...",
      "relation_signature": "...",
      "relation_aliases": ["..."],
      "required_relation_terms": ["..."],
      "reject_relation_terms": ["..."],
      "generic_relation_policy": "allow|allow_only_with_answer_object_gate|reject_generic",
      "terminal_object_allow_terms": ["..."],
      "terminal_object_reject_terms": ["..."],
      "direction": "either|forward|reverse"
    }
  ],
  "answer_constraints": {
    "allowed_terms": ["..."],
    "reject_terms": ["..."],
    "object_type_rules": ["..."]
  },
  "triggers": ["..."],
  "required_query_terms": ["..."],
  "unsafe_query_terms": ["..."],
  "answer_type": "...",
  "grounding_citations": [
    {"example_id": "...", "title": "...", "quote": "exact passage span", "supports_steps": [0]}
  ],
  "confidence": 0.0-1.0,
  "precision": 0.0-1.0,
  "rationale": "..."
}"""

VERIFIER_TASK = """Verify whether the proposed ontology plan is supported by the training examples.
The input includes a deterministic grounding_report produced by executing the
proposed relation path against the induction evidence paths. Treat this report
as authoritative; do not claim support for examples that it marks unsupported.
Use revise or drop only when a deterministic blocking violation or a concrete
failed example justifies it. If all deterministic blocking violations are resolved
and the estimates satisfy the requested thresholds, return keep; generic uncertainty
belongs in failure_modes and cannot by itself reject the plan.
Check whether terminal answer-object constraints are strong enough for generic relations; require revision if a broad relation can retrieve many wrong terminal objects.
Also inspect the supplied contrastive CQs and real-corpus hard negatives. A plan
that covers positives but accepts wrong bridges, incomplete branches, wrong answer
types, or structurally plausible near misses is over-generalized.
Return JSON:
{
  "decision": "keep|revise|drop",
  "precision_estimate": 0.0-1.0,
  "coverage_estimate": 0.0-1.0,
  "hard_negative_rejection_rate": 0.0-1.0,
  "direction_correctness": 0.0-1.0,
  "binding_correctness": 0.0-1.0,
  "branch_completeness": 0.0-1.0,
  "object_gate_quality": 0.0-1.0,
  "supported_example_ids": ["..."],
  "failed_example_ids": ["..."],
  "failure_modes": ["..."],
  "required_fixes": ["..."],
  "rationale": "..."
}"""

CRITIC_TASK = """Audit the proposed ontology plan against concrete induction
counterexamples. Do not veto a plan because of a hypothetical or generic concern.
Every blocking violation must name one of the allowed codes below, cite affected
example IDs from the input, identify the path step when applicable, and be
reproducible from the deterministic grounding_report. Concerns without such
evidence are warnings, not blockers.
Allowed blocking codes: empty_path, hallucinated_relation_signature,
insufficient_grounding_coverage, missing_terminal_object_gate, answer_leakage.
Return JSON:
{
  "risk_level": "low|medium|high",
  "violations": [
    {
      "code": "allowed code",
      "severity": "blocking|warning",
      "evidence_example_ids": ["..."],
      "step": 0,
      "deterministically_checkable": true|false,
      "evidence": "specific observed failure",
      "repair": "smallest repair"
    }
  ],
  "warnings": ["non-blocking concern"],
  "suggested_precision": 0.0-1.0,
  "suggested_confidence": 0.0-1.0
}"""

REPAIR_TASK = """Repair the proposed ontology plan using the supplied typed
blocking violations and their concrete counterexamples. Make the smallest structural
change that resolves a blocking code. A repair may correct relation signatures or
direction, remove unsafe aliases, strengthen anchor/answer constraints, or restore
a missing evidence step. Do not invent a relation, entity type, citation, or answer.
Preserve valid verbatim citations and add a citation only when its exact text occurs
in the supplied supporting passages. Separate remaining blocking violations from
non-blocking warnings; never put a general caveat in remaining_blocking_violation_codes.
Return JSON:
{
  "decision": "keep|drop",
  "repair_operations": [
    {"operation": "replace|remove|add|tighten|no_op", "target": "...", "reason": "..."}
  ],
  "resolved_violation_codes": ["..."],
  "remaining_blocking_violation_codes": ["..."],
  "warnings": ["..."],
  "plan_patch": {
    "intent": "...",
    "question_pattern": "...",
    "triggers": ["..."],
    "anchor_slots": [{"name": "anchor", "type": "entity", "source": "query_mention", "constraints": ["..."]}],
    "path": [
      {
        "step": 0,
        "from": "anchor",
        "to": "answer|node_1",
        "relation_family": "...",
        "relation_signature": "...",
        "relation_aliases": ["..."],
        "required_relation_terms": ["..."],
        "reject_relation_terms": ["..."],
        "generic_relation_policy": "allow|allow_only_with_answer_object_gate|reject_generic",
        "terminal_object_allow_terms": ["..."],
        "terminal_object_reject_terms": ["..."],
        "direction": "either|forward|reverse"
      }
    ],
    "answer_type": "...",
    "answer_constraints": {
      "allowed_terms": ["..."],
      "reject_terms": ["..."],
      "object_type_rules": ["..."]
    },
    "grounding_citations": [
      {"example_id": "...", "title": "...", "quote": "exact passage span", "supports_steps": [0]}
    ],
    "confidence": 0.0-1.0,
    "precision": 0.0-1.0
  },
  "quality": {
    "entity_score": 0.0-1.0,
    "relation_score": 0.0-1.0,
    "bridge_score": 0.0-1.0,
    "verifier_precision": 0.0-1.0,
    "verifier_coverage": 0.0-1.0,
    "object_gate_score": 0.0-1.0,
    "risk_level": "low|medium|high"
  },
  "rationale": "..."
}"""


REFINER_TASK = REPAIR_TASK


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value or "").strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n", ""}:
        return False
    return default


def clean_patch_value(value: Any) -> bool:
    return value not in (None, "", [])


BLOCKING_VIOLATION_CODES = {
    "empty_path",
    "hallucinated_relation_signature",
    "insufficient_grounding_coverage",
    "missing_terminal_object_gate",
    "answer_leakage",
}


CONSTRAINT_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by", "for",
    "from", "had", "has", "have", "in", "is", "it", "its", "of", "on", "or",
    "that", "the", "their", "this", "to", "was", "were", "with",
    "yes", "no", "unknown", "none", "not", "specified",
}


GENERIC_RELATION_SURFACES = {
    "be", "is", "are", "was", "were", "been", "being",
    "is a", "is an", "are a", "are an", "was a", "was an",
    "were a", "were an", "is the", "was the", "has", "have", "had",
}


OBJECT_GATE_RELATION_FAMILIES = {
    "occupation", "profession", "job", "position", "role",
}

OCCUPATION_OBJECT_TERMS = [
    "politician", "actor", "actress", "writer", "singer", "footballer",
    "musician", "artist", "author", "director", "producer", "composer",
    "journalist", "poet", "scientist", "professor", "coach", "athlete",
    "businessman", "businesswoman", "lawyer", "doctor", "physician",
    "engineer", "entrepreneur", "designer", "painter", "philosopher",
    "pianist", "pilot", "rapper", "songwriter", "teacher", "novelist",
    "comedian", "chemist", "judge", "banker", "researcher", "drummer",
]


def clean_string_list(value: Any, limit: int = 24) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        values = []
    seen = set()
    cleaned = []
    for item in values:
        text = normalize_text(item)
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
        if len(cleaned) >= limit:
            break
    return cleaned


def merge_string_lists(*values: Any, limit: int = 32) -> List[str]:
    merged = []
    seen = set()
    for value in values:
        for item in clean_string_list(value, limit=limit):
            if item in seen:
                continue
            seen.add(item)
            merged.append(item)
            if len(merged) >= limit:
                return merged
    return merged


def extract_answer_constraints(source: Dict[str, Any]) -> Dict[str, List[str]]:
    if not isinstance(source, dict):
        return {"allowed_terms": [], "reject_terms": [], "object_type_rules": []}
    constraints = source.get("answer_constraints")
    if not isinstance(constraints, dict):
        constraints = source.get("terminal_object_constraints")
    if not isinstance(constraints, dict):
        return {"allowed_terms": [], "reject_terms": [], "object_type_rules": []}
    return {
        "allowed_terms": merge_string_lists(
            constraints.get("allowed_terms"),
            constraints.get("allow_terms"),
            constraints.get("terminal_object_allow_terms"),
            limit=32,
        ),
        "reject_terms": merge_string_lists(
            constraints.get("reject_terms"),
            constraints.get("terminal_object_reject_terms"),
            limit=24,
        ),
        "object_type_rules": merge_string_lists(
            constraints.get("object_type_rules"),
            constraints.get("type_rules"),
            limit=16,
        ),
    }


def answer_constraint_terms_from_examples(plan: Dict[str, Any], limit: int = 32) -> Dict[str, List[str]]:
    phrase_counts: Dict[str, int] = {}
    token_counts: Dict[str, int] = {}
    type_rules = []
    answer_type = normalize_text(plan.get("answer_type", ""))
    if answer_type:
        type_rules.append(f"answer_type={answer_type}")

    for example in plan.get("examples", []) or []:
        if not isinstance(example, dict):
            continue
        candidate_values = list(example.get("answers", []) or [])
        for triple in example.get("evidence_path", []) or []:
            if isinstance(triple, (list, tuple)) and len(triple) == 3:
                candidate_values.append(triple[2])
        for value in candidate_values:
            phrase = normalize_text(value)
            if not phrase or phrase in CONSTRAINT_STOPWORDS:
                continue
            if len(phrase) <= 1 or re.fullmatch(r"#[0-9]+", phrase):
                continue
            phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
            for token in tokenize(phrase):
                if token in CONSTRAINT_STOPWORDS or len(token) <= 2 or token.isdigit():
                    continue
                token_counts[token] = token_counts.get(token, 0) + 1

    phrases = sorted(phrase_counts, key=lambda item: (-phrase_counts[item], item))
    tokens = sorted(token_counts, key=lambda item: (-token_counts[item], item))
    allowed = []
    for item in phrases[: max(8, limit // 2)] + tokens[:limit]:
        if item not in allowed:
            allowed.append(item)
        if len(allowed) >= limit:
            break

    reject_terms = []
    relation_text = " ".join(
        " ".join(str(alias) for alias in step.get("relation_aliases", []) or [])
        + " "
        + str(step.get("relation_family", ""))
        + " "
        + str(step.get("relation_signature", ""))
        for step in plan.get("path", []) or []
        if isinstance(step, dict)
    )
    relation_terms = set(tokenize(relation_text))
    if relation_terms & {"occupation", "profession", "job"}:
        for term in OCCUPATION_OBJECT_TERMS:
            if term not in allowed:
                allowed.append(term)
        type_rules.extend([
            "terminal object should denote a profession, role, job, or occupation",
            "generic copula relations are valid only when the object matches an occupation-like answer term",
        ])
        reject_terms.extend([
            "also known as", "known as", "person from", "member of",
            "born", "died", "located", "opposed", "competed",
        ])

    return {
        "allowed_terms": allowed[:limit],
        "reject_terms": clean_string_list(reject_terms, limit=16),
        "object_type_rules": clean_string_list(type_rules, limit=12),
    }


def merge_answer_constraints(*sources: Dict[str, Any], fallback_plan: Dict[str, Any]) -> Dict[str, List[str]]:
    fallback = answer_constraint_terms_from_examples(fallback_plan)
    allowed_values = [fallback.get("allowed_terms", [])]
    reject_values = [fallback.get("reject_terms", [])]
    type_rule_values = [fallback.get("object_type_rules", [])]
    for source in sources:
        constraints = extract_answer_constraints(source)
        allowed_values.append(constraints.get("allowed_terms", []))
        reject_values.append(constraints.get("reject_terms", []))
        type_rule_values.append(constraints.get("object_type_rules", []))
    return {
        "allowed_terms": merge_string_lists(*allowed_values, limit=32),
        "reject_terms": merge_string_lists(*reject_values, limit=24),
        "object_type_rules": merge_string_lists(*type_rule_values, limit=16),
    }


def step_needs_object_gate(step: Dict[str, Any], plan: Dict[str, Any]) -> bool:
    relation_family = normalize_text(step.get("relation_family") or "")
    relation_signature_value = normalize_text(step.get("relation_signature") or "")
    aliases = {normalize_text(alias) for alias in step.get("relation_aliases", []) or []}
    policy = normalize_object_gate_policy(step.get("generic_relation_policy") or "")
    if policy in {"allow_only_with_answer_object_gate", "reject_generic"}:
        return True
    if relation_family in OBJECT_GATE_RELATION_FAMILIES or relation_signature_value in OBJECT_GATE_RELATION_FAMILIES:
        return True
    if aliases & GENERIC_RELATION_SURFACES:
        return True
    if any(
        unsafe in alias
        for alias in aliases
        for unsafe in ("known as", "also known", "called", "named")
    ):
        return True
    plan_intent = normalize_text(plan.get("intent") or "")
    question_pattern = normalize_text(plan.get("question_pattern") or "")
    if relation_family == "attribute" and {"occupation", "profession", "job"} & (
        set(tokenize(plan_intent)) | set(tokenize(question_pattern))
    ):
        return True
    return False


def normalize_object_gate_policy(value: Any) -> str:
    policy = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if policy in {"allow", "allow_only_with_answer_object_gate", "reject_generic"}:
        return policy
    return ""


def plan_example_ids(plan: Dict[str, Any]) -> List[str]:
    return [
        str(example.get("id", "") or "")
        for example in plan.get("examples", []) or []
        if isinstance(example, dict) and str(example.get("id", "") or "")
    ]


def deterministic_contrastive_report(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Check whether explicit schema contracts reject typed corpus near misses."""
    path = [item for item in plan.get("path", []) or [] if isinstance(item, dict)]
    answer_type = normalize_text(plan.get("answer_type", "entity"))
    intent = normalize_text(plan.get("intent", ""))
    specific_relations = all(
        relation_signature(
            str(item.get("relation_signature") or item.get("relation_family") or "")
        ) not in {"", "attribute", "is", "was", "has"}
        for item in path
    )
    directed = bool(path) and all(
        normalize_text(item.get("direction", "either")) in {"forward", "reverse"}
        for item in path
    )
    plan_contract = [(
        relation_signature(str(
            item.get("relation_signature") or item.get("relation_family") or ""
        )),
        normalize_text(item.get("direction", "forward")) or "forward",
    ) for item in path]
    tested = rejected = 0
    decisions = []
    for example in plan.get("examples", []) or []:
        cq = (example.get("contrastive_cq", {}) or {}) if isinstance(example, dict) else {}
        expected_hops = int((cq.get("branch_structure", {}) or {}).get("hop_count", 0) or 0)
        for negative in cq.get("hard_negatives", []) or []:
            if not isinstance(negative, dict):
                continue
            types = set(negative.get("negative_types", []) or [])
            checks = []
            if "incomplete_path" in types and expected_hops:
                checks.append(len(path) >= expected_hops)
            if "wrong_bridge" in types:
                checks.append(len(path) >= 2 and all(item.get("from") and item.get("to") for item in path))
            if "branch_incomplete" in types:
                checks.append("comparison" in intent or "conjunction" in intent)
            if "wrong_answer_type" in types:
                checks.append(answer_type not in {"", "entity", "answer", "unknown"})
            if "structurally_plausible_incorrect_path" in types:
                checks.append(specific_relations and directed)
            if types & {"wrong_relation", "ontology_induced_high_score_wrong_schema"}:
                negative_contract = dict(
                    negative.get("negative_schema_contract", {}) or {}
                )
                negative_path = [(
                    relation_signature(str(
                        item.get("relation_signature")
                        or item.get("relation_family")
                        or ""
                    )),
                    normalize_text(item.get("direction", "forward")) or "forward",
                ) for item in negative_contract.get("path", []) or []
                    if isinstance(item, dict)]
                relation_mismatch = bool(
                    negative_path and plan_contract and negative_path != plan_contract
                )
                answer_mismatch = bool(
                    negative_contract.get("answer_type")
                    and normalize_text(negative_contract.get("answer_type"))
                    != answer_type
                )
                intent_mismatch = bool(
                    negative_contract.get("intent")
                    and normalize_text(negative_contract.get("intent")) != intent
                )
                checks.append(
                    relation_mismatch or answer_mismatch or intent_mismatch
                )
            if not checks:
                continue
            tested += 1
            is_rejected = all(checks)
            rejected += int(is_rejected)
            decisions.append({
                "doc_id": str(negative.get("doc_id", "") or ""),
                "negative_types": sorted(types),
                "rejected": is_rejected,
            })
    return {
        "testable_hard_negative_count": tested,
        "rejected_hard_negative_count": rejected,
        "hard_negative_rejection_rate": rejected / max(tested, 1),
        "decisions": decisions[:64],
        "source": "deterministic_typed_constraint_checks",
    }


def deterministic_plan_violations(
    plan: Dict[str, Any],
    grounding_report: Dict[str, Any],
    min_grounded_coverage: float,
) -> List[Dict[str, Any]]:
    """Produce the only violations that are allowed to block admission."""
    example_ids = plan_example_ids(plan)
    failed_ids = [
        str(value) for value in grounding_report.get("failed_example_ids", []) or []
        if str(value)
    ]
    path = [step for step in plan.get("path", []) or [] if isinstance(step, dict)]
    violations: List[Dict[str, Any]] = []

    def add(code: str, evidence: str, ids: List[str], step: int | None = None) -> None:
        payload: Dict[str, Any] = {
            "code": code,
            "severity": "blocking",
            "evidence_example_ids": ids[:24],
            "deterministically_checkable": True,
            "evidence": evidence,
            "source": "deterministic_contract",
        }
        if step is not None:
            payload["step"] = step
        violations.append(payload)

    if not path:
        add("empty_path", "The executable relation path is empty.", example_ids)

    hallucinated = [
        str(value) for value in grounding_report.get("hallucinated_steps", []) or []
        if str(value)
    ]
    if hallucinated:
        add(
            "hallucinated_relation_signature",
            f"Unobserved relation signatures: {hallucinated}",
            failed_ids or example_ids,
        )

    grounded_coverage = as_float(grounding_report.get("path_coverage"), 0.0)
    if grounded_coverage < min_grounded_coverage:
        add(
            "insufficient_grounding_coverage",
            f"Grounding coverage {grounded_coverage:.4f} is below {min_grounded_coverage:.4f}.",
            failed_ids or example_ids,
        )

    constraints = extract_answer_constraints(plan)
    for step_idx, step in enumerate(path):
        is_terminal = step_idx == len(path) - 1 or normalize_text(step.get("to", "")) == "answer"
        if not is_terminal or not step_needs_object_gate(step, plan):
            continue
        policy = normalize_object_gate_policy(step.get("generic_relation_policy")) or "allow"
        allow_terms = merge_string_lists(
            step.get("terminal_object_allow_terms"),
            constraints.get("allowed_terms"),
            limit=32,
        )
        if policy == "reject_generic":
            continue
        if policy == "allow" or not allow_terms:
            add(
                "missing_terminal_object_gate",
                "A generic terminal relation is executable without an answer-object gate.",
                example_ids,
                step=step_idx,
            )

    route_values = {
        normalize_text(value)
        for value in [
            *(plan.get("triggers", []) or []),
            *(plan.get("required_query_terms", []) or []),
        ]
        if normalize_text(value)
    }
    leaked_ids = []
    leaked_answers = set()
    for example in plan.get("examples", []) or []:
        if not isinstance(example, dict):
            continue
        for answer in example.get("answers", []) or []:
            normalized = normalize_text(answer)
            if len(normalized) >= 4 and normalized in route_values:
                leaked_ids.append(str(example.get("id", "") or ""))
                leaked_answers.add(normalized)
    if leaked_answers:
        add(
            "answer_leakage",
            f"Routing terms contain induction answers: {sorted(leaked_answers)}",
            [value for value in leaked_ids if value],
        )
    return violations


def normalize_critic_findings(
    critic: Dict[str, Any],
    deterministic_violations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Downgrade unverifiable critic claims and attach confirmed blockers."""
    deterministic_by_code = {
        str(item.get("code", "")): item for item in deterministic_violations
    }
    confirmed_codes = set(deterministic_by_code)
    normalized_violations = []
    warnings = clean_string_list(critic.get("warnings"), limit=32)
    for item in critic.get("violations", []) or []:
        if not isinstance(item, dict):
            warnings.append(normalize_text(item))
            continue
        code = normalize_text(item.get("code", "")).replace(" ", "_")
        requested_blocking = normalize_text(item.get("severity", "")) == "blocking"
        checkable = as_bool(item.get("deterministically_checkable"), False)
        confirmed = requested_blocking and checkable and code in confirmed_codes
        normalized = deepcopy(item)
        normalized["code"] = code or "untyped_critic_warning"
        normalized["severity"] = "blocking" if confirmed else "warning"
        normalized["deterministically_confirmed"] = confirmed
        normalized_violations.append(normalized)
        if requested_blocking and not confirmed:
            warnings.append(
                f"downgraded_unverified_critic_claim:{normalized['code']}"
            )
    for code, violation in deterministic_by_code.items():
        if not any(
            item.get("code") == code and item.get("severity") == "blocking"
            for item in normalized_violations
        ):
            normalized_violations.append(deepcopy(violation))
    for legacy_key in ("risks", "generic_relation_risks", "object_gate_risks"):
        warnings.extend(clean_string_list(critic.get(legacy_key), limit=16))
    return {
        "risk_level": str(critic.get("risk_level") or "unknown").lower(),
        "violations": normalized_violations,
        "blocking_violations": [
            item for item in normalized_violations if item.get("severity") == "blocking"
        ],
        "warnings": clean_string_list(warnings, limit=64),
        "suggested_precision": as_float(critic.get("suggested_precision"), 0.0),
        "suggested_confidence": as_float(critic.get("suggested_confidence"), 0.0),
    }


def verifier_admits(
    verifier: Dict[str, Any],
    allow_numeric_override: bool,
    min_precision: float,
    min_coverage: float,
) -> bool:
    if str(verifier.get("decision") or "").lower() == "keep":
        return True
    return (
        allow_numeric_override
        and as_float(verifier.get("precision_estimate"), 0.0) >= min_precision
        and as_float(verifier.get("coverage_estimate"), 0.0) >= min_coverage
    )


def build_patch_from_agents(
    original_plan: Dict[str, Any],
    entity: Dict[str, Any],
    relation: Dict[str, Any],
    bridge: Dict[str, Any],
    verifier: Dict[str, Any],
    critic: Dict[str, Any],
    final_payload: Dict[str, Any],
) -> Dict[str, Any]:
    patch = final_payload.get("plan_patch") if isinstance(final_payload.get("plan_patch"), dict) else {}
    bridge_patch = bridge if isinstance(bridge, dict) else {}
    merged = {}

    for source in (original_plan, bridge_patch, patch):
        for key in (
            "intent", "question_pattern", "triggers", "anchor_slots", "path",
            "answer_type", "required_query_terms", "unsafe_query_terms",
            "grounding_citations",
        ):
            if clean_patch_value(source.get(key)):
                merged[key] = source[key]

    repaired_constraints = extract_answer_constraints(patch)
    if any(repaired_constraints.values()):
        merged["answer_constraints"] = repaired_constraints
    else:
        merged["answer_constraints"] = merge_answer_constraints(
            original_plan,
            entity,
            relation,
            bridge_patch,
            fallback_plan=original_plan,
        )

    precision = (
        as_float(patch.get("precision"), -1.0)
        if isinstance(patch, dict) else -1.0
    )
    if precision < 0:
        precision = as_float(final_payload.get("quality", {}).get("verifier_precision"), -1.0) if isinstance(final_payload.get("quality"), dict) else -1.0
    if precision < 0:
        precision = as_float(verifier.get("precision_estimate"), -1.0)
    if precision < 0:
        precision = as_float(critic.get("suggested_precision"), -1.0)
    if precision < 0:
        precision = as_float(original_plan.get("precision"), 0.0)

    confidence = as_float(patch.get("confidence"), -1.0) if isinstance(patch, dict) else -1.0
    if confidence < 0:
        confidence = as_float(critic.get("suggested_confidence"), -1.0)
    if confidence < 0:
        confidence = as_float(bridge_patch.get("confidence"), -1.0)
    if confidence < 0:
        confidence = as_float(original_plan.get("confidence"), 0.0)

    merged["precision"] = max(0.0, min(precision, 0.99))
    merged["confidence"] = max(0.0, min(confidence, 0.99))
    return merged


def materialize_repaired_candidate(
    induction_plan: Dict[str, Any],
    working_plan: Dict[str, Any],
    bridge: Dict[str, Any],
    repair_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Materialize the exact plan that post-repair agents must review."""
    repaired = deepcopy(working_plan)
    plan_patch = (
        repair_payload.get("plan_patch")
        if isinstance(repair_payload.get("plan_patch"), dict)
        else {}
    )
    transferable_keys = (
        "intent", "question_pattern", "triggers", "anchor_slots", "path",
        "answer_type", "answer_constraints", "required_query_terms",
        "unsafe_query_terms", "grounding_citations", "confidence", "precision",
    )
    for source in (bridge, plan_patch):
        if not isinstance(source, dict):
            continue
        for key in transferable_keys:
            if clean_patch_value(source.get(key)):
                repaired[key] = deepcopy(source[key])




    induction_path = list(induction_plan.get("path", []) or [])
    repaired_path = list(repaired.get("path", []) or [])
    for step_idx, step in enumerate(repaired_path):
        fallback = induction_path[step_idx] if step_idx < len(induction_path) else {}
        step["relation_family_alternatives"] = merge_string_lists(
            step.get("relation_family_alternatives"),
            fallback.get("relation_family_alternatives"),
            [step.get("relation_family", ""), fallback.get("relation_family", "")],
            limit=8,
        )
        step["relation_signature_alternatives"] = merge_string_lists(
            step.get("relation_signature_alternatives"),
            fallback.get("relation_signature_alternatives"),
            [step.get("relation_signature", ""), fallback.get("relation_signature", "")],
            limit=12,
        )
        step["relation_aliases"] = merge_string_lists(
            step.get("relation_aliases"),
            fallback.get("relation_aliases"),
            limit=16,
        )
    repaired["path"] = repaired_path


    repaired["examples"] = deepcopy(induction_plan.get("examples", []))
    repaired_constraints = extract_answer_constraints(plan_patch)
    if any(repaired_constraints.values()):
        repaired["answer_constraints"] = repaired_constraints
    else:
        repaired["answer_constraints"] = merge_answer_constraints(
            induction_plan,
            working_plan,
            bridge,
            fallback_plan=induction_plan,
        )
    return repaired


def merge_quality(
    entity: Dict[str, Any],
    relation: Dict[str, Any],
    bridge: Dict[str, Any],
    verifier: Dict[str, Any],
    critic: Dict[str, Any],
    final_payload: Dict[str, Any],
) -> Dict[str, Any]:
    final_quality = final_payload.get("quality") if isinstance(final_payload.get("quality"), dict) else {}


    risk_level = str(critic.get("risk_level") or final_quality.get("risk_level") or "high").lower()
    return {
        "entity_score": as_float(final_quality.get("entity_score"), as_float(entity.get("quality_score"), 0.0)),
        "relation_score": as_float(final_quality.get("relation_score"), as_float(relation.get("path_quality_score"), 0.0)),
        "bridge_score": as_float(final_quality.get("bridge_score"), as_float(bridge.get("confidence"), 0.0)),
        "verifier_precision": as_float(verifier.get("precision_estimate"), as_float(final_quality.get("verifier_precision"), 0.0)),
        "verifier_coverage": as_float(verifier.get("coverage_estimate"), as_float(final_quality.get("verifier_coverage"), 0.0)),
        "hard_negative_rejection_rate": as_float(
            verifier.get("hard_negative_rejection_rate"), 0.0
        ),
        "direction_correctness": as_float(verifier.get("direction_correctness"), 0.0),
        "binding_correctness": as_float(verifier.get("binding_correctness"), 0.0),
        "branch_completeness": as_float(verifier.get("branch_completeness"), 0.0),
        "object_gate_score": as_float(verifier.get("object_gate_quality"), as_float(final_quality.get("object_gate_score"), 0.0)),
        "risk_level": risk_level,
        "critic_safe_to_keep": as_bool(critic.get("safe_to_keep"), False),
        "critic_suggested_precision": as_float(critic.get("suggested_precision"), 0.0),
        "critic_suggested_confidence": as_float(critic.get("suggested_confidence"), 0.0),
    }


def normalize_step(
    step: Dict[str, Any],
    fallback_step: Dict[str, Any],
    step_idx: int,
    plan: Dict[str, Any],
    answer_constraints: Dict[str, List[str]],
    path_len: int,
) -> Dict[str, Any]:
    relation_family = str(step.get("relation_family") or fallback_step.get("relation_family") or "").strip()
    relation_signature_value = str(step.get("relation_signature") or fallback_step.get("relation_signature") or "").strip()
    if relation_family and not relation_signature_value:
        relation_signature_value = relation_family
    aliases = step.get("relation_aliases") or fallback_step.get("relation_aliases") or []
    aliases = [str(alias).strip() for alias in aliases if str(alias).strip()]
    if not aliases and relation_signature_value:
        aliases = [relation_signature_value.replace("_", " ")]
    if relation_signature_value and relation_signature_value == relation_family and aliases:
        signature_from_alias = relation_signature(aliases[0])
        if signature_from_alias:
            relation_signature_value = signature_from_alias
    is_terminal_step = (
        step_idx == path_len - 1
        or str(step.get("to") or fallback_step.get("to") or "").strip().lower() == "answer"
    )
    normalized = {
        "step": step_idx,
        "from": step.get("from") or fallback_step.get("from") or ("anchor" if step_idx == 0 else f"node_{step_idx}"),
        "to": step.get("to") or fallback_step.get("to") or "answer",
        "relation_family": relation_family,
        "relation_family_alternatives": merge_string_lists(
            step.get("relation_family_alternatives"),
            fallback_step.get("relation_family_alternatives"),
            [relation_family],
            limit=8,
        ),
        "relation_signature": relation_signature_value,
        "relation_signature_alternatives": merge_string_lists(
            step.get("relation_signature_alternatives"),
            fallback_step.get("relation_signature_alternatives"),
            [relation_signature_value],
            limit=12,
        ),
        "relation_aliases": aliases[:8],
        "required_relation_terms": [str(term).strip() for term in (step.get("required_relation_terms") or []) if str(term).strip()][:8],
        "reject_relation_terms": [str(term).strip() for term in (step.get("reject_relation_terms") or []) if str(term).strip()][:8],
        "direction": step.get("direction") or fallback_step.get("direction") or "either",
    }
    policy = normalize_object_gate_policy(
        step.get("generic_relation_policy") or fallback_step.get("generic_relation_policy") or ""
    )
    if not policy:
        policy = "allow_only_with_answer_object_gate" if is_terminal_step and step_needs_object_gate(normalized, plan) else "allow"
    normalized["generic_relation_policy"] = policy
    if is_terminal_step and policy != "allow":
        normalized["terminal_object_allow_terms"] = merge_string_lists(
            step.get("terminal_object_allow_terms"),
            fallback_step.get("terminal_object_allow_terms"),
            answer_constraints.get("allowed_terms", []),
            limit=32,
        )
        normalized["terminal_object_reject_terms"] = merge_string_lists(
            step.get("terminal_object_reject_terms"),
            fallback_step.get("terminal_object_reject_terms"),
            answer_constraints.get("reject_terms", []),
            limit=24,
        )
    else:
        normalized["terminal_object_allow_terms"] = merge_string_lists(
            step.get("terminal_object_allow_terms"),
            fallback_step.get("terminal_object_allow_terms"),
            limit=16,
        )
        normalized["terminal_object_reject_terms"] = merge_string_lists(
            step.get("terminal_object_reject_terms"),
            fallback_step.get("terminal_object_reject_terms"),
            limit=16,
        )
    return normalized


def ground_proposed_path(
    proposed_plan: Dict[str, Any],
    induction_plan: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute a proposed schema symbolically over induction evidence paths."""
    proposed_steps = [
        step for step in proposed_plan.get("path", []) or []
        if isinstance(step, dict)
    ]
    proposed_signature_sets = []
    proposed_signatures = []
    for step in proposed_steps:
        values = [
            step.get("relation_signature"),
            *(step.get("relation_signature_alternatives", []) or []),
            *(step.get("relation_aliases", []) or []),
        ]
        signatures = {
            relation_signature(str(value)) for value in values if value
        }
        signatures.discard("")
        proposed_signature_sets.append(signatures)
        proposed_signatures.append(
            relation_signature(
                str(step.get("relation_signature") or step.get("relation_family") or "")
            )
        )
    examples = [
        item for item in induction_plan.get("examples", []) or []
        if isinstance(item, dict)
    ]
    supported_ids: List[str] = []
    failed_ids: List[str] = []
    observed_signatures = set()
    evidence_texts: Dict[str, List[Tuple[str, str]]] = {}
    for example in examples:
        evidence_path = [
            item for item in example.get("evidence_path", []) or []
            if isinstance(item, (list, tuple)) and len(item) >= 3
        ]
        signatures = [relation_signature(str(item[1])) for item in evidence_path]
        signatures = [value for value in signatures if value]
        observed_signatures.update(signatures)
        example_id = str(example.get("id", "") or "")
        evidence_texts[example_id] = [
            (
                str(passage.get("title", "") or ""),
                normalize_text(passage.get("text", "") or ""),
            )
            for passage in example.get("supporting_passages", []) or []
            if isinstance(passage, dict) and passage.get("text")
        ]
        path_supported = bool(proposed_signature_sets) and len(proposed_signature_sets) <= len(signatures)
        if path_supported:
            path_supported = all(
                observed in allowed
                for allowed, observed in zip(proposed_signature_sets, signatures)
            )
        if path_supported:
            supported_ids.append(example_id)
        else:
            failed_ids.append(example_id)
    valid_citations = []
    invalid_citations = []
    cited_steps: Dict[str, set[int]] = {}
    for citation in proposed_plan.get("grounding_citations", []) or []:
        if not isinstance(citation, dict):
            continue
        example_id = str(citation.get("example_id", "") or "")
        title = normalize_text(citation.get("title", "") or "")
        quote = normalize_text(citation.get("quote", "") or "")
        steps = {
            int(value) for value in citation.get("supports_steps", []) or []
            if str(value).lstrip("-").isdigit() and int(value) >= 0
        }
        matched = bool(quote) and len(quote) >= 12 and any(
            quote in text and (not title or title == normalize_text(passage_title))
            for passage_title, text in evidence_texts.get(example_id, [])
        )
        normalized_citation = {
            "example_id": example_id,
            "title": str(citation.get("title", "") or ""),
            "quote": str(citation.get("quote", "") or ""),
            "supports_steps": sorted(steps),
        }
        if matched:
            valid_citations.append(normalized_citation)
            cited_steps.setdefault(example_id, set()).update(steps)
        else:
            invalid_citations.append(normalized_citation)
    required_steps = set(range(len(proposed_signature_sets)))
    document_grounded_ids = sorted(
        example_id for example_id, steps in cited_steps.items()
        if required_steps and required_steps.issubset(steps)
    )
    document_grounded_steps = sorted({
        step
        for steps in cited_steps.values()
        for step in steps
        if step in required_steps
    })
    document_step_coverage = (
        len(document_grounded_steps) / len(required_steps)
        if required_steps else 0.0
    )
    examples_with_documents = sum(bool(value) for value in evidence_texts.values())
    grounding_mode = (
        "verbatim_supporting_passage_citations"
        if examples_with_documents
        else "evidence_path_signature_fallback"
    )


    hallucinated = [
        proposed_signatures[index] or f"step_{index}"
        for index, allowed in enumerate(proposed_signature_sets)
        if not (allowed & observed_signatures)
    ]
    signature_supported_id_set = set(supported_ids)
    document_grounded_id_set = set(document_grounded_ids)
    if examples_with_documents:
        jointly_grounded_ids = sorted(
            signature_supported_id_set & document_grounded_id_set
        )





        path_coverage = min(
            len(supported_ids) / max(len(examples), 1),
            document_step_coverage,
        )
    else:
        jointly_grounded_ids = sorted(signature_supported_id_set)
        path_coverage = len(supported_ids) / max(len(examples), 1)
    return {
        "executor": "deterministic_verbatim_provenance_checker",
        "grounding_mode": grounding_mode,
        "example_count": len(examples),
        "supported_example_ids": jointly_grounded_ids,
        "failed_example_ids": sorted(
            set(str(item.get("id", "") or "") for item in examples)
            - set(jointly_grounded_ids)
        ),
        "path_coverage": path_coverage,
        "coverage_unit": "relation_steps_across_induction_examples",
        "joint_grounded_example_ids": jointly_grounded_ids,
        "signature_path_coverage": len(supported_ids) / max(len(examples), 1),
        "signature_supported_example_ids": supported_ids,
        "document_grounded_example_ids": document_grounded_ids,
        "document_grounded_coverage": len(document_grounded_ids) / max(examples_with_documents, 1),
        "document_grounded_steps": document_grounded_steps,
        "document_step_coverage": document_step_coverage,
        "valid_citations": valid_citations,
        "invalid_citations": invalid_citations,
        "proposed_hop_count": len(proposed_signature_sets),
        "allowed_relation_signatures_by_step": [
            sorted(values) for values in proposed_signature_sets
        ],
        "observed_relation_signatures": sorted(observed_signatures),
        "hallucinated_steps": hallucinated,
        "hallucinated_step_count": len(hallucinated),
        "uses_gold_at_inference": False,
    }


def repair_round_converged(
    repair_payload: Dict[str, Any],
    verifier: Dict[str, Any],
    blocking_violations: List[Dict[str, Any]],
    verifier_keep_override: bool,
    min_precision: float,
    min_coverage: float,
) -> bool:
    return (
        str(repair_payload.get("decision") or "drop").lower() == "keep"
        and not blocking_violations
        and verifier_admits(
            verifier,
            allow_numeric_override=verifier_keep_override,
            min_precision=min_precision,
            min_coverage=min_coverage,
        )
    )


def refine_one_plan(
    plan: Dict[str, Any],
    client: CachedChatClient,
    max_examples: int,
    rounds: int,
    min_precision: float,
    min_confidence: float,
    min_coverage: float,
    allow_medium_risk: bool,
    verifier_keep_override: bool,
    min_grounded_coverage: float,
) -> Tuple[Dict[str, Any] | None, Dict[str, Any]]:
    candidate = compact_plan(plan, max_examples=max_examples)
    audit = {
        "plan_id": plan.get("id", ""),
        "support": plan.get("support", 0),
        "rounds": [],
        "decision": "drop",
    }

    working = deepcopy(candidate)
    final_payload: Dict[str, Any] = {}
    last_entity: Dict[str, Any] = {}
    last_relation: Dict[str, Any] = {}
    last_bridge: Dict[str, Any] = {}
    last_verifier: Dict[str, Any] = {}
    last_critic: Dict[str, Any] = {}
    for round_idx in range(max(rounds, 1)):
        payload = {
            "round": round_idx + 1,
            "candidate_plan": working,
            "previous_final": final_payload,
        }
        entity, entity_meta = client.ask_json(agent_messages("EntityAgent", ENTITY_TASK, payload), "EntityAgent")
        relation, relation_meta = client.ask_json(agent_messages("RelationAgent", RELATION_TASK, payload), "RelationAgent")
        bridge_payload = {**payload, "entity_agent": entity, "relation_agent": relation}
        bridge, bridge_meta = client.ask_json(agent_messages("BridgeAgent", BRIDGE_TASK, bridge_payload), "BridgeAgent")
        grounding_report = ground_proposed_path(bridge, working)
        contrastive_report = deterministic_contrastive_report(bridge)
        deterministic_violations = deterministic_plan_violations(
            bridge,
            grounding_report,
            min_grounded_coverage=min_grounded_coverage,
        )
        verifier_payload = {
            **bridge_payload,
            "proposed_bridge_plan": bridge,
            "grounding_report": grounding_report,
            "contrastive_verification_report": contrastive_report,
            "deterministic_blocking_violations": deterministic_violations,
        }
        verifier, verifier_meta = client.ask_json(agent_messages("VerifierAgent", VERIFIER_TASK, verifier_payload), "VerifierAgent")
        critic_payload = {**verifier_payload, "verifier_agent": verifier}
        critic, critic_meta = client.ask_json(agent_messages("CriticAgent", CRITIC_TASK, critic_payload), "CriticAgent")
        typed_critic = normalize_critic_findings(critic, deterministic_violations)
        refiner_payload = {
            **critic_payload,
            "critic_agent": typed_critic,
            "blocking_violations": typed_critic["blocking_violations"],
            "min_precision": min_precision,
            "min_confidence": min_confidence,
        }
        final_payload, repair_meta = client.ask_json(
            agent_messages("RepairAgent", REPAIR_TASK, refiner_payload),
            "RepairAgent",
        )
        repaired_candidate = materialize_repaired_candidate(
            induction_plan=candidate,
            working_plan=working,
            bridge=bridge,
            repair_payload=final_payload,
        )
        post_repair_grounding = ground_proposed_path(repaired_candidate, candidate)
        post_repair_contrastive = deterministic_contrastive_report(
            repaired_candidate
        )
        post_repair_violations = deterministic_plan_violations(
            repaired_candidate,
            post_repair_grounding,
            min_grounded_coverage=min_grounded_coverage,
        )
        post_verifier_payload = {
            **bridge_payload,
            "review_phase": "post_repair",
            "pre_repair_verifier": verifier,
            "pre_repair_critic": critic,
            "repair_agent": final_payload,
            "proposed_bridge_plan": repaired_candidate,
            "grounding_report": post_repair_grounding,
            "contrastive_verification_report": post_repair_contrastive,
            "deterministic_blocking_violations": post_repair_violations,
        }
        post_verifier, post_verifier_meta = client.ask_json(
            agent_messages("PostRepairVerifierAgent", VERIFIER_TASK, post_verifier_payload),
            "PostRepairVerifierAgent",
        )
        post_critic_payload = {
            **post_verifier_payload,
            "verifier_agent": post_verifier,
        }
        post_critic, post_critic_meta = client.ask_json(
            agent_messages("PostRepairCriticAgent", CRITIC_TASK, post_critic_payload),
            "PostRepairCriticAgent",
        )
        typed_post_critic = normalize_critic_findings(
            post_critic,
            post_repair_violations,
        )
        converged = repair_round_converged(
            repair_payload=final_payload,
            verifier=post_verifier,
            blocking_violations=typed_post_critic["blocking_violations"],
            verifier_keep_override=verifier_keep_override,
            min_precision=min_precision,
            min_coverage=min_coverage,
        )
        pre_blocking_codes = sorted(
            str(item.get("code", ""))
            for item in typed_critic["blocking_violations"]
        )
        post_blocking_codes = sorted(
            str(item.get("code", ""))
            for item in typed_post_critic["blocking_violations"]
        )
        repair_stalled = bool(post_blocking_codes) and post_blocking_codes == pre_blocking_codes
        retry_eligible = bool(post_blocking_codes) and not repair_stalled
        last_entity = entity
        last_relation = relation
        last_bridge = repaired_candidate
        last_verifier = post_verifier
        last_critic = post_critic

        audit["rounds"].append({
            "round": round_idx + 1,
            "entity": entity,
            "relation": relation,
            "bridge": bridge,
            "verifier": verifier,
            "critic": critic,
            "typed_critic": typed_critic,
            "repair": final_payload,
            "final": final_payload,
            "repaired_candidate": repaired_candidate,
            "post_repair_grounding": post_repair_grounding,
            "post_repair_contrastive_verification": post_repair_contrastive,
            "post_repair_verifier": post_verifier,
            "post_repair_critic": post_critic,
            "typed_post_repair_critic": typed_post_critic,
            "pre_repair_blocking_violation_codes": pre_blocking_codes,
            "post_repair_blocking_violation_codes": post_blocking_codes,
            "repair_stalled": repair_stalled,
            "retry_eligible": retry_eligible,
            "converged": converged,
            "metadata": {
                "entity": entity_meta,
                "relation": relation_meta,
                "bridge": bridge_meta,
                "verifier": verifier_meta,
                "critic": critic_meta,
                "repair": repair_meta,
                "refiner": repair_meta,
                "post_repair_verifier": post_verifier_meta,
                "post_repair_critic": post_critic_meta,
            },
        })
        working = repaired_candidate
        if converged or not retry_eligible:
            break

    quality = merge_quality(
        entity=last_entity,
        relation=last_relation,
        bridge=last_bridge,
        verifier=last_verifier,
        critic=last_critic,
        final_payload=final_payload,
    )
    audit["repair_iterations"] = len(audit["rounds"])
    audit["repair_converged"] = bool(
        audit["rounds"] and audit["rounds"][-1].get("converged", False)
    )
    patch = build_patch_from_agents(
        original_plan=plan,
        entity=last_entity,
        relation=last_relation,
        bridge=last_bridge,
        verifier=last_verifier,
        critic=last_critic,
        final_payload=final_payload,
    )
    precision = as_float(patch.get("precision"), 0.0)
    confidence = as_float(patch.get("confidence"), 0.0)
    coverage = as_float(quality.get("verifier_coverage"), as_float(last_verifier.get("coverage_estimate"), 0.0))
    risk_level = str(quality.get("risk_level") or "high").lower()
    decision = str(final_payload.get("decision") or "drop").lower()
    verifier_decision = str(last_verifier.get("decision") or "").lower()
    grounding_report = ground_proposed_path(last_bridge, candidate)
    contrastive_report = deterministic_contrastive_report(last_bridge)
    quality["hard_negative_rejection_rate"] = float(
        contrastive_report.get("hard_negative_rejection_rate", 0.0) or 0.0
    )
    quality["hard_negative_test_count"] = int(
        contrastive_report.get("testable_hard_negative_count", 0) or 0
    )
    final_violations = deterministic_plan_violations(
        last_bridge,
        grounding_report,
        min_grounded_coverage=min_grounded_coverage,
    )
    typed_final_critic = normalize_critic_findings(last_critic, final_violations)
    blocking_violations = typed_final_critic["blocking_violations"]
    critic_safe = not blocking_violations
    grounded_coverage = as_float(grounding_report.get("path_coverage"), 0.0)
    grounding_pass = not blocking_violations



    risk_allowed = True
    repair_keep = decision == "keep"
    verifier_keep = verifier_admits(
        last_verifier,
        allow_numeric_override=verifier_keep_override,
        min_precision=min_precision,
        min_coverage=min_coverage,
    )
    quality["critic_safe_to_keep"] = critic_safe
    quality["validated_blocking_violation_count"] = len(blocking_violations)
    quality["critic_warning_count"] = len(typed_final_critic["warnings"])
    keep_plan = (
        grounding_pass
        and risk_allowed
        and precision >= min_precision
        and confidence >= min_confidence
        and coverage >= min_coverage
        and repair_keep
        and verifier_keep
        and critic_safe
    )

    if not keep_plan:
        audit["decision"] = "drop"
        audit["drop_reason"] = {
            "decision": decision,
            "verifier_decision": verifier_decision,
            "precision": precision,
            "confidence": confidence,
            "coverage": coverage,
            "risk_level": risk_level,
            "critic_safe_to_keep": critic_safe,
            "blocking_violations": blocking_violations,
            "critic_warnings": typed_final_critic["warnings"],
            "min_precision": min_precision,
            "min_confidence": min_confidence,
            "min_coverage": min_coverage,
            "allow_medium_risk": allow_medium_risk,
            "verifier_keep_override": verifier_keep_override,
            "verifier_admitted": verifier_keep,
            "grounding_report": grounding_report,
            "min_grounded_coverage": min_grounded_coverage,
        }
        return None, audit

    fallback_path = plan.get("path", [])
    answer_constraints = patch.get("answer_constraints") or answer_constraint_terms_from_examples(plan)
    raw_path = patch.get("path") or fallback_path
    refined_path = []
    for step_idx, step in enumerate(raw_path):
        fallback_step = fallback_path[step_idx] if step_idx < len(fallback_path) else {}
        normalized = normalize_step(
            step=step,
            fallback_step=fallback_step,
            step_idx=step_idx,
            plan={**plan, **patch},
            answer_constraints=answer_constraints,
            path_len=len(raw_path),
        )
        if normalized["relation_family"]:
            refined_path.append(normalized)
    if not refined_path:
        audit["decision"] = "drop"
        audit["drop_reason"] = "empty_refined_path"
        return None, audit

    refined = deepcopy(plan)
    refined.update({
        "intent": patch.get("intent") or plan.get("intent", ""),
        "question_pattern": patch.get("question_pattern") or plan.get("question_pattern", ""),
        "triggers": (patch.get("triggers") or plan.get("triggers") or [])[:20],
        "anchor_slots": patch.get("anchor_slots") or plan.get("anchor_slots") or [{"name": "anchor", "type": "entity", "source": "query_mention"}],
        "path": refined_path,
        "answer_type": patch.get("answer_type") or plan.get("answer_type", ""),
        "answer_constraints": answer_constraints,
        "grounding_citations": patch.get("grounding_citations", []),
        "required_query_terms": patch.get("required_query_terms") or plan.get("required_query_terms", []),
        "unsafe_query_terms": patch.get("unsafe_query_terms") or plan.get("unsafe_query_terms", []),
        "confidence": round(confidence, 4),
        "precision": round(precision, 4),
        "multiagent_quality": quality,
        "symbolic_grounding": grounding_report,
        "contrastive_verification": contrastive_report,
        "repair_operations": final_payload.get("repair_operations", []),
        "resolved_violation_codes": final_payload.get("resolved_violation_codes", []),
        "remaining_blocking_violations": blocking_violations,
        "critic_warnings": typed_final_critic["warnings"],
        "repair_iterations": len(audit["rounds"]),
        "repair_converged": True,
        "multiagent_rationale": final_payload.get("rationale", ""),
        "multiagent_keep_reason": "typed_violations_resolved_and_reverified",
        "builder_stage": "counterexample_repaired_reverified",
        "atomic_proposal": {
            "operator": "MODIFY_SCHEMA",
            "source_schema_id": str(plan.get("id", "") or ""),
            "source_cq_cluster": str(plan.get("local_cq_cluster_id", "") or ""),
            "provenance": "q_ind_contrastive_local_team",
        },
    })
    audit["decision"] = "keep"
    audit["refined_precision"] = refined["precision"]
    audit["refined_confidence"] = refined["confidence"]
    audit["answer_constraints"] = answer_constraints
    audit["object_gated_steps"] = [
        {
            "step": step.get("step"),
            "relation_signature": step.get("relation_signature"),
            "policy": step.get("generic_relation_policy"),
            "allow_terms": step.get("terminal_object_allow_terms", [])[:8],
            "reject_terms": step.get("terminal_object_reject_terms", [])[:8],
        }
        for step in refined_path
        if step.get("generic_relation_policy") != "allow"
    ]
    return refined, audit


def write_quality_report(path: Path, audits: List[Dict[str, Any]], refined_plans: List[Dict[str, Any]], diagnostics: Dict[str, Any]) -> None:
    kept = [audit for audit in audits if audit.get("decision") == "keep"]
    dropped = [audit for audit in audits if audit.get("decision") != "keep"]
    lines = [
        "# Multi-Agent Query-Plan Ontology Quality",
        "",
        f"- candidate_plans: {diagnostics.get('candidate_plan_count', 0)}",
        f"- processed_plans: {diagnostics.get('processed_plan_count', 0)}",
        f"- kept_plans: {len(kept)}",
        f"- dropped_plans: {len(dropped)}",
        f"- llm_calls: {diagnostics.get('llm_calls', 0)}",
        f"- cache_hits: {diagnostics.get('cache_hits', 0)}",
        f"- prompt_tokens: {diagnostics.get('prompt_tokens', 0)}",
        f"- completion_tokens: {diagnostics.get('completion_tokens', 0)}",
        f"- repair_converged_plans: {diagnostics.get('repair_converged_plan_count', 0)}",
        f"- repair_retry_plans: {diagnostics.get('repair_retry_plan_count', 0)}",
        f"- repair_operations: {diagnostics.get('repair_operation_count', 0)}",
        "",
        "## Kept Plans",
        "",
    ]
    object_gated_plan_count = sum(
        1 for plan in refined_plans
        if any(step.get("generic_relation_policy") != "allow" for step in plan.get("path", []))
    )
    plans_with_constraints = sum(
        1 for plan in refined_plans
        if (plan.get("answer_constraints") or {}).get("allowed_terms")
    )
    lines.extend([
        f"- plans_with_answer_constraints: {plans_with_constraints}",
        f"- object_gated_plans: {object_gated_plan_count}",
        "",
    ])
    for plan in refined_plans[:50]:
        constraints = plan.get("answer_constraints") or {}
        gated_steps = [
            step for step in plan.get("path", [])
            if step.get("generic_relation_policy") != "allow"
        ]
        lines.append(
            f"- `{plan.get('id')}` support={plan.get('support')} confidence={plan.get('confidence')} "
            f"precision={plan.get('precision')} intent={plan.get('intent')} path="
            f"{' -> '.join(step.get('relation_signature', step.get('relation_family', '')) for step in plan.get('path', []))}"
        )
        if constraints.get("allowed_terms") or gated_steps:
            lines.append(
                f"  answer_allow={constraints.get('allowed_terms', [])[:8]} "
                f"answer_reject={constraints.get('reject_terms', [])[:6]} "
                f"object_gated_steps={[step.get('step') for step in gated_steps]}"
            )
    lines.extend(["", "## Dropped Plan Reasons", ""])
    for audit in dropped[:50]:
        lines.append(f"- `{audit.get('plan_id')}`: {json.dumps(audit.get('drop_reason', ''), ensure_ascii=False)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Refine query-plan ontology with multi-agent review.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-dir", default="outputs_query_plan_ontology_multiagent")
    parser.add_argument("--llm-name", default=os.environ.get("MODEL_NAME", "reader-model"))
    parser.add_argument("--llm-base-url", default=os.environ.get("MODEL_BASE_URL"))
    parser.add_argument(
        "--max-plans",
        type=int,
        default=80,
        help="Maximum plans to refine; a negative value refines every candidate plan.",
    )
    parser.add_argument(
        "--selection-policy",
        choices=("quality", "workload_impact"),
        default="quality",
    )
    parser.add_argument(
        "--passthrough-unprocessed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Retain unreviewed schemas for a downstream held-out competency gate.",
    )
    parser.add_argument(
        "--passthrough-agent-rejected",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Let the held-out gate arbitrate schemas rejected by the LLM agents.",
    )
    parser.add_argument("--max-examples-per-plan", type=int, default=6)
    parser.add_argument(
        "--split-path-shapes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Split schemas whose induction examples have different relation-path shapes.",
    )
    parser.add_argument(
        "--split-cq-clusters",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run one Entity/Relation/Bridge specialist team per CQ cluster.",
    )
    parser.add_argument(
        "--hierarchical-aggregation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reconcile local proposals before the frozen ontology is emitted.",
    )
    parser.add_argument(
        "--retain-parent-incumbents",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Retain every deterministic Q_ind parent schema as an optimizer memo "
            "incumbent; local agents add alternatives and Q_val owns deployment."
        ),
    )
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1400)
    parser.add_argument(
        "--cache-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disallow API calls and fail immediately on any LLM cache miss.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("MULTIAGENT_API_CONCURRENCY", "4")),
        help="Parallel plans; agents inside one plan remain dependency ordered.",
    )
    parser.add_argument(
        "--max-plan-errors",
        type=int,
        default=int(os.environ.get("MULTIAGENT_MAX_PLAN_ERRORS", "3")),
        help=(
            "Maximum isolated plan failures tolerated after API retries. Failed plans are not "
            "execution eligible and their deterministic parent incumbents remain available."
        ),
    )
    parser.add_argument("--min-precision", type=float, default=0.74)
    parser.add_argument("--min-confidence", type=float, default=0.70)
    parser.add_argument("--min-coverage", type=float, default=0.60)
    parser.add_argument(
        "--min-grounded-coverage",
        type=float,
        default=0.60,
        help="Minimum deterministic induction evidence-path coverage.",
    )
    parser.add_argument("--allow-medium-risk", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verifier-keep-override", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    input_path = Path(args.input_path)
    ontology = json.loads(input_path.read_text(encoding="utf-8"))
    if args.selection_policy == "workload_impact":
        sort_key = lambda plan: (
            float(plan.get("precision", 0.0)) * int(plan.get("support", 0)),
            int(plan.get("support", 0)),
            float(plan.get("precision", 0.0)),
        )
    else:
        sort_key = lambda plan: (
            float(plan.get("precision", 0.0)),
            int(plan.get("support", 0)),
        )
    raw_candidate_plans = list(ontology.get("plans", []))
    cluster_expanded_plans = [
        child
        for plan in raw_candidate_plans
        for child in (
            split_plan_by_cq_cluster(plan)
            if args.split_cq_clusters
            else [plan]
        )
    ]
    if args.split_path_shapes:
        expanded_plans = [
            child
            for plan in cluster_expanded_plans
            for child in split_plan_by_path_shape(plan)
        ]
    else:
        expanded_plans = cluster_expanded_plans
    candidate_plans = sorted(expanded_plans, key=sort_key, reverse=True)
    selected_plans = (
        candidate_plans
        if args.max_plans < 0
        else candidate_plans[:max(args.max_plans, 0)]
    )

    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    client = CachedChatClient(
        model=args.llm_name,
        base_url=args.llm_base_url,
        cache_path=output_dir / "multiagent_llm_cache.sqlite",
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        cache_only=args.cache_only,
    )

    def process_plan(index_and_plan):
        idx, plan = index_and_plan
        print(f"[{args.dataset}] refining plan {idx}/{len(selected_plans)} {plan.get('id')}", flush=True)
        try:
            refined, audit = refine_one_plan(
                plan=plan,
                client=client,
                max_examples=args.max_examples_per_plan,
                rounds=args.rounds,
                min_precision=args.min_precision,
                min_confidence=args.min_confidence,
                min_coverage=args.min_coverage,
                allow_medium_risk=args.allow_medium_risk,
                verifier_keep_override=args.verifier_keep_override,
                min_grounded_coverage=args.min_grounded_coverage,
            )
            return idx, refined, audit
        except Exception as exc:
            print(
                f"ERROR: [{args.dataset}] isolated plan failure {plan.get('id')}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            return idx, None, {
                "plan_id": str(plan.get("id", "") or ""),
                "decision": "runtime_error",
                "runtime_error": True,
                "retryable_error": retryable_api_error(exc),
                "drop_reason": {
                    "code": "multiagent_runtime_error",
                    "error_type": type(exc).__name__,
                    "message": str(exc)[:1000],
                },
                "rounds": [],
            }

    indexed_plans = list(enumerate(selected_plans, start=1))
    if max(args.workers, 1) == 1:
        results = [process_plan(item) for item in indexed_plans]
    else:
        results = []
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=max(args.workers, 1))
        futures = {executor.submit(process_plan, item): item[0] for item in indexed_plans}
        try:
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                plan_error_count = sum(
                    bool(item[2].get("runtime_error")) for item in results
                )
                if plan_error_count > max(args.max_plan_errors, 0):
                    for pending in futures:
                        if not pending.done():
                            pending.cancel()
                    raise RuntimeError(
                        f"aborting {args.dataset} multi-agent refinement after "
                        f"{plan_error_count} plan failures; allowed={args.max_plan_errors}. "
                        "Check API/proxy availability and resume with the same artifact root."
                    )
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
    plan_error_count = sum(
        bool(item[2].get("runtime_error")) for item in results
    )
    if plan_error_count > max(args.max_plan_errors, 0):
        raise RuntimeError(
            f"aborting {args.dataset} multi-agent refinement after "
            f"{plan_error_count} plan failures; allowed={args.max_plan_errors}. "
            "Check API/proxy availability and resume with the same artifact root."
        )
    results.sort(key=lambda item: item[0])
    agent_refined_plans = [item[1] for item in results if item[1] is not None]
    for plan in agent_refined_plans:
        if args.retain_parent_incumbents:
            parent_id = str(plan.get("id", "") or "")
            plan["parent_plan_id"] = parent_id
            plan["id"] = f"{parent_id}__agent_repaired"
        plan["execution_eligible"] = True
        plan["execution_eligibility_basis"] = (
            "multiagent_repaired_and_deterministically_reverified"
        )
    audits = [item[2] for item in results]
    passthrough_plans = []
    rejected_passthrough_plans = []
    if args.passthrough_agent_rejected:
        for idx, refined, _audit in results:
            if refined is not None:
                continue
            passthrough = deepcopy(selected_plans[idx - 1])
            passthrough["builder_stage"] = (
                "multiagent_rejected_pending_competency_gate"
            )
            passthrough["execution_eligible"] = False
            rejected_passthrough_plans.append(passthrough)
    if args.passthrough_unprocessed:
        for plan in candidate_plans[len(selected_plans):]:
            passthrough = deepcopy(plan)
            passthrough["builder_stage"] = "workload_induced_pending_competency_gate"
            passthrough["execution_eligible"] = False
            passthrough_plans.append(passthrough)
    parent_incumbents = (
        retain_parent_incumbents(raw_candidate_plans)
        if args.retain_parent_incumbents
        else []
    )
    refined_plans = (
        parent_incumbents
        + agent_refined_plans
        + rejected_passthrough_plans
        + passthrough_plans
    )

    if args.hierarchical_aggregation:
        refined_plans, supervisor_memory = hierarchical_aggregate(
            refined_plans, audits
        )
    else:
        supervisor_memory = {
            "aggregation": "disabled_ablation",
            "accepted_proposals": [
                str(item.get("id", "") or "") for item in refined_plans
            ],
            "rejected_proposals": [],
        }

    refined_plans.sort(key=lambda plan: (float(plan.get("precision", 0.0)), int(plan.get("support", 0))), reverse=True)
    repair_operation_count = sum(
        len(round_payload.get("repair", {}).get("repair_operations", []) or [])
        for audit in audits
        for round_payload in audit.get("rounds", [])
        if isinstance(round_payload, dict)
    )
    diagnostics = {
        "dataset": args.dataset,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "input_path": str(input_path),
        "raw_candidate_plan_count": len(raw_candidate_plans),
        "candidate_plan_count": len(candidate_plans),
        "path_shape_split_plan_count": sum(
            bool(plan.get("path_shape_split")) for plan in candidate_plans
        ),
        "cq_cluster_local_plan_count": sum(
            bool(plan.get("local_cq_cluster_id")) for plan in candidate_plans
        ),
        "processed_plan_count": len(selected_plans),
        "agent_kept_plan_count": len(agent_refined_plans),
        "parent_incumbent_count": len(parent_incumbents),
        "passthrough_plan_count": len(passthrough_plans),
        "agent_rejected_passthrough_count": len(rejected_passthrough_plans),
        "kept_plan_count": len(refined_plans),
        "dropped_plan_count": len(audits) - len(agent_refined_plans),
        "runtime_error_plan_count": sum(
            bool(audit.get("runtime_error")) for audit in audits
        ),
        "max_plan_errors": max(args.max_plan_errors, 0),
        "repair_converged_plan_count": sum(
            bool(audit.get("repair_converged")) for audit in audits
        ),
        "repair_retry_plan_count": sum(
            int(audit.get("repair_iterations", 0) or 0) > 1 for audit in audits
        ),
        "repair_operation_count": repair_operation_count,
        "repair_stalled_plan_count": sum(
            any(bool(round_payload.get("repair_stalled")) for round_payload in audit.get("rounds", []))
            for audit in audits
        ),
        "final_blocking_violation_count": sum(
            len((audit.get("drop_reason") or {}).get("blocking_violations", []) or [])
            for audit in audits
            if isinstance(audit.get("drop_reason"), dict)
        ),
        "typed_counterexample_contract": True,
        "freeform_critic_veto": False,
        "post_repair_verification_required": True,
        "rounds": args.rounds,
        "max_examples_per_plan": args.max_examples_per_plan,
        "split_path_shapes": args.split_path_shapes,
        "split_cq_clusters": args.split_cq_clusters,
        "hierarchical_aggregation": args.hierarchical_aggregation,
        "retain_parent_incumbents": args.retain_parent_incumbents,
        "global_supervisor_memory": supervisor_memory,
        "min_precision": args.min_precision,
        "min_confidence": args.min_confidence,
        "min_coverage": args.min_coverage,
        "min_grounded_coverage": args.min_grounded_coverage,
        "allow_medium_risk": args.allow_medium_risk,
        "verifier_keep_override": args.verifier_keep_override,
        "llm_name": args.llm_name,
        "llm_base_url": args.llm_base_url,
        "workers": max(args.workers, 1),
        "selection_policy": args.selection_policy,
        "passthrough_unprocessed": args.passthrough_unprocessed,
        "passthrough_agent_rejected": args.passthrough_agent_rejected,
        "llm_calls": client.calls,
        "cache_hits": client.cache_hits,
        "network_attempts": client.network_attempts,
        "api_retries": client.api_retries,
        "api_max_attempts": client.max_attempts,
        "cache_only": args.cache_only,
        "prompt_tokens": client.prompt_tokens,
        "completion_tokens": client.completion_tokens,
    }
    refined_ontology = deepcopy(ontology)
    refined_ontology.update({
        "version": 3,
        "kind": "typed_counterexample_repaired_evidence_ontology",
        "builder": "scripts/construct_evidence_ontology.py",
        "plans": refined_plans,
        "global_supervisor_memory": supervisor_memory,
        "diagnostics": {
            **ontology.get("diagnostics", {}),
            "multiagent": diagnostics,
        },
    })

    output_path = output_dir / "query_plan_ontology.json"
    audit_path = output_dir / "multiagent_audit.jsonl"
    diagnostics_path = output_dir / "multiagent_diagnostics.json"
    report_path = output_dir / "multiagent_quality.md"
    output_path.write_text(json.dumps(refined_ontology, ensure_ascii=False, indent=2), encoding="utf-8")
    diagnostics_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    with audit_path.open("w", encoding="utf-8") as handle:
        for audit in audits:
            handle.write(json.dumps(audit, ensure_ascii=False) + "\n")
    write_quality_report(report_path, audits, refined_plans, diagnostics)

    print(f"Wrote {output_path}")
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
