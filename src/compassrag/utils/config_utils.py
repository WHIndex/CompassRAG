import os
from dataclasses import dataclass, field
from typing import (
    Literal,
    Union,
    Optional
)

from .logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class BaseConfig:
    """One and only configuration."""

    llm_name: str = field(
        default="reader-model",
        metadata={"help": "Class name indicating which LLM model to use."}
    )
    llm_base_url: str = field(
        default=None,
        metadata={"help": "Base URL for the chat-completion endpoint."}
    )
    embedding_base_url: str = field(
        default=None,
        metadata={"help": "Base URL for a compatible embedding endpoint."}
    )
    azure_endpoint: str = field(
        default=None,
        metadata={"help": "Optional managed chat-completion endpoint URI."}
    )
    azure_embedding_endpoint: str = field(
        default=None,
        metadata={"help": "Optional managed embedding endpoint URI."}
    )
    max_new_tokens: Union[None, int] = field(
        default=2048,
        metadata={"help": "Max new tokens to generate in each inference."}
    )
    num_gen_choices: int = field(
        default=1,
        metadata={"help": "How many chat completion choices to generate for each input message."}
    )
    seed: Union[None, int] = field(
        default=None,
        metadata={"help": "Random seed."}
    )
    temperature: float = field(
        default=0,
        metadata={"help": "Temperature for sampling in each inference."}
    )
    response_format: Union[dict, None] = field(
        default_factory=lambda: { "type": "json_object" },
        metadata={"help": "Specifying the format that the model must output."}
    )


    max_retry_attempts: int = field(
        default=5,
        metadata={"help": "Max number of retry attempts for an asynchronous API calling."}
    )

    force_openie_from_scratch: bool = field(
        default=False,
        metadata={"help": "If set to True, will ignore all existing openie files and rebuild them from scratch."}
    )


    force_index_from_scratch: bool = field(
        default=False,
        metadata={"help": "If set to True, will ignore all existing storage files and graph data and will rebuild from scratch."}
    )
    rerank_dspy_file_path: str = field(
        default=None,
        metadata={"help": "Path to the rerank dspy file."}
    )
    passage_node_weight: float = field(
        default=0.05,
        metadata={"help": "Multiplicative factor that modified the passage node weights in PPR."}
    )
    save_openie: bool = field(
        default=True,
        metadata={"help": "If set to True, will save the OpenIE model to disk."}
    )


    text_preprocessor_class_name: str = field(
        default="TextPreprocessor",
        metadata={"help": "Name of the text-based preprocessor to use in preprocessing."}
    )
    preprocess_encoder_name: str = field(
        default="tokenizer-model",
        metadata={"help": "Name of the encoder to use in preprocessing (currently implemented specifically for doc chunking)."}
    )
    preprocess_chunk_overlap_token_size: int = field(
        default=128,
        metadata={"help": "Number of overlap tokens between neighbouring chunks."}
    )
    preprocess_chunk_max_token_size: int = field(
        default=None,
        metadata={"help": "Max number of tokens each chunk can contain. If set to None, the whole doc will treated as a single chunk."}
    )
    preprocess_chunk_func: Literal["by_token", "by_word"] = field(default='by_token')



    information_extraction_model_name: Literal["openie_remote_client", ] = field(
        default="openie_remote_client",
        metadata={"help": "Class name indicating which information extraction model to use."}
    )
    openie_mode: Literal["offline", "online"] = field(
        default="online",
        metadata={"help": "Mode of the OpenIE model to use."}
    )
    skip_graph: bool = field(
        default=False,
        metadata={"help": "Whether to skip graph construction or not. Set it to be true when running vllm offline indexing for the first time."}
    )



    embedding_model_name: str = field(
        default="nvidia/NV-Embed-v2",
        metadata={"help": "Class name indicating which embedding model to use."}
    )
    embedding_batch_size: int = field(
        default=16,
        metadata={"help": "Batch size of calling embedding model."}
    )
    embedding_return_as_normalized: bool = field(
        default=True,
        metadata={"help": "Whether to normalize encoded embeddings not."}
    )
    embedding_max_seq_len: int = field(
        default=2048,
        metadata={"help": "Max sequence length for the embedding model."}
    )
    embedding_model_dtype: Literal["float16", "float32", "bfloat16", "auto"] = field(
        default="auto",
        metadata={"help": "Data type for local embedding model."}
    )




    synonymy_edge_topk: int = field(
        default=2047,
        metadata={"help": "k for knn retrieval in buiding synonymy edges."}
    )
    synonymy_edge_query_batch_size: int = field(
        default=1000,
        metadata={"help": "Batch size for query embeddings for knn retrieval in buiding synonymy edges."}
    )
    synonymy_edge_key_batch_size: int = field(
        default=10000,
        metadata={"help": "Batch size for key embeddings for knn retrieval in buiding synonymy edges."}
    )
    synonymy_edge_sim_threshold: float = field(
        default=0.8,
        metadata={"help": "Similarity threshold to include candidate synonymy nodes."}
    )
    is_directed_graph: bool = field(
        default=False,
        metadata={"help": "Whether the graph is directed or not."}
    )




    linking_top_k: int = field(
        default=5,
        metadata={"help": "The number of linked nodes at each retrieval step"}
    )
    retrieval_top_k: int = field(
        default=200,
        metadata={"help": "Retrieving k documents at each step"}
    )
    damping: float = field(
        default=0.5,
        metadata={"help": "Damping factor for ppr algorithm."}
    )
    enable_adaptive_base_access_router: bool = field(
        default=False,
        metadata={"help": "Select a complete PPR, dense, or sparse base ranking per query."}
    )
    adaptive_base_access_collect_only: bool = field(
        default=False,
        metadata={"help": "Materialize base access alternatives and replay labels while preserving PPR."}
    )
    adaptive_base_access_model_path: str = field(
        default="",
        metadata={"help": "JSON gain/risk model used by the adaptive base access router."}
    )
    adaptive_base_access_sparse_top_k: int = field(
        default=200,
        metadata={"help": "Number of sparse candidates materialized before dense backoff."}
    )
    enable_ontology: bool = field(
        default=False,
        metadata={"help": "Whether to build and use a Saturn-inspired graph ontology for query expansion."}
    )
    force_ontology_from_scratch: bool = field(
        default=False,
        metadata={"help": "If set to True, rebuild the graph ontology even if a cached ontology exists."}
    )
    ontology_query_top_k: int = field(
        default=8,
        metadata={"help": "Number of ontology concepts/entities/relations to keep for each query."}
    )
    ontology_max_query_terms: int = field(
        default=24,
        metadata={"help": "Maximum related ontology terms appended to each query."}
    )
    ontology_embedding_mode: str = field(
        default="passage",
        metadata={"help": "Where to apply ontology-expanded query embeddings: none, fact, passage, or both."}
    )
    ontology_embedding_weight: float = field(
        default=1.0,
        metadata={"help": "Interpolation weight for ontology-expanded query embeddings. 1.0 uses only ontology-expanded text; 0.0 uses only the original query."}
    )
    ontology_fact_boost_weight: float = field(
        default=0.0,
        metadata={"help": "Interpolation weight for ontology-guided fact score boosting before fact reranking and PPR."}
    )
    ontology_fact_boost_max_facts_per_hint: int = field(
        default=500,
        metadata={"help": "Skip ontology entity/relation hints that match more than this many facts; <=0 disables the high-frequency hint guard."}
    )
    enable_query_aware_fact_expansion: bool = field(
        default=False,
        metadata={"help": "Whether to add query-aware bridge facts after fact reranking and before graph PPR."}
    )
    query_aware_candidate_top_k: int = field(
        default=80,
        metadata={"help": "Number of dense fact candidates considered by query-aware bridge fact expansion."}
    )
    query_aware_max_added_facts: int = field(
        default=5,
        metadata={"help": "Maximum bridge facts added by query-aware fact expansion."}
    )
    query_aware_max_facts_per_entity: int = field(
        default=120,
        metadata={"help": "Skip local entity neighborhoods larger than this during query-aware fact expansion; <=0 disables the guard."}
    )
    query_aware_score_weight: float = field(
        default=0.20,
        metadata={"help": "Weight used to raise selected bridge fact scores toward the current anchor fact score."}
    )
    enable_query_local_ontology: bool = field(
        default=False,
        metadata={"help": "Whether to dynamically build a query-local multi-agent ontology over top retrieved facts/passages before PPR."}
    )
    query_local_ontology_candidate_top_k: int = field(
        default=24,
        metadata={"help": "Number of dense top facts shown to the query-local ontology agents."}
    )
    query_local_ontology_top_passages: int = field(
        default=2,
        metadata={"help": "Number of dense top passages shown to the query-local ontology agents."}
    )
    query_local_ontology_passage_chars: int = field(
        default=600,
        metadata={"help": "Maximum characters kept for each passage snippet in query-local ontology prompts."}
    )
    query_local_ontology_max_added_facts: int = field(
        default=5,
        metadata={"help": "Maximum verifier-accepted bridge facts injected into PPR for each query."}
    )
    query_local_ontology_score_weight: float = field(
        default=0.60,
        metadata={"help": "Weight used to raise verifier-accepted bridge fact scores toward the current anchor fact score."}
    )
    query_local_ontology_max_completion_tokens: int = field(
        default=512,
        metadata={"help": "Maximum completion tokens for each query-local ontology agent call."}
    )
    enable_query_ontology_prior: bool = field(
        default=False,
        metadata={"help": "Whether to use a reusable historical-query ontology prior for lightweight online fact-score boosting."}
    )
    query_ontology_prior_path: str = field(
        default=None,
        metadata={"help": "Path to a historical-query ontology prior JSON. If unset, uses query_ontology_priors.json in the CompassRAG working directory."}
    )
    query_ontology_prior_boost_weight: float = field(
        default=0.12,
        metadata={"help": "Interpolation weight for historical-query ontology prior fact-score boosting."}
    )
    query_ontology_prior_top_k: int = field(
        default=8,
        metadata={"help": "Number of matched prior relation/bridge/entity hints used for each query."}
    )
    query_ontology_prior_max_boost_facts: int = field(
        default=80,
        metadata={"help": "Maximum facts whose scores can be boosted by the query ontology prior; <=0 disables this cap."}
    )
    query_ontology_schema_mode: str = field(
        default="gated",
        metadata={"help": "Historical-query ontology prior mode: gated, hints_only, templates_only, or path_gated."}
    )
    query_ontology_template_candidate_top_k: int = field(
        default=500,
        metadata={"help": "Only the current top-k fact candidates are eligible for bridge-template boosts."}
    )
    query_ontology_template_max_boost_facts: int = field(
        default=20,
        metadata={"help": "Maximum facts boosted by bridge templates before the global prior boost cap is applied."}
    )
    query_ontology_hint_candidate_top_k: int = field(
        default=500,
        metadata={"help": "Only the current top-k fact candidates are eligible for relation/entity hint boosts."}
    )
    query_ontology_require_query_anchor: bool = field(
        default=True,
        metadata={"help": "Whether prior boosts require a concrete entity/token overlap with the current query."}
    )
    enable_query_ontology_prior_ppr_injection: bool = field(
        default=False,
        metadata={"help": "Whether boosted query-ontology-prior facts should be injected as additional PPR seed facts."}
    )
    query_ontology_prior_ppr_top_k: int = field(
        default=4,
        metadata={"help": "Maximum boosted query-ontology-prior facts added as PPR seed facts per query."}
    )
    query_ontology_prior_ppr_score_weight: float = field(
        default=0.50,
        metadata={"help": "Weight used to raise prior-injected PPR fact scores toward the current anchor fact score."}
    )
    query_ontology_prior_ppr_expand_graph_paths: bool = field(
        default=True,
        metadata={"help": "Whether prior PPR injection should instantiate matched schemas by expanding graph-neighbor facts from selected anchor facts."}
    )
    query_ontology_prior_ppr_max_facts_per_entity: int = field(
        default=120,
        metadata={"help": "Skip graph-neighbor path expansion for entities with more than this many incident facts; <=0 disables this guard."}
    )
    query_ontology_prior_ppr_intent_gate: bool = field(
        default=False,
        metadata={"help": "If true, inject prior PPR path facts only when their relation family matches the query intent."}
    )
    enable_query_ontology_prior_path_instantiation: bool = field(
        default=False,
        metadata={"help": "Whether to actively instantiate matched query-ontology templates from query anchor entities in the graph."}
    )
    query_ontology_prior_path_instantiation_top_k: int = field(
        default=8,
        metadata={"help": "Maximum graph facts added per query from active ontology path instantiation."}
    )
    query_ontology_prior_path_instantiation_max_anchor_entities: int = field(
        default=16,
        metadata={"help": "Maximum query-matched graph entities used as active ontology path anchors."}
    )
    query_ontology_prior_path_instantiation_max_paths: int = field(
        default=80,
        metadata={"help": "Maximum instantiated ontology paths retained per query before fact injection."}
    )
    query_ontology_prior_path_instantiation_min_template_precision: float = field(
        default=0.45,
        metadata={"help": "Minimum historical path precision required for a bridge template to be actively instantiated."}
    )
    query_ontology_prior_path_instantiation_max_facts_per_entity: int = field(
        default=120,
        metadata={"help": "Skip active path expansion for graph entities with more incident facts than this; <=0 disables this guard."}
    )
    enable_query_ontology_prior_passage_lift: bool = field(
        default=False,
        metadata={"help": "Whether ontology-prior injected facts should lightly lift their source passages after PPR ranking."}
    )
    query_ontology_prior_passage_lift_weight: float = field(
        default=0.03,
        metadata={"help": "Late-stage normalized score interpolation weight for ontology-prior source passage lift."}
    )
    query_ontology_prior_passage_lift_top_k: int = field(
        default=20,
        metadata={"help": "Maximum source passages lifted per query from ontology-prior injected facts."}
    )
    query_ontology_prior_passage_lift_use_entity_fallback: bool = field(
        default=False,
        metadata={"help": "If true, lift passages linked to injected fact entities when exact fact-source passages are unavailable."}
    )
    query_ontology_prior_trace_path: str = field(
        default=None,
        metadata={"help": "Optional JSONL path for per-query query-ontology-prior injection diagnostics."}
    )
    enable_query_plan_ontology: bool = field(
        default=False,
        metadata={"help": "Whether to use query-conditioned micro-ontology plans for late passage reranking."}
    )
    query_plan_ontology_path: str = field(
        default=None,
        metadata={"help": "Path to query_plan_ontology.json. If unset, uses query_plan_ontology.json in the CompassRAG working directory."}
    )
    query_plan_ontology_compiler_sidecar_path: str = field(
        default=None,
        metadata={"help": "Optional proposal-only ontology used by the executable compiler while the primary ontology remains the retrieval incumbent."}
    )
    query_plan_relation_realization_path: str = field(
        default=None,
        metadata={"help": "Optional path to relation_realization_map.json for mapping query-plan conceptual relations to OpenIE relation surfaces."}
    )
    query_plan_ontology_compiler_sidecar_relation_realization_path: str = field(
        default=None,
        metadata={"help": "Optional relation realization catalog paired with the proposal-only compiler sidecar ontology."}
    )
    query_plan_use_relation_realization: bool = field(
        default=True,
        metadata={"help": "If True, query-plan execution can match OpenIE relation surfaces learned in relation_realization_map.json."}
    )
    query_plan_relation_realization_allow_family_fallback: bool = field(
        default=False,
        metadata={"help": "If True, relation realization can match by realized relation family. Defaults to False to avoid broad OpenIE relation drift."}
    )
    query_plan_relation_realization_require_ontology_hash_match: bool = field(
        default=False,
        metadata={"help": "Fail closed unless relation realizations were built from the active ontology file."}
    )
    query_plan_ontology_enable_schema_linked_router: bool = field(
        default=False,
        metadata={"help": "Route over all leaf schemas using exact query relation coverage."}
    )
    query_plan_ontology_enable_schema_linked_compiler_sidecar: bool = field(
        default=False,
        metadata={"help": "Expose schema-linked plans to the executable compiler without replacing incumbent retrieval plans."}
    )
    query_plan_ontology_matching_mode: str = field(
        default="learned_structural",
        metadata={"help": "Ontology matching ablation mode: learned_structural, semantic_only, or direct_query."}
    )
    query_plan_ontology_top_k: int = field(
        default=6,
        metadata={"help": "Maximum query-plan templates matched per query."}
    )
    query_plan_ontology_min_confidence: float = field(
        default=0.72,
        metadata={"help": "Minimum plan confidence for online matching."}
    )
    query_plan_ontology_max_paths: int = field(
        default=60,
        metadata={"help": "Maximum executable graph paths retained per query."}
    )
    query_plan_ontology_max_branch_facts: int = field(
        default=80,
        metadata={"help": "Skip plan execution branches whose current entity has more incident facts than this; <=0 disables the guard."}
    )
    query_plan_ontology_signature_match_mode: str = field(
        default="strict_then_soft",
        metadata={"help": "Relation signature matching mode for query-plan graph execution: strict, soft, family, or strict_then_soft."}
    )
    query_plan_ontology_passage_lift_weight: float = field(
        default=0.04,
        metadata={"help": "Late-stage normalized score interpolation weight for query-plan source passages."}
    )
    query_plan_ontology_passage_rerank_mode: str = field(
        default="interpolate",
        metadata={"help": "How query-plan passage evidence changes passage scores: interpolate, boost, or adaptive_promote."}
    )
    query_plan_ontology_passage_top_k: int = field(
        default=10,
        metadata={"help": "Maximum query-plan source passages lifted per query."}
    )
    query_plan_ontology_passage_reranker_high_confidence_threshold: float = field(
        default=0.70,
        metadata={"help": "Adaptive passage reranker promotion threshold for ontology-supported passages."}
    )
    query_plan_ontology_passage_reranker_promote_top_k: int = field(
        default=1,
        metadata={"help": "Maximum high-confidence ontology-supported passages promoted per query by the adaptive passage reranker."}
    )
    query_plan_ontology_passage_reranker_promote_to_rank: int = field(
        default=2,
        metadata={"help": "Target rank slot for adaptive ontology passage promotion; 2 preserves the current top passage and inserts after it."}
    )
    query_plan_ontology_passage_reranker_preserve_top_k: int = field(
        default=1,
        metadata={"help": "Number of highest baseline-ranked passages preserved before adaptive ontology promotions are inserted."}
    )
    query_plan_ontology_passage_reranker_enable_promotion_verifier: bool = field(
        default=False,
        metadata={"help": "If True, verify ontology-supported passage promotion before inserting a passage near the top."}
    )
    query_plan_ontology_passage_reranker_verifier_min_confidence: float = field(
        default=0.85,
        metadata={"help": "Minimum adaptive passage confidence required by the conservative promotion verifier."}
    )
    query_plan_ontology_passage_reranker_verifier_max_original_rank: int = field(
        default=5,
        metadata={"help": "Maximum baseline rank eligible for conservative ontology passage promotion; <=0 disables this verifier bound."}
    )
    query_plan_ontology_passage_promotion_policy_path: str = field(
        default="",
        metadata={"help": "Optional learned query-plan passage promotion policy JSON path."}
    )
    query_plan_ontology_passage_promotion_policy_mode: str = field(
        default="off",
        metadata={"help": "Learned promotion policy mode: off, gate, or override."}
    )
    query_plan_ontology_passage_promotion_policy_enable_coverage_gate: bool = field(
        default=False,
        metadata={"help": "If True, policy-approved passage promotions must add a new ontology evidence role before entering the top window."}
    )
    query_plan_ontology_passage_promotion_policy_coverage_top_k: int = field(
        default=2,
        metadata={"help": "Top-k evidence window used by the promotion policy coverage gate."}
    )
    query_plan_ontology_enable_evidence_context_builder: bool = field(
        default=False,
        metadata={"help": "If True, assemble a QA-only context from baseline retrieval plus complementary query-plan evidence passages."}
    )
    query_plan_ontology_evidence_context_builder_mode: str = field(
        default="qa_only",
        metadata={"help": "Evidence context builder mode: qa_only keeps retrieval metrics unchanged; retrieval_and_qa also changes returned retrieval docs."}
    )
    query_plan_ontology_evidence_context_builder_preserve_top_k: int = field(
        default=1,
        metadata={"help": "Number of baseline top passages preserved before adding query-plan evidence passages to the QA context."}
    )
    query_plan_ontology_evidence_context_builder_max_evidence_passages: int = field(
        default=2,
        metadata={"help": "Maximum complementary query-plan evidence passages inserted into the QA context."}
    )
    query_plan_ontology_evidence_context_builder_max_original_rank: int = field(
        default=20,
        metadata={"help": "Maximum original retrieval rank eligible for QA context insertion; <=0 disables this bound."}
    )
    query_plan_ontology_evidence_context_builder_min_candidate_confidence: float = field(
        default=0.0,
        metadata={"help": "Minimum adaptive passage confidence required for QA context insertion."}
    )
    query_plan_ontology_evidence_context_builder_require_new_evidence_role: bool = field(
        default=True,
        metadata={"help": "If True, inserted QA context passages must add a new query-plan role, anchor, plan, or relation signature."}
    )
    query_plan_ontology_reader_intervention_require_atomic_commit: bool = field(
        default=False,
        metadata={"help": "If True, uncommitted ontology candidates cannot reorder reader context or inject concrete QA evidence."}
    )
    query_plan_ontology_evidence_context_builder_enable_chain_guidance: bool = field(
        default=False,
        metadata={"help": "If True, add ontology-grounded evidence-chain and answer-form guidance to the QA prompt after assembling the QA context."}
    )
    query_plan_ontology_evidence_context_builder_enable_logical_plan_guidance: bool = field(
        default=False,
        metadata={"help": "If True, compile an admitted logical evidence plan into a passage-grounded verification order for the QA reader."}
    )
    query_plan_ontology_evidence_context_builder_logical_plan_require_commit: bool = field(
        default=True,
        metadata={"help": "If True, expose logical-plan guidance only when the complete-plan optimizer atomically commits that plan."}
    )
    query_plan_ontology_evidence_context_builder_max_chain_items: int = field(
        default=4,
        metadata={"help": "Maximum context-builder evidence-chain bullets added to the QA prompt."}
    )
    query_plan_ontology_evidence_context_builder_chain_guidance_mode: str = field(
        default="always",
        metadata={"help": "Evidence-chain guidance mode: always preserves historical behavior; gated applies availability guards; operator_safe additionally requires committed, operator-compatible, terminal-consistent chains."}
    )
    query_plan_ontology_evidence_context_builder_allow_hint_only: bool = field(
        default=True,
        metadata={"help": "If False, do not add answer-form-only chain guidance when no concrete evidence chain is available."}
    )
    query_plan_ontology_evidence_context_builder_enable_comparative_branches: bool = field(
        default=False,
        metadata={"help": "If True, comparative questions require two distinct evidence branches before chain guidance is injected."}
    )
    query_plan_ontology_enable_state_retriever: bool = field(
        default=False,
        metadata={"help": "If True, use query-plan state coverage to reassemble the retrieval top-k evidence window."}
    )
    query_plan_ontology_state_retriever_top_k: int = field(
        default=5,
        metadata={"help": "Top-k evidence window optimized by the query-plan state retriever."}
    )
    query_plan_ontology_state_retriever_candidate_top_k: int = field(
        default=50,
        metadata={"help": "Number of baseline-ranked passages considered by the query-plan state retriever."}
    )
    query_plan_ontology_state_retriever_preserve_top_k: int = field(
        default=1,
        metadata={"help": "Number of highest baseline passages preserved before state-coverage selection."}
    )
    query_plan_ontology_state_retriever_max_original_rank: int = field(
        default=50,
        metadata={"help": "Maximum original rank for ontology-supported passages considered by the state retriever; <=0 disables this bound."}
    )
    query_plan_ontology_state_retriever_baseline_weight: float = field(
        default=0.45,
        metadata={"help": "Weight of original retrieval rank/score inside the state retriever."}
    )
    query_plan_ontology_state_retriever_coverage_weight: float = field(
        default=1.00,
        metadata={"help": "Weight of new query-plan state slot coverage inside the state retriever."}
    )
    query_plan_ontology_state_retriever_evidence_weight: float = field(
        default=0.65,
        metadata={"help": "Weight of ontology evidence strength inside the state retriever."}
    )
    query_plan_ontology_state_retriever_redundancy_penalty: float = field(
        default=0.25,
        metadata={"help": "Penalty for selecting passages that repeat already covered query-plan state slots."}
    )
    query_plan_ontology_state_retriever_min_evidence_score: float = field(
        default=0.0,
        metadata={"help": "Minimum ontology evidence score required before an ontology-supported passage can be selected by the state retriever."}
    )
    query_plan_ontology_state_retriever_require_new_slot: bool = field(
        default=False,
        metadata={"help": "If True, ontology-supported passages must add a new state slot to be selected by the state retriever."}
    )
    query_plan_ontology_state_retriever_max_ontology_passages: int = field(
        default=1,
        metadata={"help": "Maximum ontology-supported passages inserted into the state-retrieved top-k window."}
    )
    query_plan_ontology_state_retriever_require_policy_approval: bool = field(
        default=True,
        metadata={"help": "If True, state retriever uses ontology passage signals only when the learned promotion policy approves them, unless a high-confidence fallback applies."}
    )
    query_plan_ontology_state_retriever_policy_fallback_min_confidence: float = field(
        default=0.85,
        metadata={"help": "Minimum passage-reranker confidence for state retriever fallback when no policy decision exists."}
    )
    query_plan_ontology_state_retriever_policy_fallback_max_original_rank: int = field(
        default=5,
        metadata={"help": "Maximum original rank for state retriever high-confidence fallback when no policy decision exists; <=0 disables this bound."}
    )
    query_plan_ontology_state_retriever_require_structural_slot: bool = field(
        default=True,
        metadata={"help": "If True, state retriever only promotes ontology passages that add a role, relation, plan, or verified path slot; terminal/anchor-only gains are treated as too weak."}
    )
    query_plan_ontology_state_policy_path: str = field(
        default="",
        metadata={"help": "Optional learned state retriever policy JSON path for query-plan ontology passage insertion."}
    )
    query_plan_ontology_state_policy_mode: str = field(
        default="off",
        metadata={"help": "State retriever policy mode: off, gate, or override. Override lets learned state rules replace the passage-promotion policy gate."}
    )
    query_plan_ontology_enable_subgraph_retriever: bool = field(
        default=False,
        metadata={"help": "If True, use query-plan ontology anchors and relation families to expand a local fact subgraph and append source passages to the evidence candidate pool."}
    )
    query_plan_ontology_subgraph_candidate_fact_top_k: int = field(
        default=240,
        metadata={"help": "Maximum ontology-subgraph facts considered per query before mapping facts back to source passages."}
    )
    query_plan_ontology_subgraph_max_facts_per_anchor: int = field(
        default=120,
        metadata={"help": "Maximum fact-graph neighbors scanned for each concrete query anchor in the ontology subgraph retriever."}
    )
    query_plan_ontology_subgraph_max_facts_per_relation: int = field(
        default=160,
        metadata={"help": "Maximum fact-graph facts scanned for each query-plan relation family in the ontology subgraph retriever."}
    )
    query_plan_ontology_subgraph_max_source_passages_per_fact: int = field(
        default=3,
        metadata={"help": "Maximum source passages exposed for each ontology-subgraph candidate fact."}
    )
    query_plan_ontology_subgraph_max_added_passages: int = field(
        default=24,
        metadata={"help": "Maximum ontology-subgraph source passages appended to the downstream evidence candidate pool."}
    )
    query_plan_ontology_subgraph_max_candidate_pool_rank: int = field(
        default=260,
        metadata={"help": "Synthetic rank cap assigned to appended ontology-subgraph passages so evidence-set selection can inspect them."}
    )
    query_plan_ontology_subgraph_min_fact_score: float = field(
        default=0.10,
        metadata={"help": "Minimum combined query/ontology score for a fact to become an ontology-subgraph candidate."}
    )
    query_plan_ontology_subgraph_min_passage_score: float = field(
        default=0.18,
        metadata={"help": "Minimum aggregated ontology-subgraph score for a source passage to be exposed downstream."}
    )
    query_plan_ontology_subgraph_relation_weight: float = field(
        default=0.34,
        metadata={"help": "Weight of query-plan relation-family match in ontology-subgraph fact scoring."}
    )
    query_plan_ontology_subgraph_anchor_weight: float = field(
        default=0.30,
        metadata={"help": "Weight of concrete anchor overlap in ontology-subgraph fact scoring."}
    )
    query_plan_ontology_subgraph_path_neighbor_weight: float = field(
        default=0.22,
        metadata={"help": "Weight of executable-path seed or neighbor evidence in ontology-subgraph fact scoring."}
    )
    query_plan_ontology_subgraph_query_fact_weight: float = field(
        default=0.14,
        metadata={"help": "Weight of CompassRAG query-fact relevance in ontology-subgraph fact scoring."}
    )
    query_plan_ontology_subgraph_use_entity_fallback: bool = field(
        default=True,
        metadata={"help": "If True, fall back to entity-neighbor source passages when direct fact source passages are unavailable."}
    )
    query_plan_ontology_subgraph_append_score_margin: float = field(
        default=1e-7,
        metadata={"help": "Tiny decreasing score margin used when appending ontology-subgraph passages after the original retrieval list."}
    )
    query_plan_ontology_subgraph_min_original_rank: int = field(
        default=1,
        metadata={"help": "Minimum original retrieval rank for ontology-subgraph passages; use >5 to only add candidates that can improve Recall@5."}
    )
    query_plan_ontology_subgraph_max_original_rank: int = field(
        default=0,
        metadata={"help": "Maximum original retrieval rank for ontology-subgraph passages; <=0 disables this bound."}
    )
    query_plan_ontology_subgraph_require_strong_fact_gate: bool = field(
        default=False,
        metadata={"help": "If True, candidate facts must be path seeds, anchor+relation matches, or high query-score relation-only matches."}
    )
    query_plan_ontology_subgraph_relation_only_min_query_fact_score: float = field(
        default=0.35,
        metadata={"help": "Minimum query-fact score required for relation-only ontology-subgraph facts when the strong fact gate is enabled."}
    )
    query_plan_ontology_enable_support_evidence_retriever: bool = field(
        default=False,
        metadata={"help": "If True, issue query-plan ontology dense support probes and expose verified passages to evidence-set selection."}
    )
    query_plan_ontology_support_evidence_dense_top_k: int = field(
        default=30,
        metadata={"help": "Dense top-k passages considered for each query-plan support evidence probe."}
    )
    query_plan_ontology_support_evidence_max_probes: int = field(
        default=10,
        metadata={"help": "Maximum ontology-derived support evidence probes issued per query."}
    )
    query_plan_ontology_support_evidence_max_candidates: int = field(
        default=4,
        metadata={"help": "Maximum verifier-approved support evidence passages exposed downstream."}
    )
    query_plan_ontology_support_evidence_min_confidence: float = field(
        default=0.70,
        metadata={"help": "Minimum confidence required before a dense-retrieved support passage is accepted."}
    )
    query_plan_ontology_support_evidence_preserve_top_k: int = field(
        default=2,
        metadata={"help": "Number of top baseline passages protected from support evidence insertion."}
    )
    query_plan_ontology_support_evidence_synthetic_original_rank: int = field(
        default=40,
        metadata={"help": "Synthetic original rank assigned to support evidence passages not present in the baseline retrieval list."}
    )
    query_plan_ontology_support_evidence_require_anchor_hit: bool = field(
        default=True,
        metadata={"help": "If True, support evidence candidates with concrete anchors must mention the anchor."}
    )
    query_plan_ontology_support_evidence_require_relation_or_answer_hit: bool = field(
        default=True,
        metadata={"help": "If True, support evidence candidates must mention relation, terminal, or answer-type clues."}
    )
    query_plan_ontology_support_evidence_candidate_only: bool = field(
        default=True,
        metadata={"help": "If True, support evidence retrieval only appends candidates and lets evidence-set selection decide ranking."}
    )
    query_plan_ontology_support_evidence_score_margin: float = field(
        default=1e-7,
        metadata={"help": "Small score margin used when appending support evidence candidates after the original retrieval list."}
    )
    query_plan_ontology_support_evidence_probe_policy_path: str = field(
        default="",
        metadata={"help": "Optional path to a trace-feedback support probe policy JSON."}
    )
    query_plan_ontology_support_evidence_probe_policy_mode: str = field(
        default="off",
        metadata={"help": "How to use support probe policy: off or filter."}
    )
    query_plan_ontology_enable_schema_route_retriever: bool = field(
        default=False,
        metadata={"help": "If True, decompose query-plan schema into route-specific dense retrieval probes."}
    )
    query_plan_ontology_schema_route_dense_top_k: int = field(
        default=80,
        metadata={"help": "Dense top-k passages considered for each schema-route probe."}
    )
    query_plan_ontology_schema_route_max_subqueries: int = field(
        default=8,
        metadata={"help": "Maximum schema-guided route subqueries issued per query."}
    )
    query_plan_ontology_schema_route_max_candidates: int = field(
        default=6,
        metadata={"help": "Maximum verified schema-route candidates exposed downstream."}
    )
    query_plan_ontology_schema_route_min_confidence: float = field(
        default=0.58,
        metadata={"help": "Minimum confidence required before accepting a schema-route candidate."}
    )
    query_plan_ontology_schema_route_preserve_top_k: int = field(
        default=2,
        metadata={"help": "Number of top baseline passages protected from schema-route insertion."}
    )
    query_plan_ontology_schema_route_synthetic_original_rank: int = field(
        default=35,
        metadata={"help": "Synthetic original rank used so schema-route candidates enter evidence-set selection."}
    )
    query_plan_ontology_schema_route_candidate_only: bool = field(
        default=True,
        metadata={"help": "If True, schema-route retrieval only appends candidates for evidence-set selection."}
    )
    query_plan_ontology_schema_route_score_margin: float = field(
        default=1e-7,
        metadata={"help": "Small score margin used when appending schema-route candidates."}
    )
    query_plan_ontology_schema_route_require_anchor_hit: bool = field(
        default=True,
        metadata={"help": "If True, anchored schema routes require the candidate passage to mention the anchor."}
    )
    query_plan_ontology_schema_route_require_route_signal: bool = field(
        default=True,
        metadata={"help": "If True, schema-route candidates must match relation, terminal, answer, or route signal."}
    )
    query_plan_ontology_schema_route_enable_community_route: bool = field(
        default=True,
        metadata={"help": "If True, add community/topic-style schema route probes for weakly anchored queries."}
    )
    query_plan_ontology_schema_route_enable_dfs_route: bool = field(
        default=True,
        metadata={"help": "If True, multi-relation schema paths are treated as DFS-style route probes."}
    )
    query_plan_ontology_schema_route_policy_model_path: str = field(
        default="",
        metadata={"help": "Optional learned utility policy for schema-route candidates."}
    )
    query_plan_ontology_schema_route_policy_mode: str = field(
        default="off",
        metadata={"help": "How to use the schema-route utility policy: off, score, or filter."}
    )
    query_plan_ontology_schema_route_policy_min_score: float = field(
        default=0.55,
        metadata={"help": "Minimum schema-route policy score for filter mode."}
    )
    query_plan_ontology_schema_route_policy_reject_below: float = field(
        default=0.0,
        metadata={"help": "Reject schema-route candidates whose policy score is below this value."}
    )
    query_plan_ontology_schema_route_policy_score_weight: float = field(
        default=0.12,
        metadata={"help": "Score interpolation weight for schema-route policy-approved candidates."}
    )
    query_plan_ontology_enable_schema_slot_recall_expander: bool = field(
        default=False,
        metadata={"help": "If True, issue schema-slot constrained dense probes to add missing-gold recall candidates before evidence-set selection."}
    )
    query_plan_ontology_schema_slot_recall_dense_top_k: int = field(
        default=80,
        metadata={"help": "Dense top-k passages considered for each schema-slot recall probe."}
    )
    query_plan_ontology_schema_slot_recall_max_probes: int = field(
        default=12,
        metadata={"help": "Maximum schema-slot recall probes issued per query."}
    )
    query_plan_ontology_schema_slot_recall_max_candidates: int = field(
        default=10,
        metadata={"help": "Maximum schema-slot recall candidates exposed downstream."}
    )
    query_plan_ontology_schema_slot_recall_min_confidence: float = field(
        default=0.48,
        metadata={"help": "Minimum confidence required before accepting a schema-slot recall candidate."}
    )
    query_plan_ontology_schema_slot_recall_synthetic_original_rank: int = field(
        default=90,
        metadata={"help": "Synthetic original rank used so schema-slot recall candidates enter the evidence-set candidate pool."}
    )
    query_plan_ontology_schema_slot_recall_require_anchor_or_relation: bool = field(
        default=True,
        metadata={"help": "If True, schema-slot recall candidates must match an anchor, relation token, terminal, or answer-type cue."}
    )
    query_plan_ontology_schema_slot_recall_include_original_query: bool = field(
        default=True,
        metadata={"help": "If True, include original-query augmented probes in the schema-slot recall expander."}
    )
    query_plan_ontology_schema_slot_recall_enable_multi_probe: bool = field(
        default=False,
        metadata={"help": "If True, build multiple ontology-slot dense probes targeted at missing anchor/relation/terminal/answer slots."}
    )
    query_plan_ontology_schema_slot_recall_target_datasets: str = field(
        default="",
        metadata={"help": "Comma-separated dataset names where schema-slot multi-probe recall is enabled; empty means all datasets."}
    )
    query_plan_ontology_schema_slot_recall_max_missing_slot_terms: int = field(
        default=8,
        metadata={"help": "Maximum missing ontology slot terms used to prioritize multi-probe recall templates."}
    )
    query_plan_ontology_schema_slot_recall_min_relation_overlap: float = field(
        default=0.20,
        metadata={"help": "Minimum relation-token overlap required for weak multi-probe candidates when no anchor/terminal/answer cue is present."}
    )
    query_plan_ontology_schema_slot_recall_answer_type_boost: float = field(
        default=0.08,
        metadata={"help": "Confidence boost for schema-slot recall candidates that match answer-type cues."}
    )
    query_plan_ontology_enable_bind_probe_join: bool = field(
        default=False,
        metadata={"help": "If True, bind bridge entities from first-hop OpenIE triples and execute bound second-hop dense probes."}
    )
    query_plan_ontology_bind_probe_join_target_datasets: str = field(
        default="",
        metadata={"help": "Comma-separated dataset names where ontology bind-and-probe is enabled; empty means all datasets."}
    )
    query_plan_ontology_bind_probe_join_anchor_top_k: int = field(
        default=8,
        metadata={"help": "Number of current top passages inspected for grounded bridge bindings."}
    )
    query_plan_ontology_bind_probe_join_max_bindings: int = field(
        default=4,
        metadata={"help": "Maximum distinct grounded bridge bindings retained per query."}
    )
    query_plan_ontology_bind_probe_join_max_probes: int = field(
        default=8,
        metadata={"help": "Maximum bound second-hop dense probes issued per query."}
    )
    query_plan_ontology_bind_probe_join_dense_top_k: int = field(
        default=64,
        metadata={"help": "Dense retrieval window inspected for each bound second-hop probe."}
    )
    query_plan_ontology_bind_probe_join_max_candidates: int = field(
        default=8,
        metadata={"help": "Maximum bind-and-probe candidates exposed to the evidence CBO."}
    )
    query_plan_ontology_bind_probe_join_min_binding_score: float = field(
        default=0.52,
        metadata={"help": "Minimum score for accepting a bridge entity bound from a first-hop triple."}
    )
    query_plan_ontology_bind_probe_join_min_candidate_score: float = field(
        default=0.62,
        metadata={"help": "Minimum score for exposing a bound second-hop passage candidate."}
    )
    query_plan_ontology_bind_probe_join_min_relation_overlap: float = field(
        default=0.20,
        metadata={"help": "Minimum unresolved-relation overlap required when the second-hop passage lacks an answer cue."}
    )
    query_plan_ontology_bind_probe_join_synthetic_original_rank: int = field(
        default=60,
        metadata={"help": "Synthetic rank used to place out-of-window bound candidates in the CBO candidate pool."}
    )
    query_plan_ontology_bind_probe_join_require_relation_or_answer: bool = field(
        default=True,
        metadata={"help": "If True, second-hop candidates must match the unresolved relation or an answer-type cue."}
    )
    query_plan_ontology_bind_probe_join_enable_relation_role_constraints: bool = field(
        default=False,
        metadata={"help": "If True, decompose bridge and terminal relation roles before issuing bound probes."}
    )
    query_plan_ontology_bind_probe_join_require_role_consistent_candidate: bool = field(
        default=False,
        metadata={"help": "If True, accept only candidates whose bridge-connected terminal triple matches the terminal relation and answer type."}
    )
    query_plan_ontology_bind_probe_join_min_terminal_relation_overlap: float = field(
        default=0.40,
        metadata={"help": "Minimum predicate overlap between a bridge-connected triple and the typed terminal relation."}
    )
    query_plan_ontology_bind_probe_join_min_terminal_type_score: float = field(
        default=0.60,
        metadata={"help": "Minimum compatibility between the terminal triple object and the requested answer type."}
    )
    query_plan_ontology_bind_probe_join_max_role_candidate_dense_rank: int = field(
        default=0,
        metadata={"help": "Maximum dense rank inspected per typed terminal probe; zero disables this bounded access path."}
    )
    query_plan_ontology_enable_bind_probe_transaction_optimizer: bool = field(
        default=False,
        metadata={"help": "If True, isolate bind-probe candidates and commit complete source-terminal evidence actions after evidence-set ranking."}
    )
    query_plan_ontology_bind_probe_transaction_target_datasets: str = field(
        default="",
        metadata={"help": "Comma-separated datasets where bind-probe evidence transactions are evaluated; empty means all datasets."}
    )
    query_plan_ontology_bind_probe_transaction_top2_target_datasets: str = field(
        default="2wikimultihopqa,hotpotqa",
        metadata={"help": "Comma-separated datasets where a complete bind-probe chain may atomically replace the final top-2 pair."}
    )
    query_plan_ontology_bind_probe_transaction_min_action_utility: float = field(
        default=0.80,
        metadata={"help": "Minimum typed-chain utility required before evaluating a bind-probe transaction."}
    )
    query_plan_ontology_bind_probe_transaction_min_net_benefit: float = field(
        default=0.10,
        metadata={"help": "Minimum counterfactual utility minus displacement and probe cost required to commit a top-2 transaction."}
    )
    query_plan_ontology_bind_probe_transaction_max_terminal_rank: int = field(
        default=5,
        metadata={"help": "Maximum current final rank of a terminal passage eligible for top-2 promotion."}
    )
    query_plan_ontology_bind_probe_transaction_require_source_top1: bool = field(
        default=True,
        metadata={"help": "If True, top-2 transactions preserve the current rank-1 binding source and replace only rank 2."}
    )
    query_plan_ontology_bind_probe_transaction_displacement_cost: float = field(
        default=0.62,
        metadata={"help": "Estimated risk of displacing the current rank-2 passage in a bind-probe transaction."}
    )
    query_plan_ontology_bind_probe_transaction_probe_cost_weight: float = field(
        default=0.025,
        metadata={"help": "Weight applied to normalized bound-probe execution cost in transaction net benefit."}
    )
    query_plan_ontology_enable_entity_keyed_top5_optimizer: bool = field(
        default=False,
        metadata={"help": "If True, admit strict external entity-keyed bind-probe terminals at rank 5 while protecting top-2."}
    )
    query_plan_ontology_entity_keyed_top5_target_datasets: str = field(
        default="2wikimultihopqa,musique",
        metadata={"help": "Comma-separated datasets eligible for rank-isolated entity-keyed top-5 admission."}
    )
    query_plan_ontology_entity_keyed_top5_min_confidence: float = field(
        default=0.84,
        metadata={"help": "Minimum bind-probe confidence for entity-keyed rank-5 admission."}
    )
    query_plan_ontology_entity_keyed_top5_min_relation_score: float = field(
        default=0.80,
        metadata={"help": "Minimum terminal relation score for entity-keyed rank-5 admission."}
    )
    query_plan_ontology_entity_keyed_top5_min_type_score: float = field(
        default=0.75,
        metadata={"help": "Minimum terminal answer-type score for entity-keyed rank-5 admission."}
    )
    query_plan_ontology_entity_keyed_top5_min_binding_score: float = field(
        default=0.58,
        metadata={"help": "Minimum source entity binding score for entity-keyed rank-5 admission."}
    )
    query_plan_ontology_entity_keyed_top5_min_action_utility: float = field(
        default=0.80,
        metadata={"help": "Minimum composite action utility for entity-keyed rank-5 admission."}
    )
    query_plan_ontology_entity_keyed_top5_max_source_rank: int = field(
        default=2,
        metadata={"help": "Maximum rank of the binding source in the protected prefix."}
    )
    query_plan_ontology_enable_factorized_query_executor: bool = field(
        default=False,
        metadata={"help": "If True, execute ontology plans with costed relational physical operators."}
    )
    query_plan_ontology_factorized_target_datasets: str = field(
        default="2wikimultihopqa,musique,nq_rear,popqa",
        metadata={"help": "Comma-separated datasets eligible for factorized query execution."}
    )
    query_plan_ontology_factorized_enable_entity_relation_index_scan: bool = field(
        default=True,
        metadata={"help": "Enable materialized entity-relation index scans."}
    )
    query_plan_ontology_factorized_enable_cascaded_bind_join: bool = field(
        default=True,
        metadata={"help": "Enable bounded multi-hop cascaded bind joins."}
    )
    query_plan_ontology_factorized_enable_dual_branch_comparison_join: bool = field(
        default=True,
        metadata={"help": "Enable atomic dual-branch comparison joins."}
    )
    query_plan_ontology_factorized_dense_top_k: int = field(
        default=48,
        metadata={"help": "Dense fallback window per factorized probe."}
    )
    query_plan_ontology_factorized_max_probes: int = field(
        default=10,
        metadata={"help": "Maximum physical probes issued per query."}
    )
    query_plan_ontology_factorized_max_candidates: int = field(
        default=8,
        metadata={"help": "Maximum validated factorized candidates retained per query."}
    )
    query_plan_ontology_factorized_max_hops: int = field(
        default=4,
        metadata={"help": "Maximum cascaded bind-join depth."}
    )
    query_plan_ontology_factorized_beam_width: int = field(
        default=4,
        metadata={"help": "Maximum intermediate bindings retained per join step."}
    )
    query_plan_ontology_factorized_max_index_rows: int = field(
        default=64,
        metadata={"help": "Maximum entity-relation index rows inspected per access."}
    )
    query_plan_ontology_factorized_index_max_rows_per_entity: int = field(
        default=256,
        metadata={"help": "Maximum OpenIE rows materialized for one normalized entity key."}
    )
    query_plan_ontology_factorized_min_relation_score: float = field(
        default=0.34,
        metadata={"help": "Minimum predicate compatibility for a factorized edge."}
    )
    query_plan_ontology_factorized_min_type_score: float = field(
        default=0.55,
        metadata={"help": "Minimum terminal answer-type compatibility."}
    )
    query_plan_ontology_factorized_min_candidate_utility: float = field(
        default=0.68,
        metadata={"help": "Minimum cost-adjusted utility retained by a physical operator."}
    )
    query_plan_ontology_factorized_cost_weight: float = field(
        default=0.035,
        metadata={"help": "Weight of physical access cost in factorized utility."}
    )
    query_plan_ontology_factorized_enable_top2_admission: bool = field(
        default=True,
        metadata={"help": "Allow strict factorized actions to update top-2."}
    )
    query_plan_ontology_factorized_enable_top5_admission: bool = field(
        default=True,
        metadata={"help": "Allow factorized candidates and chains to update top-5."}
    )
    query_plan_ontology_factorized_shadow_only: bool = field(
        default=False,
        metadata={"help": "Compile factorized candidates without mutating the incumbent ranking; a downstream set optimizer must commit them."}
    )
    query_plan_ontology_factorized_min_top2_utility: float = field(
        default=0.86,
        metadata={"help": "Minimum factorized utility for top-2 admission."}
    )
    query_plan_ontology_factorized_min_top5_utility: float = field(
        default=0.72,
        metadata={"help": "Minimum factorized utility for top-5 admission."}
    )
    query_plan_ontology_factorized_enable_typed_semantic_compiler: bool = field(
        default=False,
        metadata={"help": "Compile typed query contracts before factorized execution."}
    )
    query_plan_ontology_factorized_enable_lossless_typed_plan_ir: bool = field(
        default=False,
        metadata={"help": "Preserve ontology variables, directions, and topology in the logical-plan IR."}
    )
    query_plan_ontology_factorized_require_typed_variable_unification: bool = field(
        default=False,
        metadata={"help": "Execute typed plans with explicit variable unification and binding certificates."}
    )
    query_plan_ontology_factorized_enable_certificate_aware_plan_portfolio: bool = field(
        default=False,
        metadata={"help": "Try compatible typed plans and a structural fallback only when the primary typed certificate fails."}
    )
    query_plan_ontology_factorized_enable_certificate_aware_access_path_reoptimization: bool = field(
        default=False,
        metadata={"help": "Retry a failed typed plan through bound dense access and require a complete binding certificate."}
    )
    query_plan_ontology_factorized_enable_typed_certificate_top2: bool = field(
        default=False,
        metadata={"help": "Allow a complete typed binding certificate to produce a branch-balanced top-2 slate."}
    )
    query_plan_ontology_factorized_typed_certificate_top2_min_utility: float = field(
        default=0.90,
        metadata={"help": "Minimum complete typed-proof utility for top-2 admission."}
    )
    query_plan_ontology_factorized_typed_certificate_top2_min_semantic_score: float = field(
        default=0.90,
        metadata={"help": "Minimum semantic score for a typed-certificate top-2 action."}
    )
    query_plan_ontology_factorized_require_exact_entity_keys: bool = field(
        default=False,
        metadata={"help": "Require exact entity keys for factorized probes."}
    )
    query_plan_ontology_factorized_require_typed_join_keys: bool = field(
        default=False,
        metadata={"help": "Reject literal intermediate join bindings."}
    )
    query_plan_ontology_factorized_min_query_predicate_coverage: float = field(
        default=0.75,
        metadata={"help": "Minimum query predicate coverage for a compiled path."}
    )
    query_plan_ontology_factorized_enable_simple_top2_admission: bool = field(
        default=True,
        metadata={"help": "Allow single-index actions to update top-2."}
    )
    query_plan_ontology_factorized_require_chain_source_top2: bool = field(
        default=False,
        metadata={"help": "Require cascaded joins to bind from protected top-2 evidence."}
    )
    query_plan_ontology_factorized_min_counterfactual_net_benefit: float = field(
        default=0.0,
        metadata={"help": "Minimum metric-aligned benefit minus displacement-risk proxy."}
    )
    query_plan_ontology_factorized_enable_branch_complete_compiler: bool = field(
        default=False,
        metadata={"help": "Compile exactly two complete comparison branches."}
    )
    query_plan_ontology_factorized_require_branch_qualifier_evidence: bool = field(
        default=False,
        metadata={"help": "Require branch qualifiers in selected evidence."}
    )
    query_plan_ontology_factorized_enable_evidence_query_compiler: bool = field(
        default=False,
        metadata={"help": "Compile ontology matches into typed logical evidence plans."}
    )
    query_plan_ontology_factorized_enable_executable_ontology_compiler: bool = field(
        default=False,
        metadata={"help": "Compile query-grounded slots into branch-complete executable plans."}
    )
    query_plan_ontology_factorized_enable_relation_realization: bool = field(
        default=False,
        metadata={"help": "Resolve ontology relations to observed OpenIE predicate surfaces."}
    )
    query_plan_ontology_factorized_enable_relation_keyed_catalog: bool = field(
        default=False,
        metadata={"help": "Resolve logical relations by semantic relation key across all physical artifact steps."}
    )
    query_plan_ontology_factorized_relation_keyed_allow_matched_schema_fallback: bool = field(
        default=False,
        metadata={"help": "Fill unresolved selected-schema relations from other runtime-matched schemas."}
    )
    query_plan_ontology_factorized_enable_query_relation_alternative_recovery: bool = field(
        default=False,
        metadata={"help": "Execute a bounded portfolio of ontology relation paths when the surface relation contract cannot be materialized."}
    )
    query_plan_ontology_factorized_max_query_relation_alternatives: int = field(
        default=3,
        metadata={"help": "Maximum non-primary ontology relation paths executed per query."}
    )
    query_plan_ontology_factorized_enable_execution_aware_contract: bool = field(
        default=False,
        metadata={"help": "Rank compiler schemas by query coverage, answer type, precision, and executable structure."}
    )
    query_plan_ontology_factorized_enable_provenance_plan_portfolio: bool = field(
        default=False,
        metadata={"help": "Retain validated relation-slot evidence when a complete physical join is empty."}
    )
    query_plan_ontology_factorized_enable_candidate_window_path_compiler: bool = field(
        default=False,
        metadata={"help": "Compile complete entity-bound ontology paths over the query-local candidate graph."}
    )
    query_plan_ontology_factorized_candidate_window_path_size: int = field(
        default=200,
        metadata={"help": "Number of incumbent passages materialized into the query-local evidence graph."}
    )
    query_plan_ontology_factorized_candidate_window_path_beam_width: int = field(
        default=12,
        metadata={"help": "Maximum grounded bindings retained after each local graph hop."}
    )
    query_plan_ontology_factorized_candidate_window_path_max_paths: int = field(
        default=8,
        metadata={"help": "Maximum complete provenance paths emitted per query."}
    )
    query_plan_ontology_factorized_candidate_window_path_min_anchor_score: float = field(
        default=0.86,
        metadata={"help": "Minimum candidate-graph endpoint score for query anchor grounding."}
    )
    query_plan_ontology_factorized_candidate_window_path_min_relation_score: float = field(
        default=0.50,
        metadata={"help": "Minimum local OpenIE edge score for a logical ontology relation."}
    )
    query_plan_ontology_factorized_candidate_window_path_min_path_score: float = field(
        default=0.58,
        metadata={"help": "Minimum mean score for a complete query-local provenance path."}
    )
    query_plan_ontology_factorized_candidate_window_path_allow_contextual_terminal_edges: bool = field(
        default=True,
        metadata={"help": "Validate generic terminal predicates using relation and answer-type context."}
    )
    query_plan_ontology_factorized_candidate_window_path_enable_entity_index_expansion: bool = field(
        default=False,
        metadata={"help": "Ground each bound logical variable through the corpus entity-relation index."}
    )
    query_plan_ontology_factorized_candidate_window_path_enable_bound_dense_expansion: bool = field(
        default=False,
        metadata={"help": "Issue relation-conditioned dense probes for unresolved bound variables."}
    )
    query_plan_ontology_factorized_candidate_window_path_entity_index_top_k: int = field(
        default=64,
        metadata={"help": "Maximum entity-relation index rows materialized for each bound variable."}
    )
    query_plan_ontology_factorized_candidate_window_path_bound_dense_top_k: int = field(
        default=32,
        metadata={"help": "Maximum passages inspected for each relation-conditioned dense probe."}
    )
    query_plan_ontology_factorized_candidate_window_path_max_access_probes: int = field(
        default=8,
        metadata={"help": "Per-query budget for relation-conditioned dense access probes."}
    )
    query_plan_ontology_factorized_candidate_window_path_max_external_passages: int = field(
        default=96,
        metadata={"help": "Maximum passages admitted from access paths outside the incumbent semantic window."}
    )
    query_plan_ontology_factorized_candidate_window_path_enable_relation_agnostic_bridge: bool = field(
        default=False,
        metadata={"help": "Compile bounded entity-connected evidence bridges when ontology relation labels cannot be realized."}
    )
    query_plan_ontology_factorized_candidate_window_path_bridge_max_paths: int = field(
        default=8,
        metadata={"help": "Maximum observed evidence bridges emitted per query."}
    )
    query_plan_ontology_factorized_candidate_window_path_bridge_max_edges_per_state: int = field(
        default=48,
        metadata={"help": "Maximum local OpenIE edges traversed from one bridge binding."}
    )
    query_plan_ontology_factorized_candidate_window_path_bridge_min_path_score: float = field(
        default=0.55,
        metadata={"help": "Minimum evidence-first score for an observed relation-agnostic bridge."}
    )
    query_plan_ontology_factorized_enable_inverse_direction_fallback: bool = field(
        default=False,
        metadata={"help": "Recover exact-key OpenIE triples whose stored orientation opposes the logical relation direction."}
    )
    query_plan_ontology_factorized_inverse_direction_penalty: float = field(
        default=0.12,
        metadata={"help": "Score penalty for exact-key inverse-direction physical access candidates."}
    )
    query_plan_ontology_factorized_enable_whole_warehouse_probe: bool = field(
        default=False,
        metadata={"help": "Probe approximate entity keys across the materialized evidence warehouse."}
    )
    query_plan_ontology_factorized_warehouse_max_entity_keys: int = field(
        default=12,
        metadata={"help": "Maximum approximate warehouse entity keys inspected per probe."}
    )
    query_plan_ontology_factorized_enable_cost_based_physical_planner: bool = field(
        default=False,
        metadata={"help": "Choose a physical evidence access plan from a train-split catalog."}
    )
    query_plan_ontology_factorized_evidence_catalog_path: str = field(
        default="",
        metadata={"help": "Path to a validated ontology evidence catalog."}
    )
    query_plan_ontology_factorized_physical_planner_quality_weight: float = field(
        default=0.45,
        metadata={"help": "Quality weight in physical-plan utility."}
    )
    query_plan_ontology_factorized_physical_planner_recall_weight: float = field(
        default=0.25,
        metadata={"help": "Expected recall-delta weight in physical-plan utility."}
    )
    query_plan_ontology_factorized_physical_planner_proof_weight: float = field(
        default=0.30,
        metadata={"help": "Proof-coverage weight in physical-plan utility."}
    )
    query_plan_ontology_factorized_physical_planner_cost_weight: float = field(
        default=0.15,
        metadata={"help": "Execution-cost weight in physical-plan utility."}
    )
    query_plan_ontology_factorized_physical_planner_risk_weight: float = field(
        default=0.35,
        metadata={"help": "Estimated displacement-risk weight in physical-plan utility."}
    )
    query_plan_ontology_factorized_physical_planner_min_proof_coverage: float = field(
        default=0.20,
        metadata={"help": "Minimum estimated proof coverage for a physical plan."}
    )
    query_plan_ontology_factorized_physical_planner_max_estimated_cost: float = field(
        default=8.0,
        metadata={"help": "Maximum catalog-estimated cost per physical plan."}
    )
    query_plan_ontology_factorized_physical_planner_no_op_utility: float = field(
        default=0.0,
        metadata={"help": "Utility of retaining the incumbent retrieval plan."}
    )
    query_plan_ontology_factorized_physical_planner_selection_mode: str = field(
        default="weighted_utility",
        metadata={"help": "weighted_utility or constrained_lexicographic."}
    )
    query_plan_ontology_factorized_physical_planner_max_loss_probability: float = field(
        default=0.25,
        metadata={"help": "Maximum Wilson upper bound on retrieval-loss probability."}
    )
    query_plan_ontology_factorized_physical_planner_confidence_delta: float = field(
        default=0.05,
        metadata={"help": "Failure probability used for empirical confidence bounds."}
    )
    query_plan_ontology_factorized_physical_planner_min_catalog_samples: int = field(
        default=5,
        metadata={"help": "Minimum train-trace support for physical-plan admission."}
    )
    query_plan_ontology_factorized_physical_access_strategy: str = field(
        default="adaptive",
        metadata={"help": "Physical access strategy used when the cost-based planner is disabled."}
    )
    query_plan_ontology_factorized_enable_physical_shadow_collection: bool = field(
        default=False,
        metadata={"help": "Collect physical alternatives on shared retrieval and index state."}
    )
    query_plan_ontology_factorized_physical_shadow_strategies: str = field(
        default="adaptive,index_only,dense_first",
        metadata={"help": "Physical strategies included in counterfactual shadow collection."}
    )
    query_plan_ontology_factorized_enable_missing_slot_executor: bool = field(
        default=False,
        metadata={"help": "Recover failed plans with missing-slot dependent joins."}
    )
    query_plan_ontology_factorized_missing_slot_trigger_reasons: str = field(
        default=(
            "no_validated_candidates,insufficient_query_predicate_coverage,"
            "no_exact_query_anchor,no_relational_access_path"
        ),
        metadata={"help": "Factorized failure reasons eligible for missing-slot recovery."}
    )
    query_plan_ontology_factorized_missing_slot_dense_top_k: int = field(
        default=64,
        metadata={"help": "Dense retrieval window for each missing-slot probe."}
    )
    query_plan_ontology_factorized_missing_slot_max_probes: int = field(
        default=12,
        metadata={"help": "Maximum missing-slot probe stages per query."}
    )
    query_plan_ontology_factorized_missing_slot_max_bridge_entities: int = field(
        default=6,
        metadata={"help": "Maximum bridge bindings retained per join stage."}
    )
    query_plan_ontology_factorized_missing_slot_max_candidates: int = field(
        default=12,
        metadata={"help": "Maximum recovered terminal candidates."}
    )
    query_plan_ontology_factorized_missing_slot_max_hops: int = field(
        default=3,
        metadata={"help": "Maximum relation hops in a recovery plan."}
    )
    query_plan_ontology_factorized_missing_slot_min_relation_score: float = field(
        default=0.20,
        metadata={"help": "Minimum relation compatibility for recovery joins."}
    )
    query_plan_ontology_factorized_missing_slot_min_terminal_type_score: float = field(
        default=0.30,
        metadata={"help": "Minimum terminal answer-type compatibility."}
    )
    query_plan_ontology_factorized_missing_slot_min_candidate_utility: float = field(
        default=0.38,
        metadata={"help": "Minimum utility for a recovered terminal passage."}
    )
    query_plan_ontology_factorized_missing_slot_enable_top2_admission: bool = field(
        default=True,
        metadata={"help": "Expose recovered terminals as counterfactual top-2 actions."}
    )
    query_plan_ontology_factorized_enable_typed_proof_join: bool = field(
        default=False,
        metadata={"help": "Use typed, proof-carrying dependent joins for missing slots."}
    )
    query_plan_ontology_factorized_require_branch_complete_proof: bool = field(
        default=True,
        metadata={"help": "Require every logical query branch in a proof set."}
    )
    query_plan_ontology_factorized_require_provenance_validation: bool = field(
        default=True,
        metadata={"help": "Require exact bindings and source triples for proof admission."}
    )
    query_plan_ontology_factorized_require_strict_branch_identity: bool = field(
        default=False,
        metadata={"help": "Require exact branch endpoints and qualifier provenance."}
    )
    query_plan_ontology_factorized_project_full_proof_set: bool = field(
        default=True,
        metadata={"help": "Project the complete proof set instead of terminal-only evidence."}
    )
    query_plan_ontology_factorized_min_proof_confidence: float = field(
        default=0.72,
        metadata={"help": "Minimum confidence for a proof-carrying dependent join."}
    )
    query_plan_ontology_factorized_enable_proof_gap_access_path: bool = field(
        default=False,
        metadata={"help": "Execute bound ontology paths for relation slots absent from the current top-k."}
    )
    query_plan_ontology_factorized_proof_gap_top_window_k: int = field(
        default=5,
        metadata={"help": "Top-k evidence window audited for proof gaps."}
    )
    query_plan_ontology_factorized_proof_gap_max_paths: int = field(
        default=4,
        metadata={"help": "Maximum ontology paths executed for one proof-gap query."}
    )
    query_plan_ontology_factorized_proof_gap_min_missing_slot_fraction: float = field(
        default=0.25,
        metadata={"help": "Minimum uncovered relation-slot fraction required to trigger bound probes."}
    )
    query_plan_ontology_factorized_proof_gap_min_binding_score: float = field(
        default=0.80,
        metadata={"help": "Minimum dependent binding score for a proof-gap join edge."}
    )
    query_plan_ontology_factorized_proof_gap_min_proof_confidence: float = field(
        default=0.20,
        metadata={"help": "Minimum conjunctive proof certificate exposed to learned admission."}
    )
    query_plan_ontology_factorized_proof_gap_require_external_member: bool = field(
        default=True,
        metadata={"help": "Require every proposed proof set to add evidence outside the current top-k."}
    )
    query_plan_ontology_factorized_proof_gap_merge_candidate_pool: bool = field(
        default=False,
        metadata={"help": "Union proof-gap output with the incumbent physical candidate pool."}
    )
    query_plan_ontology_factorized_enable_proof_residual_commit: bool = field(
        default=False,
        metadata={"help": "Commit only proof members missing from the current top-k."}
    )
    query_plan_ontology_factorized_proof_residual_direct_commit: bool = field(
        default=False,
        metadata={"help": "Let a strict proof certificate bypass the learned action gate."}
    )
    query_plan_ontology_factorized_proof_residual_post_optimizer_commit: bool = field(
        default=False,
        metadata={"help": "Recompile and commit a proof residual after the learned primary action."}
    )
    query_plan_ontology_factorized_proof_residual_target_datasets: str = field(
        default="2wikimultihopqa",
        metadata={"help": "Comma-separated datasets eligible for proof-residual mutation."}
    )
    query_plan_ontology_factorized_proof_residual_min_certificate_confidence: float = field(
        default=0.95,
        metadata={"help": "Minimum confidence for a direct proof-certified commit."}
    )
    query_plan_ontology_factorized_proof_residual_max_top5_insertions: int = field(
        default=1,
        metadata={"help": "Maximum missing proof members inserted into top-5 per action."}
    )
    query_plan_ontology_factorized_proof_residual_enable_top2: bool = field(
        default=True,
        metadata={"help": "Allow certified two-branch terminal projections into top-2."}
    )
    query_plan_ontology_factorized_enable_proof_materialized_view: bool = field(
        default=False,
        metadata={"help": "Materialize a compact certified proof view for the reader."}
    )
    query_plan_ontology_factorized_proof_materialized_view_target_datasets: str = field(
        default="2wikimultihopqa",
        metadata={"help": "Datasets eligible for compact proof-view materialization."}
    )
    query_plan_ontology_factorized_proof_materialized_view_max_edges: int = field(
        default=8,
        metadata={"help": "Maximum proof edges exposed in the compact reader view."}
    )
    query_plan_ontology_factorized_enable_learned_delta_admission: bool = field(
        default=False,
        metadata={"help": "Gate actions with a train-split metric-delta model."}
    )
    query_plan_ontology_factorized_delta_admission_model_path: str = field(
        default="",
        metadata={"help": "Path to the learned factorized delta-admission model."}
    )
    query_plan_ontology_factorized_delta_admission_min_expected_delta: float = field(
        default=0.0,
        metadata={"help": "Minimum predicted recall delta for admission."}
    )
    query_plan_ontology_factorized_delta_admission_max_loss_probability: float = field(
        default=0.20,
        metadata={"help": "Maximum predicted displacement-loss probability."}
    )
    query_plan_ontology_factorized_delta_admission_fail_open: bool = field(
        default=False,
        metadata={"help": "Admit when the learned model has no compatible section."}
    )
    query_plan_ontology_factorized_delta_admission_collect_only: bool = field(
        default=False,
        metadata={"help": "Collect train actions without enforcing learned admission."}
    )
    query_plan_ontology_factorized_enable_set_level_optimizer: bool = field(
        default=False,
        metadata={"help": "Jointly optimize Top-2/Top-5 evidence sets against no-op."}
    )
    query_plan_ontology_factorized_set_level_optimizer_model_path: str = field(
        default="",
        metadata={"help": "Path to the learned query-level evidence-set optimizer."}
    )
    query_plan_ontology_factorized_set_level_optimizer_min_expected_utility: float = field(
        default=-1.0,
        metadata={"help": "Optional set-utility threshold override; negative uses model policy."}
    )
    query_plan_ontology_factorized_set_level_optimizer_max_loss_probability: float = field(
        default=-1.0,
        metadata={"help": "Optional set-loss threshold override; negative uses model policy."}
    )
    query_plan_ontology_factorized_set_level_optimizer_fail_open: bool = field(
        default=False,
        metadata={"help": "Allow a set action when the set-level model is unavailable."}
    )
    query_plan_ontology_factorized_enable_incumbent_reoptimizer: bool = field(
        default=False,
        metadata={"help": "Admit only risk-bounded improvements over the incumbent plan."}
    )
    query_plan_ontology_factorized_incumbent_reoptimizer_model_path: str = field(
        default="",
        metadata={"help": "Path to the incumbent-relative residual optimizer."}
    )
    query_plan_ontology_factorized_incumbent_reoptimizer_min_expected_delta: float = field(
        default=-1.0,
        metadata={"help": "Optional incumbent-relative delta threshold override."}
    )
    query_plan_ontology_factorized_incumbent_reoptimizer_max_loss_probability: float = field(
        default=-1.0,
        metadata={"help": "Optional incumbent-relative loss-risk threshold override."}
    )
    query_plan_ontology_factorized_incumbent_reoptimizer_set_trigger_mode: str = field(
        default="incumbent_noop_or_proof_deficit",
        metadata={"help": "Trigger policy for compiling residual evidence sets."}
    )
    query_plan_ontology_factorized_entity_scan_mutation_target_datasets: str = field(
        default="2wikimultihopqa,musique,nq_rear,popqa",
        metadata={"help": "Datasets where entity scans may mutate top-k."}
    )
    query_plan_ontology_factorized_enable_counterfactual_action_replay: bool = field(
        default=False,
        metadata={"help": "Enumerate independent top-k actions and select them with one learned delta gate."}
    )
    query_plan_ontology_factorized_counterfactual_action_replay_max_actions: int = field(
        default=24,
        metadata={"help": "Maximum no-op, factorized, and warehouse actions replayed per query."}
    )
    query_plan_ontology_factorized_action_replay_include_evidence_warehouse: bool = field(
        default=True,
        metadata={"help": "Include evidence-warehouse candidate scans in the replay action space."}
    )
    query_plan_ontology_factorized_action_replay_enable_atomic_chain_top2: bool = field(
        default=True,
        metadata={"help": "Include complete two-document chain groups as atomic top-2 actions."}
    )
    query_plan_ontology_factorized_enable_evidence_set_compiler: bool = field(
        default=False,
        metadata={"help": "Compile warehouse candidates into ontology-complementary set actions."}
    )
    query_plan_ontology_factorized_evidence_set_max_candidates: int = field(
        default=10,
        metadata={"help": "Maximum candidates considered by the evidence-set compiler."}
    )
    query_plan_ontology_factorized_evidence_set_beam_width: int = field(
        default=16,
        metadata={"help": "Beam width for evidence-set action enumeration."}
    )
    query_plan_ontology_factorized_evidence_set_max_top2_actions: int = field(
        default=6,
        metadata={"help": "Maximum compiled top-2 set actions per query."}
    )
    query_plan_ontology_factorized_evidence_set_max_top5_actions: int = field(
        default=10,
        metadata={"help": "Maximum compiled top-5 set actions per query."}
    )
    query_plan_ontology_enable_top5_listwise_optimizer: bool = field(
        default=False,
        metadata={"help": "If True, invoke a grounded listwise judge over the current top-5 and bounded external ontology candidates."}
    )
    query_plan_ontology_top5_listwise_target_datasets: str = field(
        default="2wikimultihopqa,hotpotqa,musique,nq_rear,popqa",
        metadata={"help": "Comma-separated datasets eligible for adaptive top-5 listwise re-optimization."}
    )
    query_plan_ontology_top5_listwise_require_ontology_candidate: bool = field(
        default=True,
        metadata={"help": "If True, call and commit the listwise operator only for provenance-bearing ontology candidates."}
    )
    query_plan_ontology_top5_listwise_max_passage_chars: int = field(
        default=1200,
        metadata={"help": "Maximum passage characters supplied to the listwise judge per top-5 item."}
    )
    query_plan_ontology_top5_listwise_max_completion_tokens: int = field(
        default=384,
        metadata={"help": "Maximum completion tokens for one listwise judge call."}
    )
    query_plan_ontology_top5_listwise_max_external_candidates: int = field(
        default=3,
        metadata={"help": "Maximum ontology candidates outside the current top-5 exposed to one listwise call."}
    )
    query_plan_ontology_top5_listwise_min_external_access_score: float = field(
        default=0.58,
        metadata={"help": "Minimum estimated access-path utility for an ontology candidate outside the current top-5."}
    )
    query_plan_ontology_top5_listwise_min_candidate_score: float = field(
        default=0.72,
        metadata={"help": "Minimum judge score required to promote a rank-3-to-5 ontology candidate."}
    )
    query_plan_ontology_top5_listwise_min_score_margin: float = field(
        default=0.16,
        metadata={"help": "Minimum listwise score advantage over the current rank-2 passage required for commit."}
    )
    query_plan_ontology_top5_listwise_min_judge_confidence: float = field(
        default=0.78,
        metadata={"help": "Minimum judge confidence required for a listwise top-2 commit."}
    )
    query_plan_ontology_top5_listwise_min_evidence_span_chars: int = field(
        default=3,
        metadata={"help": "Minimum exact evidence-span length required to ground a listwise promotion."}
    )
    query_plan_ontology_enable_projected_graph_retriever: bool = field(
        default=False,
        metadata={"help": "If True, project corpus OpenIE facts onto query-plan ontology schema steps and retrieve source passages from the projected graph."}
    )
    query_plan_ontology_projected_graph_max_schema_relations: int = field(
        default=512,
        metadata={"help": "Maximum ontology schema steps materialized in the projected graph retriever."}
    )
    query_plan_ontology_projected_graph_max_facts_per_schema: int = field(
        default=5000,
        metadata={"help": "Maximum projected fact indices retained per ontology schema step."}
    )
    query_plan_ontology_projected_graph_max_facts_per_step_query: int = field(
        default=600,
        metadata={"help": "Maximum projected facts scanned per schema step for a query."}
    )
    query_plan_ontology_projected_graph_max_candidates: int = field(
        default=4,
        metadata={"help": "Maximum projected-graph passage candidates exposed downstream."}
    )
    query_plan_ontology_projected_graph_min_score: float = field(
        default=0.62,
        metadata={"help": "Minimum projected-graph candidate score required for insertion."}
    )
    query_plan_ontology_projected_graph_preserve_top_k: int = field(
        default=2,
        metadata={"help": "Number of top baseline passages protected from projected-graph insertion."}
    )
    query_plan_ontology_projected_graph_candidate_only: bool = field(
        default=True,
        metadata={"help": "If True, projected-graph retrieval only appends candidates for downstream evidence-set selection."}
    )
    query_plan_ontology_projected_graph_score_margin: float = field(
        default=1e-7,
        metadata={"help": "Small score margin used when appending projected-graph candidates."}
    )
    query_plan_ontology_projected_graph_require_anchor_overlap: bool = field(
        default=True,
        metadata={"help": "If True, projected facts must overlap query/anchor tokens before becoming candidates."}
    )
    query_plan_ontology_projected_graph_min_relation_token_overlap: float = field(
        default=0.50,
        metadata={"help": "Minimum relation-token overlap for mapping graph relations to ontology schema steps."}
    )
    query_plan_ontology_projected_graph_enable_bridge_paths: bool = field(
        default=True,
        metadata={"help": "If True, prefer two-hop projected graph evidence when ontology paths have multiple steps."}
    )
    query_plan_ontology_enable_evidence_index_retriever: bool = field(
        default=False,
        metadata={"help": "If True, retrieve high-precision passage candidates from an ontology-slot-to-fact evidence index."}
    )
    query_plan_ontology_evidence_index_max_schema_slots: int = field(
        default=1024,
        metadata={"help": "Maximum ontology slots materialized by the evidence-index retriever."}
    )
    query_plan_ontology_evidence_index_max_facts_per_slot: int = field(
        default=5000,
        metadata={"help": "Maximum fact indices retained per ontology evidence slot."}
    )
    query_plan_ontology_evidence_index_max_facts_per_query_slot: int = field(
        default=400,
        metadata={"help": "Maximum indexed facts scanned for one query-slot match."}
    )
    query_plan_ontology_evidence_index_max_candidates: int = field(
        default=4,
        metadata={"help": "Maximum evidence-index passage candidates exposed downstream."}
    )
    query_plan_ontology_evidence_index_min_score: float = field(
        default=0.84,
        metadata={"help": "Minimum evidence-index candidate score required for insertion."}
    )
    query_plan_ontology_evidence_index_preserve_top_k: int = field(
        default=2,
        metadata={"help": "Number of top baseline passages protected from evidence-index insertion."}
    )
    query_plan_ontology_evidence_index_candidate_only: bool = field(
        default=True,
        metadata={"help": "If True, evidence-index retrieval only appends candidates for downstream evidence-set selection."}
    )
    query_plan_ontology_evidence_index_score_margin: float = field(
        default=1e-7,
        metadata={"help": "Small score margin used when appending evidence-index candidates."}
    )
    query_plan_ontology_evidence_index_require_anchor_overlap: bool = field(
        default=True,
        metadata={"help": "If True, evidence-index facts must overlap query/anchor tokens."}
    )
    query_plan_ontology_evidence_index_min_relation_token_overlap: float = field(
        default=0.35,
        metadata={"help": "Minimum relation-token overlap for mapping facts to ontology evidence slots."}
    )
    query_plan_ontology_evidence_index_synthetic_original_rank: int = field(
        default=45,
        metadata={"help": "Synthetic original rank used so high-precision evidence-index candidates enter evidence-set selection."}
    )
    query_plan_ontology_evidence_index_top2_min_score: float = field(
        default=0.92,
        metadata={"help": "When evidence-index is allowed to affect rank 2, minimum score required for a candidate to enter that slot."}
    )
    query_plan_ontology_evidence_index_top2_max_original_rank: int = field(
        default=30,
        metadata={"help": "When evidence-index is allowed to affect rank 2, maximum original rank eligible; <=0 disables this bound."}
    )
    query_plan_ontology_evidence_index_top2_require_terminal: bool = field(
        default=True,
        metadata={"help": "If True, evidence-index rank-2 interventions require both anchor and terminal endpoint evidence."}
    )
    query_plan_ontology_evidence_index_recall_utility_model_path: str = field(
        default="",
        metadata={"help": "Optional supervised model for evidence-index recall-utility candidate selection."}
    )
    query_plan_ontology_evidence_index_recall_utility_mode: str = field(
        default="off",
        metadata={"help": "Evidence-index recall-utility mode: off, score, or selector."}
    )
    query_plan_ontology_evidence_index_recall_utility_min_score: float = field(
        default=0.70,
        metadata={"help": "Minimum recall-utility score that marks an evidence-index candidate as recall-promoted."}
    )
    query_plan_ontology_evidence_index_recall_utility_score_weight: float = field(
        default=0.35,
        metadata={"help": "Score weight added to evidence-index candidates from the recall-utility model."}
    )
    query_plan_ontology_evidence_index_recall_utility_promote_confidence: float = field(
        default=0.98,
        metadata={"help": "Confidence floor for evidence-index candidates promoted by the recall-utility model."}
    )
    query_plan_ontology_evidence_index_recall_utility_top2_min_score: float = field(
        default=0.80,
        metadata={"help": "Minimum recall-utility score required for evidence-index top-2 intervention."}
    )
    query_plan_ontology_enable_path_executor_retriever: bool = field(
        default=False,
        metadata={"help": "If True, execute matched ontology slot paths over OpenIE facts and expose path-source passages downstream."}
    )
    query_plan_ontology_path_executor_max_schema_slots: int = field(
        default=1024,
        metadata={"help": "Maximum ontology slots materialized by the ontology path executor."}
    )
    query_plan_ontology_path_executor_max_facts_per_slot: int = field(
        default=5000,
        metadata={"help": "Maximum fact indices retained per ontology path-executor slot."}
    )
    query_plan_ontology_path_executor_max_facts_per_step: int = field(
        default=160,
        metadata={"help": "Maximum indexed facts scanned per path step during ontology path execution."}
    )
    query_plan_ontology_path_executor_max_runtime_paths: int = field(
        default=8,
        metadata={"help": "Maximum runtime ontology paths executed per query."}
    )
    query_plan_ontology_path_executor_max_candidates: int = field(
        default=4,
        metadata={"help": "Maximum ontology path-executor candidates exposed downstream."}
    )
    query_plan_ontology_path_executor_min_score: float = field(
        default=0.78,
        metadata={"help": "Minimum ontology path-executor candidate score required for insertion."}
    )
    query_plan_ontology_path_executor_preserve_top_k: int = field(
        default=2,
        metadata={"help": "Number of top baseline passages protected from ontology path-executor insertion."}
    )
    query_plan_ontology_path_executor_candidate_only: bool = field(
        default=True,
        metadata={"help": "If True, ontology path execution only appends candidates for downstream evidence-set selection."}
    )
    query_plan_ontology_path_executor_score_margin: float = field(
        default=1e-7,
        metadata={"help": "Small score margin used when appending ontology path-executor candidates."}
    )
    query_plan_ontology_path_executor_synthetic_original_rank: int = field(
        default=35,
        metadata={"help": "Synthetic original rank used so ontology path-executor candidates enter evidence-set selection."}
    )
    query_plan_ontology_path_executor_beam_size: int = field(
        default=8,
        metadata={"help": "Beam size for ontology path execution over OpenIE facts."}
    )
    query_plan_ontology_path_executor_max_steps: int = field(
        default=3,
        metadata={"help": "Maximum ontology slots followed by the path executor."}
    )
    query_plan_ontology_path_executor_min_relation_token_overlap: float = field(
        default=0.35,
        metadata={"help": "Minimum relation-token overlap for mapping facts to ontology path-executor slots."}
    )
    query_plan_ontology_path_executor_enable_answer_type_completion: bool = field(
        default=False,
        metadata={"help": "If True, complete bridge/query-condition slots with answer-type-compatible terminal ontology slots from the same plan."}
    )
    query_plan_ontology_path_executor_enable_terminal_type_filter: bool = field(
        default=False,
        metadata={"help": "If True, filter executed paths whose terminal entity does not match the query answer type."}
    )
    query_plan_ontology_path_executor_allow_query_condition_single: bool = field(
        default=True,
        metadata={"help": "If True, allow single-slot query-condition paths when answer-type completion cannot extend them."}
    )
    query_plan_ontology_path_executor_require_terminal_type_for_single_step: bool = field(
        default=False,
        metadata={"help": "If True, single-step path candidates must pass terminal answer-type verification."}
    )
    query_plan_ontology_path_executor_min_terminal_type_score: float = field(
        default=0.45,
        metadata={"help": "Minimum terminal answer-type score for verified ontology path-executor candidates."}
    )
    query_plan_ontology_path_executor_direction_penalty: float = field(
        default=1.0,
        metadata={"help": "Soft penalty applied when an OpenIE fact is traversed opposite the ontology step direction."}
    )
    query_plan_ontology_path_executor_max_relation_repeats: int = field(
        default=0,
        metadata={"help": "Maximum times the same relation can repeat in one executed path; <=0 disables this check."}
    )
    query_plan_ontology_path_executor_strict_query_relation_binding: bool = field(
        default=False,
        metadata={"help": "If True, executed ontology paths must cover strong relation tokens explicitly present in the query."}
    )
    query_plan_ontology_path_executor_min_query_relation_coverage: float = field(
        default=1.0,
        metadata={"help": "Minimum fraction of strong query relation tokens that an ontology path must cover when strict relation binding is enabled."}
    )
    query_plan_ontology_enable_failure_evidence_retriever: bool = field(
        default=False,
        metadata={"help": "If True, use retrieval-failure evidence rules to generate extra dense support candidates."}
    )
    query_plan_ontology_failure_evidence_path: str = field(
        default="",
        metadata={"help": "Optional explicit failure_evidence_ontology.json path."}
    )
    query_plan_ontology_failure_evidence_root: str = field(
        default="",
        metadata={"help": "Root containing <dataset>/failure_evidence_ontology.json files."}
    )
    query_plan_ontology_failure_evidence_dense_top_k: int = field(
        default=120,
        metadata={"help": "Dense top-k passages considered for each retrieval-failure evidence probe."}
    )
    query_plan_ontology_failure_evidence_max_rules: int = field(
        default=8,
        metadata={"help": "Maximum retrieval-failure rules matched per query."}
    )
    query_plan_ontology_failure_evidence_max_probes: int = field(
        default=16,
        metadata={"help": "Maximum retrieval-failure evidence probes issued per query."}
    )
    query_plan_ontology_failure_evidence_max_candidates: int = field(
        default=8,
        metadata={"help": "Maximum retrieval-failure evidence candidates exposed downstream."}
    )
    query_plan_ontology_failure_evidence_min_confidence: float = field(
        default=0.56,
        metadata={"help": "Minimum confidence required before a retrieval-failure evidence candidate is accepted."}
    )
    query_plan_ontology_failure_evidence_preserve_top_k: int = field(
        default=2,
        metadata={"help": "Number of top baseline passages protected from retrieval-failure evidence insertion."}
    )
    query_plan_ontology_failure_evidence_synthetic_original_rank: int = field(
        default=60,
        metadata={"help": "Synthetic original rank for retrieval-failure evidence passages not present in baseline retrieval."}
    )
    query_plan_ontology_failure_evidence_require_anchor_hit: bool = field(
        default=False,
        metadata={"help": "If True, retrieval-failure evidence candidates with concrete anchors must mention the anchor."}
    )
    query_plan_ontology_failure_evidence_require_rule_term_hit: bool = field(
        default=True,
        metadata={"help": "If True, retrieval-failure evidence candidates must mention rule relation/query/answer hints."}
    )
    query_plan_ontology_failure_evidence_candidate_only: bool = field(
        default=True,
        metadata={"help": "If True, retrieval-failure evidence retrieval only appends candidates and lets evidence-set selection decide ranking."}
    )
    query_plan_ontology_failure_evidence_score_margin: float = field(
        default=1e-7,
        metadata={"help": "Small score margin used when appending retrieval-failure evidence candidates."}
    )
    query_plan_ontology_failure_evidence_min_rule_gold_precision: float = field(
        default=0.0,
        metadata={"help": "Minimum validated gold precision required for a retrieval-failure evidence rule."}
    )
    query_plan_ontology_failure_evidence_min_rule_gt5_precision: float = field(
        default=0.0,
        metadata={"help": "Minimum validated new-gold-from-gt5 precision required for a retrieval-failure evidence rule."}
    )
    query_plan_ontology_failure_evidence_min_rule_gold_hits: int = field(
        default=0,
        metadata={"help": "Minimum validated gold hits required for a retrieval-failure evidence rule."}
    )
    query_plan_ontology_failure_evidence_min_rule_gt5_hits: int = field(
        default=0,
        metadata={"help": "Minimum validated new-gold-from-gt5 hits required for a retrieval-failure evidence rule."}
    )
    enable_evidence_grounded_ontology: bool = field(
        default=False,
        metadata={"help": "If True, load an evidence-grounded ontology built from training retrieval failures/support docs."}
    )
    evidence_grounded_ontology_path: str = field(
        default="",
        metadata={"help": "Optional explicit evidence-grounded ontology JSON path."}
    )
    evidence_grounded_ontology_root: str = field(
        default="outputs_evidence_grounded_ontology_v51",
        metadata={"help": "Root containing <dataset>/evidence_grounded_ontology.json files."}
    )
    evidence_grounded_ontology_dense_top_k: int = field(
        default=80,
        metadata={"help": "Dense top-k passages considered for each evidence-grounded ontology probe."}
    )
    evidence_grounded_ontology_top_k_plans: int = field(
        default=6,
        metadata={"help": "Maximum evidence-grounded ontology plans matched per query."}
    )
    evidence_grounded_ontology_max_probes: int = field(
        default=18,
        metadata={"help": "Maximum evidence-grounded dense probes issued per query."}
    )
    evidence_grounded_ontology_max_candidates: int = field(
        default=10,
        metadata={"help": "Maximum evidence-grounded candidate passages exposed downstream."}
    )
    evidence_grounded_ontology_min_confidence: float = field(
        default=0.58,
        metadata={"help": "Minimum confidence for accepting evidence-grounded dense candidates."}
    )
    evidence_grounded_ontology_preserve_top_k: int = field(
        default=2,
        metadata={"help": "Number of top baseline passages protected from evidence-grounded insertion."}
    )
    evidence_grounded_ontology_candidate_only: bool = field(
        default=True,
        metadata={"help": "If True, evidence-grounded retrieval appends candidates and lets evidence-set selection rank them."}
    )
    evidence_grounded_ontology_min_match_score: float = field(
        default=0.05,
        metadata={"help": "Minimum query-plan match score for an evidence-grounded plan to issue probes."}
    )
    evidence_grounded_leaf_policy_path: str = field(
        default="",
        metadata={"help": "Optional learned utility policy for evidence-grounded ontology tree leaves."}
    )
    evidence_grounded_leaf_policy_reject_below: float = field(
        default=0.0,
        metadata={"help": "Reject matched evidence-grounded ontology leaves with utility score below this value."}
    )
    evidence_grounded_leaf_policy_boost: float = field(
        default=0.0,
        metadata={"help": "Add this multiplier times the learned leaf utility score to the plan match score."}
    )
    query_plan_ontology_enable_llm_evidence_verifier: bool = field(
        default=False,
        metadata={"help": "If True, use the LLM to verify ontology support evidence candidates before evidence-set selection."}
    )
    query_plan_ontology_llm_evidence_verifier_max_candidates: int = field(
        default=8,
        metadata={"help": "Maximum ontology support candidates judged by the LLM evidence verifier per query."}
    )
    query_plan_ontology_llm_evidence_verifier_min_original_rank: int = field(
        default=3,
        metadata={"help": "Minimum original retrieval rank for candidates judged by the LLM evidence verifier."}
    )
    query_plan_ontology_llm_evidence_verifier_max_original_rank: int = field(
        default=30,
        metadata={"help": "Maximum original retrieval rank for candidates judged by the LLM evidence verifier; <=0 disables."}
    )
    query_plan_ontology_llm_evidence_verifier_min_approval_score: float = field(
        default=0.70,
        metadata={"help": "Minimum LLM approval score required to promote an ontology evidence candidate."}
    )
    query_plan_ontology_llm_evidence_verifier_approved_confidence: float = field(
        default=0.98,
        metadata={"help": "Confidence assigned to LLM-approved ontology evidence candidates before evidence-set selection."}
    )
    query_plan_ontology_llm_evidence_verifier_rejected_confidence: float = field(
        default=0.10,
        metadata={"help": "Confidence cap assigned to LLM-rejected ontology evidence candidates."}
    )
    query_plan_ontology_llm_evidence_verifier_max_passage_chars: int = field(
        default=850,
        metadata={"help": "Maximum passage characters shown to the LLM evidence verifier."}
    )
    query_plan_ontology_llm_evidence_verifier_max_completion_tokens: int = field(
        default=700,
        metadata={"help": "Maximum completion tokens for each LLM evidence verifier call."}
    )
    query_plan_ontology_enable_supervised_evidence_ranker: bool = field(
        default=False,
        metadata={"help": "If True, score ontology evidence candidates with a supervised ranker trained from trace gold-doc labels."}
    )
    query_plan_ontology_supervised_evidence_ranker_model_path: str = field(
        default="",
        metadata={"help": "Path to the JSON model used by the supervised ontology evidence ranker."}
    )
    query_plan_ontology_supervised_evidence_ranker_min_score: float = field(
        default=0.55,
        metadata={"help": "Minimum supervised probability needed to promote an ontology evidence candidate."}
    )
    query_plan_ontology_supervised_evidence_ranker_promote_confidence: float = field(
        default=0.90,
        metadata={"help": "Confidence floor assigned to supervised-ranker-promoted ontology evidence candidates."}
    )
    query_plan_ontology_supervised_evidence_ranker_reject_below: float = field(
        default=0.20,
        metadata={"help": "Candidates below this supervised probability have their confidence capped."}
    )
    query_plan_ontology_supervised_evidence_ranker_rejected_confidence: float = field(
        default=0.05,
        metadata={"help": "Confidence cap for candidates rejected by the supervised ontology evidence ranker."}
    )
    query_plan_ontology_supervised_evidence_ranker_max_candidates: int = field(
        default=16,
        metadata={"help": "Maximum candidates promoted by the supervised ontology evidence ranker per query; <=0 disables the cap."}
    )
    query_plan_ontology_enable_metric_aligned_slate_ranker: bool = field(
        default=False,
        metadata={"help": "If True, apply the OOF-certified gold-free Top-2/Top-5 slate ranker after all evidence access operators."}
    )
    query_plan_ontology_metric_aligned_slate_ranker_model_path: str = field(
        default="",
        metadata={"help": "Path to the metric-aligned slate-ranker manifest."}
    )
    query_plan_ontology_enable_post_physical_delta_optimizer: bool = field(
        default=False,
        metadata={"help": "If True, run the composable delta optimizer after all base and physical evidence operators."}
    )
    query_plan_ontology_post_physical_delta_optimizer_model_path: str = field(
        default="",
        metadata={"help": "Path to the post-physical counterfactual delta model."}
    )
    query_plan_ontology_post_physical_delta_optimizer_candidate_mode: str = field(
        default="unified",
        metadata={"help": "Candidate relation used by the post-physical delta optimizer."}
    )
    query_plan_ontology_post_physical_delta_optimizer_enforce_cq: bool = field(
        default=True,
        metadata={"help": "Whether post-physical actions must satisfy competency-query constraints."}
    )
    query_plan_ontology_post_physical_delta_optimizer_shadow_only: bool = field(
        default=False,
        metadata={"help": "Compile post-physical actions without committing one, for aligned collection."}
    )
    query_plan_ontology_post_physical_delta_optimizer_max_candidates: int = field(
        default=24,
        metadata={"help": "Maximum candidates exposed to the post-physical delta optimizer."}
    )
    query_plan_ontology_enable_complete_plan_optimizer: bool = field(
        default=False,
        metadata={"help": "If True, choose among complete semantic, ontology, and explicit merge physical plans."}
    )
    query_plan_ontology_complete_plan_optimizer_model_path: str = field(
        default="",
        metadata={"help": "Path to the calibrated complete-plan portfolio CBO model."}
    )
    query_plan_ontology_complete_plan_optimizer_shadow_only: bool = field(
        default=False,
        metadata={"help": "Materialize all complete physical plans without changing the retrieval ranking."}
    )
    query_plan_ontology_complete_plan_optimizer_top_k: int = field(
        default=5,
        metadata={"help": "Evidence-set size jointly selected by the complete-plan optimizer."}
    )
    query_plan_ontology_complete_plan_optimizer_allowed_plans: str = field(
        default="",
        metadata={"help": "Optional comma-separated complete-plan families materialized before model selection; empty keeps the full catalog."}
    )
    query_plan_ontology_complete_plan_selector: str = field(
        default="full_cbo",
        metadata={"help": "Complete-plan selector used by controlled experiments: full_cbo, fixed, utility_only, random, or offline_oracle."}
    )
    query_plan_ontology_complete_plan_integrate_semantic_access: bool = field(
        default=False,
        metadata={"help": "Expose semantic/fact retrieval as an access operator inside the ontology-guided complete plan."}
    )
    query_plan_ontology_complete_plan_immutable_incumbent: bool = field(
        default=False,
        metadata={"help": "Use the frozen pre-ontology semantic/fact ranking as the atomic no-op incumbent."}
    )
    query_plan_ontology_enable_agentic_evidence_repair: bool = field(
        default=False,
        metadata={"help": "Run a bounded verifier-planner-critic loop over materialized evidence plans."}
    )
    query_plan_ontology_agentic_evidence_repair_shadow_only: bool = field(
        default=False,
        metadata={"help": "Audit agentic repair decisions without changing the reader slate."}
    )
    query_plan_ontology_agentic_evidence_repair_top_k: int = field(
        default=5,
        metadata={"help": "Atomic evidence-set size used by the agentic repair loop."}
    )
    query_plan_ontology_agentic_evidence_repair_max_iterations: int = field(
        default=2,
        metadata={"help": "Maximum bounded verify-repair-critic iterations per query."}
    )
    query_plan_ontology_agentic_evidence_repair_enable_top2: bool = field(
        default=True,
        metadata={"help": "Allow certified query-local plans to repair the top-2 slate."}
    )
    query_plan_ontology_agentic_evidence_repair_enable_top5: bool = field(
        default=True,
        metadata={"help": "Allow proof-complete residual repairs below the protected top-2."}
    )
    query_plan_ontology_agentic_evidence_repair_min_relation_gain: float = field(
        default=0.10,
        metadata={"help": "Minimum verifier-observed relation-coverage gain for a repair."}
    )
    query_plan_ontology_agentic_evidence_repair_max_top5_displacements: int = field(
        default=2,
        metadata={"help": "Maximum top-5 members displaced by a certified top-2 repair."}
    )
    query_plan_ontology_agentic_evidence_repair_max_estimated_cost: float = field(
        default=384.0,
        metadata={"help": "Maximum materialized physical-plan cost accepted by the critic."}
    )
    query_plan_ontology_agentic_evidence_repair_require_query_local_proof: bool = field(
        default=True,
        metadata={"help": "Restrict repair actions to proof-carrying query-local plan families."}
    )
    query_plan_ontology_agentic_evidence_repair_action_model_path: str = field(
        default="",
        metadata={"help": "Q_ind/Q_cal/Q_val-certified action model used by the agentic Delta-Critic."}
    )
    query_plan_ontology_agentic_evidence_repair_require_calibrated_delta: bool = field(
        default=False,
        metadata={"help": "Reject every agentic action that is not admitted by the certified Delta-Critic."}
    )
    query_plan_ontology_agentic_evidence_repair_skip_after_upstream_commit: bool = field(
        default=True,
        metadata={"help": "Preserve endpoint alignment by skipping agentic repair after an upstream CBO transaction commits."}
    )
    query_plan_ontology_enable_agentic_residual_retrieval: bool = field(
        default=False,
        metadata={"help": "Run the native five-role residual retrieval plan after complete-plan selection."}
    )
    query_plan_ontology_agentic_residual_shadow_only: bool = field(
        default=False,
        metadata={"help": "Materialize and trace the residual slate without committing its RRF result."}
    )
    query_plan_ontology_agentic_residual_max_rounds: int = field(
        default=3,
        metadata={"help": "Maximum verifier-planner-retriever-merger rounds."}
    )
    query_plan_ontology_agentic_residual_max_queries: int = field(
        default=3,
        metadata={"help": "Maximum residual retrieval queries issued in one round."}
    )
    query_plan_ontology_agentic_residual_max_snippets: int = field(
        default=3,
        metadata={"help": "Maximum current evidence snippets shown to each residual agent."}
    )
    query_plan_ontology_agentic_residual_max_tokens: int = field(
        default=160,
        metadata={"help": "Maximum completion tokens for residual planner and verifier calls."}
    )
    query_plan_ontology_agentic_residual_protect_top_k: int = field(
        default=1,
        metadata={"help": "Number of incumbent ranks protected inside the residual merger."}
    )
    query_plan_ontology_agentic_residual_max_promotions: int = field(
        default=2,
        metadata={"help": "Maximum residual candidates promoted by one merge."}
    )
    query_plan_ontology_agentic_residual_plan_top_k: int = field(
        default=20,
        metadata={"help": "Candidates considered from each residual retrieval query."}
    )
    query_plan_ontology_agentic_residual_min_plan_score: float = field(
        default=0.18,
        metadata={"help": "Minimum normalized residual retrieval score considered by the merger."}
    )
    query_plan_ontology_agentic_residual_merge_mode: str = field(
        default="guarded_promote",
        metadata={"help": "Residual merger policy copied from the strong retrieval loop."}
    )
    query_plan_ontology_agentic_residual_guarded_min_score: float = field(
        default=0.82,
        metadata={"help": "Minimum guarded promotion score."}
    )
    query_plan_ontology_agentic_residual_guarded_max_original_rank: int = field(
        default=80,
        metadata={"help": "Maximum original rank eligible for guarded residual promotion."}
    )
    query_plan_ontology_agentic_residual_guarded_max_plan_rank: int = field(
        default=8,
        metadata={"help": "Maximum within-query rank eligible for guarded residual promotion."}
    )
    query_plan_ontology_agentic_residual_guarded_min_query_overlap: float = field(
        default=0.08,
        metadata={"help": "Minimum lexical overlap between the original and residual queries."}
    )
    query_plan_ontology_agentic_residual_route_mode: str = field(
        default="adaptive",
        metadata={"help": "Label-free routing mode for residual retrieval."}
    )
    query_plan_ontology_agentic_residual_fusion_depth: int = field(
        default=5,
        metadata={"help": "Prefix depth used by complete-slate reciprocal-rank fusion."}
    )
    query_plan_ontology_agentic_residual_rrf_k: float = field(
        default=60.0,
        metadata={"help": "Rank constant for complete-slate reciprocal-rank fusion."}
    )
    query_plan_ontology_agentic_residual_base_weight: float = field(
        default=1.0,
        metadata={"help": "RRF weight for the incumbent complete plan."}
    )
    query_plan_ontology_agentic_residual_residual_weight: float = field(
        default=1.0,
        metadata={"help": "RRF weight for the agentic residual plan."}
    )
    query_plan_ontology_enable_certified_direct_execution: bool = field(
        default=False,
        metadata={"help": "Atomically send a certified ontology-derived top-k to the reader, with baseline used only as fallback."}
    )
    query_plan_ontology_require_certified_schema_mask: bool = field(
        default=False,
        metadata={"help": "Fail closed and restrict every ontology matcher, including compiler sidecars, to the frozen Q_val-certified schema mask."}
    )
    query_plan_ontology_certified_direct_policy_path: str = field(
        default="",
        metadata={"help": "Frozen Q_cal/Q_val deployment certificate for ontology-direct execution."}
    )
    query_plan_ontology_certified_direct_shadow_only: bool = field(
        default=False,
        metadata={"help": "Evaluate ontology-direct eligibility without changing the reader slate."}
    )
    query_plan_ontology_certified_direct_require_policy: bool = field(
        default=True,
        metadata={"help": "Require a hash-bound Q_val deployment certificate before direct execution."}
    )
    query_plan_ontology_certified_direct_top_k: int = field(
        default=5,
        metadata={"help": "Atomic ontology evidence-set size sent to the reader."}
    )
    query_plan_ontology_certified_direct_min_matched_plans: int = field(
        default=1,
        metadata={"help": "Minimum routed ontology schemas for direct execution."}
    )
    query_plan_ontology_certified_direct_min_strict_paths: int = field(
        default=1,
        metadata={"help": "Minimum strictly executable ontology paths for direct execution."}
    )
    query_plan_ontology_certified_direct_min_relation_coverage: float = field(
        default=0.35,
        metadata={"help": "Minimum mean relation-slot coverage of the ontology evidence slate."}
    )
    query_plan_ontology_certified_direct_min_answer_type_fraction: float = field(
        default=0.20,
        metadata={"help": "Minimum answer-type-compatible fraction of ontology evidence candidates."}
    )
    query_plan_ontology_certified_direct_require_complete_proof: bool = field(
        default=True,
        metadata={"help": "Require a complete local or single-passage proof before direct execution."}
    )
    query_plan_ontology_enable_missing_evidence_completer: bool = field(
        default=False,
        metadata={"help": "If True, use query-plan fact slots missing from the current top window to issue dense evidence-completion subqueries."}
    )
    query_plan_ontology_missing_evidence_dense_top_k: int = field(
        default=40,
        metadata={"help": "Dense top-k passages considered for each missing-evidence subquery."}
    )
    query_plan_ontology_missing_evidence_top_window_k: int = field(
        default=5,
        metadata={"help": "Top retrieval window inspected for already-covered query-plan fact slots."}
    )
    query_plan_ontology_missing_evidence_max_missing_slots: int = field(
        default=6,
        metadata={"help": "Maximum uncovered query-plan fact slots expanded into missing-evidence subqueries."}
    )
    query_plan_ontology_missing_evidence_max_query_variants: int = field(
        default=8,
        metadata={"help": "Maximum ontology-derived dense subqueries issued by the missing-evidence completer."}
    )
    query_plan_ontology_missing_evidence_max_candidates: int = field(
        default=3,
        metadata={"help": "Maximum verifier-approved missing-evidence passages exposed to downstream evidence-set selection."}
    )
    query_plan_ontology_missing_evidence_min_confidence: float = field(
        default=0.70,
        metadata={"help": "Minimum verifier confidence required before a dense-retrieved passage is accepted as missing evidence."}
    )
    query_plan_ontology_missing_evidence_promote_to_rank: int = field(
        default=5,
        metadata={"help": "Target rank for direct missing-evidence insertion when candidate-only mode is disabled."}
    )
    query_plan_ontology_missing_evidence_preserve_top_k: int = field(
        default=2,
        metadata={"help": "Number of top-ranked baseline passages protected from missing-evidence insertion."}
    )
    query_plan_ontology_missing_evidence_require_relation_hit: bool = field(
        default=True,
        metadata={"help": "If True, missing-evidence candidates must contain at least one non-generic relation token when the fact has relation clues."}
    )
    query_plan_ontology_missing_evidence_require_both_endpoints: bool = field(
        default=False,
        metadata={"help": "If True, missing-evidence candidates must mention both subject and object endpoints of the uncovered fact slot."}
    )
    query_plan_ontology_missing_evidence_candidate_only: bool = field(
        default=True,
        metadata={"help": "If True, missing-evidence retrieval only adds candidates to evidence-set selection instead of directly changing ranks."}
    )
    query_plan_ontology_missing_evidence_include_covered_slots: bool = field(
        default=False,
        metadata={"help": "If True, also retrieve alternative support passages for query-plan fact slots already covered by the current top window."}
    )
    query_plan_ontology_missing_evidence_max_covered_slots: int = field(
        default=4,
        metadata={"help": "Maximum already-covered fact slots expanded as alternative supporting evidence queries."}
    )
    query_plan_ontology_missing_evidence_score_margin: float = field(
        default=1e-6,
        metadata={"help": "Small score margin used for direct missing-evidence insertion when candidate-only mode is disabled."}
    )
    query_plan_ontology_enable_evidence_set_retriever: bool = field(
        default=False,
        metadata={"help": "If True, reselect the top evidence set by query-plan ontology slot coverage instead of boosting passages independently."}
    )
    query_plan_ontology_evidence_set_final_top_k: int = field(
        default=5,
        metadata={"help": "Top-k evidence window optimized by the query-plan ontology evidence-set retriever."}
    )
    query_plan_ontology_evidence_set_preserve_top_k: int = field(
        default=1,
        metadata={"help": "Number of strongest baseline passages preserved before ontology evidence-set selection."}
    )
    query_plan_ontology_evidence_set_candidate_pool_top_k: int = field(
        default=80,
        metadata={"help": "Maximum original retrieval rank considered by the ontology evidence-set retriever."}
    )
    query_plan_ontology_evidence_set_enable_optimizer_candidate_pool: bool = field(
        default=False,
        metadata={"help": "Expose bounded pre-filter evidence candidates to the learned optimizer."}
    )
    query_plan_ontology_evidence_set_optimizer_candidate_pool_max: int = field(
        default=80,
        metadata={"help": "Maximum pre-filter evidence candidates exposed to the learned optimizer."}
    )
    query_plan_ontology_evidence_set_enable_source_gap_aperture: bool = field(
        default=False,
        metadata={"help": "If True, allow high-utility source-gap candidates beyond the fixed evidence-set candidate pool."}
    )
    query_plan_ontology_evidence_set_source_gap_aperture_max_original_rank: int = field(
        default=900,
        metadata={"help": "Maximum original rank for source-gap aperture candidates; <=0 disables this bound."}
    )
    query_plan_ontology_evidence_set_source_gap_aperture_min_confidence: float = field(
        default=0.34,
        metadata={"help": "Minimum candidate confidence required by source-gap aperture admission."}
    )
    query_plan_ontology_evidence_set_source_gap_aperture_min_source_score: float = field(
        default=0.05,
        metadata={"help": "Minimum source prior score required by source-gap aperture admission unless the source is a recall operator."}
    )
    query_plan_ontology_evidence_set_source_gap_aperture_min_query_pressure: float = field(
        default=0.06,
        metadata={"help": "Minimum query-level recall pressure required by source-gap aperture admission unless the source is a recall operator."}
    )
    query_plan_ontology_evidence_set_source_gap_aperture_max_candidates: int = field(
        default=18,
        metadata={"help": "Maximum source-gap aperture candidates admitted per query; <=0 allows all approved candidates."}
    )
    query_plan_ontology_evidence_set_source_gap_aperture_score_weight: float = field(
        default=0.10,
        metadata={"help": "Score bonus weight assigned to source-gap aperture candidates."}
    )
    query_plan_ontology_evidence_set_source_gap_aperture_allow_no_new_slot: bool = field(
        default=True,
        metadata={"help": "If True, approved source-gap aperture candidates can bypass the strict new-slot filter."}
    )
    query_plan_ontology_evidence_set_max_ontology_passages: int = field(
        default=3,
        metadata={"help": "Maximum ontology-supported passages inserted into the selected evidence set."}
    )
    query_plan_ontology_evidence_set_min_candidate_confidence: float = field(
        default=0.72,
        metadata={"help": "Minimum ontology confidence required before a passage can enter the evidence set."}
    )
    query_plan_ontology_evidence_set_top2_min_candidate_confidence: float = field(
        default=0.92,
        metadata={"help": "Minimum confidence required before an ontology passage can alter the top-2 evidence window."}
    )
    query_plan_ontology_evidence_set_top2_max_original_rank: int = field(
        default=8,
        metadata={"help": "Maximum original rank for ontology passages allowed to enter top-2; <=0 disables this bound."}
    )
    query_plan_ontology_evidence_set_require_new_slot: bool = field(
        default=True,
        metadata={"help": "If True, ontology passages must add a new plan/role/relation/entity slot to enter the evidence set."}
    )
    query_plan_ontology_evidence_set_baseline_weight: float = field(
        default=0.45,
        metadata={"help": "Baseline rank weight used by ontology evidence-set selection."}
    )
    query_plan_ontology_evidence_set_ontology_weight: float = field(
        default=0.75,
        metadata={"help": "Ontology confidence weight used by ontology evidence-set selection."}
    )
    query_plan_ontology_evidence_set_coverage_weight: float = field(
        default=0.95,
        metadata={"help": "New slot coverage weight used by ontology evidence-set selection."}
    )
    query_plan_ontology_evidence_set_redundancy_penalty: float = field(
        default=0.28,
        metadata={"help": "Penalty for ontology evidence-set passages that repeat already covered slots."}
    )
    query_plan_ontology_evidence_set_coverage_reference_top_k: int = field(
        default=0,
        metadata={"help": "Top baseline passages used to decide whether an ontology candidate adds a genuinely missing slot; <=0 uses preserve_top_k."}
    )
    query_plan_ontology_evidence_set_enable_intervention_policy: bool = field(
        default=False,
        metadata={"help": "If True, ontology evidence-set candidates must pass a strong intervention gate before they can change the top evidence window."}
    )
    query_plan_ontology_evidence_set_intervention_min_confidence: float = field(
        default=0.90,
        metadata={"help": "Minimum candidate confidence required by the ontology intervention gate."}
    )
    query_plan_ontology_evidence_set_intervention_relation_only_min_confidence: float = field(
        default=0.995,
        metadata={"help": "Minimum confidence for relation-only candidates in simple relation queries when the ontology intervention gate is enabled."}
    )
    query_plan_ontology_evidence_set_intervention_relation_only_max_original_rank: int = field(
        default=20,
        metadata={"help": "Maximum original rank for relation-only candidates allowed by the ontology intervention gate; <=0 disables this bound."}
    )
    query_plan_ontology_evidence_set_intervention_allow_relation_only_simple: bool = field(
        default=True,
        metadata={"help": "If True, allow very high-confidence relation-only candidates for simple relation queries under the ontology intervention gate."}
    )
    query_plan_ontology_evidence_set_enable_counterfactual_policy: bool = field(
        default=False,
        metadata={"help": "If True, ontology evidence-set candidates must beat the baseline passage they would displace from top-k."}
    )
    query_plan_ontology_evidence_set_counterfactual_min_candidate_value: float = field(
        default=0.74,
        metadata={"help": "Minimum estimated evidence value required by the counterfactual replacement policy."}
    )
    query_plan_ontology_evidence_set_counterfactual_margin: float = field(
        default=0.03,
        metadata={"help": "Required estimated value margin over the displaced baseline passage."}
    )
    query_plan_ontology_evidence_set_policy_model_path: str = field(
        default="",
        metadata={"help": "Optional learned top-5 evidence-set intervention policy JSON."}
    )
    query_plan_ontology_evidence_set_policy_mode: str = field(
        default="off",
        metadata={"help": "How to use the learned evidence-set policy: off, score, filter, or selector."}
    )
    query_plan_ontology_evidence_set_policy_min_score: float = field(
        default=0.55,
        metadata={"help": "Policy score needed to promote an ontology candidate before evidence-set selection."}
    )
    query_plan_ontology_evidence_set_policy_reject_below: float = field(
        default=0.0,
        metadata={"help": "In filter mode, reject ontology evidence-set candidates below this policy score."}
    )
    query_plan_ontology_evidence_set_policy_score_weight: float = field(
        default=0.45,
        metadata={"help": "Score weight added to evidence-set selection from the learned policy score."}
    )
    query_plan_ontology_evidence_set_policy_promote_confidence: float = field(
        default=0.90,
        metadata={"help": "Confidence floor assigned when the learned evidence-set policy promotes a candidate."}
    )
    query_plan_ontology_evidence_set_policy_counterfactual_weight: float = field(
        default=0.20,
        metadata={"help": "Policy score contribution to candidate replacement value in counterfactual selection."}
    )
    query_plan_ontology_evidence_set_recall_selector_model_path: str = field(
        default="",
        metadata={"help": "Optional learned recall-oriented selector JSON for promoting gold-from-gt5 ontology candidates."}
    )
    query_plan_ontology_evidence_set_recall_selector_mode: str = field(
        default="off",
        metadata={"help": "How to use the recall selector: off, score, or selector."}
    )
    query_plan_ontology_evidence_set_recall_selector_min_score: float = field(
        default=0.62,
        metadata={"help": "Minimum recall-selector score required to promote an ontology candidate."}
    )
    query_plan_ontology_evidence_set_recall_selector_score_weight: float = field(
        default=0.55,
        metadata={"help": "Score weight added to evidence-set selection from the recall selector."}
    )
    query_plan_ontology_evidence_set_recall_selector_promote_confidence: float = field(
        default=0.97,
        metadata={"help": "Confidence floor assigned when the recall selector promotes a candidate."}
    )
    query_plan_ontology_evidence_set_recall_selector_counterfactual_weight: float = field(
        default=0.35,
        metadata={"help": "Recall-selector score contribution to counterfactual replacement value."}
    )
    query_plan_ontology_evidence_set_recall_selector_bypass_router_min_score: float = field(
        default=0.80,
        metadata={"help": "Minimum recall-selector score allowed to override a query-level router skip decision."}
    )
    query_plan_ontology_evidence_set_recall_swapper_model_path: str = field(
        default="",
        metadata={"help": "Optional learned oracle-gap recall swapper JSON for directly replacing top-k passages."}
    )
    query_plan_ontology_evidence_set_recall_swapper_mode: str = field(
        default="off",
        metadata={"help": "How to use the learned recall swapper: off, score, or selector."}
    )
    query_plan_ontology_evidence_set_recall_swapper_min_score: float = field(
        default=0.70,
        metadata={"help": "Minimum recall-swapper score required to promote an ontology candidate."}
    )
    query_plan_ontology_evidence_set_recall_swapper_top2_min_score: float = field(
        default=0.88,
        metadata={"help": "Minimum recall-swapper score required to bypass top-2 protection."}
    )
    query_plan_ontology_evidence_set_recall_swapper_score_weight: float = field(
        default=1.10,
        metadata={"help": "Score weight added to final evidence-window selection from the recall swapper."}
    )
    query_plan_ontology_evidence_set_recall_swapper_promote_confidence: float = field(
        default=0.98,
        metadata={"help": "Confidence floor assigned when the recall swapper promotes a candidate."}
    )
    query_plan_ontology_evidence_set_recall_swapper_counterfactual_weight: float = field(
        default=0.45,
        metadata={"help": "Recall-swapper score contribution to counterfactual replacement value."}
    )
    query_plan_ontology_evidence_set_recall_swapper_max_top2_swaps: int = field(
        default=1,
        metadata={"help": "Maximum recall-swapper bypass insertions allowed in the top-2 window per query."}
    )
    query_plan_ontology_evidence_set_recall_swapper_allow_relation_only_top2: bool = field(
        default=False,
        metadata={"help": "If True, allow high-scoring relation-only recall-swapper candidates to enter top-2."}
    )
    query_plan_ontology_evidence_set_rank2_rescue_model_path: str = field(
        default="",
        metadata={"help": "Optional learned high-precision rank-2 rescue policy JSON."}
    )
    query_plan_ontology_evidence_set_rank2_rescue_mode: str = field(
        default="off",
        metadata={"help": "How to use the learned rank-2 rescue policy: off, score, rescue, or high_confidence."}
    )
    query_plan_ontology_evidence_set_rank2_rescue_min_score: float = field(
        default=0.95,
        metadata={"help": "Minimum rank-2 rescue policy score required to replace baseline rank 2."}
    )
    query_plan_ontology_evidence_set_rank2_rescue_min_confidence: float = field(
        default=0.94,
        metadata={"help": "Minimum ontology candidate confidence for rank-2 rescue."}
    )
    query_plan_ontology_evidence_set_rank2_rescue_max_original_rank: int = field(
        default=80,
        metadata={"help": "Maximum original retrieval rank allowed for rank-2 rescue; <=0 disables this guard."}
    )
    query_plan_ontology_evidence_set_rank2_rescue_min_source_precision: float = field(
        default=0.20,
        metadata={"help": "Minimum retrieval-feedback source precision required for rank-2 rescue."}
    )
    query_plan_ontology_evidence_set_rank2_rescue_min_selected_precision: float = field(
        default=0.05,
        metadata={"help": "Minimum historical selected precision required for rank-2 rescue."}
    )
    query_plan_ontology_evidence_set_rank2_rescue_require_terminal_or_strict: bool = field(
        default=True,
        metadata={"help": "If True, rank-2 rescue requires terminal/strict/multihop/path evidence rather than relation-only evidence."}
    )
    query_plan_ontology_evidence_set_rank2_rescue_max_swaps: int = field(
        default=1,
        metadata={"help": "Maximum baseline rank-2 replacements allowed per query."}
    )
    query_plan_ontology_evidence_set_optimizer_mode: str = field(
        default="off",
        metadata={"help": "How to use the provenance-aware evidence-set optimizer: off or top2."}
    )
    query_plan_ontology_evidence_set_optimizer_min_confidence: float = field(
        default=0.90,
        metadata={"help": "Minimum ontology candidate confidence for the evidence-set optimizer."}
    )
    query_plan_ontology_evidence_set_optimizer_min_source_precision: float = field(
        default=0.30,
        metadata={"help": "Minimum retrieval-feedback source precision required by the evidence-set optimizer."}
    )
    query_plan_ontology_evidence_set_optimizer_min_path_source_precision: float = field(
        default=0.20,
        metadata={"help": "Lower source-precision threshold allowed when path-aware evidence approves the candidate."}
    )
    query_plan_ontology_evidence_set_optimizer_max_original_rank: int = field(
        default=20,
        metadata={"help": "Maximum original retrieval rank eligible for optimizer top-2 replacement; <=0 disables this guard."}
    )
    query_plan_ontology_evidence_set_optimizer_rank2_margin: float = field(
        default=0.03,
        metadata={"help": "Minimum estimated utility margin over the current rank-2 passage for optimizer replacement."}
    )
    query_plan_ontology_evidence_set_optimizer_require_complex_or_path: bool = field(
        default=True,
        metadata={"help": "If True, optimizer top-2 swaps require a complex query, path-approved evidence, or strict/multihop evidence."}
    )
    query_plan_ontology_evidence_set_optimizer_max_swaps: int = field(
        default=1,
        metadata={"help": "Maximum optimizer top-2 replacements allowed per query."}
    )
    query_plan_ontology_evidence_set_rank2_pairwise_optimizer_model_path: str = field(
        default="",
        metadata={"help": "Optional learned pairwise optimizer JSON for replacing the current rank-2 passage."}
    )
    query_plan_ontology_evidence_set_rank2_pairwise_optimizer_mode: str = field(
        default="off",
        metadata={"help": "How to use the learned rank-2 pairwise optimizer: off or top2."}
    )
    query_plan_ontology_evidence_set_rank2_pairwise_optimizer_min_score: float = field(
        default=0.85,
        metadata={"help": "Minimum pairwise optimizer score required to replace the current rank-2 passage."}
    )
    query_plan_ontology_evidence_set_rank2_pairwise_optimizer_min_confidence: float = field(
        default=0.0,
        metadata={"help": "Minimum ontology candidate confidence required by the rank-2 pairwise optimizer."}
    )
    query_plan_ontology_evidence_set_rank2_pairwise_optimizer_max_original_rank: int = field(
        default=120,
        metadata={"help": "Maximum original retrieval rank eligible for learned pairwise rank-2 replacement; <=0 disables this guard."}
    )
    query_plan_ontology_evidence_set_rank2_pairwise_optimizer_max_swaps: int = field(
        default=1,
        metadata={"help": "Maximum learned pairwise rank-2 replacements allowed per query."}
    )
    query_plan_ontology_evidence_set_rank2_pairwise_optimizer_min_value_margin: float = field(
        default=-1.0,
        metadata={"help": "Minimum estimated candidate-vs-current-rank2 utility margin required by the learned pairwise optimizer; < -0.999 disables this cost guard."}
    )
    query_plan_ontology_evidence_set_rank2_pairwise_optimizer_path_override_min_value_margin: float = field(
        default=-1.0,
        metadata={"help": "Relaxed value-margin threshold for strong path/terminal candidates; < -0.999 disables path override."}
    )
    query_plan_ontology_evidence_set_rank2_pairwise_optimizer_require_complex_or_path: bool = field(
        default=False,
        metadata={"help": "If True, learned pairwise top-2 swaps require a complex query or strong path/strict/multihop evidence."}
    )
    query_plan_ontology_evidence_set_rank2_pairwise_optimizer_after_evidence_optimizer: bool = field(
        default=False,
        metadata={"help": "If True, run the learned pairwise rank-2 optimizer after the provenance evidence-set optimizer."}
    )
    query_plan_ontology_evidence_set_utility_policy_model_path: str = field(
        default="",
        metadata={"help": "Optional learned utility policy JSON for deciding whether recall-oriented ontology candidates should alter top-k."}
    )
    query_plan_ontology_evidence_set_utility_policy_mode: str = field(
        default="off",
        metadata={"help": "How to use the utility policy: off, score, filter, or selector."}
    )
    query_plan_ontology_evidence_set_utility_policy_min_score: float = field(
        default=0.90,
        metadata={"help": "Utility-policy score required to promote an ontology candidate."}
    )
    query_plan_ontology_evidence_set_utility_policy_reject_below: float = field(
        default=0.0,
        metadata={"help": "In filter/selector mode, reject ontology candidates below this utility-policy score."}
    )
    query_plan_ontology_evidence_set_utility_policy_score_weight: float = field(
        default=0.35,
        metadata={"help": "Score weight added to evidence-set selection from the utility-policy score."}
    )
    query_plan_ontology_evidence_set_utility_policy_promote_confidence: float = field(
        default=0.97,
        metadata={"help": "Confidence floor assigned when the utility policy promotes a candidate."}
    )
    query_plan_ontology_evidence_set_utility_policy_counterfactual_weight: float = field(
        default=0.40,
        metadata={"help": "Utility-policy score contribution to candidate replacement value in counterfactual selection."}
    )
    query_plan_ontology_evidence_set_utility_policy_apply_scope: str = field(
        default="all",
        metadata={"help": "Candidate scope for utility policy: all, recall_selector, recall_only, or weak_only."}
    )
    query_plan_ontology_evidence_set_utility_planner_model_path: str = field(
        default="",
        metadata={"help": "Optional learned cost-aware utility planner JSON for evidence-set selection."}
    )
    query_plan_ontology_evidence_set_utility_planner_mode: str = field(
        default="off",
        metadata={"help": "How to use the cost-aware utility planner: off, score, planner, safe_planner, or position_planner."}
    )
    query_plan_ontology_evidence_set_utility_planner_min_score: float = field(
        default=0.78,
        metadata={"help": "Minimum planner score required for top-5 ontology insertion."}
    )
    query_plan_ontology_evidence_set_utility_planner_top2_min_score: float = field(
        default=0.90,
        metadata={"help": "Minimum planner score required for rank-2 replacement."}
    )
    query_plan_ontology_evidence_set_utility_planner_score_weight: float = field(
        default=0.65,
        metadata={"help": "Score weight added to evidence-set selection from the utility planner."}
    )
    query_plan_ontology_evidence_set_utility_planner_counterfactual_weight: float = field(
        default=0.55,
        metadata={"help": "Planner score contribution to candidate replacement value."}
    )
    query_plan_ontology_evidence_set_utility_planner_min_value_margin: float = field(
        default=0.02,
        metadata={"help": "Minimum candidate-vs-displaced top-5 utility margin for planner insertion."}
    )
    query_plan_ontology_evidence_set_utility_planner_top2_min_value_margin: float = field(
        default=0.04,
        metadata={"help": "Minimum candidate-vs-rank2 utility margin for planner rank-2 replacement."}
    )
    query_plan_ontology_evidence_set_utility_planner_max_original_rank: int = field(
        default=120,
        metadata={"help": "Maximum original retrieval rank eligible for utility-planner insertion; <=0 disables."}
    )
    query_plan_ontology_evidence_set_utility_planner_require_complex_or_path: bool = field(
        default=True,
        metadata={"help": "If True, planner insertions require complex query or strong path/endpoint evidence."}
    )
    query_plan_ontology_evidence_set_recall_plan_optimizer_mode: str = field(
        default="off",
        metadata={"help": "How to use the cost-based recall plan optimizer: off, score, or recall."}
    )
    query_plan_ontology_evidence_set_recall_plan_optimizer_stats_path: str = field(
        default="",
        metadata={"help": "Optional JSON source/operator utility stats for the cost-based recall plan optimizer."}
    )
    query_plan_ontology_evidence_set_recall_plan_optimizer_min_score: float = field(
        default=0.52,
        metadata={"help": "Minimum cost-benefit score required to promote a recall-plan candidate."}
    )
    query_plan_ontology_evidence_set_recall_plan_optimizer_score_weight: float = field(
        default=0.40,
        metadata={"help": "Score weight added to evidence-set selection from the recall plan optimizer."}
    )
    query_plan_ontology_evidence_set_recall_plan_optimizer_confidence_boost: float = field(
        default=0.06,
        metadata={"help": "Confidence boost scale for promoted recall-plan candidates."}
    )
    query_plan_ontology_evidence_set_recall_plan_optimizer_router_bypass_min_score: float = field(
        default=0.72,
        metadata={"help": "Minimum recall-plan score allowed to override a query-level router skip/precision gate."}
    )
    query_plan_ontology_evidence_set_recall_plan_optimizer_allow_no_new_slot: bool = field(
        default=True,
        metadata={"help": "If True, high-utility recall-plan candidates may bypass the strict new-slot guard."}
    )
    query_plan_ontology_evidence_set_recall_plan_optimizer_max_original_rank: int = field(
        default=220,
        metadata={"help": "Maximum original rank eligible for recall-plan promotion; <=0 disables this guard."}
    )
    query_plan_ontology_evidence_set_recall_plan_optimizer_max_selected: int = field(
        default=2,
        metadata={"help": "Maximum no-new-slot recall-plan insertions per query; <=0 disables this limit."}
    )
    query_plan_ontology_evidence_set_recall_plan_optimizer_min_source_precision: float = field(
        default=0.035,
        metadata={"help": "Minimum source/operator precision prior required for recall-plan promotion."}
    )
    query_plan_ontology_evidence_set_enable_rank_window_recall: bool = field(
        default=False,
        metadata={"help": "Expose dense rank-window passages as a cost-based recall access path."}
    )
    query_plan_ontology_evidence_set_rank_window_model_path: str = field(
        default="",
        metadata={"help": "Optional JSON scorer for the rank-window recall access path."}
    )
    query_plan_ontology_evidence_set_rank_window_target_datasets: str = field(
        default="",
        metadata={"help": "Comma-separated allowlist of datasets where rank-window recall is allowed; empty enables all datasets."}
    )
    query_plan_ontology_evidence_set_rank_window_start_rank: int = field(
        default=3,
        metadata={"help": "First original dense rank admitted by the rank-window access path."}
    )
    query_plan_ontology_evidence_set_rank_window_end_rank: int = field(
        default=24,
        metadata={"help": "Last original dense rank admitted by the rank-window access path."}
    )
    query_plan_ontology_evidence_set_rank_window_min_score: float = field(
        default=0.56,
        metadata={"help": "Minimum rank-window benefit score for top-5 recall insertion."}
    )
    query_plan_ontology_evidence_set_rank_window_top2_min_score: float = field(
        default=0.80,
        metadata={"help": "Minimum rank-window benefit score for rank-2 replacement."}
    )
    query_plan_ontology_evidence_set_rank_window_router_bypass_min_score: float = field(
        default=0.74,
        metadata={"help": "Rank-window score required to bypass query-level skip/precision routing."}
    )
    query_plan_ontology_evidence_set_rank_window_score_weight: float = field(
        default=0.18,
        metadata={"help": "Score weight assigned to rank-window candidates during evidence-set selection."}
    )
    query_plan_ontology_evidence_set_rank_window_confidence_floor: float = field(
        default=0.52,
        metadata={"help": "Confidence floor assigned to approved rank-window candidates."}
    )
    query_plan_ontology_evidence_set_rank_window_max_selected: int = field(
        default=2,
        metadata={"help": "Maximum rank-window candidates inserted per query."}
    )
    query_plan_ontology_evidence_set_rank_window_max_top2_swaps: int = field(
        default=1,
        metadata={"help": "Maximum rank-window replacements allowed in the top-2 window."}
    )
    query_plan_ontology_evidence_set_rank_window_min_margin: float = field(
        default=-0.12,
        metadata={"help": "Minimum candidate-value margin for rank-window top-5 replacement."}
    )
    query_plan_ontology_evidence_set_rank_window_top2_min_margin: float = field(
        default=-0.08,
        metadata={"help": "Minimum candidate-value margin for rank-window rank-2 replacement."}
    )
    query_plan_ontology_evidence_set_rank_window_require_text_signal: bool = field(
        default=False,
        metadata={"help": "Require non-empty text verifier evidence for rank-window candidates."}
    )
    query_plan_ontology_evidence_set_rank_window_min_text_score: float = field(
        default=0.0,
        metadata={"help": "Minimum text verifier score required for rank-window candidates."}
    )
    query_plan_ontology_evidence_set_rank_window_defer_to_planner: bool = field(
        default=False,
        metadata={"help": "Generate pure rank-window candidates as access-path candidates and let the replacement planner decide whether to insert them."}
    )
    query_plan_ontology_evidence_set_rank_window_protect_original_top_k: int = field(
        default=0,
        metadata={"help": "Protect baseline documents with original rank <= this value from rank-window replacement."}
    )
    query_plan_ontology_evidence_set_rank_window_min_replacement_rank: int = field(
        default=3,
        metadata={"help": "Minimum final rank position that rank-window planner may replace."}
    )
    query_plan_ontology_evidence_set_rank_window_query_router_mode: str = field(
        default="off",
        metadata={"help": "Rank-window query router mode: off/global or adaptive."}
    )
    query_plan_ontology_evidence_set_rank_window_complex_datasets: str = field(
        default="2wikimultihopqa,hotpotqa,musique",
        metadata={"help": "Datasets where rank-window is treated as a multi-hop/path access path."}
    )
    query_plan_ontology_evidence_set_rank_window_simple_datasets: str = field(
        default="popqa,nq_rear,narrativeqa_dev_10_doc,lveval",
        metadata={"help": "Datasets where rank-window needs stricter answer/entity evidence."}
    )
    query_plan_ontology_evidence_set_rank_window_complex_min_score: float = field(
        default=0.58,
        metadata={"help": "Adaptive-router min rank-window score for complex/path queries."}
    )
    query_plan_ontology_evidence_set_rank_window_complex_min_text_score: float = field(
        default=0.04,
        metadata={"help": "Adaptive-router min text score for complex/path queries."}
    )
    query_plan_ontology_evidence_set_rank_window_simple_min_score: float = field(
        default=0.86,
        metadata={"help": "Adaptive-router min rank-window score for simple/entity queries."}
    )
    query_plan_ontology_evidence_set_rank_window_simple_min_text_score: float = field(
        default=0.16,
        metadata={"help": "Adaptive-router min text score for simple/entity queries."}
    )
    query_plan_ontology_evidence_set_rank_window_simple_require_answer_signal: bool = field(
        default=True,
        metadata={"help": "Require terminal/answer-type/query text evidence for simple/entity rank-window candidates."}
    )
    query_plan_ontology_evidence_set_rank_window_simple_answer_signal_min_text_score: float = field(
        default=0.32,
        metadata={"help": "Text verifier score that counts as answer signal for simple/entity rank-window routing."}
    )
    query_plan_ontology_evidence_set_rank_window_enable_net_benefit_gate: bool = field(
        default=False,
        metadata={"help": "Require rank-window append/replacement candidates to pass a cost-based net-benefit gate."}
    )
    query_plan_ontology_evidence_set_rank_window_allow_no_new_slot_bypass: bool = field(
        default=True,
        metadata={"help": "Allow rank-window candidates to bypass the strict no-new-slot guard when the gate permits it."}
    )
    query_plan_ontology_evidence_set_rank_window_no_new_slot_min_score: float = field(
        default=0.0,
        metadata={"help": "Minimum rank-window score for a no-new-slot bypass under the net-benefit gate."}
    )
    query_plan_ontology_evidence_set_rank_window_no_new_slot_min_text_score: float = field(
        default=0.0,
        metadata={"help": "Minimum text score for a no-new-slot bypass under the net-benefit gate."}
    )
    query_plan_ontology_evidence_set_rank_window_no_new_slot_require_path_or_terminal: bool = field(
        default=False,
        metadata={"help": "Require path, terminal, or answer-overlap evidence for no-new-slot rank-window bypasses."}
    )
    query_plan_ontology_evidence_set_rank_window_net_benefit_min_margin: float = field(
        default=0.0,
        metadata={"help": "Minimum net-benefit margin for rank-window replacement when the gate is enabled."}
    )
    query_plan_ontology_evidence_set_rank_window_append_min_margin: float = field(
        default=0.0,
        metadata={"help": "Minimum net-benefit margin for rank-window append when the gate is enabled."}
    )
    query_plan_ontology_evidence_set_rank_window_displacement_cost_weight: float = field(
        default=0.85,
        metadata={"help": "Cost weight for the document displaced by a rank-window candidate."}
    )
    query_plan_ontology_evidence_set_enable_evidence_warehouse_optimizer: bool = field(
        default=False,
        metadata={"help": "Build a runtime evidence warehouse and run a cost-based evidence query optimizer over the final top-k window."}
    )
    query_plan_ontology_evidence_set_enable_final_rank_evidence_warehouse: bool = field(
        default=False,
        metadata={"help": "Apply the evidence warehouse optimizer after the full final rank has been assembled."}
    )
    query_plan_ontology_evidence_set_evidence_warehouse_target_datasets: str = field(
        default="",
        metadata={"help": "Comma-separated allowlist of datasets where the evidence warehouse optimizer is enabled; empty enables all datasets."}
    )
    query_plan_ontology_evidence_set_evidence_warehouse_max_candidates: int = field(
        default=96,
        metadata={"help": "Maximum evidence warehouse candidate rows optimized per query."}
    )
    query_plan_ontology_evidence_set_evidence_warehouse_max_top5_swaps: int = field(
        default=1,
        metadata={"help": "Maximum evidence warehouse replacements allowed inside the top-5 window after protected ranks."}
    )
    query_plan_ontology_evidence_set_evidence_warehouse_max_top2_swaps: int = field(
        default=0,
        metadata={"help": "Maximum evidence warehouse replacements allowed inside the top-2 window."}
    )
    query_plan_ontology_evidence_set_evidence_warehouse_protect_top_k: int = field(
        default=2,
        metadata={"help": "Protect baseline documents in final ranks <= this value from evidence warehouse replacement."}
    )
    query_plan_ontology_evidence_set_evidence_warehouse_max_original_rank: int = field(
        default=220,
        metadata={"help": "Maximum original rank eligible for evidence warehouse optimization; <=0 disables this bound."}
    )
    query_plan_ontology_evidence_set_evidence_warehouse_min_net_benefit: float = field(
        default=0.035,
        metadata={"help": "Minimum net benefit required for non-top-2 evidence warehouse actions."}
    )
    query_plan_ontology_evidence_set_evidence_warehouse_top2_min_net_benefit: float = field(
        default=0.16,
        metadata={"help": "Minimum net benefit required before the evidence warehouse optimizer can change top-2."}
    )
    query_plan_ontology_evidence_set_evidence_warehouse_min_benefit: float = field(
        default=0.52,
        metadata={"help": "Minimum raw utility benefit required before a warehouse row is eligible."}
    )
    query_plan_ontology_evidence_set_evidence_warehouse_require_answer_signal: bool = field(
        default=True,
        metadata={"help": "Require text/path/endpoint evidence related to the expected answer before a warehouse row is eligible."}
    )
    query_plan_ontology_evidence_set_evidence_warehouse_allow_relation_only: bool = field(
        default=False,
        metadata={"help": "Allow relation-only warehouse rows to be selected."}
    )
    query_plan_ontology_evidence_set_evidence_warehouse_cost_weight: float = field(
        default=0.56,
        metadata={"help": "Weight for replacement/displacement cost in the warehouse optimizer."}
    )
    query_plan_ontology_evidence_set_evidence_warehouse_risk_weight: float = field(
        default=0.36,
        metadata={"help": "Weight for hard-negative risk in the warehouse optimizer."}
    )
    query_plan_ontology_evidence_set_evidence_warehouse_text_weight: float = field(
        default=0.20,
        metadata={"help": "Weight for answer-aware text support in warehouse row benefit."}
    )
    query_plan_ontology_evidence_set_evidence_warehouse_path_weight: float = field(
        default=0.16,
        metadata={"help": "Weight for path-aware evidence support in warehouse row benefit."}
    )
    query_plan_ontology_evidence_set_evidence_warehouse_slot_weight: float = field(
        default=0.22,
        metadata={"help": "Weight for missing/novel schema slot coverage in warehouse row benefit."}
    )
    query_plan_ontology_evidence_set_evidence_warehouse_source_weight: float = field(
        default=0.18,
        metadata={"help": "Weight for source/provenance reliability in warehouse row benefit."}
    )
    query_plan_ontology_evidence_set_evidence_warehouse_selectivity_model_path: str = field(
        default="",
        metadata={"help": "Optional learned selectivity model for final-rank evidence warehouse actions."}
    )
    query_plan_ontology_evidence_set_evidence_warehouse_selectivity_min_score: float = field(
        default=0.54,
        metadata={"help": "Minimum learned selectivity score required for top-5 warehouse actions."}
    )
    query_plan_ontology_evidence_set_evidence_warehouse_selectivity_top2_min_score: float = field(
        default=0.68,
        metadata={"help": "Minimum learned selectivity score required for top-2 warehouse actions."}
    )
    query_plan_ontology_evidence_set_evidence_warehouse_selectivity_score_weight: float = field(
        default=0.18,
        metadata={"help": "Weight used to add learned selectivity margin into warehouse net-benefit ranking."}
    )
    query_plan_ontology_evidence_set_evidence_warehouse_selectivity_enable_recall_bypass: bool = field(
        default=False,
        metadata={"help": "If True, allow learned selectivity to re-admit high-signal candidates rejected by conservative warehouse row rules."}
    )
    query_plan_ontology_evidence_set_evidence_warehouse_selectivity_bypass_datasets: str = field(
        default="musique,nq_rear",
        metadata={"help": "Comma-separated datasets where learned recall bypass is allowed."}
    )
    query_plan_ontology_evidence_set_evidence_warehouse_selectivity_bypass_reasons: str = field(
        default="low_benefit,missing_answer_signal,relation_only",
        metadata={"help": "Comma-separated warehouse row rejection reasons eligible for learned recall bypass."}
    )
    query_plan_ontology_evidence_set_evidence_warehouse_selectivity_bypass_min_score: float = field(
        default=0.56,
        metadata={"help": "Minimum learned score required by the recall bypass gate."}
    )
    query_plan_ontology_evidence_set_evidence_warehouse_selectivity_bypass_max_original_rank: int = field(
        default=18,
        metadata={"help": "Maximum original rank eligible for learned recall bypass; <=0 disables the rank cap."}
    )
    query_plan_ontology_evidence_set_evidence_warehouse_selectivity_bypass_min_confidence: float = field(
        default=0.88,
        metadata={"help": "Minimum candidate confidence required for learned recall bypass."}
    )
    query_plan_ontology_evidence_set_evidence_warehouse_selectivity_bypass_min_text_score: float = field(
        default=0.18,
        metadata={"help": "Minimum text support score for weak non-structural learned recall bypass candidates."}
    )
    query_plan_ontology_evidence_set_evidence_warehouse_selectivity_bypass_max_risk: float = field(
        default=0.64,
        metadata={"help": "Maximum warehouse row risk allowed for learned recall bypass."}
    )
    query_plan_ontology_evidence_set_evidence_warehouse_selectivity_bypass_score_relaxation: float = field(
        default=0.08,
        metadata={"help": "Amount by which recall bypass may relax top-5 learned selectivity thresholds."}
    )
    query_plan_ontology_evidence_set_evidence_warehouse_selectivity_bypass_net_bonus: float = field(
        default=0.12,
        metadata={"help": "Controlled net-benefit bonus for recall-bypass candidates that pass learned selectivity."}
    )
    query_plan_ontology_evidence_set_evidence_warehouse_selectivity_enable_top2_guard_bypass: bool = field(
        default=False,
        metadata={"help": "Allow high-confidence learned replay selectivity to override selected conservative top-2 warehouse guards."}
    )
    query_plan_ontology_evidence_set_evidence_warehouse_selectivity_top2_guard_bypass_min_score: float = field(
        default=0.86,
        metadata={"help": "Minimum learned replay score required to override bypassable top-2 warehouse guards."}
    )
    query_plan_ontology_evidence_set_evidence_warehouse_selectivity_top2_guard_bypass_max_risk: float = field(
        default=0.16,
        metadata={"help": "Maximum warehouse row risk allowed for learned top-2 guard bypass."}
    )
    query_plan_ontology_evidence_set_evidence_warehouse_enable_action_counterfactual_gate: bool = field(
        default=False,
        metadata={"help": "Enable metric-aligned counterfactual action gates for final-rank evidence warehouse swaps."}
    )
    query_plan_ontology_evidence_set_evidence_warehouse_action_gate_top2_min_text_score: float = field(
        default=0.52,
        metadata={"help": "Minimum text score used by weak top-2 high-replacement warehouse action gates."}
    )
    query_plan_ontology_evidence_set_evidence_warehouse_action_gate_top2_min_selectivity_score: float = field(
        default=0.98,
        metadata={"help": "Minimum learned selectivity score used by weak top-2 high-replacement warehouse action gates."}
    )
    query_plan_ontology_evidence_set_evidence_warehouse_action_gate_top5_min_text_score: float = field(
        default=0.43,
        metadata={"help": "Minimum text score used by weak top-5 high-replacement warehouse action gates."}
    )
    query_plan_ontology_evidence_set_evidence_warehouse_action_gate_top5_min_selectivity_score: float = field(
        default=0.78,
        metadata={"help": "Minimum learned selectivity score used by weak top-5 high-replacement warehouse action gates."}
    )
    query_plan_ontology_evidence_set_evidence_warehouse_action_gate_nq_top5_min_rule_net: float = field(
        default=0.0,
        metadata={"help": "Minimum rule net benefit for NQ-Rear top-5 high-replacement warehouse actions."}
    )
    query_plan_ontology_evidence_set_evidence_warehouse_enable_set_coverage_cbo: bool = field(
        default=False,
        metadata={"help": "Enable set-level slot coverage optimization for final-rank evidence warehouse actions."}
    )
    query_plan_ontology_evidence_set_evidence_warehouse_set_coverage_weight: float = field(
        default=0.20,
        metadata={"help": "Weight added to warehouse net benefit for evidence-set slot coverage gain."}
    )
    query_plan_ontology_evidence_set_evidence_warehouse_set_coverage_min_gain: float = field(
        default=0.08,
        metadata={"help": "Minimum normalized coverage gain required for set-coverage warehouse bypass."}
    )
    query_plan_ontology_evidence_set_evidence_warehouse_set_coverage_max_risk: float = field(
        default=0.56,
        metadata={"help": "Maximum warehouse row risk allowed for set-coverage warehouse bypass."}
    )
    query_plan_ontology_evidence_set_evidence_warehouse_answer_aware_mode: str = field(
        default="off",
        metadata={"help": "Answer-aware cost model mode for final-rank warehouse actions: off, prior, model, or auto."}
    )
    query_plan_ontology_evidence_set_evidence_warehouse_answer_aware_model_path: str = field(
        default="",
        metadata={"help": "Optional learned answer-aware cost model path for final-rank warehouse actions."}
    )
    query_plan_ontology_evidence_set_evidence_warehouse_answer_aware_min_score: float = field(
        default=0.58,
        metadata={"help": "Minimum answer-aware score required for top-5 warehouse actions."}
    )
    query_plan_ontology_evidence_set_evidence_warehouse_answer_aware_top2_min_score: float = field(
        default=0.72,
        metadata={"help": "Minimum answer-aware score required for top-2 warehouse actions."}
    )
    query_plan_ontology_evidence_set_evidence_warehouse_answer_aware_score_weight: float = field(
        default=0.24,
        metadata={"help": "Weight used to add answer-aware score margin into warehouse net-benefit ranking."}
    )
    query_plan_ontology_evidence_set_evidence_warehouse_answer_aware_require_direct_signal: bool = field(
        default=False,
        metadata={"help": "If true, answer-aware warehouse actions require direct answer/terminal signal unless path-grounded."}
    )
    query_plan_ontology_evidence_set_evidence_warehouse_answer_aware_allow_top2_guard_bypass: bool = field(
        default=False,
        metadata={"help": "Allow high answer-aware score to bypass selected conservative top-2 warehouse guards."}
    )
    query_plan_ontology_evidence_set_evidence_warehouse_answer_aware_top2_guard_bypass_min_score: float = field(
        default=0.86,
        metadata={"help": "Minimum answer-aware score required to bypass selected top-2 warehouse guards."}
    )
    query_plan_ontology_evidence_set_evidence_warehouse_answer_aware_top2_guard_bypass_max_risk: float = field(
        default=0.18,
        metadata={"help": "Maximum warehouse row risk allowed for answer-aware top-2 guard bypass."}
    )
    query_plan_ontology_evidence_set_enable_top5_to_top2_promotion: bool = field(
        default=False,
        metadata={"help": "Enable a rank-2 evidence promotion operator that reorders high-confidence ontology-supported top-5 passages into rank 2."}
    )
    query_plan_ontology_evidence_set_top5_to_top2_promotion_target_datasets: str = field(
        default="",
        metadata={"help": "Comma-separated allowlist of datasets for top5-to-top2 evidence promotion; empty enables all datasets."}
    )
    query_plan_ontology_evidence_set_top5_to_top2_promotion_min_score: float = field(
        default=0.62,
        metadata={"help": "Minimum support score required before a top-5 evidence passage can be promoted to rank 2."}
    )
    query_plan_ontology_evidence_set_top5_to_top2_promotion_min_margin: float = field(
        default=0.055,
        metadata={"help": "Minimum support-score margin over the current rank-2 passage before promotion."}
    )
    query_plan_ontology_evidence_set_top5_to_top2_promotion_max_risk: float = field(
        default=0.48,
        metadata={"help": "Maximum warehouse risk allowed for rank-2 evidence promotion."}
    )
    query_plan_ontology_evidence_set_top5_to_top2_promotion_min_text_score: float = field(
        default=0.06,
        metadata={"help": "Minimum text score that can satisfy the promotion support gate."}
    )
    query_plan_ontology_evidence_set_top5_to_top2_promotion_min_path_score: float = field(
        default=0.48,
        metadata={"help": "Minimum path-aware score that can satisfy the promotion support gate."}
    )
    query_plan_ontology_evidence_set_top5_to_top2_promotion_min_source_precision: float = field(
        default=0.02,
        metadata={"help": "Minimum source precision that can satisfy the promotion support gate."}
    )
    query_plan_ontology_evidence_set_top5_to_top2_promotion_min_slot_value: float = field(
        default=0.04,
        metadata={"help": "Minimum query-slot value that can satisfy the promotion support gate."}
    )
    query_plan_ontology_evidence_set_top5_to_top2_promotion_require_ontology_signal: bool = field(
        default=True,
        metadata={"help": "Require promoted top-5 passages to have ontology-derived evidence contexts."}
    )
    query_plan_ontology_evidence_set_top5_to_top2_promotion_max_promotions: int = field(
        default=1,
        metadata={"help": "Maximum number of top5-to-top2 promotions per query."}
    )
    query_plan_ontology_evidence_set_enable_pairwise_top2_join: bool = field(
        default=False,
        metadata={"help": "Enable pairwise evidence-join optimization that directly rewrites the final top-2 evidence set."}
    )
    query_plan_ontology_evidence_set_pairwise_top2_join_target_datasets: str = field(
        default="",
        metadata={"help": "Comma-separated dataset allowlist for pairwise top-2 evidence join; empty enables all datasets."}
    )
    query_plan_ontology_evidence_set_pairwise_top2_join_candidate_pool: int = field(
        default=24,
        metadata={"help": "Maximum candidate documents considered by pairwise top-2 evidence join."}
    )
    query_plan_ontology_evidence_set_pairwise_top2_join_min_pair_score: float = field(
        default=0.60,
        metadata={"help": "Minimum absolute pair utility score required before pairwise top-2 join can rewrite the rank."}
    )
    query_plan_ontology_evidence_set_pairwise_top2_join_min_margin: float = field(
        default=0.015,
        metadata={"help": "Minimum pair utility margin over the current top-2 required by pairwise top-2 join."}
    )
    query_plan_ontology_evidence_set_pairwise_top2_join_min_text_score: float = field(
        default=0.10,
        metadata={"help": "Minimum best-document text support score required by pairwise top-2 join."}
    )
    query_plan_ontology_evidence_set_pairwise_top2_join_min_second_text_score: float = field(
        default=0.03,
        metadata={"help": "Minimum second-document text support score unless the pair has structural/path/answer evidence."}
    )
    query_plan_ontology_evidence_set_pairwise_top2_join_min_coverage_gain: float = field(
        default=0.08,
        metadata={"help": "Minimum slot coverage gain over the current top-2 for normal pairwise top-2 join actions."}
    )
    query_plan_ontology_evidence_set_pairwise_top2_join_max_original_rank: int = field(
        default=140,
        metadata={"help": "Maximum original rank eligible for pairwise top-2 join; <=0 disables this bound."}
    )
    query_plan_ontology_evidence_set_pairwise_top2_join_protect_top1: bool = field(
        default=False,
        metadata={"help": "If True, pairwise top-2 join must keep the original rank-1 document."}
    )
    query_plan_ontology_evidence_set_pairwise_top2_join_require_answer_signal: bool = field(
        default=False,
        metadata={"help": "If True, pairwise top-2 join requires answer/terminal/path evidence in the selected pair."}
    )
    query_plan_ontology_evidence_set_pairwise_top2_join_max_swaps: int = field(
        default=1,
        metadata={"help": "Maximum number of documents outside the current top-2 allowed in a pairwise top-2 rewrite."}
    )
    query_plan_ontology_evidence_set_enable_decomposed_chain_join: bool = field(
        default=False,
        metadata={"help": "Enable ontology-slot decomposition plus evidence-chain join for final top-2 optimization."}
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_target_datasets: str = field(
        default="",
        metadata={"help": "Comma-separated dataset allowlist for decomposed chain join; empty enables all datasets."}
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_candidate_pool: int = field(
        default=36,
        metadata={"help": "Maximum candidate documents considered by decomposed chain join."}
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_min_chain_score: float = field(
        default=0.58,
        metadata={"help": "Minimum absolute chain utility score required before decomposed chain join rewrites top-2."}
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_min_margin: float = field(
        default=0.030,
        metadata={"help": "Minimum chain-score margin over the current top-2 required by decomposed chain join."}
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_min_coverage_gain: float = field(
        default=0.06,
        metadata={"help": "Minimum ontology slot coverage gain required by decomposed chain join."}
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_max_original_rank: int = field(
        default=180,
        metadata={"help": "Maximum original rank eligible for decomposed chain join; <=0 disables this bound."}
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_max_swaps: int = field(
        default=2,
        metadata={"help": "Maximum number of documents outside the current top-2 allowed in a decomposed chain rewrite."}
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_protect_top1: bool = field(
        default=False,
        metadata={"help": "If True, decomposed chain join must keep the original rank-1 document."}
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_enable_decomposition: bool = field(
        default=True,
        metadata={"help": "Ablation switch: use ontology/query slot decomposition in decomposed chain join."}
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_enable_subquery_access: bool = field(
        default=True,
        metadata={"help": "Ablation switch: use role-aware subquery access-path scoring in decomposed chain join."}
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_enable_chain_join: bool = field(
        default=True,
        metadata={"help": "Ablation switch: use pairwise evidence-chain join instead of single-document support."}
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_require_bridge_signal: bool = field(
        default=True,
        metadata={"help": "If True, decomposed chain join requires anchor/bridge/answer role complementarity or bridge evidence."}
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_top2_target_datasets: str = field(
        default="",
        metadata={"help": "Comma-separated dataset allowlist where decomposed chain join may rewrite top-2; empty enables all targeted datasets."}
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_enable_top5_fallback: bool = field(
        default=False,
        metadata={"help": "If True, safe-router-rejected decomposed chains may enter top-5/context without changing top-2."}
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_top5_fallback_target_datasets: str = field(
        default="",
        metadata={"help": "Comma-separated dataset allowlist where decomposed chain join may use top-5 fallback; empty enables all targeted datasets."}
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_top5_require_answer_signal: bool = field(
        default=False,
        metadata={"help": "If True, top-5 fallback requires answer/terminal evidence in the selected chain."}
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_top5_min_chain_score: float = field(
        default=0.0,
        metadata={"help": "Minimum chain utility required by the top-5 fallback gate."}
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_top5_min_net_margin: float = field(
        default=-1.0,
        metadata={"help": "Minimum net margin required by the top-5 fallback gate."}
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_top5_min_text_score: float = field(
        default=0.0,
        metadata={"help": "Minimum best document text score required by the top-5 fallback gate."}
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_top5_max_risk: float = field(
        default=1.0,
        metadata={"help": "Maximum displacement/noise risk allowed by the top-5 fallback gate."}
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_top2_min_chain_score: float = field(
        default=0.58,
        metadata={"help": "Minimum chain utility required by the safe top-2 router."}
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_top2_min_margin: float = field(
        default=0.030,
        metadata={"help": "Minimum chain margin over current top-2 required by the safe top-2 router."}
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_top2_min_net_margin: float = field(
        default=0.030,
        metadata={"help": "Minimum chain net margin after displacement risk required by the safe top-2 router."}
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_top2_min_text_score: float = field(
        default=0.0,
        metadata={"help": "Minimum best document text score required by the safe top-2 router."}
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_top2_min_second_text_score: float = field(
        default=0.0,
        metadata={"help": "Minimum weaker document text score required by the safe top-2 router."}
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_top2_min_source_precision: float = field(
        default=0.0,
        metadata={"help": "Minimum evidence-source precision required by the safe top-2 router."}
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_top2_max_risk: float = field(
        default=1.0,
        metadata={"help": "Maximum displacement/noise risk allowed by the safe top-2 router."}
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_top2_require_answer_signal: bool = field(
        default=False,
        metadata={"help": "If True, safe top-2 routing requires answer/terminal evidence in the selected chain."}
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_top2_require_non_simple: bool = field(
        default=False,
        metadata={"help": "If True, safe top-2 routing is disabled for simple lookup/entity operators."}
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_enable_slot_completion_promotion: bool = field(
        default=False,
        metadata={"help": "Enable slot-completion promotion for partial evidence chains on targeted datasets."}
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_slot_completion_target_datasets: str = field(
        default="musique,nq_rear,popqa",
        metadata={"help": "Comma-separated dataset allowlist for slot-completion promotion."}
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_slot_completion_min_chain_score: float = field(
        default=0.46,
        metadata={"help": "Minimum chain score required by slot-completion promotion."}
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_slot_completion_min_required_gain: float = field(
        default=0.12,
        metadata={"help": "Minimum required-slot coverage gain required by slot-completion promotion."}
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_slot_completion_min_coverage_gain: float = field(
        default=0.05,
        metadata={"help": "Minimum ontology slot-value coverage gain required by slot-completion promotion."}
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_slot_completion_min_role_gain: float = field(
        default=0.0,
        metadata={"help": "Minimum anchor/bridge/answer role gain required by slot-completion promotion when positive."}
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_slot_completion_max_current_coverage: float = field(
        default=0.82,
        metadata={"help": "Maximum current top-2 required-slot coverage eligible for slot-completion promotion."}
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_slot_completion_min_text_score: float = field(
        default=0.03,
        metadata={"help": "Minimum best-document text score required by slot-completion promotion."}
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_slot_completion_min_second_text_score: float = field(
        default=0.0,
        metadata={"help": "Minimum weaker-document text score required by slot-completion promotion."}
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_slot_completion_max_risk: float = field(
        default=0.72,
        metadata={"help": "Maximum risk allowed by slot-completion promotion."}
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_slot_completion_allow_top1_replacement: bool = field(
        default=False,
        metadata={"help": "If True, slot-completion promotion may replace the original rank-1 document."}
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_slot_completion_require_bridge_or_answer: bool = field(
        default=True,
        metadata={"help": "If True, slot-completion promotion requires bridge or answer role evidence."}
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_enable_metric_regret_gate: bool = field(
        default=False,
        metadata={"help": "Enable metric-aligned regret gate before slot-completion chains can alter top-2."}
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_metric_regret_target_datasets: str = field(
        default="musique,nq_rear,popqa",
        metadata={"help": "Comma-separated dataset allowlist for decomposed-chain metric-regret gating."}
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_metric_regret_protect_top1: bool = field(
        default=True,
        metadata={"help": "If True, metric-regret gating protects the original rank-1 document unless an exceptional gain profile is observed."}
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_metric_regret_allow_top1_min_net_margin: float = field(
        default=0.55,
        metadata={"help": "Minimum net margin required before metric-regret gating allows rank-1 replacement."}
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_metric_regret_allow_top1_min_required_gain: float = field(
        default=0.85,
        metadata={"help": "Minimum required-slot gain required before metric-regret gating allows rank-1 replacement."}
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_metric_regret_allow_top1_min_text_score: float = field(
        default=0.55,
        metadata={"help": "Minimum weaker-document text score required before metric-regret gating allows rank-1 replacement."}
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_metric_regret_allow_top1_min_source_precision: float = field(
        default=0.80,
        metadata={"help": "Minimum source precision required before metric-regret gating allows rank-1 replacement."}
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_metric_regret_allow_top1_max_risk: float = field(
        default=0.08,
        metadata={"help": "Maximum risk allowed before metric-regret gating allows rank-1 replacement."}
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_metric_regret_rank2_min_net_margin: float = field(
        default=0.08,
        metadata={"help": "Minimum net margin required for rank-2-only decomposed-chain promotion under metric-regret gating."}
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_metric_regret_rank2_min_required_gain: float = field(
        default=0.12,
        metadata={"help": "Minimum required-slot gain required for rank-2-only decomposed-chain promotion under metric-regret gating."}
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_metric_regret_rank2_min_text_score: float = field(
        default=0.16,
        metadata={"help": "Minimum best-document text score required for rank-2-only decomposed-chain promotion under metric-regret gating."}
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_metric_regret_enable_top5_append: bool = field(
        default=True,
        metadata={"help": "If True, metric-regret vetoed top-2 actions may still append supporting evidence to top-5/context."}
    )
    query_plan_ontology_evidence_set_enable_regret_calibrated_set_optimizer: bool = field(
        default=False,
        metadata={"help": "Enable regret-calibrated set optimization over the final evidence window."}
    )
    query_plan_ontology_evidence_set_regret_calibrated_set_target_datasets: str = field(
        default="",
        metadata={"help": "Comma-separated dataset allowlist for regret-calibrated set optimization; empty enables all datasets."}
    )
    query_plan_ontology_evidence_set_regret_calibrated_set_max_candidates: int = field(
        default=160,
        metadata={"help": "Maximum candidate rows considered by regret-calibrated set optimization."}
    )
    query_plan_ontology_evidence_set_regret_calibrated_set_max_top2_swaps: int = field(
        default=1,
        metadata={"help": "Maximum regret-calibrated replacements allowed inside the top-2 window."}
    )
    query_plan_ontology_evidence_set_regret_calibrated_set_max_top5_swaps: int = field(
        default=1,
        metadata={"help": "Maximum regret-calibrated replacements allowed inside the top-5 window."}
    )
    query_plan_ontology_evidence_set_regret_calibrated_set_protect_top1: bool = field(
        default=True,
        metadata={"help": "If True, regret-calibrated set optimization cannot replace the original rank-1 document."}
    )
    query_plan_ontology_evidence_set_regret_calibrated_set_max_original_rank: int = field(
        default=220,
        metadata={"help": "Maximum original retrieval rank eligible for regret-calibrated set optimization; <=0 disables this bound."}
    )
    query_plan_ontology_evidence_set_regret_calibrated_set_min_regret: float = field(
        default=0.045,
        metadata={"help": "Minimum estimated metric regret reduction required for normal regret-calibrated set actions."}
    )
    query_plan_ontology_evidence_set_regret_calibrated_set_top2_min_regret: float = field(
        default=0.165,
        metadata={"help": "Minimum estimated metric regret reduction required for top-2 regret-calibrated set actions."}
    )
    query_plan_ontology_evidence_set_regret_calibrated_set_min_text_score: float = field(
        default=0.08,
        metadata={"help": "Minimum text support score required for weak regret-calibrated candidates."}
    )
    query_plan_ontology_evidence_set_regret_calibrated_set_top2_min_text_score: float = field(
        default=0.38,
        metadata={"help": "Minimum text support score required for weak top-2 regret-calibrated candidates."}
    )
    query_plan_ontology_evidence_set_regret_calibrated_set_min_source_precision: float = field(
        default=0.02,
        metadata={"help": "Minimum source/provenance precision required for weak regret-calibrated candidates."}
    )
    query_plan_ontology_evidence_set_regret_calibrated_set_require_answer_signal: bool = field(
        default=False,
        metadata={"help": "If True, regret-calibrated candidates must expose answer, terminal, path, or text evidence."}
    )
    query_plan_ontology_evidence_set_regret_calibrated_set_cost_weight: float = field(
        default=0.50,
        metadata={"help": "Cost weight for replacing existing evidence in regret-calibrated set optimization."}
    )
    query_plan_ontology_evidence_set_regret_calibrated_set_risk_weight: float = field(
        default=0.42,
        metadata={"help": "Hard-negative risk weight in regret-calibrated set optimization."}
    )
    query_plan_ontology_evidence_set_regret_calibrated_set_gap_weight: float = field(
        default=0.24,
        metadata={"help": "Weight for query-level recall pressure in regret-calibrated set optimization."}
    )
    query_plan_ontology_evidence_set_regret_calibrated_set_slot_weight: float = field(
        default=0.18,
        metadata={"help": "Weight for missing schema-slot repair in regret-calibrated set optimization."}
    )
    query_plan_ontology_evidence_set_regret_calibrated_set_max_risk: float = field(
        default=0.58,
        metadata={"help": "Maximum hard-negative risk allowed for normal regret-calibrated candidates."}
    )
    query_plan_ontology_evidence_set_regret_calibrated_set_top2_max_risk: float = field(
        default=0.22,
        metadata={"help": "Maximum hard-negative risk allowed for top-2 regret-calibrated candidates."}
    )
    query_plan_ontology_evidence_set_regret_calibrated_set_min_gap_pressure: float = field(
        default=0.04,
        metadata={"help": "Minimum recall pressure required before non-top-2 low-regret replacements are allowed."}
    )
    query_plan_ontology_evidence_set_regret_calibrated_set_enable_top5_safe_bypass: bool = field(
        default=False,
        metadata={"help": "If True, allow strong top-5 regret repairs to bypass conservative high-risk candidate gates."}
    )
    query_plan_ontology_evidence_set_regret_calibrated_set_top5_safe_bypass_max_risk: float = field(
        default=0.72,
        metadata={"help": "Maximum candidate risk allowed by top-5 safe regret bypass."}
    )
    query_plan_ontology_evidence_set_regret_calibrated_set_top5_safe_bypass_min_support: float = field(
        default=0.30,
        metadata={"help": "Minimum support signal required by top-5 safe regret bypass."}
    )
    query_plan_ontology_evidence_set_regret_calibrated_set_top5_safe_bypass_min_regret: float = field(
        default=0.075,
        metadata={"help": "Minimum scored regret required by top-5 safe regret bypass."}
    )
    query_plan_ontology_evidence_set_enable_evidence_sidecar: bool = field(
        default=False,
        metadata={"help": "If True, expose high-confidence ontology candidates as QA-only sidecar evidence without changing retrieved top-k order."}
    )
    query_plan_ontology_evidence_set_evidence_sidecar_max_items: int = field(
        default=2,
        metadata={"help": "Maximum QA-only evidence sidecar items emitted per query."}
    )
    query_plan_ontology_evidence_set_evidence_sidecar_min_score: float = field(
        default=0.48,
        metadata={"help": "Minimum sidecar utility score required before a candidate is exposed to QA."}
    )
    query_plan_ontology_evidence_set_evidence_sidecar_min_text_score: float = field(
        default=0.12,
        metadata={"help": "Minimum text support score required for weak non-structural sidecar candidates."}
    )
    query_plan_ontology_evidence_set_evidence_sidecar_max_original_rank: int = field(
        default=320,
        metadata={"help": "Maximum original retrieval rank eligible for QA-only evidence sidecar; <=0 disables this bound."}
    )
    query_plan_ontology_evidence_set_evidence_sidecar_allow_relation_only: bool = field(
        default=False,
        metadata={"help": "If True, relation-only candidates can be emitted as QA-only sidecar evidence."}
    )
    query_plan_ontology_evidence_set_evidence_sidecar_preferred_sources: str = field(
        default="evidence_grounded,support,path_executor,supervised_ranker,schema_slot_recall,rank_window,recall_plan,utility_planner,lifted",
        metadata={"help": "Comma-separated source labels that receive a sidecar utility bonus."}
    )
    query_plan_ontology_evidence_set_evidence_sidecar_gate_mode: str = field(
        default="loose",
        metadata={"help": "Sidecar gate mode: loose, strict, or complex_only_strict."}
    )
    query_plan_ontology_evidence_set_evidence_sidecar_complex_datasets: str = field(
        default="2wikimultihopqa,hotpotqa,musique",
        metadata={"help": "Comma-separated datasets routed to strict complex sidecar gating."}
    )
    query_plan_ontology_evidence_set_evidence_sidecar_simple_datasets: str = field(
        default="popqa,nq_rear,narrativeqa_dev_10_doc,lveval",
        metadata={"help": "Comma-separated datasets routed to conservative simple sidecar gating."}
    )
    query_plan_ontology_evidence_set_evidence_sidecar_strict_min_text_score: float = field(
        default=0.45,
        metadata={"help": "Minimum text support score required by strict sidecar gating."}
    )
    query_plan_ontology_evidence_set_evidence_sidecar_strict_min_path_score: float = field(
        default=0.68,
        metadata={"help": "Minimum path-aware score required by strict sidecar gating for path-derived sidecar items."}
    )
    query_plan_ontology_evidence_set_evidence_sidecar_require_anchor_hit: bool = field(
        default=False,
        metadata={"help": "If True, strict sidecar gating requires an anchor text hit."}
    )
    query_plan_ontology_evidence_set_evidence_sidecar_require_terminal_hit: bool = field(
        default=False,
        metadata={"help": "If True, strict sidecar gating requires a terminal or answer text hit."}
    )
    query_plan_ontology_evidence_set_evidence_sidecar_require_preferred_source: bool = field(
        default=False,
        metadata={"help": "If True, strict sidecar gating only admits candidates from preferred source labels."}
    )
    query_plan_ontology_evidence_set_evidence_sidecar_disable_simple_without_answer_overlap: bool = field(
        default=False,
        metadata={"help": "If True, simple-query sidecar items require terminal or answer overlap."}
    )
    query_plan_ontology_evidence_set_evidence_sidecar_disable_answer_type_conflict: bool = field(
        default=False,
        metadata={"help": "If True, strict sidecar gating rejects obvious expected-answer-type conflicts."}
    )
    query_plan_ontology_evidence_set_strict_backbone_scope: bool = field(
        default=False,
        metadata={"help": "When enabled, bare anchor-relation candidates are treated as weak evidence and must pass the utility policy unless retrieval feedback marks them as reliable."}
    )
    query_plan_ontology_evidence_set_anchor_relation_min_source_precision: float = field(
        default=0.30,
        metadata={"help": "Minimum retrieval-feedback precision for an anchor-relation candidate to be treated as backbone evidence under strict backbone scope."}
    )
    query_plan_ontology_evidence_set_anchor_relation_min_selected_precision: float = field(
        default=0.05,
        metadata={"help": "Minimum selected/gold@5 feedback precision for anchor-relation candidates to be treated as backbone evidence under strict backbone scope."}
    )
    query_plan_ontology_evidence_set_window_selector_model_path: str = field(
        default="",
        metadata={"help": "Optional learned query-level evidence-window selector JSON for choosing final top-k ontology insertions."}
    )
    query_plan_ontology_evidence_set_window_selector_mode: str = field(
        default="off",
        metadata={"help": "How to use the learned evidence-window selector: off, score, selector, or veto_rescue."}
    )
    query_plan_ontology_evidence_set_window_selector_min_score: float = field(
        default=0.60,
        metadata={"help": "Minimum window-selector score required before an ontology candidate may enter final top-k in selector mode."}
    )
    query_plan_ontology_evidence_set_window_selector_top2_min_score: float = field(
        default=0.90,
        metadata={"help": "Minimum window-selector score allowed to compete for top-2 slots when top-2 replacement is otherwise protected."}
    )
    query_plan_ontology_evidence_set_window_selector_score_weight: float = field(
        default=0.85,
        metadata={"help": "Score weight added to final evidence-window selection from the learned window selector."}
    )
    query_plan_ontology_evidence_set_window_selector_max_ontology_passages: int = field(
        default=0,
        metadata={"help": "Optional max ontology passages used when the learned window selector is active; <=0 keeps the global max."}
    )
    query_plan_ontology_evidence_set_window_selector_veto_below: float = field(
        default=0.0,
        metadata={"help": "In veto_rescue mode, veto non-backbone ontology candidates whose window-selector score is below this threshold."}
    )
    query_plan_ontology_evidence_set_window_selector_backbone_veto_below: float = field(
        default=0.0,
        metadata={"help": "In veto_rescue mode, veto backbone ontology candidates whose window-selector score is below this threshold."}
    )
    query_plan_ontology_evidence_set_window_selector_rescue_min_score: float = field(
        default=0.85,
        metadata={"help": "In veto_rescue mode, minimum window-selector score for an extra source-aware evidence-grounded rescue candidate."}
    )
    query_plan_ontology_evidence_set_window_selector_rescue_max_passages: int = field(
        default=0,
        metadata={"help": "In veto_rescue mode, maximum number of rescue candidates allowed beyond the source-aware evidence-grounded one-per-query guard."}
    )
    query_plan_ontology_evidence_set_router_model_path: str = field(
        default="",
        metadata={"help": "Optional learned query-level router JSON for deciding whether ontology evidence-set intervention should run."}
    )
    query_plan_ontology_evidence_set_router_mode: str = field(
        default="off",
        metadata={"help": "How to use the query-level evidence-set intervention router: off, filter, or action."}
    )
    query_plan_ontology_evidence_set_router_min_score: float = field(
        default=0.50,
        metadata={"help": "Minimum query-level router score required before ontology evidence-set intervention can alter top-k."}
    )
    query_plan_ontology_evidence_set_router_precision_min_score: float = field(
        default=0.50,
        metadata={"help": "In action mode, minimum router score for high-precision low-budget ontology intervention."}
    )
    query_plan_ontology_evidence_set_router_precision_min_confidence: float = field(
        default=0.90,
        metadata={"help": "In action mode, minimum candidate confidence in high-precision intervention mode."}
    )
    query_plan_ontology_evidence_set_router_precision_max_ontology_passages: int = field(
        default=1,
        metadata={"help": "In action mode, maximum ontology passages selected for high-precision intervention."}
    )
    query_plan_ontology_evidence_set_schema_route_adapter_mode: str = field(
        default="off",
        metadata={"help": "Youtu-style schema route adapter mode for evidence-set candidates: off, score, filter, or selector."}
    )
    query_plan_ontology_evidence_set_schema_route_adapter_score_weight: float = field(
        default=0.18,
        metadata={"help": "Score weight added from schema-guided entity/triple/path/community route coverage."}
    )
    query_plan_ontology_evidence_set_schema_route_adapter_confidence_boost: float = field(
        default=0.04,
        metadata={"help": "Candidate confidence boost scale when schema-route coverage is strong."}
    )
    query_plan_ontology_evidence_set_schema_route_adapter_min_score: float = field(
        default=0.30,
        metadata={"help": "Schema-route score threshold used for promotion-style confidence boosts."}
    )
    query_plan_ontology_evidence_set_schema_route_adapter_reject_below: float = field(
        default=0.0,
        metadata={"help": "In filter/selector mode, reject schema-route candidates below this score."}
    )
    query_plan_ontology_evidence_set_schema_route_adapter_max_subqueries: int = field(
        default=6,
        metadata={"help": "Maximum schema-guided atomic route probes recorded per query."}
    )
    query_plan_ontology_evidence_set_enable_schema_slot_utility: bool = field(
        default=False,
        metadata={"help": "If True, score schema-slot recall candidates with a utility gate before final evidence-set ranking."}
    )
    query_plan_ontology_evidence_set_schema_slot_utility_min_score: float = field(
        default=0.70,
        metadata={"help": "Minimum schema-slot utility score required before a schema-slot candidate is trusted."}
    )
    query_plan_ontology_evidence_set_schema_slot_utility_top2_min_score: float = field(
        default=0.86,
        metadata={"help": "Minimum schema-slot utility score required before a schema-slot candidate can replace rank 2."}
    )
    query_plan_ontology_evidence_set_schema_slot_utility_score_weight: float = field(
        default=0.45,
        metadata={"help": "Score weight added to evidence-set ranking from schema-slot utility."}
    )
    query_plan_ontology_evidence_set_schema_slot_utility_confidence_boost: float = field(
        default=0.08,
        metadata={"help": "Confidence boost applied to high-utility schema-slot candidates."}
    )
    query_plan_ontology_evidence_set_schema_slot_utility_top2_max_swaps: int = field(
        default=1,
        metadata={"help": "Maximum number of rank-2 replacements allowed by schema-slot utility per query."}
    )
    query_plan_ontology_evidence_set_enable_provenance_groupwise_selector: bool = field(
        default=False,
        metadata={"help": "Use provenance/text/source-aware groupwise replacement for evidence-set top-5 repair."}
    )
    query_plan_ontology_evidence_set_provenance_groupwise_min_score: float = field(
        default=0.68,
        metadata={"help": "Minimum provenance-groupwise score required before a candidate can repair top-5."}
    )
    query_plan_ontology_evidence_set_provenance_groupwise_top2_min_score: float = field(
        default=0.92,
        metadata={"help": "Minimum provenance-groupwise score required before a candidate can replace rank 2."}
    )
    query_plan_ontology_evidence_set_provenance_groupwise_score_weight: float = field(
        default=0.45,
        metadata={"help": "Score weight added from provenance-groupwise candidate utility."}
    )
    query_plan_ontology_evidence_set_provenance_groupwise_max_top5_swaps: int = field(
        default=2,
        metadata={"help": "Maximum number of top-5 replacements allowed by provenance-groupwise selector per query."}
    )
    query_plan_ontology_evidence_set_provenance_groupwise_max_top2_swaps: int = field(
        default=1,
        metadata={"help": "Maximum number of rank-2 replacements allowed by provenance-groupwise selector per query."}
    )
    query_plan_ontology_evidence_set_provenance_groupwise_min_source_precision: float = field(
        default=0.06,
        metadata={"help": "Minimum source-level precision needed by provenance-groupwise selector unless structure is strong."}
    )
    query_plan_ontology_evidence_set_provenance_groupwise_require_text_signal: bool = field(
        default=True,
        metadata={"help": "Require passage text signal for provenance-groupwise candidates unless structure is strong."}
    )
    query_plan_ontology_evidence_set_enable_evidence_join_planner: bool = field(
        default=False,
        metadata={"help": "Use cost-based evidence join optimization for ontology evidence-set top-k selection."}
    )
    query_plan_ontology_evidence_set_evidence_join_planner_min_score: float = field(
        default=0.58,
        metadata={"help": "Minimum static evidence-join profile score before a candidate is eligible."}
    )
    query_plan_ontology_evidence_set_evidence_join_planner_top2_min_score: float = field(
        default=0.82,
        metadata={"help": "Minimum evidence-join profile score before a candidate can replace rank 2."}
    )
    query_plan_ontology_evidence_set_evidence_join_planner_min_marginal_gain: float = field(
        default=0.30,
        metadata={"help": "Minimum marginal slot gain required by evidence-join planner."}
    )
    query_plan_ontology_evidence_set_evidence_join_planner_min_value_margin: float = field(
        default=-0.02,
        metadata={"help": "Minimum utility margin over the replaced passage for evidence-join planner."}
    )
    query_plan_ontology_evidence_set_evidence_join_planner_score_weight: float = field(
        default=0.50,
        metadata={"help": "Selection score weight from evidence-join profile score."}
    )
    query_plan_ontology_evidence_set_evidence_join_planner_marginal_gain_weight: float = field(
        default=0.45,
        metadata={"help": "Selection score weight from evidence-join marginal slot gain."}
    )
    query_plan_ontology_evidence_set_evidence_join_planner_max_top5_swaps: int = field(
        default=2,
        metadata={"help": "Maximum top-5 replacements allowed by evidence-join planner."}
    )
    query_plan_ontology_evidence_set_evidence_join_planner_max_top2_swaps: int = field(
        default=1,
        metadata={"help": "Maximum rank-2 replacements allowed by evidence-join planner."}
    )
    query_plan_ontology_evidence_set_evidence_join_planner_router_bypass_min_score: float = field(
        default=0.68,
        metadata={"help": "Evidence-join profile score needed to bypass the old intervention policy."}
    )
    query_plan_ontology_evidence_set_evidence_join_planner_require_text_or_path: bool = field(
        default=True,
        metadata={"help": "Require text, path, or structural support for evidence-join candidates."}
    )
    query_plan_ontology_evidence_set_evidence_join_planner_enable_answer_slot_rescue: bool = field(
        default=False,
        metadata={"help": "Allow evidence-join candidates with strong answer-slot evidence to bypass conservative gates."}
    )
    query_plan_ontology_evidence_set_evidence_join_planner_answer_slot_score_weight: float = field(
        default=0.22,
        metadata={"help": "Selection score weight from evidence-join answer-slot compatibility."}
    )
    query_plan_ontology_evidence_set_evidence_join_planner_answer_slot_min_score: float = field(
        default=0.52,
        metadata={"help": "Minimum answer-slot score for evidence-join top-5 rescue eligibility."}
    )
    query_plan_ontology_evidence_set_evidence_join_planner_answer_slot_top2_min_score: float = field(
        default=0.70,
        metadata={"help": "Minimum answer-slot score for evidence-join rank-2 rescue eligibility."}
    )
    query_plan_ontology_evidence_set_evidence_join_planner_answer_slot_router_bypass_min_score: float = field(
        default=0.76,
        metadata={"help": "Answer-slot score needed to bypass the intervention router for evidence-join candidates."}
    )
    query_plan_ontology_evidence_set_evidence_join_planner_enable_operator_aware: bool = field(
        default=False,
        metadata={"help": "Make evidence join scoring and replacement gates aware of query operator and missing join slots."}
    )
    query_plan_ontology_evidence_set_evidence_join_planner_operator_score_weight: float = field(
        default=0.18,
        metadata={"help": "Selection score weight from operator-aware evidence join compatibility."}
    )
    query_plan_ontology_evidence_set_evidence_join_planner_operator_min_gain: float = field(
        default=0.14,
        metadata={"help": "Minimum operator-slot marginal gain for top-5 evidence join replacements."}
    )
    query_plan_ontology_evidence_set_evidence_join_planner_operator_top2_min_gain: float = field(
        default=0.24,
        metadata={"help": "Minimum operator-slot marginal gain for rank-2 evidence join replacements."}
    )
    query_plan_ontology_evidence_set_evidence_join_planner_operator_chain_pressure_weight: float = field(
        default=0.10,
        metadata={"help": "Extra score weight for complex operator chain pressure when candidate operator evidence matches."}
    )
    query_plan_ontology_evidence_set_evidence_join_planner_enable_risk_gate: bool = field(
        default=False,
        metadata={"help": "Enable counterfactual risk gate before evidence-join top-k replacement."}
    )
    query_plan_ontology_evidence_set_evidence_join_planner_risk_gate_top2_min_source_precision: float = field(
        default=0.18,
        metadata={"help": "Minimum source precision for evidence-join rank-2 replacement unless answer-slot rescue fires."}
    )
    query_plan_ontology_evidence_set_evidence_join_planner_risk_gate_top5_min_source_precision: float = field(
        default=0.06,
        metadata={"help": "Minimum source precision for evidence-join top-5 replacement without strong answer signal."}
    )
    query_plan_ontology_evidence_set_evidence_join_planner_risk_gate_top2_min_margin: float = field(
        default=0.10,
        metadata={"help": "Minimum utility margin for evidence-join rank-2 replacement when risk gate is enabled."}
    )
    query_plan_ontology_evidence_set_evidence_join_planner_risk_gate_top5_min_margin: float = field(
        default=0.03,
        metadata={"help": "Minimum utility margin for evidence-join top-5 replacement when risk gate is enabled."}
    )
    query_plan_ontology_evidence_set_evidence_join_planner_risk_gate_max_lost_value: float = field(
        default=0.36,
        metadata={"help": "Maximum allowed lost query-slot value for evidence-join replacement without new query slots."}
    )
    query_plan_ontology_evidence_set_enable_answer_pairwise_verifier: bool = field(
        default=False,
        metadata={"help": "Use answer-aware pairwise verification before ontology evidence-set replacement."}
    )
    query_plan_ontology_evidence_set_answer_pairwise_min_score: float = field(
        default=0.56,
        metadata={"help": "Minimum answer-pairwise score before a candidate can replace top-5."}
    )
    query_plan_ontology_evidence_set_answer_pairwise_top2_min_score: float = field(
        default=0.64,
        metadata={"help": "Minimum answer-pairwise score before a candidate can replace rank 2."}
    )
    query_plan_ontology_evidence_set_answer_pairwise_min_margin: float = field(
        default=0.04,
        metadata={"help": "Minimum pairwise utility margin for top-5 answer-aware replacement."}
    )
    query_plan_ontology_evidence_set_answer_pairwise_top2_min_margin: float = field(
        default=0.08,
        metadata={"help": "Minimum pairwise utility margin for rank-2 answer-aware replacement."}
    )
    query_plan_ontology_evidence_set_answer_pairwise_max_top5_swaps: int = field(
        default=2,
        metadata={"help": "Maximum top-5 replacements allowed by answer-aware pairwise verifier."}
    )
    query_plan_ontology_evidence_set_answer_pairwise_max_top2_swaps: int = field(
        default=1,
        metadata={"help": "Maximum rank-2 replacements allowed by answer-aware pairwise verifier."}
    )
    query_plan_ontology_evidence_set_answer_pairwise_require_answer_signal: bool = field(
        default=True,
        metadata={"help": "Require answer/terminal/text/path signal before answer-aware pairwise replacement."}
    )
    query_plan_ontology_evidence_set_answer_pairwise_score_weight: float = field(
        default=0.30,
        metadata={"help": "Selection score weight from answer-aware pairwise verifier score."}
    )
    query_plan_ontology_evidence_set_enable_counterfactual_text_verifier: bool = field(
        default=False,
        metadata={"help": "Use passage-text counterfactual comparison before final evidence-set replacements."}
    )
    query_plan_ontology_evidence_set_counterfactual_text_verifier_min_score: float = field(
        default=0.62,
        metadata={"help": "Minimum counterfactual text score before a candidate can repair top-5."}
    )
    query_plan_ontology_evidence_set_counterfactual_text_verifier_top2_min_score: float = field(
        default=0.74,
        metadata={"help": "Minimum counterfactual text score before a candidate can replace rank 2."}
    )
    query_plan_ontology_evidence_set_counterfactual_text_verifier_min_margin: float = field(
        default=0.10,
        metadata={"help": "Minimum utility margin over the replaced top-5 passage."}
    )
    query_plan_ontology_evidence_set_counterfactual_text_verifier_top2_min_margin: float = field(
        default=0.16,
        metadata={"help": "Minimum utility margin over the current rank-2 passage."}
    )
    query_plan_ontology_evidence_set_counterfactual_text_verifier_max_top5_swaps: int = field(
        default=1,
        metadata={"help": "Maximum top-5 swaps allowed by counterfactual text verification per query."}
    )
    query_plan_ontology_evidence_set_counterfactual_text_verifier_max_top2_swaps: int = field(
        default=1,
        metadata={"help": "Maximum rank-2 swaps allowed by counterfactual text verification per query."}
    )
    query_plan_ontology_evidence_set_counterfactual_text_verifier_max_original_rank: int = field(
        default=80,
        metadata={"help": "Maximum original rank eligible for counterfactual text verification; <=0 disables."}
    )
    query_plan_ontology_evidence_set_counterfactual_text_verifier_min_candidate_text_score: float = field(
        default=0.38,
        metadata={"help": "Minimum candidate text score unless answer/path evidence is strong."}
    )
    query_plan_ontology_evidence_set_counterfactual_text_verifier_score_weight: float = field(
        default=0.0,
        metadata={"help": "Optional soft ranking weight from counterfactual text score."}
    )
    query_plan_ontology_evidence_set_counterfactual_text_verifier_require_answer_signal: bool = field(
        default=True,
        metadata={"help": "Require answer/terminal/path/text evidence before counterfactual replacement."}
    )
    query_plan_ontology_evidence_set_enable_decomposition_slots: bool = field(
        default=False,
        metadata={"help": "Use schema-decomposed sub-question slots as an evidence-set coverage signal."}
    )
    query_plan_ontology_evidence_set_decomposition_slot_weight: float = field(
        default=0.35,
        metadata={"help": "Score weight added when a candidate covers a new schema-decomposition slot."}
    )
    query_plan_ontology_evidence_set_decomposition_slot_max_slots: int = field(
        default=6,
        metadata={"help": "Maximum schema-decomposition slots extracted per candidate passage."}
    )
    query_plan_ontology_evidence_set_enable_path_aware_selector: bool = field(
        default=False,
        metadata={"help": "If True, use verified ontology path features as an explicit evidence-set selection signal."}
    )
    query_plan_ontology_evidence_set_path_aware_score_weight: float = field(
        default=0.0,
        metadata={"help": "Score weight added to evidence-set selection from verified ontology path support."}
    )
    query_plan_ontology_evidence_set_path_aware_confidence_boost: float = field(
        default=0.0,
        metadata={"help": "Confidence boost applied to candidates with approved verified ontology path support."}
    )
    query_plan_ontology_evidence_set_path_aware_min_score: float = field(
        default=0.65,
        metadata={"help": "Minimum path-aware score required before a path-supported candidate is promoted."}
    )
    query_plan_ontology_evidence_set_path_aware_min_terminal_type_score: float = field(
        default=0.45,
        metadata={"help": "Minimum terminal answer-type score required for path-aware candidate promotion."}
    )
    query_plan_ontology_evidence_set_path_aware_require_query_relation_coverage: bool = field(
        default=True,
        metadata={"help": "If True, path-aware promotion requires full coverage of strong query relation tokens."}
    )
    query_plan_ontology_evidence_set_text_verifier_mode: str = field(
        default="off",
        metadata={"help": "Use passage text to verify ontology evidence candidates: off, score, or filter."}
    )
    query_plan_ontology_evidence_set_text_verifier_min_score: float = field(
        default=0.55,
        metadata={"help": "Text-verifier score required to promote an ontology evidence candidate."}
    )
    query_plan_ontology_evidence_set_text_verifier_reject_below: float = field(
        default=0.0,
        metadata={"help": "In filter mode, reject weak non-strict candidates below this text-verifier score."}
    )
    query_plan_ontology_evidence_set_text_verifier_score_weight: float = field(
        default=0.20,
        metadata={"help": "Score weight added to evidence-set selection from passage-text verification."}
    )
    query_plan_ontology_evidence_set_text_verifier_confidence_boost: float = field(
        default=0.04,
        metadata={"help": "Small confidence boost for candidates whose passage text supports query/schema evidence."}
    )
    query_plan_ontology_evidence_set_text_verifier_max_chars: int = field(
        default=1600,
        metadata={"help": "Maximum passage characters used by the lightweight text verifier."}
    )
    query_plan_ontology_lift_min_original_rank: int = field(
        default=1,
        metadata={"help": "Minimum original passage rank eligible for query-plan lift; <=0 disables the lower bound."}
    )
    query_plan_ontology_lift_max_original_rank: int = field(
        default=20,
        metadata={"help": "Maximum original passage rank eligible for query-plan lift; <=0 disables the upper bound."}
    )
    query_plan_ontology_soft_fallback_lift_max_original_rank: int = field(
        default=5,
        metadata={"help": "Maximum original passage rank eligible when all query-plan evidence came from soft fallback; <=0 disables this soft-only bound."}
    )
    query_plan_ontology_soft_fallback_weight_scale: float = field(
        default=0.35,
        metadata={"help": "Multiplier applied to passage lift evidence that only came from soft fallback paths."}
    )
    query_plan_ontology_validated_soft_max_paths: int = field(
        default=4,
        metadata={"help": "Maximum provenance-validated soft paths added beside strict paths in strict_plus_validated_soft mode."}
    )
    query_plan_ontology_validated_soft_min_answer_path_score: float = field(
        default=0.50,
        metadata={"help": "Minimum query/answer-path compatibility score for provenance-validated soft paths."}
    )
    query_plan_ontology_gate_require_fact_endpoints: bool = field(
        default=True,
        metadata={"help": "If True, lift only passages that contain both endpoints of the lifted fact."}
    )
    query_plan_ontology_gate_require_anchor_or_terminal: bool = field(
        default=True,
        metadata={"help": "If True, first-hop passages must contain the query anchor and last-hop passages must contain the path terminal."}
    )
    query_plan_ontology_gate_min_relation_token_overlap: int = field(
        default=0,
        metadata={"help": "Minimum relation/signature token overlap required in a passage before query-plan lift; 0 disables this gate."}
    )
    query_plan_ontology_enable_simple_evidence_validator: bool = field(
        default=False,
        metadata={"help": "If True, route simple attribute queries to dense-candidate evidence validation instead of source-passage path lift."}
    )
    query_plan_ontology_simple_validator_candidate_top_k: int = field(
        default=30,
        metadata={"help": "Number of current dense/PPR candidate passages considered by the simple query evidence validator."}
    )
    query_plan_ontology_simple_validator_max_evidence_paths: int = field(
        default=16,
        metadata={"help": "Maximum executable one-hop paths used as evidence objects for simple query validation."}
    )
    query_plan_ontology_simple_validator_weight: float = field(
        default=0.10,
        metadata={"help": "Boost weight for simple query evidence validator reranking."}
    )
    query_plan_ontology_simple_validator_fallback_to_path_lift: bool = field(
        default=False,
        metadata={"help": "If True, simple queries with no validator evidence can still fall back to source-passage path lift."}
    )
    query_plan_ontology_enable_qa_evidence: bool = field(
        default=False,
        metadata={"help": "If True, pass query-plan ontology evidence bullets into the QA reader prompt."}
    )
    query_plan_ontology_enable_adapter_policy: bool = field(
        default=False,
        metadata={"help": "If True, use the query-plan ontology adapter policy to route PPR injection, QA evidence style, and answer-format hints by query type."}
    )
    query_plan_ontology_qa_evidence_max_items: int = field(
        default=6,
        metadata={"help": "Maximum query-plan ontology evidence bullets added to each QA prompt."}
    )
    query_plan_ontology_qa_evidence_min_score: float = field(
        default=0.0,
        metadata={"help": "Minimum query-plan evidence score required before adding a bullet to the QA prompt."}
    )
    query_plan_ontology_qa_evidence_style: str = field(
        default="structured_chain",
        metadata={"help": "QA evidence rendering style. structured_chain keeps terminal/type hints; facts_only only passes graph facts as non-answer evidence; bridge_cues passes compact relation/path hints; auto routes by query type and evidence reliability."}
    )
    query_plan_ontology_enable_evidence_chain_materializer: bool = field(
        default=False,
        metadata={"help": "If True, materialize top-window evidence snippets that are missing from the QA-only context as evidence-chain prompt items."}
    )
    query_plan_ontology_evidence_chain_materializer_max_items: int = field(
        default=3,
        metadata={"help": "Maximum evidence-chain materializer snippets added to the QA prompt."}
    )
    query_plan_ontology_evidence_chain_materializer_source_top_k: int = field(
        default=5,
        metadata={"help": "Number of original retrieved passages inspected by the evidence-chain materializer."}
    )
    query_plan_ontology_evidence_chain_materializer_max_doc_chars: int = field(
        default=1200,
        metadata={"help": "Maximum passage characters read when extracting a materialized evidence snippet."}
    )
    query_plan_ontology_evidence_chain_materializer_min_query_overlap: int = field(
        default=2,
        metadata={"help": "Minimum query-token overlap required unless the passage contains an answer-like span."}
    )
    query_plan_ontology_evidence_chain_materializer_max_spans: int = field(
        default=4,
        metadata={"help": "Maximum answer-like spans exposed per materialized evidence snippet."}
    )
    query_plan_ontology_evidence_chain_materializer_require_context_gap: bool = field(
        default=True,
        metadata={"help": "If True, only materialize original top-window passages that are absent from the QA-only context."}
    )
    query_plan_ontology_evidence_chain_materializer_add_docs_to_context: bool = field(
        default=True,
        metadata={"help": "If True, add materialized top-window passages back into the QA-only context without changing retrieval metrics."}
    )
    query_plan_ontology_evidence_chain_materializer_preserve_context_top_k: int = field(
        default=1,
        metadata={"help": "Number of existing QA-context passages preserved before materialized companion passages are inserted."}
    )
    query_plan_ontology_evidence_chain_materializer_enable_chain_complete: bool = field(
        default=False,
        metadata={"help": "If True, greedily materialize a chain-complete QA context that covers distinct ontology roles instead of only high-overlap snippets."}
    )
    query_plan_ontology_evidence_chain_materializer_chain_complete_target_datasets: str = field(
        default="",
        metadata={"help": "Comma-separated dataset allowlist for chain-complete materialization; empty enables all datasets."}
    )
    query_plan_ontology_evidence_chain_materializer_chain_complete_min_roles: int = field(
        default=2,
        metadata={"help": "Minimum distinct ontology evidence-role categories targeted by chain-complete materialization."}
    )
    query_plan_ontology_evidence_chain_materializer_chain_complete_role_weight: float = field(
        default=0.24,
        metadata={"help": "Score bonus per distinct ontology role covered by a materialized passage."}
    )
    query_plan_ontology_evidence_chain_materializer_require_chain_complete_route: bool = field(
        default=False,
        metadata={"help": "If True, skip materialization unless the chain-complete router approves the query."}
    )
    query_plan_ontology_evidence_chain_materializer_strict_admission: bool = field(
        default=False,
        metadata={"help": "If True, materialized passages must cover concrete ontology roles rather than only query overlap or answer-like spans."}
    )
    query_plan_ontology_evidence_chain_materializer_min_real_roles: int = field(
        default=1,
        metadata={"help": "Minimum distinct concrete ontology roles required by strict materializer admission; concrete roles are anchor, relation, and terminal."}
    )
    query_plan_ontology_evidence_chain_materializer_context_gap_use_qa_context: bool = field(
        default=False,
        metadata={"help": "If True, the materializer treats existing QA-context documents as the context-gap baseline even when the context builder did not add ontology evidence."}
    )
    query_plan_ontology_evidence_chain_materializer_emit_answer_spans: bool = field(
        default=True,
        metadata={"help": "If False, do not render answer_like_spans in materialized QA evidence, reducing reader bias toward incidental spans."}
    )
    query_plan_ontology_evidence_chain_materializer_suppress_answer_spans_for_slots: str = field(
        default="",
        metadata={"help": "Comma-separated answer slots for which materialized evidence should suppress answer_like_spans."}
    )
    query_plan_ontology_enable_answer_candidate_operator: bool = field(
        default=False,
        metadata={"help": "If True, add a provenance-aware answer-candidate table to the QA prompt from retrieved passages and ontology evidence."}
    )
    query_plan_ontology_answer_candidate_max_items: int = field(
        default=3,
        metadata={"help": "Maximum answer candidates shown in the provenance-aware answer-candidate table."}
    )
    query_plan_ontology_answer_candidate_min_score: float = field(
        default=0.62,
        metadata={"help": "Minimum answer-candidate confidence score required before a candidate is shown to the QA reader."}
    )
    query_plan_ontology_answer_candidate_max_doc_chars: int = field(
        default=1400,
        metadata={"help": "Maximum characters read from each retrieved passage when verifying answer-candidate support."}
    )
    query_plan_ontology_answer_candidate_require_passage_support: bool = field(
        default=True,
        metadata={"help": "If True, answer candidates must occur in the retrieved passages before they are exposed to QA."}
    )
    query_plan_ontology_answer_candidate_use_doc_spans: bool = field(
        default=False,
        metadata={"help": "If True, date/number spans can be proposed directly from top passages when ontology candidates are sparse."}
    )
    query_plan_ontology_answer_candidate_gate_mode: str = field(
        default="loose",
        metadata={"help": "Answer-candidate exposure gate. loose uses default behavior; strict/strict_unique only expose high-margin candidates."}
    )
    query_plan_ontology_answer_candidate_min_margin: float = field(
        default=0.0,
        metadata={"help": "Minimum score margin between the top two answer candidates under strict gates."}
    )
    query_plan_ontology_answer_candidate_disable_ambiguous_slots: bool = field(
        default=False,
        metadata={"help": "If True, suppress answer-candidate tables for ambiguous slots such as occupation when multiple candidates survive."}
    )
    query_plan_ontology_answer_candidate_enable_slot_hint: bool = field(
        default=False,
        metadata={"help": "If True, add a slot-specific answer extraction constraint even when the candidate table is suppressed."}
    )
    query_plan_ontology_enable_answer_canonicalizer: bool = field(
        default=False,
        metadata={"help": "If True, conservatively post-process QA answers using supported answer candidates and span normalization."}
    )
    query_plan_ontology_answer_canonicalizer_min_candidate_score: float = field(
        default=0.82,
        metadata={"help": "Minimum answer-candidate score needed before answer canonicalization can adopt it."}
    )
    query_plan_ontology_answer_canonicalizer_require_candidate_support: bool = field(
        default=True,
        metadata={"help": "If True, answer canonicalization only adopts candidates that occur in retrieved passages."}
    )
    query_plan_ontology_answer_canonicalizer_fix_year_only: bool = field(
        default=True,
        metadata={"help": "If True, shorten temporal answers to a year when the question explicitly asks for a year."}
    )
    query_plan_ontology_answer_canonicalizer_replace_unknown: bool = field(
        default=False,
        metadata={"help": "If True, replace unknown-like answers with a supported high-confidence answer candidate."}
    )
    query_plan_ontology_answer_canonicalizer_allow_expansion: bool = field(
        default=False,
        metadata={"help": "If True, allow answer canonicalization to expand a shorter reader answer to a longer supported candidate."}
    )
    query_plan_ontology_answer_canonicalizer_enable_slot_shrink: bool = field(
        default=True,
        metadata={"help": "If True, shrink over-specific answers for known slots such as city, year, and occupation."}
    )
    query_plan_ontology_answer_canonicalizer_allow_candidate_shrink: bool = field(
        default=False,
        metadata={"help": "If True, allow generic supported-candidate span shrink; disabled by default because it can shorten exact names/dates incorrectly."}
    )
    query_plan_ontology_enable_grounded_answer_verifier: bool = field(
        default=False,
        metadata={"help": "If True, conservatively verify the reader answer against the submitted passages."}
    )
    query_plan_ontology_grounded_answer_verifier_min_confidence: float = field(
        default=0.93,
        metadata={"help": "Minimum confidence for a passage-grounded answer replacement."}
    )
    query_plan_ontology_grounded_answer_verifier_max_passage_chars: int = field(
        default=1800,
        metadata={"help": "Maximum characters per passage shown to the grounded answer verifier."}
    )
    query_plan_ontology_enable_answer_operator_calibrator: bool = field(
        default=False,
        metadata={"help": "If True, apply operator-aware final answer constraints for boolean, comparison, and single-slot questions."}
    )
    query_plan_ontology_answer_operator_calibrator_min_candidate_score: float = field(
        default=0.82,
        metadata={"help": "Minimum supported candidate score before operator-aware answer calibration can override the reader."}
    )
    query_plan_ontology_answer_operator_calibrator_min_margin: float = field(
        default=0.08,
        metadata={"help": "Minimum score margin required by operator-aware answer calibration."}
    )
    query_plan_ontology_answer_operator_calibrator_single_value_slots: str = field(
        default="nationality,occupation,type",
        metadata={"help": "Comma-separated slots where compound reader answers may be shrunk to the first requested value."}
    )
    query_plan_ontology_enable_typed_answer_projection: bool = field(
        default=False,
        metadata={"help": "If True, post-process QA answers with a typed final-answer projection scorer over ontology/passage candidates."}
    )
    query_plan_ontology_typed_answer_projection_min_score: float = field(
        default=0.82,
        metadata={"help": "Minimum score for typed answer projection to override the reader answer."}
    )
    query_plan_ontology_typed_answer_projection_min_margin: float = field(
        default=0.06,
        metadata={"help": "Minimum score margin between the best typed projection candidate and alternatives/current answer."}
    )
    query_plan_ontology_typed_answer_projection_allowed_slots: str = field(
        default="occupation,birth_place,death_place,location,date",
        metadata={"help": "Comma-separated answer slots where typed answer projection may override the reader answer."}
    )
    query_plan_ontology_typed_answer_projection_require_support: bool = field(
        default=True,
        metadata={"help": "If True, typed projection candidates must occur in retrieved passages."}
    )
    query_plan_ontology_typed_answer_projection_prefer_chain_order: bool = field(
        default=True,
        metadata={"help": "If True, prefer earlier ontology-chain candidates when scores are otherwise similar."}
    )
    query_plan_ontology_typed_answer_projection_operator_gate: bool = field(
        default=True,
        metadata={"help": "If True, only allow typed answer projection for configured query operators."}
    )
    query_plan_ontology_typed_answer_projection_allowed_operators: str = field(
        default="select_attr",
        metadata={"help": "Comma-separated query operators where typed answer projection may override the reader answer."}
    )
    query_plan_ontology_typed_answer_projection_require_relation_or_slot_source: bool = field(
        default=True,
        metadata={"help": "If True, projection candidates must have a relation match or a slot-specific provenance source."}
    )
    query_plan_ontology_typed_answer_projection_protect_supported_reader_answer: bool = field(
        default=True,
        metadata={"help": "If True, keep a non-unknown reader answer when it is directly supported by retrieved passages."}
    )
    query_plan_ontology_enable_span_constrained_answer_resolver: bool = field(
        default=False,
        metadata={"help": "If True, resolve unsupported or unknown reader answers to high-confidence answer-like spans supported by the QA context."}
    )
    query_plan_ontology_span_resolver_min_score: float = field(
        default=0.86,
        metadata={"help": "Minimum span/candidate score required before the span-constrained resolver can override the reader answer."}
    )
    query_plan_ontology_span_resolver_min_margin: float = field(
        default=0.08,
        metadata={"help": "Minimum margin between the best span candidate and alternatives/current answer."}
    )
    query_plan_ontology_span_resolver_allowed_slots: str = field(
        default="date,number,occupation",
        metadata={"help": "Comma-separated answer slots where span-constrained resolution may override the reader answer."}
    )
    query_plan_ontology_span_resolver_allowed_operators: str = field(
        default="select_attr,bridge",
        metadata={"help": "Comma-separated query operators where span-constrained resolution may override the reader answer."}
    )
    query_plan_ontology_span_resolver_max_doc_spans: int = field(
        default=8,
        metadata={"help": "Maximum answer-like spans collected from QA-context passages by the span resolver."}
    )
    query_plan_ontology_span_resolver_replace_unsupported_only: bool = field(
        default=True,
        metadata={"help": "If True, only override unknown or unsupported reader answers unless the candidate is a safe contained shrink."}
    )
    query_plan_ontology_span_resolver_protect_supported_reader_answer: bool = field(
        default=True,
        metadata={"help": "If True, keep a non-unknown reader answer when it is directly supported by retrieved passages."}
    )
    query_plan_ontology_enable_quality_gate: bool = field(
        default=False,
        metadata={"help": "If True, suppress noisy query-plan QA evidence using trace quality and query fan-out."}
    )
    query_plan_ontology_quality_gate_min_plan_precision: float = field(
        default=0.0,
        metadata={"help": "Minimum effective plan precision required before non-simple QA evidence can be added; <=0 disables."}
    )
    query_plan_ontology_quality_gate_min_plan_scale: float = field(
        default=0.0,
        metadata={"help": "Minimum trace lift_weight_scale required before non-simple QA evidence can be added; <=0 disables."}
    )
    query_plan_ontology_quality_gate_max_matched_plans_for_qa: int = field(
        default=0,
        metadata={"help": "Disable non-simple QA evidence when a query matches more plans than this; <=0 disables."}
    )
    query_plan_ontology_quality_gate_max_executable_paths_for_qa: int = field(
        default=0,
        metadata={"help": "Disable non-simple QA evidence when a query has more executable paths than this; <=0 disables."}
    )
    query_plan_ontology_quality_gate_max_avg_paths_per_plan_for_qa: float = field(
        default=0.0,
        metadata={"help": "Disable non-simple QA evidence when executable paths per matched plan exceed this; <=0 disables."}
    )
    query_plan_ontology_quality_gate_allow_simple_validator: bool = field(
        default=True,
        metadata={"help": "If True, simple validator evidence can bypass fan-out quality gates."}
    )
    query_plan_ontology_enable_qa_format_hint: bool = field(
        default=False,
        metadata={"help": "If True, add a query-form answer-format hint to QA prompts even when no query-plan graph evidence is safe to expose."}
    )
    query_plan_ontology_qa_format_hint_mode: str = field(
        default="typed",
        metadata={"help": "Answer-format hint mode. typed uses query-form/type-specific rules; generic only asks for the shortest supported answer span; auto routes generic hints by query type and evidence quality."}
    )
    query_plan_ontology_enable_answer_path_verifier: bool = field(
        default=False,
        metadata={"help": "If True, score query-plan paths as answer paths before using them for complex passage lift or bridge-cue QA evidence."}
    )
    query_plan_ontology_answer_path_min_score: float = field(
        default=0.50,
        metadata={"help": "Minimum answer-path verifier score required for non-simple query-plan paths."}
    )
    query_plan_ontology_answer_path_score_weight: float = field(
        default=0.0,
        metadata={"help": "Soft multiplicative bonus weight for paths that look like answer paths; 0 disables score adjustment."}
    )
    query_plan_ontology_answer_path_filter_lift: bool = field(
        default=True,
        metadata={"help": "If True with answer-path verifier enabled, only verified non-simple paths can lift passages."}
    )
    query_plan_ontology_answer_path_filter_qa_evidence: bool = field(
        default=True,
        metadata={"help": "If True with answer-path verifier enabled, bridge-cue QA evidence is built only from verified paths."}
    )
    query_plan_ontology_enable_answer_object_gate: bool = field(
        default=True,
        metadata={"help": "If True, query-plan execution uses multi-agent terminal answer/object constraints to filter broad generic relation paths."}
    )
    query_plan_ontology_enable_query_conditioning: bool = field(
        default=False,
        metadata={"help": "If True, refine matched query-plan schemas with the current query's inferred answer type and relation intent before execution."}
    )
    query_plan_ontology_condition_min_score: float = field(
        default=0.0,
        metadata={"help": "Minimum conditioned query-plan score after query-aware refinement."}
    )
    query_plan_ontology_condition_require_relation_overlap: bool = field(
        default=False,
        metadata={"help": "If True, conditioned query-plan matching drops plans with no inferred query relation overlap when query relation hints exist."}
    )
    query_plan_ontology_enable_simple_fallback: bool = field(
        default=False,
        metadata={"help": "If True, create a conservative one-hop query-plan from inferred query relations when no learned plan covers the query."}
    )
    query_plan_ontology_simple_fallback_only_when_unmatched: bool = field(
        default=True,
        metadata={"help": "If True, add simple fallback plans only when learned query-plan matching returned no usable plan."}
    )
    query_plan_ontology_simple_fallback_precision: float = field(
        default=0.62,
        metadata={"help": "Synthetic precision assigned to simple one-hop query-plan fallback templates."}
    )
    query_plan_ontology_simple_fallback_weight_scale: float = field(
        default=0.65,
        metadata={"help": "Passage lift weight multiplier for simple one-hop query-plan fallback evidence."}
    )
    query_plan_ontology_enable_plan_repair: bool = field(
        default=False,
        metadata={"help": "If True, add low-weight repaired query plans from inferred query slots and relation-realization aliases."}
    )
    query_plan_ontology_plan_repair_max_repairs: int = field(
        default=4,
        metadata={"help": "Maximum repaired query-plan templates appended per query."}
    )
    query_plan_ontology_plan_repair_min_query_token_overlap: int = field(
        default=1,
        metadata={"help": "Minimum relation/query token overlap required before relaxing a matched plan step."}
    )
    query_plan_ontology_plan_repair_min_plan_precision: float = field(
        default=0.30,
        metadata={"help": "Minimum effective precision for a matched plan to receive a relaxed repair copy."}
    )
    query_plan_ontology_plan_repair_score_floor: float = field(
        default=0.46,
        metadata={"help": "Minimum match score assigned to condition-fallback repaired plans."}
    )
    query_plan_ontology_plan_repair_enable_query_condition: bool = field(
        default=True,
        metadata={"help": "If True, use inferred query-condition relation slots to synthesize repair plans."}
    )
    query_plan_ontology_plan_repair_enable_relation_realization: bool = field(
        default=True,
        metadata={"help": "If True, create soft relation-realization repair copies of matched plans."}
    )
    query_plan_ontology_anchor_mode: str = field(
        default="coverage",
        metadata={"help": "Anchor matching mode for query-plan execution: coverage or exact. Exact only uses graph entities explicitly mentioned as query phrases."}
    )
    query_plan_ontology_anchor_require_concrete: bool = field(
        default=False,
        metadata={"help": "If True, query-plan execution drops role/schema anchors such as director, company, label, or place and keeps concrete entity anchors only."}
    )
    query_plan_ontology_anchor_prefer_longest_exact: bool = field(
        default=False,
        metadata={"help": "If True with exact anchor mode, drop exact anchors contained in a longer exact query entity mention."}
    )
    enable_query_plan_ontology_ppr_injection: bool = field(
        default=False,
        metadata={"help": "If True, instantiate query-plan ontology paths before graph PPR and inject selected path facts as additional PPR seed facts."}
    )
    query_plan_ontology_ppr_enable_query_router: bool = field(
        default=True,
        metadata={"help": "If True, skip query-plan PPR injection for query forms where ontology path injection has been empirically noisy, such as explanatory or measurement questions."}
    )
    query_plan_ontology_ppr_enable_path_quality_gate: bool = field(
        default=False,
        metadata={"help": "If True, skip query-plan PPR injection when executable paths are too diffuse or low-quality before adding seed facts."}
    )
    query_plan_ontology_ppr_max_executable_paths_for_quality_gate: int = field(
        default=0,
        metadata={"help": "With the PPR path quality gate enabled, skip injection when executable path count exceeds this value; <=0 disables this condition."}
    )
    query_plan_ontology_ppr_max_unique_terminals_for_quality_gate: int = field(
        default=0,
        metadata={"help": "With the PPR path quality gate enabled, skip injection when unique terminal entity count exceeds this value; <=0 disables this condition."}
    )
    query_plan_ontology_ppr_min_plan_precision: float = field(
        default=0.0,
        metadata={"help": "Minimum trace/plan precision required before a query-plan can inject facts into PPR."}
    )
    query_plan_ontology_ppr_require_anchor_fact: bool = field(
        default=True,
        metadata={"help": "If True, injected PPR facts must contain one of the exact query-plan anchor entities unless their source passage is dense-top-k."}
    )
    query_plan_ontology_ppr_source_top_k: int = field(
        default=0,
        metadata={"help": "If >0, allow PPR fact injection only when a fact source passage is in the dense top-k, or the fact contains a query anchor."}
    )
    query_plan_ontology_ppr_top_k: int = field(
        default=6,
        metadata={"help": "Maximum query-plan path facts injected into graph PPR per query."}
    )
    query_plan_ontology_ppr_score_weight: float = field(
        default=0.55,
        metadata={"help": "Weight used to raise query-plan injected PPR fact scores toward the current anchor fact score."}
    )
    query_plan_ontology_ppr_max_paths: int = field(
        default=40,
        metadata={"help": "Maximum executable query-plan paths considered for PPR fact injection."}
    )
    query_plan_ontology_ppr_relation_match_mode: str = field(
        default="strict",
        metadata={"help": "Relation matching mode used for query-plan PPR injection: strict, soft, family, or strict_then_soft."}
    )
    query_plan_ontology_enable_ppr_passage_seeds: bool = field(
        default=False,
        metadata={"help": "If True, use source passages of high-quality query-plan PPR facts as extra passage reset seeds in graph PPR."}
    )
    query_plan_ontology_ppr_passage_seed_weight: float = field(
        default=0.08,
        metadata={"help": "Maximum reset weight assigned to ontology-guided source passage seeds."}
    )
    query_plan_ontology_ppr_passage_seed_top_k: int = field(
        default=4,
        metadata={"help": "Maximum ontology-guided source passages added as graph PPR reset seeds per query."}
    )
    query_plan_ontology_ppr_passage_seed_min_plan_precision: float = field(
        default=0.65,
        metadata={"help": "Minimum query-plan effective precision required before a source passage can become a PPR passage seed."}
    )
    query_plan_ontology_ppr_passage_seed_require_query_anchor: bool = field(
        default=True,
        metadata={"help": "If True, source passage PPR seeds must come from facts containing a query anchor."}
    )
    query_plan_ontology_ppr_passage_seed_dense_bonus: float = field(
        default=0.25,
        metadata={"help": "Additional score multiplier for ontology passage seeds whose source passage also appears in dense top-k."}
    )
    enable_query_plan_ontology_graph_scorer: bool = field(
        default=False,
        metadata={"help": "If True, use executable query-plan path facts to boost graph-search fact seeds before PPR."}
    )
    query_plan_ontology_graph_scorer_score_weight: float = field(
        default=0.28,
        metadata={"help": "Weight used to raise query-plan graph-scorer fact scores toward the current anchor fact score."}
    )
    query_plan_ontology_graph_scorer_top_k: int = field(
        default=6,
        metadata={"help": "Maximum new query-plan graph-scorer facts appended to graph-search seeds per query."}
    )
    query_plan_ontology_graph_scorer_min_confidence: float = field(
        default=0.45,
        metadata={"help": "Minimum confidence required for query-plan graph-scorer fact boosting."}
    )
    query_plan_ontology_graph_scorer_require_anchor_or_dense_source: bool = field(
        default=True,
        metadata={"help": "If True, graph-scorer facts must contain a query anchor or originate from dense-top-k source passages."}
    )
    query_plan_ontology_graph_scorer_boost_existing: bool = field(
        default=True,
        metadata={"help": "If True, boost already-selected top facts when they are part of executable query-plan paths."}
    )
    query_plan_ontology_graph_scorer_reorder_facts: bool = field(
        default=True,
        metadata={"help": "If True, reorder graph-search fact seeds by the graph-scorer-adjusted fact scores."}
    )
    query_plan_ontology_enable_structural_candidate_retriever: bool = field(
        default=False,
        metadata={"help": "If True, scan retrieved passages with query-plan entity/relation clues to add ontology-supported candidate passages."}
    )
    query_plan_ontology_structural_candidate_retriever_scan_top_k: int = field(
        default=80,
        metadata={"help": "Top retrieved passages scanned by the structural query-plan candidate retriever."}
    )
    query_plan_ontology_structural_candidate_retriever_max_original_rank: int = field(
        default=120,
        metadata={"help": "Maximum original rank eligible for structural candidate retrieval; <=0 disables this bound."}
    )
    query_plan_ontology_structural_candidate_retriever_max_added_candidates: int = field(
        default=8,
        metadata={"help": "Maximum structural candidates added before candidate promotion."}
    )
    query_plan_ontology_structural_candidate_retriever_min_score: float = field(
        default=0.56,
        metadata={"help": "Minimum structural candidate score required before adding a passage candidate."}
    )
    query_plan_ontology_structural_candidate_retriever_require_anchor: bool = field(
        default=True,
        metadata={"help": "If True, structural candidates must mention a concrete query-plan anchor."}
    )
    query_plan_ontology_structural_candidate_retriever_require_terminal_or_relation: bool = field(
        default=True,
        metadata={"help": "If True, structural candidates must mention a terminal entity or relation clue."}
    )
    query_plan_ontology_structural_candidate_retriever_min_relation_token_overlap: int = field(
        default=1,
        metadata={"help": "Minimum relation-token overlap for structural candidates when no terminal entity is matched."}
    )
    query_plan_ontology_structural_candidate_retriever_reject_generic_anchors: bool = field(
        default=True,
        metadata={"help": "If True, reject generic/directional anchors such as north, region, person, or location."}
    )
    query_plan_ontology_structural_candidate_retriever_score_weight: float = field(
        default=0.85,
        metadata={"help": "Score scale used when structural candidates are merged with source-linked candidate scores."}
    )
    query_plan_ontology_structural_candidate_retriever_require_existing_source: bool = field(
        default=False,
        metadata={"help": "If True, structural scan can only enrich passages already linked by executable source facts."}
    )
    query_plan_ontology_enable_candidate_retriever: bool = field(
        default=False,
        metadata={"help": "If True, directly promotes high-confidence query-plan source passages into the retrieval ranking before final passage lift."}
    )
    query_plan_ontology_candidate_retriever_top_k: int = field(
        default=2,
        metadata={"help": "Maximum ontology-supported passages promoted by the query-plan candidate retriever per query."}
    )
    query_plan_ontology_candidate_retriever_promote_to_rank: int = field(
        default=5,
        metadata={"help": "Target rank for candidate-retriever promotions. Top baseline passages are preserved separately."}
    )
    query_plan_ontology_candidate_retriever_preserve_top_k: int = field(
        default=2,
        metadata={"help": "Number of highest baseline-ranked passages protected from candidate-retriever insertion."}
    )
    query_plan_ontology_candidate_retriever_max_original_rank: int = field(
        default=120,
        metadata={"help": "Maximum original rank eligible for candidate-retriever promotion; <=0 disables this bound."}
    )
    query_plan_ontology_candidate_retriever_min_confidence: float = field(
        default=0.72,
        metadata={"help": "Minimum structural confidence required before candidate-retriever promotion."}
    )
    query_plan_ontology_candidate_retriever_score_margin: float = field(
        default=1e-6,
        metadata={"help": "Small score margin used to place promoted candidates at the requested rank."}
    )
    query_plan_ontology_candidate_retriever_aggressive_promote_to_rank: int = field(
        default=2,
        metadata={"help": "Target rank for very high-confidence candidate-retriever promotions."}
    )
    query_plan_ontology_candidate_retriever_aggressive_min_confidence: float = field(
        default=0.84,
        metadata={"help": "Minimum confidence required before using the aggressive candidate promotion rank."}
    )
    query_plan_ontology_candidate_retriever_aggressive_require_strict: bool = field(
        default=True,
        metadata={"help": "If True, aggressive candidate promotions require at least one strict query-plan context."}
    )
    query_plan_ontology_candidate_retriever_aggressive_require_multihop_or_multi_context: bool = field(
        default=True,
        metadata={"help": "If True, aggressive candidate promotions require a multi-hop path or multiple evidence contexts."}
    )
    query_plan_ontology_candidate_retriever_enable_anchor_guard: bool = field(
        default=True,
        metadata={"help": "If True, candidate promotions require a concrete anchor or terminal entity."}
    )
    query_plan_ontology_candidate_retriever_allow_structural_only: bool = field(
        default=True,
        metadata={"help": "If False, passages supported only by structural lexical matching cannot be directly promoted."}
    )
    query_plan_ontology_candidate_retriever_structural_only_min_confidence: float = field(
        default=0.92,
        metadata={"help": "Minimum confidence required when structural-only candidate promotion is allowed."}
    )
    query_plan_ontology_enable_terminal_evidence_retriever: bool = field(
        default=False,
        metadata={"help": "If True, use query-plan terminal/anchor/relation clues to launch short dense subqueries and promote answer-terminal evidence passages."}
    )
    query_plan_ontology_terminal_evidence_dense_top_k: int = field(
        default=40,
        metadata={"help": "Dense top-k passages considered for each query-plan terminal evidence subquery."}
    )
    query_plan_ontology_terminal_evidence_max_query_variants: int = field(
        default=4,
        metadata={"help": "Maximum ontology-derived dense subqueries issued by the terminal evidence retriever."}
    )
    query_plan_ontology_terminal_evidence_max_selected: int = field(
        default=1,
        metadata={"help": "Maximum terminal evidence passages inserted into the retrieval ranking per query."}
    )
    query_plan_ontology_terminal_evidence_promote_to_rank: int = field(
        default=5,
        metadata={"help": "Target rank slot for terminal evidence passage insertion."}
    )
    query_plan_ontology_terminal_evidence_preserve_top_k: int = field(
        default=2,
        metadata={"help": "Number of highest baseline-ranked passages protected from terminal evidence insertion."}
    )
    query_plan_ontology_terminal_evidence_min_confidence: float = field(
        default=0.58,
        metadata={"help": "Minimum terminal evidence confidence required before inserting a dense-retrieved passage."}
    )
    query_plan_ontology_terminal_evidence_require_relation_hint: bool = field(
        default=True,
        metadata={"help": "If True, terminal evidence candidates must mention query-plan relation clues."}
    )
    query_plan_ontology_terminal_evidence_require_entity_hint: bool = field(
        default=True,
        metadata={"help": "If True, terminal evidence candidates must mention query-plan anchor or terminal entities."}
    )
    query_plan_ontology_terminal_evidence_max_entity_terms: int = field(
        default=6,
        metadata={"help": "Maximum anchor/terminal entity phrases used to create terminal evidence subqueries."}
    )
    query_plan_ontology_terminal_evidence_score_margin: float = field(
        default=1e-6,
        metadata={"help": "Small score margin used to place terminal evidence passages at the requested rank."}
    )
    query_plan_ontology_terminal_evidence_candidate_only: bool = field(
        default=False,
        metadata={"help": "If True, terminal evidence retrieval only produces candidates and does not directly alter passage ranks."}
    )
    query_plan_ontology_enable_terminal_evidence_verifier: bool = field(
        default=False,
        metadata={"help": "If True, verify terminal evidence candidates before inserting them into the passage ranking."}
    )
    query_plan_ontology_terminal_evidence_verifier_min_confidence: float = field(
        default=0.74,
        metadata={"help": "Minimum terminal evidence confidence required by the verifier."}
    )
    query_plan_ontology_terminal_evidence_verifier_min_relation_hits: int = field(
        default=1,
        metadata={"help": "Minimum relation-token hits required by the terminal evidence verifier."}
    )
    query_plan_ontology_terminal_evidence_verifier_max_original_rank: int = field(
        default=120,
        metadata={"help": "Farthest original passage rank eligible for terminal evidence verifier insertion."}
    )
    query_plan_ontology_terminal_evidence_verifier_max_matched_plans: int = field(
        default=3,
        metadata={"help": "Maximum matched query-plan count eligible for terminal evidence verifier insertion."}
    )
    query_plan_ontology_terminal_evidence_verifier_max_executable_paths: int = field(
        default=6,
        metadata={"help": "Maximum executable path count eligible for terminal evidence verifier insertion."}
    )
    query_plan_ontology_terminal_evidence_verifier_require_new_coverage: bool = field(
        default=True,
        metadata={"help": "If True, terminal evidence verifier requires the candidate to add entity/relation coverage absent from the current top window."}
    )
    query_plan_ontology_terminal_evidence_verifier_promote_to_rank: int = field(
        default=5,
        metadata={"help": "Target rank for verifier-approved terminal evidence insertion."}
    )
    query_plan_ontology_terminal_evidence_verifier_preserve_top_k: int = field(
        default=2,
        metadata={"help": "Number of top-ranked passages protected from verifier-approved terminal evidence insertion."}
    )
    query_plan_ontology_terminal_evidence_verifier_max_inserted: int = field(
        default=1,
        metadata={"help": "Maximum verifier-approved terminal evidence passages inserted per query."}
    )
    query_plan_ontology_terminal_evidence_verifier_score_margin: float = field(
        default=1e-6,
        metadata={"help": "Small score margin used by verifier-approved terminal evidence insertion."}
    )
    query_plan_ontology_trace_path: str = field(
        default=None,
        metadata={"help": "Optional JSONL path for per-query query-plan ontology diagnostics."}
    )
    query_plan_ontology_trace_compact_physical_shadow: bool = field(
        default=False,
        metadata={"help": "Write a compact trace containing only physical-shadow evidence."}
    )
    query_plan_ontology_trace_compact_end_to_end_slate: bool = field(
        default=False,
        metadata={"help": "Write a compact trace for end-to-end counterfactual slate training."}
    )
    ontology_include_concepts: bool = field(
        default=False,
        metadata={"help": "Whether ontology query expansion should append coarse concept names."}
    )
    ontology_include_related_terms: bool = field(
        default=False,
        metadata={"help": "Whether ontology query expansion should append neighbor/schema related terms."}
    )
    ontology_min_concept_score: int = field(
        default=3,
        metadata={"help": "Minimum concept match score required before a concept can be appended."}
    )
    ontology_require_entity_match: bool = field(
        default=False,
        metadata={"help": "If True, ontology query expansion is applied only when a concrete entity is matched."}
    )
    ontology_max_entities_per_query: int = field(
        default=3,
        metadata={"help": "Maximum matched ontology entities appended to each query."}
    )
    ontology_max_relations_per_query: int = field(
        default=3,
        metadata={"help": "Maximum matched ontology relations appended to each query."}
    )
    ontology_max_entity_terms: int = field(
        default=5000,
        metadata={"help": "Maximum entity profiles stored in the graph ontology."}
    )
    ontology_max_relation_schemas: int = field(
        default=300,
        metadata={"help": "Maximum relation schemas stored in the graph ontology."}
    )



    max_qa_steps: int = field(
        default=1,
        metadata={"help": "For answering a single question, the max steps that we use to interleave retrieval and reasoning."}
    )
    qa_top_k: int = field(
        default=5,
        metadata={"help": "Feeding top k documents to the QA model for reading."}
    )
    qa_api_concurrency: int = field(
        default=1,
        metadata={"help": "Maximum concurrent remote QA API requests; result order is preserved."}
    )


    save_dir: str = field(
        default=None,
        metadata={"help": "Directory to save all related information. If it's given, will overwrite all default save_dir setups. If it's not given, then if we're not running specific datasets, default to `outputs`, otherwise, default to a dataset-customized output dir."}
    )





    dataset: Optional[Literal['hotpotqa', 'hotpotqa_train', 'musique', '2wikimultihopqa']] = field(
        default=None,
        metadata={"help": "Dataset to use. If specified, it means we will run specific datasets. If not specified, it means we're running freely."}
    )

    graph_type: Literal[
        'dpr_only',
        'entity',
        'passage_entity', 'relation_aware_passage_entity',
        'passage_entity_relation',
        'facts_and_sim_passage_node_unidirectional',
    ] = field(
        default="facts_and_sim_passage_node_unidirectional",
        metadata={"help": "Type of graph to use in the experiment."}
    )
    corpus_len: Optional[int] = field(
        default=None,
        metadata={"help": "Length of the corpus to use."}
    )


    def __post_init__(self):
        if self.save_dir is None:
            if self.dataset is None: self.save_dir = 'outputs'
            else: self.save_dir = os.path.join('outputs', self.dataset)
        logger.debug(f"Initializing the highest level of save_dir to be {self.save_dir}")
