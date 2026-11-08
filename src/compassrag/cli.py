import hashlib
import os
from typing import List
import json

from compassrag.system import CompassRAG
from compassrag.utils.misc_utils import compute_mdhash_id, string_to_bool
from compassrag.utils.config_utils import BaseConfig

import argparse


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging


def _snapshot_file_signature(path: str):
    stat = os.stat(path)
    return stat.st_size, stat.st_mtime_ns


def validate_read_only_retrieval_snapshot(compassrag: CompassRAG, docs: List[str]):
    """Fail closed unless the loaded retrieval index exactly matches ``docs``."""
    if len(docs) != len(set(docs)):
        raise ValueError("read-only retrieval corpus contains duplicate passages")

    expected_chunk_ids = {
        compute_mdhash_id(doc, prefix="chunk-") for doc in docs
    }
    actual_chunk_ids = set(compassrag.chunk_embedding_store.get_all_ids())
    missing_chunk_ids = expected_chunk_ids - actual_chunk_ids
    extra_chunk_ids = actual_chunk_ids - expected_chunk_ids
    if missing_chunk_ids or extra_chunk_ids:
        raise ValueError(
            "read-only retrieval index does not exactly match the requested corpus "
            f"(corpus={len(expected_chunk_ids)} index={len(actual_chunk_ids)} "
            f"missing={len(missing_chunk_ids)} extra={len(extra_chunk_ids)})"
        )

    if not os.path.isfile(compassrag.openie_results_path):
        raise FileNotFoundError(
            f"read-only retrieval OpenIE snapshot is missing: {compassrag.openie_results_path}"
        )
    with open(compassrag.openie_results_path, "r", encoding="utf-8") as handle:
        openie_payload = json.load(handle)
    openie_docs = openie_payload.get("docs", []) if isinstance(openie_payload, dict) else []
    actual_openie_ids = {
        compute_mdhash_id(str(row.get("passage", "")), prefix="chunk-")
        for row in openie_docs
        if isinstance(row, dict)
    }
    missing_openie_ids = expected_chunk_ids - actual_openie_ids
    extra_openie_ids = actual_openie_ids - expected_chunk_ids
    if (
        len(openie_docs) != len(expected_chunk_ids)
        or missing_openie_ids
        or extra_openie_ids
    ):
        raise ValueError(
            "read-only OpenIE snapshot does not exactly match the requested corpus "
            f"(corpus={len(expected_chunk_ids)} openie={len(openie_docs)} "
            f"missing={len(missing_openie_ids)} extra={len(extra_openie_ids)})"
        )

    entity_ids = set(compassrag.entity_embedding_store.get_all_ids())
    expected_graph_nodes = expected_chunk_ids | entity_ids
    actual_graph_nodes = {
        str(vertex["name"])
        for vertex in compassrag.graph.vs
        if "name" in vertex.attributes()
    }
    if (
        compassrag.graph.vcount() != len(expected_graph_nodes)
        or actual_graph_nodes != expected_graph_nodes
    ):
        raise ValueError(
            "read-only graph does not match the frozen passage/entity stores "
            f"(expected={len(expected_graph_nodes)} graph={len(actual_graph_nodes)} "
            f"missing={len(expected_graph_nodes - actual_graph_nodes)} "
            f"extra={len(actual_graph_nodes - expected_graph_nodes)})"
        )

    immutable_paths = [
        compassrag.openie_results_path,
        compassrag._graph_pickle_filename,
        compassrag.chunk_embedding_store.filename,
        compassrag.entity_embedding_store.filename,
        compassrag.fact_embedding_store.filename,
    ]
    missing_paths = [path for path in immutable_paths if not os.path.isfile(path)]
    if missing_paths:
        raise FileNotFoundError(
            "read-only retrieval snapshot is incomplete: " + ", ".join(missing_paths)
        )
    signatures = {path: _snapshot_file_signature(path) for path in immutable_paths}
    with open(compassrag.openie_results_path, "rb") as handle:
        openie_sha256 = hashlib.sha256(handle.read()).hexdigest()
    logging.info(
        "Validated immutable retrieval snapshot: passages=%d entities=%d facts=%d openie_sha256=%s",
        len(actual_chunk_ids),
        len(entity_ids),
        len(compassrag.fact_embedding_store.get_all_ids()),
        openie_sha256,
    )
    return signatures, openie_sha256


def assert_read_only_retrieval_snapshot_unchanged(
    compassrag: CompassRAG,
    snapshot_contract,
):
    signatures, expected_openie_sha256 = snapshot_contract
    changed = [
        path
        for path, signature in signatures.items()
        if not os.path.isfile(path) or _snapshot_file_signature(path) != signature
    ]
    with open(compassrag.openie_results_path, "rb") as handle:
        current_openie_sha256 = hashlib.sha256(handle.read()).hexdigest()
    if changed or current_openie_sha256 != expected_openie_sha256:
        raise RuntimeError(
            "immutable retrieval snapshot changed during read-only execution "
            f"(changed_files={changed}, openie_before={expected_openie_sha256[:12]}, "
            f"openie_after={current_openie_sha256[:12]})"
        )

def get_gold_docs(samples: List, dataset_name: str = None) -> List:
    gold_docs = []
    for sample in samples:
        if 'supporting_facts' in sample:
            gold_title = set([item[0] for item in sample['supporting_facts']])
            gold_title_and_content_list = [item for item in sample['context'] if item[0] in gold_title]
            if dataset_name.startswith('hotpotqa'):
                gold_doc = [item[0] + '\n' + ''.join(item[1]) for item in gold_title_and_content_list]
            else:
                gold_doc = [item[0] + '\n' + ' '.join(item[1]) for item in gold_title_and_content_list]
        elif 'contexts' in sample:
            gold_doc = [item['title'] + '\n' + item['text'] for item in sample['contexts'] if item['is_supporting']]
        else:
            assert 'paragraphs' in sample, "`paragraphs` should be in sample, or consider the setting not to evaluate retrieval"
            gold_paragraphs = []
            for item in sample['paragraphs']:
                if 'is_supporting' in item and item['is_supporting'] is False:
                    continue
                gold_paragraphs.append(item)
            gold_doc = [item['title'] + '\n' + (item['text'] if 'text' in item else item['paragraph_text']) for item in gold_paragraphs]

        gold_doc = list(set(gold_doc))
        gold_docs.append(gold_doc)
    return gold_docs


def get_gold_answers(samples):
    gold_answers = []
    for sample_idx in range(len(samples)):
        gold_ans = None
        sample = samples[sample_idx]

        if 'answer' in sample or 'gold_ans' in sample:
            gold_ans = sample['answer'] if 'answer' in sample else sample['gold_ans']
        elif 'reference' in sample:
            gold_ans = sample['reference']
        elif 'obj' in sample:
            gold_ans = set(
                [sample['obj']] + [sample['possible_answers']] + [sample['o_wiki_title']] + [sample['o_aliases']])
            gold_ans = list(gold_ans)
        assert gold_ans is not None
        if isinstance(gold_ans, str):
            gold_ans = [gold_ans]
        assert isinstance(gold_ans, list)
        gold_ans = set(gold_ans)
        if 'answer_aliases' in sample:
            gold_ans.update(sample['answer_aliases'])

        gold_answers.append(gold_ans)

    return gold_answers

def main():
    parser = argparse.ArgumentParser(description="CompassRAG retrieval and QA")
    parser.add_argument('--dataset', type=str, default='musique', help='Dataset name')
    parser.add_argument('--query_path', type=str, default=None,
                        help='Optional query/evaluation JSON path. Defaults to reproduce/dataset/{dataset}.json.')
    parser.add_argument('--corpus_path', type=str, default=None,
                        help='Optional corpus JSON path. Defaults to reproduce/dataset/{dataset}_corpus.json.')
    parser.add_argument('--qa_output_path', type=str, default=None,
                        help='Optional JSONL path for per-query QA predictions, retrieved docs, and ontology evidence.')
    parser.add_argument('--qa_api_concurrency', type=int, default=1,
                        help='Maximum concurrent QA API requests. Responses remain aligned to input order.')
    parser.add_argument('--llm_base_url', type=str, default=os.environ.get('MODEL_BASE_URL'), help='Chat-completion endpoint')
    parser.add_argument('--llm_name', type=str, default=os.environ.get('MODEL_NAME', 'reader-model'), help='Model identifier')
    parser.add_argument('--max_new_tokens', type=int, default=4096,
                        help='Maximum completion tokens for OpenIE and QA LLM calls.')
    parser.add_argument('--embedding_name', type=str, default='nvidia/NV-Embed-v2', help='embedding model name')
    parser.add_argument('--embedding_batch_size', type=int, default=8,
                        help='Batch size for local embedding inference. Lower this to reduce peak GPU memory.')
    parser.add_argument('--force_index_from_scratch', type=str, default='false',
                        help='If set to True, will ignore all existing storage files and graph data and will rebuild from scratch.')
    parser.add_argument('--force_openie_from_scratch', type=str, default='false', help='If set to False, will try to first reuse openie results for the corpus if they exist.')
    parser.add_argument('--reuse_index_read_only', type=str,
                        default=os.environ.get('REUSE_INDEX_READ_ONLY', 'false'),
                        help='If True, require an exact prebuilt retrieval snapshot and never call index(docs).')
    parser.add_argument('--openie_mode', choices=['online', 'offline'], default='online',
                        help="OpenIE mode, offline denotes using VLLM offline batch mode for indexing, while online denotes")
    parser.add_argument('--enable_adaptive_base_access_router', type=str,
                        default=os.environ.get('ENABLE_ADAPTIVE_BASE_ACCESS_ROUTER', 'false'),
                        help='Select a complete PPR, dense, or sparse base ranking for each query.')
    parser.add_argument('--adaptive_base_access_collect_only', type=str,
                        default=os.environ.get('ADAPTIVE_BASE_ACCESS_COLLECT_ONLY', 'false'),
                        help='Collect counterfactual base rankings and labels without changing the PPR incumbent.')
    parser.add_argument('--adaptive_base_access_model_path', type=str,
                        default=os.environ.get('ADAPTIVE_BASE_ACCESS_MODEL_PATH', ''),
                        help='JSON gain/risk router learned from historical OOF replay.')
    parser.add_argument('--adaptive_base_access_sparse_top_k', type=int,
                        default=int(os.environ.get('ADAPTIVE_BASE_ACCESS_SPARSE_TOP_K', '200')),
                        help='Number of sparse candidates generated before dense backoff.')
    parser.add_argument('--enable_ontology', type=str, default='false',
                        help='If set to True, build and use a Saturn-inspired graph ontology for query expansion.')
    parser.add_argument('--force_ontology_from_scratch', type=str, default='false',
                        help='If set to True, rebuild the cached graph ontology from OpenIE results.')
    parser.add_argument('--ontology_query_top_k', type=int, default=8,
                        help='Number of matched ontology concepts/entities/relations to keep per query.')
    parser.add_argument('--ontology_max_query_terms', type=int, default=24,
                        help='Maximum related ontology terms appended to each query.')
    parser.add_argument('--ontology_embedding_mode', choices=['none', 'fact', 'passage', 'both'], default='passage',
                        help='Where to apply ontology-expanded query embeddings.')
    parser.add_argument('--ontology_embedding_weight', type=float, default=1.0,
                        help='Interpolation weight for ontology-expanded query embeddings. 1.0 uses only ontology-expanded text; 0.0 uses only the original query.')
    parser.add_argument('--ontology_fact_boost_weight', type=float, default=0.0,
                        help='Interpolation weight for ontology-guided fact score boosting before fact reranking and PPR.')
    parser.add_argument('--ontology_fact_boost_max_facts_per_hint', type=int, default=500,
                        help='Skip ontology hints that match too many facts; <=0 disables this high-frequency hint guard.')
    parser.add_argument('--enable_query_aware_fact_expansion', type=str, default='false',
                        help='If True, add query-aware bridge facts after fact reranking and before graph PPR.')
    parser.add_argument('--query_aware_candidate_top_k', type=int, default=80,
                        help='Number of dense fact candidates considered by query-aware bridge fact expansion.')
    parser.add_argument('--query_aware_max_added_facts', type=int, default=5,
                        help='Maximum bridge facts added by query-aware fact expansion.')
    parser.add_argument('--query_aware_max_facts_per_entity', type=int, default=120,
                        help='Skip entity neighborhoods larger than this during query-aware fact expansion; <=0 disables this guard.')
    parser.add_argument('--query_aware_score_weight', type=float, default=0.20,
                        help='Weight used to raise selected bridge fact scores toward the current anchor fact score.')
    parser.add_argument('--enable_query_local_ontology', type=str, default='false',
                        help='If True, dynamically call query-local entity/relation/bridge/verifier agents before PPR.')
    parser.add_argument('--query_local_ontology_candidate_top_k', type=int, default=24,
                        help='Number of dense top facts shown to query-local ontology agents.')
    parser.add_argument('--query_local_ontology_top_passages', type=int, default=2,
                        help='Number of dense top passages shown to query-local ontology agents.')
    parser.add_argument('--query_local_ontology_passage_chars', type=int, default=600,
                        help='Maximum characters per passage snippet in query-local ontology prompts.')
    parser.add_argument('--query_local_ontology_max_added_facts', type=int, default=5,
                        help='Maximum verifier-accepted bridge facts injected into PPR.')
    parser.add_argument('--query_local_ontology_score_weight', type=float, default=0.60,
                        help='Weight used to raise verifier-accepted bridge fact scores toward the current anchor fact score.')
    parser.add_argument('--query_local_ontology_max_completion_tokens', type=int, default=512,
                        help='Maximum completion tokens for each query-local ontology agent call.')
    parser.add_argument('--enable_query_ontology_prior', type=str, default='false',
                        help='If True, use a reusable historical-query ontology prior for lightweight online fact-score boosting.')
    parser.add_argument('--query_ontology_prior_path', type=str, default=None,
                        help='Path to historical-query ontology prior JSON. If unset, uses query_ontology_priors.json in the working directory.')
    parser.add_argument('--query_ontology_prior_boost_weight', type=float, default=0.12,
                        help='Interpolation weight for historical-query ontology prior fact-score boosting.')
    parser.add_argument('--query_ontology_prior_top_k', type=int, default=8,
                        help='Number of matched ontology prior hints used for each query.')
    parser.add_argument('--query_ontology_prior_max_boost_facts', type=int, default=80,
                        help='Maximum facts whose scores can be boosted by query ontology prior; <=0 disables this cap.')
    parser.add_argument('--query_ontology_schema_mode', type=str, default='gated',
                        choices=['gated', 'hints_only', 'templates_only', 'path_gated'],
                        help='Historical-query ontology prior mode for bridge-template application.')
    parser.add_argument('--query_ontology_template_candidate_top_k', type=int, default=500,
                        help='Only the current top-k fact candidates are eligible for bridge-template boosts.')
    parser.add_argument('--query_ontology_template_max_boost_facts', type=int, default=20,
                        help='Maximum bridge-template boosted facts before the global prior cap is applied.')
    parser.add_argument('--query_ontology_hint_candidate_top_k', type=int, default=500,
                        help='Only the current top-k fact candidates are eligible for relation/entity hint boosts.')
    parser.add_argument('--query_ontology_require_query_anchor', type=str, default='true',
                        help='If True, prior boosts require concrete entity/token overlap with the current query.')
    parser.add_argument('--enable_query_ontology_prior_ppr_injection', type=str, default='false',
                        help='If True, add boosted query ontology prior facts as additional PPR seed facts.')
    parser.add_argument('--query_ontology_prior_ppr_top_k', type=int, default=4,
                        help='Maximum boosted query ontology prior facts injected into PPR per query.')
    parser.add_argument('--query_ontology_prior_ppr_score_weight', type=float, default=0.50,
                        help='Weight used to raise prior-injected PPR fact scores toward the current anchor fact score.')
    parser.add_argument('--query_ontology_prior_ppr_expand_graph_paths', type=str, default='true',
                        help='If True, instantiate matched ontology schemas by expanding graph-neighbor path facts from selected anchors.')
    parser.add_argument('--query_ontology_prior_ppr_max_facts_per_entity', type=int, default=120,
                        help='Skip graph-neighbor path expansion for entities with more than this many incident facts; <=0 disables this guard.')
    parser.add_argument('--query_ontology_prior_ppr_intent_gate', type=str, default='false',
                        help='If True, inject prior PPR path facts only when their relation family matches the query intent.')
    parser.add_argument('--enable_query_ontology_prior_path_instantiation', type=str, default='false',
                        help='If True, actively instantiate matched ontology path templates from query anchor entities in the graph.')
    parser.add_argument('--query_ontology_prior_path_instantiation_top_k', type=int, default=8,
                        help='Maximum graph facts added per query from active ontology path instantiation.')
    parser.add_argument('--query_ontology_prior_path_instantiation_max_anchor_entities', type=int, default=16,
                        help='Maximum query-matched graph entities used as active ontology path anchors.')
    parser.add_argument('--query_ontology_prior_path_instantiation_max_paths', type=int, default=80,
                        help='Maximum instantiated ontology paths retained per query before fact injection.')
    parser.add_argument('--query_ontology_prior_path_instantiation_min_template_precision', type=float, default=0.45,
                        help='Minimum historical path precision required for active bridge-template instantiation.')
    parser.add_argument('--query_ontology_prior_path_instantiation_max_facts_per_entity', type=int, default=120,
                        help='Skip active path expansion for graph entities with more incident facts than this; <=0 disables this guard.')
    parser.add_argument('--enable_query_ontology_prior_passage_lift', type=str, default='false',
                        help='If True, lightly lift source passages of ontology-prior injected facts after PPR ranking.')
    parser.add_argument('--query_ontology_prior_passage_lift_weight', type=float, default=0.03,
                        help='Late-stage normalized score interpolation weight for ontology-prior source passage lift.')
    parser.add_argument('--query_ontology_prior_passage_lift_top_k', type=int, default=20,
                        help='Maximum source passages lifted per query from ontology-prior injected facts.')
    parser.add_argument('--query_ontology_prior_passage_lift_use_entity_fallback', type=str, default='false',
                        help='If True, lift passages linked to injected fact entities when exact fact-source passages are unavailable.')
    parser.add_argument('--query_ontology_prior_trace_path', type=str, default=None,
                        help='Optional JSONL path for per-query ontology-prior boost/injection diagnostics.')
    parser.add_argument('--enable_query_plan_ontology', type=str, default='false',
                        help='If True, use query-conditioned micro-ontology plans for late passage reranking.')
    parser.add_argument('--query_plan_ontology_path', type=str, default=None,
                        help='Path to query_plan_ontology.json.')
    parser.add_argument('--query_plan_ontology_compiler_sidecar_path', type=str, default=None,
                        help='Optional proposal-only ontology for executable compilation; it never replaces the incumbent retrieval ontology.')
    parser.add_argument('--query_plan_relation_realization_path', type=str, default=None,
                        help='Optional path to relation_realization_map.json for mapping query-plan conceptual relations to OpenIE relation surfaces.')
    parser.add_argument('--query_plan_ontology_compiler_sidecar_relation_realization_path', type=str, default=None,
                        help='Optional relation realization map paired with the proposal-only compiler ontology.')
    parser.add_argument('--query_plan_use_relation_realization', type=str, default='true',
                        help='If True, query-plan graph execution can use relation_realization_map.json when conceptual family/signature matching fails.')
    parser.add_argument('--query_plan_relation_realization_allow_family_fallback', type=str, default='false',
                        help='If True, relation realization can match by realized relation family. Defaults to false to avoid broad OpenIE relation drift.')
    parser.add_argument('--query_plan_relation_realization_require_ontology_hash_match', type=str, default='false',
                        help='If True, fail closed unless the realization artifact was built from the active ontology file.')
    parser.add_argument('--query_plan_ontology_enable_schema_linked_router', type=str, default='false',
                        help='If True, route directly over leaf schemas using exact query relation coverage.')
    parser.add_argument('--query_plan_ontology_enable_schema_linked_compiler_sidecar', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_ENABLE_SCHEMA_LINKED_COMPILER_SIDECAR', 'false'),
                        help='Expose schema-linked plans to the compiler without replacing incumbent retrieval plans.')
    parser.add_argument('--query_plan_ontology_matching_mode', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_MATCHING_MODE', 'learned_structural'),
                        choices=['learned_structural', 'semantic_only', 'direct_query'],
                        help='Controlled ontology matching mode for reusable-schema ablations.')
    parser.add_argument('--query_plan_ontology_top_k', type=int, default=6,
                        help='Maximum query-plan templates matched per query.')
    parser.add_argument('--query_plan_ontology_min_confidence', type=float, default=0.72,
                        help='Minimum query-plan confidence for online matching.')
    parser.add_argument('--query_plan_ontology_max_paths', type=int, default=60,
                        help='Maximum executable graph paths retained per query.')
    parser.add_argument('--query_plan_ontology_max_branch_facts', type=int, default=80,
                        help='Skip plan branches whose current entity has too many incident facts; <=0 disables guard.')
    parser.add_argument('--query_plan_ontology_signature_match_mode', type=str, default='strict_then_soft',
                        choices=['strict', 'soft', 'family', 'strict_then_soft', 'strict_plus_validated_soft'],
                        help='Relation signature matching mode for query-plan graph execution.')
    parser.add_argument('--query_plan_ontology_passage_lift_weight', type=float, default=0.04,
                        help='Late-stage normalized score interpolation weight for query-plan source passage lift.')
    parser.add_argument('--query_plan_ontology_passage_rerank_mode', type=str, default='interpolate',
                        choices=['interpolate', 'boost', 'adaptive_promote'],
                        help='How query-plan passage evidence changes passage scores.')
    parser.add_argument('--query_plan_ontology_passage_top_k', type=int, default=10,
                        help='Maximum query-plan source passages lifted per query.')
    parser.add_argument('--query_plan_ontology_passage_reranker_high_confidence_threshold', type=float, default=0.70,
                        help='Adaptive passage reranker promotion threshold for ontology-supported passages.')
    parser.add_argument('--query_plan_ontology_passage_reranker_promote_top_k', type=int, default=1,
                        help='Maximum high-confidence ontology-supported passages promoted per query.')
    parser.add_argument('--query_plan_ontology_passage_reranker_promote_to_rank', type=int, default=2,
                        help='Target rank slot for adaptive ontology passage promotion.')
    parser.add_argument('--query_plan_ontology_passage_reranker_preserve_top_k', type=int, default=1,
                        help='Number of baseline top passages preserved before adaptive ontology promotions.')
    parser.add_argument('--query_plan_ontology_passage_reranker_enable_promotion_verifier', type=str, default='false',
                        help='If True, verify ontology-supported passage promotions before inserting passages near the top.')
    parser.add_argument('--query_plan_ontology_passage_reranker_verifier_min_confidence', type=float, default=0.85,
                        help='Minimum adaptive passage confidence required by the conservative promotion verifier.')
    parser.add_argument('--query_plan_ontology_passage_reranker_verifier_max_original_rank', type=int, default=5,
                        help='Maximum baseline rank eligible for conservative ontology passage promotion; <=0 disables this verifier bound.')
    parser.add_argument('--query_plan_ontology_passage_promotion_policy_path', type=str, default='',
                        help='Optional learned query-plan passage promotion policy JSON path.')
    parser.add_argument('--query_plan_ontology_passage_promotion_policy_mode', type=str, default='off',
                        choices=['off', 'gate', 'override'],
                        help='Learned query-plan passage promotion policy mode.')
    parser.add_argument('--query_plan_ontology_passage_promotion_policy_enable_coverage_gate', type=str, default='false',
                        help='If True, policy-approved passage promotions must add a new ontology evidence role before entering the top window.')
    parser.add_argument('--query_plan_ontology_passage_promotion_policy_coverage_top_k', type=int, default=2,
                        help='Top-k evidence window used by the promotion policy coverage gate.')
    parser.add_argument('--query_plan_ontology_enable_evidence_context_builder', type=str, default='false',
                        help='If True, assemble a QA context from baseline retrieval plus complementary query-plan evidence passages.')
    parser.add_argument('--query_plan_ontology_evidence_context_builder_mode', type=str, default='qa_only',
                        choices=['qa_only', 'retrieval_and_qa'],
                        help='Evidence context builder mode: qa_only keeps retrieval metrics unchanged; retrieval_and_qa also changes returned retrieval docs.')
    parser.add_argument('--query_plan_ontology_evidence_context_builder_preserve_top_k', type=int, default=1,
                        help='Number of baseline top passages preserved before adding query-plan evidence passages to the QA context.')
    parser.add_argument('--query_plan_ontology_evidence_context_builder_max_evidence_passages', type=int, default=2,
                        help='Maximum complementary query-plan evidence passages inserted into the QA context.')
    parser.add_argument('--query_plan_ontology_evidence_context_builder_max_original_rank', type=int, default=20,
                        help='Maximum original retrieval rank eligible for QA context insertion; <=0 disables this bound.')
    parser.add_argument('--query_plan_ontology_evidence_context_builder_min_candidate_confidence', type=float, default=0.0,
                        help='Minimum adaptive passage confidence required for QA context insertion.')
    parser.add_argument('--query_plan_ontology_evidence_context_builder_require_new_evidence_role', type=str, default='true',
                        help='If True, inserted QA context passages must add a new query-plan role, anchor, plan, or relation signature.')
    parser.add_argument(
        '--query_plan_ontology_reader_intervention_require_atomic_commit',
        type=str,
        default=os.environ.get(
            'QUERY_PLAN_ONTOLOGY_READER_INTERVENTION_REQUIRE_ATOMIC_COMMIT',
            'false',
        ),
        help='If True, only atomically committed ontology plans may reorder reader context or inject concrete QA evidence.',
    )
    parser.add_argument('--query_plan_ontology_evidence_context_builder_enable_chain_guidance', type=str, default='false',
                        help='If True, add ontology-grounded evidence-chain and answer-form guidance to the QA prompt.')
    parser.add_argument('--query_plan_ontology_evidence_context_builder_enable_logical_plan_guidance', type=str, default='false',
                        help='If True, compile an admitted logical evidence plan into a passage-grounded reader verification order.')
    parser.add_argument('--query_plan_ontology_evidence_context_builder_logical_plan_require_commit', type=str, default='true',
                        help='If True, expose logical-plan guidance only after an atomic complete-plan commit.')
    parser.add_argument('--query_plan_ontology_evidence_context_builder_max_chain_items', type=int, default=4,
                        help='Maximum context-builder evidence-chain bullets added to the QA prompt.')
    parser.add_argument('--query_plan_ontology_evidence_context_builder_chain_guidance_mode', type=str, default='always',
                        choices=['always', 'gated', 'operator_safe', 'hint_only'],
                        help='Evidence-chain guidance mode. operator_safe admits concrete chains only from committed plans with operator-compatible, unambiguous terminals; hint_only exposes instructions but no concrete ontology terminal.')
    parser.add_argument('--query_plan_ontology_evidence_context_builder_allow_hint_only', type=str, default='true',
                        help='If False, do not add answer-form-only chain guidance when no concrete chain evidence is available.')
    parser.add_argument('--query_plan_ontology_evidence_context_builder_enable_comparative_branches', type=str, default='false',
                        help='If True, comparative questions require two distinct evidence branches before chain guidance is injected.')
    parser.add_argument('--query_plan_ontology_enable_state_retriever', type=str, default='false',
                        help='If True, use query-plan state coverage to reassemble the retrieval top-k evidence window.')
    parser.add_argument('--query_plan_ontology_state_retriever_top_k', type=int, default=5,
                        help='Top-k evidence window optimized by the query-plan state retriever.')
    parser.add_argument('--query_plan_ontology_state_retriever_candidate_top_k', type=int, default=50,
                        help='Number of baseline-ranked passages considered by the query-plan state retriever.')
    parser.add_argument('--query_plan_ontology_state_retriever_preserve_top_k', type=int, default=1,
                        help='Number of highest baseline passages preserved before state-coverage selection.')
    parser.add_argument('--query_plan_ontology_state_retriever_max_original_rank', type=int, default=50,
                        help='Maximum original rank for ontology-supported passages considered by the state retriever; <=0 disables this bound.')
    parser.add_argument('--query_plan_ontology_state_retriever_baseline_weight', type=float, default=0.45,
                        help='Weight of original retrieval rank/score inside the state retriever.')
    parser.add_argument('--query_plan_ontology_state_retriever_coverage_weight', type=float, default=1.00,
                        help='Weight of new query-plan state slot coverage inside the state retriever.')
    parser.add_argument('--query_plan_ontology_state_retriever_evidence_weight', type=float, default=0.65,
                        help='Weight of ontology evidence strength inside the state retriever.')
    parser.add_argument('--query_plan_ontology_state_retriever_redundancy_penalty', type=float, default=0.25,
                        help='Penalty for selecting passages that repeat already covered query-plan state slots.')
    parser.add_argument('--query_plan_ontology_state_retriever_min_evidence_score', type=float, default=0.0,
                        help='Minimum ontology evidence score required before an ontology-supported passage can be selected by the state retriever.')
    parser.add_argument('--query_plan_ontology_state_retriever_require_new_slot', type=str, default='false',
                        help='If True, ontology-supported passages must add a new state slot to be selected by the state retriever.')
    parser.add_argument('--query_plan_ontology_state_retriever_max_ontology_passages', type=int, default=1,
                        help='Maximum ontology-supported passages inserted into the state-retrieved top-k window.')
    parser.add_argument('--query_plan_ontology_state_retriever_require_policy_approval', type=str, default='true',
                        help='If True, state retriever uses ontology passage signals only when the learned promotion policy approves them, unless a high-confidence fallback applies.')
    parser.add_argument('--query_plan_ontology_state_retriever_policy_fallback_min_confidence', type=float, default=0.85,
                        help='Minimum passage-reranker confidence for state retriever fallback when no policy decision exists.')
    parser.add_argument('--query_plan_ontology_state_retriever_policy_fallback_max_original_rank', type=int, default=5,
                        help='Maximum original rank for state retriever high-confidence fallback when no policy decision exists; <=0 disables this bound.')
    parser.add_argument('--query_plan_ontology_state_retriever_require_structural_slot', type=str, default='true',
                        help='If True, state retriever only promotes ontology passages that add a role, relation, plan, or verified path slot.')
    parser.add_argument('--query_plan_ontology_state_policy_path', type=str, default='',
                        help='Optional learned state retriever policy JSON path.')
    parser.add_argument('--query_plan_ontology_state_policy_mode', type=str, default='off',
                        choices=['off', 'gate', 'override'],
                        help='Learned state retriever policy mode. gate requires learned approval; override lets learned state rules replace the passage policy gate.')
    parser.add_argument('--query_plan_ontology_enable_subgraph_retriever', type=str, default='false',
                        help='If True, expand a query-plan-guided local fact subgraph and append source passages to the evidence candidate pool.')
    parser.add_argument('--query_plan_ontology_subgraph_candidate_fact_top_k', type=int, default=240,
                        help='Maximum ontology-subgraph facts considered per query.')
    parser.add_argument('--query_plan_ontology_subgraph_max_facts_per_anchor', type=int, default=120,
                        help='Maximum fact-graph neighbors scanned for each concrete query anchor.')
    parser.add_argument('--query_plan_ontology_subgraph_max_facts_per_relation', type=int, default=160,
                        help='Maximum fact-graph facts scanned for each query-plan relation family.')
    parser.add_argument('--query_plan_ontology_subgraph_max_source_passages_per_fact', type=int, default=3,
                        help='Maximum source passages exposed for each ontology-subgraph candidate fact.')
    parser.add_argument('--query_plan_ontology_subgraph_max_added_passages', type=int, default=24,
                        help='Maximum ontology-subgraph source passages appended to downstream candidate selection.')
    parser.add_argument('--query_plan_ontology_subgraph_max_candidate_pool_rank', type=int, default=260,
                        help='Synthetic rank cap assigned to appended ontology-subgraph passages.')
    parser.add_argument('--query_plan_ontology_subgraph_min_fact_score', type=float, default=0.10,
                        help='Minimum combined query/ontology score for a fact to become an ontology-subgraph candidate.')
    parser.add_argument('--query_plan_ontology_subgraph_min_passage_score', type=float, default=0.18,
                        help='Minimum aggregated ontology-subgraph score for a source passage to be exposed downstream.')
    parser.add_argument('--query_plan_ontology_subgraph_relation_weight', type=float, default=0.34,
                        help='Weight of query-plan relation-family match in ontology-subgraph fact scoring.')
    parser.add_argument('--query_plan_ontology_subgraph_anchor_weight', type=float, default=0.30,
                        help='Weight of concrete anchor overlap in ontology-subgraph fact scoring.')
    parser.add_argument('--query_plan_ontology_subgraph_path_neighbor_weight', type=float, default=0.22,
                        help='Weight of executable-path seed or neighbor evidence in ontology-subgraph fact scoring.')
    parser.add_argument('--query_plan_ontology_subgraph_query_fact_weight', type=float, default=0.14,
                        help='Weight of CompassRAG query-fact relevance in ontology-subgraph fact scoring.')
    parser.add_argument('--query_plan_ontology_subgraph_use_entity_fallback', type=str, default='true',
                        help='If True, fall back to entity-neighbor source passages when direct fact source passages are unavailable.')
    parser.add_argument('--query_plan_ontology_subgraph_append_score_margin', type=float, default=1e-7,
                        help='Tiny decreasing score margin used when appending ontology-subgraph passages after the original retrieval list.')
    parser.add_argument('--query_plan_ontology_subgraph_min_original_rank', type=int, default=1,
                        help='Minimum original retrieval rank for ontology-subgraph passages; use >5 to only add candidates that can improve Recall@5.')
    parser.add_argument('--query_plan_ontology_subgraph_max_original_rank', type=int, default=0,
                        help='Maximum original retrieval rank for ontology-subgraph passages; <=0 disables.')
    parser.add_argument('--query_plan_ontology_subgraph_require_strong_fact_gate', type=str, default='false',
                        help='If True, candidate facts must be path seeds, anchor+relation matches, or high query-score relation-only matches.')
    parser.add_argument('--query_plan_ontology_subgraph_relation_only_min_query_fact_score', type=float, default=0.35,
                        help='Minimum query-fact score required for relation-only ontology-subgraph facts when strong fact gate is enabled.')
    parser.add_argument('--query_plan_ontology_enable_support_evidence_retriever', type=str, default='false',
                        help='If True, issue query-plan ontology dense support probes and expose verified passages downstream.')
    parser.add_argument('--query_plan_ontology_support_evidence_dense_top_k', type=int, default=30,
                        help='Dense top-k passages considered for each support evidence probe.')
    parser.add_argument('--query_plan_ontology_support_evidence_max_probes', type=int, default=10,
                        help='Maximum ontology-derived support evidence probes issued per query.')
    parser.add_argument('--query_plan_ontology_support_evidence_max_candidates', type=int, default=4,
                        help='Maximum verified support evidence candidates exposed downstream.')
    parser.add_argument('--query_plan_ontology_support_evidence_min_confidence', type=float, default=0.70,
                        help='Minimum confidence required before a support evidence candidate is accepted.')
    parser.add_argument('--query_plan_ontology_support_evidence_preserve_top_k', type=int, default=2,
                        help='Number of top baseline passages protected from support evidence insertion.')
    parser.add_argument('--query_plan_ontology_support_evidence_synthetic_original_rank', type=int, default=40,
                        help='Synthetic original rank for support evidence passages not present in baseline retrieval.')
    parser.add_argument('--query_plan_ontology_support_evidence_require_anchor_hit', type=str, default='true',
                        help='If True, support evidence candidates with concrete anchors must mention the anchor.')
    parser.add_argument('--query_plan_ontology_support_evidence_require_relation_or_answer_hit', type=str, default='true',
                        help='If True, support evidence candidates must mention relation, terminal, or answer-type clues.')
    parser.add_argument('--query_plan_ontology_support_evidence_candidate_only', type=str, default='true',
                        help='If True, support evidence retrieval appends candidates and lets evidence-set selection decide.')
    parser.add_argument('--query_plan_ontology_support_evidence_score_margin', type=float, default=1e-7,
                        help='Small score margin used when appending support evidence candidates.')
    parser.add_argument('--query_plan_ontology_support_evidence_probe_policy_path', type=str, default='',
                        help='Optional path to a trace-feedback support probe policy JSON.')
    parser.add_argument('--query_plan_ontology_support_evidence_probe_policy_mode', type=str, default='off',
                        help='How to use support probe policy: off or filter.')
    parser.add_argument('--query_plan_ontology_enable_schema_route_retriever', type=str, default='false',
                        help='If True, decompose query-plan schema into route-specific dense retrieval probes.')
    parser.add_argument('--query_plan_ontology_schema_route_dense_top_k', type=int, default=80,
                        help='Dense top-k passages considered for each schema-route probe.')
    parser.add_argument('--query_plan_ontology_schema_route_max_subqueries', type=int, default=8,
                        help='Maximum schema-guided route subqueries issued per query.')
    parser.add_argument('--query_plan_ontology_schema_route_max_candidates', type=int, default=6,
                        help='Maximum verified schema-route candidates exposed downstream.')
    parser.add_argument('--query_plan_ontology_schema_route_min_confidence', type=float, default=0.58,
                        help='Minimum confidence required before accepting a schema-route candidate.')
    parser.add_argument('--query_plan_ontology_schema_route_preserve_top_k', type=int, default=2,
                        help='Number of top baseline passages protected from schema-route insertion.')
    parser.add_argument('--query_plan_ontology_schema_route_synthetic_original_rank', type=int, default=35,
                        help='Synthetic original rank used so schema-route candidates enter evidence-set selection.')
    parser.add_argument('--query_plan_ontology_schema_route_candidate_only', type=str, default='true',
                        help='If True, schema-route retrieval appends candidates for evidence-set selection.')
    parser.add_argument('--query_plan_ontology_schema_route_score_margin', type=float, default=1e-7,
                        help='Small score margin used when appending schema-route candidates.')
    parser.add_argument('--query_plan_ontology_schema_route_require_anchor_hit', type=str, default='true',
                        help='If True, anchored schema routes require passage-anchor match.')
    parser.add_argument('--query_plan_ontology_schema_route_require_route_signal', type=str, default='true',
                        help='If True, schema-route candidates must match route relation/terminal/answer signal.')
    parser.add_argument('--query_plan_ontology_schema_route_enable_community_route', type=str, default='true',
                        help='If True, add community/topic-style schema route probes for weakly anchored queries.')
    parser.add_argument('--query_plan_ontology_schema_route_enable_dfs_route', type=str, default='true',
                        help='If True, multi-relation schema paths are treated as DFS-style route probes.')
    parser.add_argument('--query_plan_ontology_schema_route_policy_model_path', type=str, default='',
                        help='Optional learned utility policy JSON for schema-route candidates.')
    parser.add_argument('--query_plan_ontology_schema_route_policy_mode', type=str, default='off',
                        choices=['off', 'score', 'filter'],
                        help='How to use schema-route utility policy: off, score, or filter.')
    parser.add_argument('--query_plan_ontology_schema_route_policy_min_score', type=float, default=0.55,
                        help='Minimum schema-route policy score in filter mode.')
    parser.add_argument('--query_plan_ontology_schema_route_policy_reject_below', type=float, default=0.0,
                        help='Always reject schema-route candidates with policy score below this threshold.')
    parser.add_argument('--query_plan_ontology_schema_route_policy_score_weight', type=float, default=0.12,
                        help='Score interpolation weight for schema-route utility policy.')
    parser.add_argument('--query_plan_ontology_enable_schema_slot_recall_expander', type=str, default='false',
                        help='If True, issue schema-slot constrained dense probes to add missing-gold recall candidates.')
    parser.add_argument('--query_plan_ontology_schema_slot_recall_dense_top_k', type=int, default=80,
                        help='Dense top-k passages considered for each schema-slot recall probe.')
    parser.add_argument('--query_plan_ontology_schema_slot_recall_max_probes', type=int, default=12,
                        help='Maximum schema-slot recall probes issued per query.')
    parser.add_argument('--query_plan_ontology_schema_slot_recall_max_candidates', type=int, default=10,
                        help='Maximum schema-slot recall candidates exposed downstream.')
    parser.add_argument('--query_plan_ontology_schema_slot_recall_min_confidence', type=float, default=0.48,
                        help='Minimum confidence required before accepting a schema-slot recall candidate.')
    parser.add_argument('--query_plan_ontology_schema_slot_recall_synthetic_original_rank', type=int, default=90,
                        help='Synthetic original rank used so schema-slot recall candidates enter the evidence-set candidate pool.')
    parser.add_argument('--query_plan_ontology_schema_slot_recall_require_anchor_or_relation', type=str, default='true',
                        help='If True, schema-slot recall candidates must match anchor/relation/terminal/answer cues.')
    parser.add_argument('--query_plan_ontology_schema_slot_recall_include_original_query', type=str, default='true',
                        help='If True, include original-query augmented schema-slot probes.')
    parser.add_argument('--query_plan_ontology_schema_slot_recall_enable_multi_probe', type=str, default='false',
                        help='If True, build multiple missing-slot ontology dense probes before evidence-set selection.')
    parser.add_argument('--query_plan_ontology_schema_slot_recall_target_datasets', type=str, default='',
                        help='Comma-separated dataset names where schema-slot multi-probe recall is enabled; empty means all datasets.')
    parser.add_argument('--query_plan_ontology_schema_slot_recall_max_missing_slot_terms', type=int, default=8,
                        help='Maximum missing ontology slot terms used to prioritize multi-probe recall templates.')
    parser.add_argument('--query_plan_ontology_schema_slot_recall_min_relation_overlap', type=float, default=0.20,
                        help='Minimum relation-token overlap required for weak multi-probe candidates.')
    parser.add_argument('--query_plan_ontology_schema_slot_recall_answer_type_boost', type=float, default=0.08,
                        help='Confidence boost for schema-slot recall candidates that match answer-type cues.')
    parser.add_argument('--query_plan_ontology_enable_bind_probe_join', type=str, default='false',
                        help='If True, bind bridge entities from first-hop OpenIE triples and execute bound second-hop probes.')
    parser.add_argument('--query_plan_ontology_bind_probe_join_target_datasets', type=str, default='',
                        help='Comma-separated datasets where ontology bind-and-probe is enabled; empty means all datasets.')
    parser.add_argument('--query_plan_ontology_bind_probe_join_anchor_top_k', type=int, default=8,
                        help='Number of current top passages inspected for grounded bridge bindings.')
    parser.add_argument('--query_plan_ontology_bind_probe_join_max_bindings', type=int, default=4,
                        help='Maximum distinct bridge bindings retained per query.')
    parser.add_argument('--query_plan_ontology_bind_probe_join_max_probes', type=int, default=8,
                        help='Maximum bound second-hop probes issued per query.')
    parser.add_argument('--query_plan_ontology_bind_probe_join_dense_top_k', type=int, default=64,
                        help='Dense retrieval window inspected for each bound probe.')
    parser.add_argument('--query_plan_ontology_bind_probe_join_max_candidates', type=int, default=8,
                        help='Maximum bind-and-probe candidates exposed to evidence-set selection.')
    parser.add_argument('--query_plan_ontology_bind_probe_join_min_binding_score', type=float, default=0.52,
                        help='Minimum score for a bridge binding extracted from a first-hop triple.')
    parser.add_argument('--query_plan_ontology_bind_probe_join_min_candidate_score', type=float, default=0.62,
                        help='Minimum score for a bound second-hop candidate.')
    parser.add_argument('--query_plan_ontology_bind_probe_join_min_relation_overlap', type=float, default=0.20,
                        help='Minimum unresolved-relation overlap when no answer cue is present.')
    parser.add_argument('--query_plan_ontology_bind_probe_join_synthetic_original_rank', type=int, default=60,
                        help='Synthetic rank assigned to out-of-window bind-and-probe candidates.')
    parser.add_argument('--query_plan_ontology_bind_probe_join_require_relation_or_answer', type=str, default='true',
                        help='If True, bound second-hop candidates must match the unresolved relation or answer cue.')
    parser.add_argument('--query_plan_ontology_bind_probe_join_enable_relation_role_constraints', type=str, default='false',
                        help='If True, separate bridge and terminal relation roles before issuing bound probes.')
    parser.add_argument('--query_plan_ontology_bind_probe_join_require_role_consistent_candidate', type=str, default='false',
                        help='If True, require a bridge-connected terminal triple with the requested predicate and answer type.')
    parser.add_argument('--query_plan_ontology_bind_probe_join_min_terminal_relation_overlap', type=float, default=0.40,
                        help='Minimum typed terminal-relation overlap for a bridge-connected OpenIE triple.')
    parser.add_argument('--query_plan_ontology_bind_probe_join_min_terminal_type_score', type=float, default=0.60,
                        help='Minimum compatibility between the terminal object and requested answer type.')
    parser.add_argument('--query_plan_ontology_bind_probe_join_max_role_candidate_dense_rank', type=int, default=0,
                        help='Maximum dense rank inspected per typed terminal probe; zero disables the bound.')
    parser.add_argument('--query_plan_ontology_enable_bind_probe_transaction_optimizer', type=str, default='false',
                        help='If True, isolate bind-probe candidates and atomically commit complete source-terminal evidence actions.')
    parser.add_argument('--query_plan_ontology_bind_probe_transaction_target_datasets', type=str, default='',
                        help='Comma-separated datasets where bind-probe evidence transactions are evaluated.')
    parser.add_argument('--query_plan_ontology_bind_probe_transaction_top2_target_datasets', type=str,
                        default='2wikimultihopqa,hotpotqa',
                        help='Comma-separated datasets where complete bind-probe chains may replace the final top-2 pair.')
    parser.add_argument('--query_plan_ontology_bind_probe_transaction_min_action_utility', type=float, default=0.80,
                        help='Minimum typed-chain utility required before transaction evaluation.')
    parser.add_argument('--query_plan_ontology_bind_probe_transaction_min_net_benefit', type=float, default=0.10,
                        help='Minimum action utility minus displacement and probe cost required for commit.')
    parser.add_argument('--query_plan_ontology_bind_probe_transaction_max_terminal_rank', type=int, default=5,
                        help='Maximum final rank of a terminal passage eligible for top-2 promotion.')
    parser.add_argument('--query_plan_ontology_bind_probe_transaction_require_source_top1', type=str, default='true',
                        help='If True, preserve the rank-1 binding source and replace only rank 2.')
    parser.add_argument('--query_plan_ontology_bind_probe_transaction_displacement_cost', type=float, default=0.62,
                        help='Estimated risk of displacing the current rank-2 passage.')
    parser.add_argument('--query_plan_ontology_bind_probe_transaction_probe_cost_weight', type=float, default=0.025,
                        help='Weight applied to normalized bound-probe execution cost.')
    parser.add_argument('--query_plan_ontology_enable_entity_keyed_top5_optimizer', type=str, default='false',
                        help='If True, admit strict external entity-keyed bind-probe terminals at rank 5 while protecting top-2.')
    parser.add_argument('--query_plan_ontology_entity_keyed_top5_target_datasets', type=str,
                        default='2wikimultihopqa,musique',
                        help='Comma-separated datasets eligible for rank-isolated entity-keyed top-5 admission.')
    parser.add_argument('--query_plan_ontology_entity_keyed_top5_min_confidence', type=float, default=0.84,
                        help='Minimum bind-probe confidence for entity-keyed rank-5 admission.')
    parser.add_argument('--query_plan_ontology_entity_keyed_top5_min_relation_score', type=float, default=0.80,
                        help='Minimum terminal relation score for entity-keyed rank-5 admission.')
    parser.add_argument('--query_plan_ontology_entity_keyed_top5_min_type_score', type=float, default=0.75,
                        help='Minimum terminal answer-type score for entity-keyed rank-5 admission.')
    parser.add_argument('--query_plan_ontology_entity_keyed_top5_min_binding_score', type=float, default=0.58,
                        help='Minimum source entity binding score for entity-keyed rank-5 admission.')
    parser.add_argument('--query_plan_ontology_entity_keyed_top5_min_action_utility', type=float, default=0.80,
                        help='Minimum composite action utility for entity-keyed rank-5 admission.')
    parser.add_argument('--query_plan_ontology_entity_keyed_top5_max_source_rank', type=int, default=2,
                        help='Maximum rank of the binding source in the protected prefix.')
    parser.add_argument('--query_plan_ontology_enable_factorized_query_executor', type=str, default='false',
                        help='If True, execute ontology plans with costed relational physical operators.')
    parser.add_argument('--query_plan_ontology_factorized_target_datasets', type=str,
                        default='2wikimultihopqa,musique,nq_rear,popqa',
                        help='Comma-separated datasets eligible for factorized query execution.')
    parser.add_argument('--query_plan_ontology_factorized_enable_entity_relation_index_scan', type=str, default='true',
                        help='Enable the materialized entity-relation index scan operator.')
    parser.add_argument('--query_plan_ontology_factorized_enable_cascaded_bind_join', type=str, default='true',
                        help='Enable bounded multi-hop cascaded bind joins.')
    parser.add_argument('--query_plan_ontology_factorized_enable_dual_branch_comparison_join', type=str, default='true',
                        help='Enable atomic dual-branch comparison joins.')
    parser.add_argument('--query_plan_ontology_factorized_dense_top_k', type=int, default=48,
                        help='Dense fallback window per factorized probe.')
    parser.add_argument('--query_plan_ontology_factorized_max_probes', type=int, default=10,
                        help='Maximum physical probes issued for one query.')
    parser.add_argument('--query_plan_ontology_factorized_max_candidates', type=int, default=8,
                        help='Maximum validated factorized candidates retained per query.')
    parser.add_argument('--query_plan_ontology_factorized_max_hops', type=int, default=4,
                        help='Maximum number of cascaded join steps.')
    parser.add_argument('--query_plan_ontology_factorized_beam_width', type=int, default=4,
                        help='Maximum intermediate bindings retained after each join.')
    parser.add_argument('--query_plan_ontology_factorized_max_index_rows', type=int, default=64,
                        help='Maximum materialized entity-relation rows inspected per access.')
    parser.add_argument('--query_plan_ontology_factorized_index_max_rows_per_entity', type=int, default=256,
                        help='Maximum OpenIE rows materialized for one normalized entity key.')
    parser.add_argument('--query_plan_ontology_factorized_min_relation_score', type=float, default=0.34,
                        help='Minimum predicate compatibility for a factorized join edge.')
    parser.add_argument('--query_plan_ontology_factorized_min_type_score', type=float, default=0.55,
                        help='Minimum answer-type compatibility at a terminal edge.')
    parser.add_argument('--query_plan_ontology_factorized_min_candidate_utility', type=float, default=0.68,
                        help='Minimum cost-adjusted utility retained by a physical operator.')
    parser.add_argument('--query_plan_ontology_factorized_cost_weight', type=float, default=0.035,
                        help='Weight of probe and index-access cost in factorized utility.')
    parser.add_argument('--query_plan_ontology_factorized_enable_top2_admission', type=str, default='true',
                        help='Allow high-confidence simple or comparison actions to update top-2.')
    parser.add_argument('--query_plan_ontology_factorized_enable_top5_admission', type=str, default='true',
                        help='Allow validated factorized candidates and chains to update top-5.')
    parser.add_argument('--query_plan_ontology_factorized_shadow_only', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_SHADOW_ONLY', 'false'),
                        help='Compile and trace factorized candidates without mutating the incumbent ranking.')
    parser.add_argument('--query_plan_ontology_factorized_min_top2_utility', type=float, default=0.86,
                        help='Minimum action utility required for factorized top-2 admission.')
    parser.add_argument('--query_plan_ontology_factorized_min_top5_utility', type=float, default=0.72,
                        help='Minimum action utility required for factorized top-5 admission.')
    parser.add_argument('--query_plan_ontology_factorized_enable_typed_semantic_compiler', type=str, default='false',
                        help='Compile exact anchors, ordered query predicates, and typed join keys before physical execution.')
    parser.add_argument('--query_plan_ontology_factorized_enable_lossless_typed_plan_ir', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_ENABLE_LOSSLESS_TYPED_PLAN_IR', 'false'),
                        help='Preserve ontology variables, edge directions, and branch topology in a typed logical-plan IR.')
    parser.add_argument('--query_plan_ontology_factorized_require_typed_variable_unification', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_REQUIRE_TYPED_VARIABLE_UNIFICATION', 'false'),
                        help='Execute the typed IR with strict variable binding and emit a binding certificate.')
    parser.add_argument('--query_plan_ontology_factorized_enable_certificate_aware_plan_portfolio', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_ENABLE_CERTIFICATE_AWARE_PLAN_PORTFOLIO', 'false'),
                        help='Try compatible typed plans and a structural fallback only after primary certificate failure.')
    parser.add_argument('--query_plan_ontology_factorized_enable_certificate_aware_access_path_reoptimization', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_ENABLE_CERTIFICATE_AWARE_ACCESS_PATH_REOPTIMIZATION', 'false'),
                        help='Retry a failed typed plan through bound dense access and retain it only with a complete binding certificate.')
    parser.add_argument('--query_plan_ontology_factorized_enable_typed_certificate_top2', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_ENABLE_TYPED_CERTIFICATE_TOP2', 'false'),
                        help='Allow complete typed binding certificates to construct branch-balanced top-2 slates.')
    parser.add_argument('--query_plan_ontology_factorized_typed_certificate_top2_min_utility', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_TYPED_CERTIFICATE_TOP2_MIN_UTILITY', '0.90')),
                        help='Minimum complete typed-proof utility for top-2 admission.')
    parser.add_argument('--query_plan_ontology_factorized_typed_certificate_top2_min_semantic_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_TYPED_CERTIFICATE_TOP2_MIN_SEMANTIC_SCORE', '0.90')),
                        help='Minimum semantic score for typed-certificate top-2 admission.')
    parser.add_argument('--query_plan_ontology_factorized_require_exact_entity_keys', type=str, default='false',
                        help='Require exact normalized entity-key matches for factorized probes.')
    parser.add_argument('--query_plan_ontology_factorized_require_typed_join_keys', type=str, default='false',
                        help='Reject dates, numbers, and literal values as intermediate entity join keys.')
    parser.add_argument('--query_plan_ontology_factorized_min_query_predicate_coverage', type=float, default=0.75,
                        help='Minimum query-predicate coverage required by a compiled relation path.')
    parser.add_argument('--query_plan_ontology_factorized_enable_simple_top2_admission', type=str, default='true',
                        help='Allow a single entity-index candidate to modify top-2.')
    parser.add_argument('--query_plan_ontology_factorized_require_chain_source_top2', type=str, default='false',
                        help='Require a cascaded join to originate from a passage already protected in top-2.')
    parser.add_argument('--query_plan_ontology_factorized_min_counterfactual_net_benefit', type=float, default=0.0,
                        help='Minimum metric-aligned benefit minus displacement-risk proxy for commit.')
    parser.add_argument('--query_plan_ontology_factorized_enable_branch_complete_compiler', type=str, default='false',
                        help='Compile coordinated comparison anchors into two complete entity branches.')
    parser.add_argument('--query_plan_ontology_factorized_require_branch_qualifier_evidence', type=str, default='false',
                        help='Require parenthetical or comma qualifiers to occur in branch evidence.')
    parser.add_argument('--query_plan_ontology_factorized_enable_evidence_query_compiler', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_ENABLE_EVIDENCE_QUERY_COMPILER', 'false'),
                        help='Compile ontology matches into typed logical evidence plans.')
    parser.add_argument('--query_plan_ontology_factorized_enable_executable_ontology_compiler', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_ENABLE_EXECUTABLE_ONTOLOGY_COMPILER', 'false'),
                        help='Bind query-surface slots and compile branch-complete executable ontology plans.')
    parser.add_argument('--query_plan_ontology_factorized_enable_relation_realization', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_ENABLE_RELATION_REALIZATION', 'false'),
                        help='Resolve conceptual ontology relations to corpus-observed OpenIE predicates.')
    parser.add_argument('--query_plan_ontology_factorized_enable_relation_keyed_catalog', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_ENABLE_RELATION_KEYED_CATALOG', 'false'),
                        help='Resolve logical relations across all schema artifact steps by semantic relation key.')
    parser.add_argument('--query_plan_ontology_factorized_relation_keyed_allow_matched_schema_fallback', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_RELATION_KEYED_ALLOW_MATCHED_SCHEMA_FALLBACK', 'false'),
                        help='Fill unresolved selected-schema relations from other runtime-matched schemas.')
    parser.add_argument('--query_plan_ontology_factorized_enable_query_relation_alternative_recovery', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_ENABLE_QUERY_RELATION_ALTERNATIVE_RECOVERY', 'false'),
                        help='Execute a bounded portfolio of ontology relation paths when the primary relation contract cannot be materialized.')
    parser.add_argument('--query_plan_ontology_factorized_max_query_relation_alternatives', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_MAX_QUERY_RELATION_ALTERNATIVES', 3)),
                        help='Maximum non-primary ontology relation paths executed per query.')
    parser.add_argument('--query_plan_ontology_factorized_enable_execution_aware_contract', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_ENABLE_EXECUTION_AWARE_CONTRACT', 'false'),
                        help='Select compiler schemas using coverage, answer type, precision, and structure.')
    parser.add_argument('--query_plan_ontology_factorized_enable_provenance_plan_portfolio', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_ENABLE_PROVENANCE_PLAN_PORTFOLIO', 'false'),
                        help='Recover independently validated relation-slot evidence when a full physical join returns no candidates.')
    parser.add_argument('--query_plan_ontology_factorized_enable_candidate_window_path_compiler', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_ENABLE_CANDIDATE_WINDOW_PATH_COMPILER', 'false'),
                        help='Compile complete entity-bound ontology paths over the query-local candidate passage graph.')
    parser.add_argument('--query_plan_ontology_factorized_candidate_window_path_size', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_CANDIDATE_WINDOW_PATH_SIZE', 200)))
    parser.add_argument('--query_plan_ontology_factorized_candidate_window_path_beam_width', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_CANDIDATE_WINDOW_PATH_BEAM_WIDTH', 12)))
    parser.add_argument('--query_plan_ontology_factorized_candidate_window_path_max_paths', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_CANDIDATE_WINDOW_PATH_MAX_PATHS', 8)))
    parser.add_argument('--query_plan_ontology_factorized_candidate_window_path_min_anchor_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_CANDIDATE_WINDOW_PATH_MIN_ANCHOR_SCORE', 0.86)))
    parser.add_argument('--query_plan_ontology_factorized_candidate_window_path_min_relation_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_CANDIDATE_WINDOW_PATH_MIN_RELATION_SCORE', 0.50)))
    parser.add_argument('--query_plan_ontology_factorized_candidate_window_path_min_path_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_CANDIDATE_WINDOW_PATH_MIN_PATH_SCORE', 0.58)))
    parser.add_argument('--query_plan_ontology_factorized_candidate_window_path_allow_contextual_terminal_edges', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_CANDIDATE_WINDOW_PATH_ALLOW_CONTEXTUAL_TERMINAL_EDGES', 'true'),
                        help='Allow typed terminal context to validate generic OpenIE predicates inside a complete bound path.')
    parser.add_argument('--query_plan_ontology_factorized_candidate_window_path_enable_entity_index_expansion', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_CANDIDATE_WINDOW_PATH_ENABLE_ENTITY_INDEX_EXPANSION', 'false'),
                        help='Ground bound logical variables through the corpus entity-relation index.')
    parser.add_argument('--query_plan_ontology_factorized_candidate_window_path_enable_bound_dense_expansion', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_CANDIDATE_WINDOW_PATH_ENABLE_BOUND_DENSE_EXPANSION', 'false'),
                        help='Issue relation-conditioned dense probes from intermediate bindings.')
    parser.add_argument('--query_plan_ontology_factorized_candidate_window_path_entity_index_top_k', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_CANDIDATE_WINDOW_PATH_ENTITY_INDEX_TOP_K', 64)))
    parser.add_argument('--query_plan_ontology_factorized_candidate_window_path_bound_dense_top_k', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_CANDIDATE_WINDOW_PATH_BOUND_DENSE_TOP_K', 32)))
    parser.add_argument('--query_plan_ontology_factorized_candidate_window_path_max_access_probes', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_CANDIDATE_WINDOW_PATH_MAX_ACCESS_PROBES', 8)))
    parser.add_argument('--query_plan_ontology_factorized_candidate_window_path_max_external_passages', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_CANDIDATE_WINDOW_PATH_MAX_EXTERNAL_PASSAGES', 96)))
    parser.add_argument('--query_plan_ontology_factorized_candidate_window_path_enable_relation_agnostic_bridge', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_CANDIDATE_WINDOW_PATH_ENABLE_RELATION_AGNOSTIC_BRIDGE', 'false'),
                        help='Compile bounded observed entity bridges when ontology relation labels cannot be realized.')
    parser.add_argument('--query_plan_ontology_factorized_candidate_window_path_bridge_max_paths', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_CANDIDATE_WINDOW_PATH_BRIDGE_MAX_PATHS', 8)))
    parser.add_argument('--query_plan_ontology_factorized_candidate_window_path_bridge_max_edges_per_state', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_CANDIDATE_WINDOW_PATH_BRIDGE_MAX_EDGES_PER_STATE', 48)))
    parser.add_argument('--query_plan_ontology_factorized_candidate_window_path_bridge_min_path_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_CANDIDATE_WINDOW_PATH_BRIDGE_MIN_PATH_SCORE', 0.55)))
    parser.add_argument('--query_plan_ontology_factorized_enable_inverse_direction_fallback', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_ENABLE_INVERSE_DIRECTION_FALLBACK', 'false'),
                        help='Try the opposite exact-key endpoint only when the declared relation direction yields no indexed evidence.')
    parser.add_argument('--query_plan_ontology_factorized_inverse_direction_penalty', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_INVERSE_DIRECTION_PENALTY', 0.12)),
                        help='Score penalty applied to inverse-direction physical access candidates.')
    parser.add_argument('--query_plan_ontology_factorized_enable_whole_warehouse_probe', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_ENABLE_WHOLE_WAREHOUSE_PROBE', 'false'),
                        help='Probe approximate entity keys over the full materialized evidence warehouse.')
    parser.add_argument('--query_plan_ontology_factorized_warehouse_max_entity_keys', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_WAREHOUSE_MAX_ENTITY_KEYS', 12)),
                        help='Maximum approximate warehouse entity keys inspected per physical probe.')
    parser.add_argument('--query_plan_ontology_factorized_enable_cost_based_physical_planner', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_ENABLE_COST_BASED_PHYSICAL_PLANNER', 'false'),
                        help='Choose physical evidence access plans using a train-split catalog.')
    parser.add_argument('--query_plan_ontology_factorized_evidence_catalog_path', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_EVIDENCE_CATALOG_PATH', ''),
                        help='Path to the train-split ontology evidence catalog.')
    parser.add_argument('--query_plan_ontology_factorized_physical_planner_quality_weight', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_PHYSICAL_PLANNER_QUALITY_WEIGHT', 0.45)))
    parser.add_argument('--query_plan_ontology_factorized_physical_planner_recall_weight', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_PHYSICAL_PLANNER_RECALL_WEIGHT', 0.25)))
    parser.add_argument('--query_plan_ontology_factorized_physical_planner_proof_weight', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_PHYSICAL_PLANNER_PROOF_WEIGHT', 0.30)))
    parser.add_argument('--query_plan_ontology_factorized_physical_planner_cost_weight', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_PHYSICAL_PLANNER_COST_WEIGHT', 0.15)))
    parser.add_argument('--query_plan_ontology_factorized_physical_planner_risk_weight', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_PHYSICAL_PLANNER_RISK_WEIGHT', 0.35)))
    parser.add_argument('--query_plan_ontology_factorized_physical_planner_min_proof_coverage', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_PHYSICAL_PLANNER_MIN_PROOF_COVERAGE', 0.20)))
    parser.add_argument('--query_plan_ontology_factorized_physical_planner_max_estimated_cost', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_PHYSICAL_PLANNER_MAX_ESTIMATED_COST', 8.0)))
    parser.add_argument('--query_plan_ontology_factorized_physical_planner_no_op_utility', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_PHYSICAL_PLANNER_NO_OP_UTILITY', 0.0)))
    parser.add_argument('--query_plan_ontology_factorized_physical_planner_selection_mode', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_PHYSICAL_PLANNER_SELECTION_MODE', 'weighted_utility'),
                        choices=['weighted_utility', 'constrained_lexicographic'])
    parser.add_argument('--query_plan_ontology_factorized_physical_planner_max_loss_probability', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_PHYSICAL_PLANNER_MAX_LOSS_PROBABILITY', 0.25)))
    parser.add_argument('--query_plan_ontology_factorized_physical_planner_confidence_delta', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_PHYSICAL_PLANNER_CONFIDENCE_DELTA', 0.05)))
    parser.add_argument('--query_plan_ontology_factorized_physical_planner_min_catalog_samples', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_PHYSICAL_PLANNER_MIN_CATALOG_SAMPLES', 5)))
    parser.add_argument('--query_plan_ontology_factorized_physical_access_strategy', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_PHYSICAL_ACCESS_STRATEGY', 'adaptive'),
                        choices=['adaptive', 'index_only', 'dense_first'],
                        help='Physical access strategy used when cost-based planning is disabled.')
    parser.add_argument('--query_plan_ontology_factorized_enable_physical_shadow_collection', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_ENABLE_PHYSICAL_SHADOW_COLLECTION', 'false'),
                        help='Collect counterfactual physical access paths on the same cached query execution.')
    parser.add_argument('--query_plan_ontology_factorized_physical_shadow_strategies', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_FACTORIZED_PHYSICAL_SHADOW_STRATEGIES', 'adaptive,index_only,dense_first'),
                        help='Comma-separated physical access paths recorded by shadow collection.')
    parser.add_argument('--query_plan_ontology_factorized_enable_missing_slot_executor', type=str, default='false',
                        help='Recover failed typed plans with ontology-guided subqueries and dependent joins.')
    parser.add_argument('--query_plan_ontology_factorized_missing_slot_trigger_reasons', type=str,
                        default='no_validated_candidates,insufficient_query_predicate_coverage,no_exact_query_anchor,no_relational_access_path',
                        help='Comma-separated factorized failure reasons eligible for missing-slot recovery.')
    parser.add_argument('--query_plan_ontology_factorized_missing_slot_dense_top_k', type=int, default=64,
                        help='Dense retrieval window for each missing-slot subquery.')
    parser.add_argument('--query_plan_ontology_factorized_missing_slot_max_probes', type=int, default=12,
                        help='Maximum missing-slot index and dense probe stages per query.')
    parser.add_argument('--query_plan_ontology_factorized_missing_slot_max_bridge_entities', type=int, default=6,
                        help='Maximum bridge bindings retained after each dependent join stage.')
    parser.add_argument('--query_plan_ontology_factorized_missing_slot_max_candidates', type=int, default=12,
                        help='Maximum terminal-only recovery candidates exposed to action replay.')
    parser.add_argument('--query_plan_ontology_factorized_missing_slot_max_hops', type=int, default=3,
                        help='Maximum relation hops compiled by missing-slot recovery.')
    parser.add_argument('--query_plan_ontology_factorized_missing_slot_min_relation_score', type=float, default=0.20,
                        help='Minimum relation compatibility for a recovery join edge.')
    parser.add_argument('--query_plan_ontology_factorized_missing_slot_min_terminal_type_score', type=float, default=0.30,
                        help='Minimum answer-type compatibility for a recovered terminal edge.')
    parser.add_argument('--query_plan_ontology_factorized_missing_slot_min_candidate_utility', type=float, default=0.38,
                        help='Minimum cost-adjusted utility for a recovered terminal passage.')
    parser.add_argument('--query_plan_ontology_factorized_missing_slot_enable_top2_admission', type=str, default='true',
                        help='Expose recovered terminal candidates as counterfactual top-2 actions.')
    parser.add_argument('--query_plan_ontology_factorized_enable_typed_proof_join', type=str, default='false',
                        help='Use a typed compiler and proof-carrying dependent joins for failed plans.')
    parser.add_argument('--query_plan_ontology_factorized_require_branch_complete_proof', type=str, default='true',
                        help='Require every logical branch before admitting a proof set.')
    parser.add_argument('--query_plan_ontology_factorized_require_provenance_validation', type=str, default='true',
                        help='Require exact bindings and source triples for every proof edge.')
    parser.add_argument('--query_plan_ontology_factorized_require_strict_branch_identity', type=str, default='false',
                        help='Require exact branch endpoints and query qualifiers in first-hop proof provenance.')
    parser.add_argument('--query_plan_ontology_factorized_project_full_proof_set', type=str, default='true',
                        help='Project all proof passages as one atomic counterfactual action.')
    parser.add_argument('--query_plan_ontology_factorized_min_proof_confidence', type=float, default=0.72,
                        help='Minimum confidence for a proof-carrying dependent join.')
    parser.add_argument('--query_plan_ontology_factorized_enable_proof_gap_access_path', type=str, default='false',
                        help='Execute bound ontology paths for relation slots missing from the current top-k.')
    parser.add_argument('--query_plan_ontology_factorized_proof_gap_top_window_k', type=int, default=5,
                        help='Top-k evidence window audited for proof gaps.')
    parser.add_argument('--query_plan_ontology_factorized_proof_gap_max_paths', type=int, default=4,
                        help='Maximum ontology paths executed for a proof-gap query.')
    parser.add_argument('--query_plan_ontology_factorized_proof_gap_min_missing_slot_fraction', type=float, default=0.25,
                        help='Minimum uncovered relation-slot fraction required to trigger bound probes.')
    parser.add_argument('--query_plan_ontology_factorized_proof_gap_min_binding_score', type=float, default=0.80,
                        help='Minimum dependent binding score for a proof-gap join edge.')
    parser.add_argument('--query_plan_ontology_factorized_proof_gap_min_proof_confidence', type=float, default=0.20,
                        help='Minimum conjunctive proof certificate exposed to learned admission.')
    parser.add_argument('--query_plan_ontology_factorized_proof_gap_require_external_member', type=str, default='true',
                        help='Require a proposed proof set to add evidence outside the audited top-k.')
    parser.add_argument('--query_plan_ontology_factorized_proof_gap_merge_candidate_pool', type=str, default='false',
                        help='Union proof-gap candidates and groups with the incumbent physical candidate pool.')
    parser.add_argument('--query_plan_ontology_factorized_enable_proof_residual_commit', type=str, default='false',
                        help='Compile only proof members missing from the current top-k into an action.')
    parser.add_argument('--query_plan_ontology_factorized_proof_residual_direct_commit', type=str, default='false',
                        help='Allow strict proof certificates to bypass the learned action gate.')
    parser.add_argument('--query_plan_ontology_factorized_proof_residual_post_optimizer_commit',
                        type=str, default='false',
                        help='Recompile a proof residual after the learned primary action commits.')
    parser.add_argument('--query_plan_ontology_factorized_proof_residual_target_datasets', type=str,
                        default='2wikimultihopqa',
                        help='Comma-separated datasets where proof-certified residuals may mutate top-k.')
    parser.add_argument('--query_plan_ontology_factorized_proof_residual_min_certificate_confidence',
                        type=float, default=0.95,
                        help='Minimum proof confidence for certificate-based direct commit.')
    parser.add_argument('--query_plan_ontology_factorized_proof_residual_max_top5_insertions', type=int,
                        default=1,
                        help='Maximum missing proof members inserted into top-5 by one action.')
    parser.add_argument('--query_plan_ontology_factorized_proof_residual_enable_top2', type=str,
                        default='true',
                        help='Enable complete comparison-terminal residual actions for top-2.')
    parser.add_argument('--query_plan_ontology_factorized_enable_proof_materialized_view', type=str,
                        default='false',
                        help='Place a compact provenance-preserving proof view before verbose QA passages.')
    parser.add_argument('--query_plan_ontology_factorized_proof_materialized_view_target_datasets', type=str,
                        default='2wikimultihopqa',
                        help='Comma-separated datasets eligible for proof-view QA materialization.')
    parser.add_argument('--query_plan_ontology_factorized_proof_materialized_view_max_edges', type=int,
                        default=8,
                        help='Maximum certified proof edges projected into one compact reader view.')
    parser.add_argument('--query_plan_ontology_factorized_enable_learned_delta_admission', type=str, default='false',
                        help='Use a train-split model to predict top-k gain and displacement loss.')
    parser.add_argument('--query_plan_ontology_factorized_delta_admission_model_path', type=str, default='',
                        help='JSON model produced by train_factorized_delta_admission.py.')
    parser.add_argument('--query_plan_ontology_factorized_delta_admission_min_expected_delta', type=float, default=0.0,
                        help='Minimum predicted metric delta required for an action.')
    parser.add_argument('--query_plan_ontology_factorized_delta_admission_max_loss_probability', type=float, default=0.20,
                        help='Maximum predicted displacement-loss probability.')
    parser.add_argument('--query_plan_ontology_factorized_delta_admission_fail_open', type=str, default='false',
                        help='If true, admit actions when no compatible learned model section exists.')
    parser.add_argument('--query_plan_ontology_factorized_delta_admission_collect_only', type=str, default='false',
                        help='Collect train-split actions without applying the learned gate.')
    parser.add_argument('--query_plan_ontology_factorized_enable_set_level_optimizer', type=str, default='false',
                        help='Jointly select one Top-2/Top-5 evidence set against an explicit no-op.')
    parser.add_argument('--query_plan_ontology_factorized_set_level_optimizer_model_path', type=str, default='',
                        help='Manifest or joblib artifact produced by train_query_plan_set_level_optimizer.py.')
    parser.add_argument('--query_plan_ontology_factorized_set_level_optimizer_min_expected_utility', type=float, default=-1.0,
                        help='Optional set-utility threshold override; negative uses model calibration.')
    parser.add_argument('--query_plan_ontology_factorized_set_level_optimizer_max_loss_probability', type=float, default=-1.0,
                        help='Optional set-loss threshold override; negative uses model calibration.')
    parser.add_argument('--query_plan_ontology_factorized_set_level_optimizer_fail_open', type=str, default='false',
                        help='If true, allow the best set when the configured model cannot be loaded.')
    parser.add_argument('--query_plan_ontology_factorized_enable_incumbent_reoptimizer', type=str, default='false',
                        help='Only admit actions predicted to improve on the incumbent CBO plan.')
    parser.add_argument('--query_plan_ontology_factorized_incumbent_reoptimizer_model_path', type=str, default='',
                        help='Manifest or joblib artifact produced by train_incumbent_residual_reoptimizer.py.')
    parser.add_argument('--query_plan_ontology_factorized_incumbent_reoptimizer_min_expected_delta', type=float, default=-1.0,
                        help='Optional relative-delta threshold override; negative uses model calibration.')
    parser.add_argument('--query_plan_ontology_factorized_incumbent_reoptimizer_max_loss_probability', type=float, default=-1.0,
                        help='Optional incumbent-relative loss threshold; negative uses model calibration.')
    parser.add_argument('--query_plan_ontology_factorized_incumbent_reoptimizer_set_trigger_mode', type=str,
                        default='incumbent_noop_or_proof_deficit',
                        help='When the residual optimizer may compile ontology evidence-set alternatives.')
    parser.add_argument('--query_plan_ontology_factorized_entity_scan_mutation_target_datasets', type=str,
                        default='2wikimultihopqa,musique,nq_rear,popqa',
                        help='Datasets where entity index scans may mutate top-k; empty disables scan mutation.')
    parser.add_argument('--query_plan_ontology_factorized_enable_counterfactual_action_replay', type=str, default='false',
                        help='Enumerate independent top-k actions and select against an explicit no-op.')
    parser.add_argument('--query_plan_ontology_factorized_counterfactual_action_replay_max_actions', type=int, default=24,
                        help='Maximum counterfactual factorized and warehouse actions per query.')
    parser.add_argument('--query_plan_ontology_factorized_action_replay_include_evidence_warehouse', type=str, default='true',
                        help='Include Evidence Warehouse candidates in action replay.')
    parser.add_argument('--query_plan_ontology_factorized_action_replay_enable_atomic_chain_top2', type=str, default='true',
                        help='Include complete two-document chain groups as atomic top-2 actions.')
    parser.add_argument('--query_plan_ontology_factorized_enable_evidence_set_compiler', type=str, default='false',
                        help='Compile ontology-bearing warehouse candidates into bounded top-k set actions.')
    parser.add_argument('--query_plan_ontology_factorized_evidence_set_max_candidates', type=int, default=10,
                        help='Maximum warehouse candidates considered by the evidence-set compiler.')
    parser.add_argument('--query_plan_ontology_factorized_evidence_set_beam_width', type=int, default=16,
                        help='Beam width for ontology-guided evidence-set enumeration.')
    parser.add_argument('--query_plan_ontology_factorized_evidence_set_max_top2_actions', type=int, default=6,
                        help='Maximum compiled two-document top-2 actions per query.')
    parser.add_argument('--query_plan_ontology_factorized_evidence_set_max_top5_actions', type=int, default=10,
                        help='Maximum compiled two/three-document top-5 actions per query.')
    parser.add_argument('--query_plan_ontology_enable_top5_listwise_optimizer', type=str, default='false',
                        help='If True, use a grounded listwise judge over the top-5 and bounded external ontology candidates.')
    parser.add_argument('--query_plan_ontology_top5_listwise_target_datasets', type=str,
                        default='2wikimultihopqa,hotpotqa,musique,nq_rear,popqa',
                        help='Comma-separated datasets eligible for adaptive top-5 listwise re-optimization.')
    parser.add_argument('--query_plan_ontology_top5_listwise_require_ontology_candidate', type=str, default='true',
                        help='If True, listwise calls and commits require a provenance-bearing ontology candidate.')
    parser.add_argument('--query_plan_ontology_top5_listwise_max_passage_chars', type=int, default=1200,
                        help='Maximum passage characters supplied to the listwise judge per top-5 item.')
    parser.add_argument('--query_plan_ontology_top5_listwise_max_completion_tokens', type=int, default=384,
                        help='Maximum completion tokens for one top-5 listwise judge call.')
    parser.add_argument('--query_plan_ontology_top5_listwise_max_external_candidates', type=int, default=3,
                        help='Maximum ontology candidates outside the current top-5 exposed to one listwise call.')
    parser.add_argument('--query_plan_ontology_top5_listwise_min_external_access_score', type=float, default=0.58,
                        help='Minimum access-path utility for an ontology candidate outside the current top-5.')
    parser.add_argument('--query_plan_ontology_top5_listwise_min_candidate_score', type=float, default=0.72,
                        help='Minimum judge score required for rank-3-to-5 candidate promotion.')
    parser.add_argument('--query_plan_ontology_top5_listwise_min_score_margin', type=float, default=0.16,
                        help='Minimum judge-score advantage over the current rank-2 passage required for commit.')
    parser.add_argument('--query_plan_ontology_top5_listwise_min_judge_confidence', type=float, default=0.78,
                        help='Minimum judge confidence required for listwise top-2 commit.')
    parser.add_argument('--query_plan_ontology_top5_listwise_min_evidence_span_chars', type=int, default=3,
                        help='Minimum exact evidence-span length required to ground a listwise promotion.')
    parser.add_argument('--query_plan_ontology_enable_projected_graph_retriever', type=str, default='false',
                        help='If True, project OpenIE facts onto query-plan ontology schema steps and retrieve source passages from the projected graph.')
    parser.add_argument('--query_plan_ontology_projected_graph_max_schema_relations', type=int, default=512,
                        help='Maximum ontology schema steps materialized by the projected graph retriever.')
    parser.add_argument('--query_plan_ontology_projected_graph_max_facts_per_schema', type=int, default=5000,
                        help='Maximum projected facts retained per ontology schema step.')
    parser.add_argument('--query_plan_ontology_projected_graph_max_facts_per_step_query', type=int, default=600,
                        help='Maximum projected facts scanned per schema step for one query.')
    parser.add_argument('--query_plan_ontology_projected_graph_max_candidates', type=int, default=4,
                        help='Maximum projected-graph passage candidates exposed downstream.')
    parser.add_argument('--query_plan_ontology_projected_graph_min_score', type=float, default=0.62,
                        help='Minimum projected-graph candidate score required for insertion.')
    parser.add_argument('--query_plan_ontology_projected_graph_preserve_top_k', type=int, default=2,
                        help='Number of top baseline passages protected from projected-graph insertion.')
    parser.add_argument('--query_plan_ontology_projected_graph_candidate_only', type=str, default='true',
                        help='If True, projected-graph retrieval appends candidates for evidence-set selection.')
    parser.add_argument('--query_plan_ontology_projected_graph_score_margin', type=float, default=1e-7,
                        help='Small score margin used when appending projected-graph candidates.')
    parser.add_argument('--query_plan_ontology_projected_graph_require_anchor_overlap', type=str, default='true',
                        help='If True, projected facts must overlap query/anchor tokens.')
    parser.add_argument('--query_plan_ontology_projected_graph_min_relation_token_overlap', type=float, default=0.50,
                        help='Minimum relation-token overlap for mapping graph relations to ontology schema steps.')
    parser.add_argument('--query_plan_ontology_projected_graph_enable_bridge_paths', type=str, default='true',
                        help='If True, prefer two-hop projected graph evidence for multi-step ontology paths.')
    parser.add_argument('--query_plan_ontology_enable_evidence_index_retriever', type=str, default='false',
                        help='If True, retrieve high-precision passage candidates from an ontology-slot evidence index.')
    parser.add_argument('--query_plan_ontology_evidence_index_max_schema_slots', type=int, default=1024,
                        help='Maximum ontology slots materialized by the evidence-index retriever.')
    parser.add_argument('--query_plan_ontology_evidence_index_max_facts_per_slot', type=int, default=5000,
                        help='Maximum facts retained per ontology evidence-index slot.')
    parser.add_argument('--query_plan_ontology_evidence_index_max_facts_per_query_slot', type=int, default=400,
                        help='Maximum indexed facts scanned for one query-slot match.')
    parser.add_argument('--query_plan_ontology_evidence_index_max_candidates', type=int, default=4,
                        help='Maximum evidence-index candidates exposed downstream.')
    parser.add_argument('--query_plan_ontology_evidence_index_min_score', type=float, default=0.84,
                        help='Minimum evidence-index candidate score required for insertion.')
    parser.add_argument('--query_plan_ontology_evidence_index_preserve_top_k', type=int, default=2,
                        help='Number of top baseline passages protected from evidence-index insertion.')
    parser.add_argument('--query_plan_ontology_evidence_index_candidate_only', type=str, default='true',
                        help='If True, evidence-index retrieval appends candidates for evidence-set selection.')
    parser.add_argument('--query_plan_ontology_evidence_index_score_margin', type=float, default=1e-7,
                        help='Small score margin used when appending evidence-index candidates.')
    parser.add_argument('--query_plan_ontology_evidence_index_require_anchor_overlap', type=str, default='true',
                        help='If True, evidence-index facts must overlap query/anchor tokens.')
    parser.add_argument('--query_plan_ontology_evidence_index_min_relation_token_overlap', type=float, default=0.35,
                        help='Minimum relation-token overlap for mapping facts to ontology evidence slots.')
    parser.add_argument('--query_plan_ontology_evidence_index_synthetic_original_rank', type=int, default=45,
                        help='Synthetic original rank for high-precision evidence-index candidates.')
    parser.add_argument('--query_plan_ontology_evidence_index_top2_min_score', type=float, default=0.92,
                        help='Minimum evidence-index score required before a candidate can replace rank 2.')
    parser.add_argument('--query_plan_ontology_evidence_index_top2_max_original_rank', type=int, default=30,
                        help='Maximum original rank eligible for evidence-index rank-2 intervention; <=0 disables this bound.')
    parser.add_argument('--query_plan_ontology_evidence_index_top2_require_terminal', type=str, default='true',
                        help='If True, evidence-index rank-2 intervention requires both anchor and terminal endpoint evidence.')
    parser.add_argument('--query_plan_ontology_evidence_index_recall_utility_model_path', type=str, default='',
                        help='Optional supervised model for evidence-index recall-utility candidate selection.')
    parser.add_argument('--query_plan_ontology_evidence_index_recall_utility_mode', type=str, default='off',
                        choices=['off', 'score', 'selector'],
                        help='Evidence-index recall-utility mode: off, score, or selector.')
    parser.add_argument('--query_plan_ontology_evidence_index_recall_utility_min_score', type=float, default=0.70,
                        help='Minimum recall-utility score that marks an evidence-index candidate as promoted.')
    parser.add_argument('--query_plan_ontology_evidence_index_recall_utility_score_weight', type=float, default=0.35,
                        help='Score weight added to evidence-index candidates from the recall-utility model.')
    parser.add_argument('--query_plan_ontology_evidence_index_recall_utility_promote_confidence', type=float, default=0.98,
                        help='Confidence floor for evidence-index candidates promoted by recall utility.')
    parser.add_argument('--query_plan_ontology_evidence_index_recall_utility_top2_min_score', type=float, default=0.80,
                        help='Minimum recall-utility score required for evidence-index top-2 intervention.')
    parser.add_argument('--query_plan_ontology_enable_path_executor_retriever', type=str, default='false',
                        help='If True, execute matched ontology slot paths over OpenIE facts and retrieve source passages.')
    parser.add_argument('--query_plan_ontology_path_executor_max_schema_slots', type=int, default=1024,
                        help='Maximum ontology slots materialized by the ontology path executor.')
    parser.add_argument('--query_plan_ontology_path_executor_max_facts_per_slot', type=int, default=5000,
                        help='Maximum facts retained per ontology path-executor slot.')
    parser.add_argument('--query_plan_ontology_path_executor_max_facts_per_step', type=int, default=160,
                        help='Maximum indexed facts scanned for one path step.')
    parser.add_argument('--query_plan_ontology_path_executor_max_runtime_paths', type=int, default=8,
                        help='Maximum runtime ontology paths executed per query.')
    parser.add_argument('--query_plan_ontology_path_executor_max_candidates', type=int, default=4,
                        help='Maximum ontology path-executor candidates exposed downstream.')
    parser.add_argument('--query_plan_ontology_path_executor_min_score', type=float, default=0.78,
                        help='Minimum ontology path-executor candidate score required for insertion.')
    parser.add_argument('--query_plan_ontology_path_executor_preserve_top_k', type=int, default=2,
                        help='Number of top baseline passages protected from path-executor insertion.')
    parser.add_argument('--query_plan_ontology_path_executor_candidate_only', type=str, default='true',
                        help='If True, ontology path execution appends candidates for evidence-set selection.')
    parser.add_argument('--query_plan_ontology_path_executor_score_margin', type=float, default=1e-7,
                        help='Small score margin used when appending ontology path-executor candidates.')
    parser.add_argument('--query_plan_ontology_path_executor_synthetic_original_rank', type=int, default=35,
                        help='Synthetic original rank for ontology path-executor candidates.')
    parser.add_argument('--query_plan_ontology_path_executor_beam_size', type=int, default=8,
                        help='Beam size for ontology path execution over OpenIE facts.')
    parser.add_argument('--query_plan_ontology_path_executor_max_steps', type=int, default=3,
                        help='Maximum ontology slots followed by the path executor.')
    parser.add_argument('--query_plan_ontology_path_executor_min_relation_token_overlap', type=float, default=0.35,
                        help='Minimum relation-token overlap for mapping facts to ontology path-executor slots.')
    parser.add_argument('--query_plan_ontology_path_executor_enable_answer_type_completion', type=str, default='false',
                        help='If True, complete bridge/query-condition slots with answer-type-compatible terminal ontology slots.')
    parser.add_argument('--query_plan_ontology_path_executor_enable_terminal_type_filter', type=str, default='false',
                        help='If True, filter paths whose terminal entity does not match the query answer type.')
    parser.add_argument('--query_plan_ontology_path_executor_allow_query_condition_single', type=str, default='true',
                        help='If True, allow single-slot query-condition paths when completion cannot extend them.')
    parser.add_argument('--query_plan_ontology_path_executor_require_terminal_type_for_single_step', type=str, default='false',
                        help='If True, single-step path candidates must pass terminal answer-type verification.')
    parser.add_argument('--query_plan_ontology_path_executor_min_terminal_type_score', type=float, default=0.45,
                        help='Minimum terminal answer-type score for verified ontology path candidates.')
    parser.add_argument('--query_plan_ontology_path_executor_direction_penalty', type=float, default=1.0,
                        help='Soft penalty when traversing an OpenIE fact opposite the ontology step direction.')
    parser.add_argument('--query_plan_ontology_path_executor_max_relation_repeats', type=int, default=0,
                        help='Maximum times the same relation can repeat in one executed path; <=0 disables.')
    parser.add_argument('--query_plan_ontology_path_executor_strict_query_relation_binding', type=str, default='false',
                        help='If True, executed ontology paths must cover strong relation tokens explicitly present in the query.')
    parser.add_argument('--query_plan_ontology_path_executor_min_query_relation_coverage', type=float, default=1.0,
                        help='Minimum strong query-relation coverage for strict ontology path execution.')
    parser.add_argument('--query_plan_ontology_enable_failure_evidence_retriever', type=str, default='false',
                        help='If True, use retrieval-failure evidence rules as an extra dense support candidate source.')
    parser.add_argument('--query_plan_ontology_failure_evidence_path', type=str, default='',
                        help='Optional explicit failure_evidence_ontology.json path.')
    parser.add_argument('--query_plan_ontology_failure_evidence_root', type=str, default='',
                        help='Root containing <dataset>/failure_evidence_ontology.json files.')
    parser.add_argument('--query_plan_ontology_failure_evidence_dense_top_k', type=int, default=120,
                        help='Dense top-k passages considered for each retrieval-failure evidence probe.')
    parser.add_argument('--query_plan_ontology_failure_evidence_max_rules', type=int, default=8,
                        help='Maximum retrieval-failure rules matched per query.')
    parser.add_argument('--query_plan_ontology_failure_evidence_max_probes', type=int, default=16,
                        help='Maximum retrieval-failure evidence probes issued per query.')
    parser.add_argument('--query_plan_ontology_failure_evidence_max_candidates', type=int, default=8,
                        help='Maximum retrieval-failure evidence candidates exposed downstream.')
    parser.add_argument('--query_plan_ontology_failure_evidence_min_confidence', type=float, default=0.56,
                        help='Minimum confidence required before a retrieval-failure evidence candidate is accepted.')
    parser.add_argument('--query_plan_ontology_failure_evidence_preserve_top_k', type=int, default=2,
                        help='Number of top baseline passages protected from retrieval-failure evidence insertion.')
    parser.add_argument('--query_plan_ontology_failure_evidence_synthetic_original_rank', type=int, default=60,
                        help='Synthetic original rank for retrieval-failure evidence passages not present in baseline retrieval.')
    parser.add_argument('--query_plan_ontology_failure_evidence_require_anchor_hit', type=str, default='false',
                        help='If True, retrieval-failure evidence candidates with concrete anchors must mention the anchor.')
    parser.add_argument('--query_plan_ontology_failure_evidence_require_rule_term_hit', type=str, default='true',
                        help='If True, retrieval-failure evidence candidates must mention rule relation/query/answer hints.')
    parser.add_argument('--query_plan_ontology_failure_evidence_candidate_only', type=str, default='true',
                        help='If True, retrieval-failure evidence retrieval appends candidates and lets evidence-set selection decide.')
    parser.add_argument('--query_plan_ontology_failure_evidence_score_margin', type=float, default=1e-7,
                        help='Small score margin used when appending retrieval-failure evidence candidates.')
    parser.add_argument('--query_plan_ontology_failure_evidence_min_rule_gold_precision', type=float, default=0.0,
                        help='Minimum validated gold precision required for a retrieval-failure evidence rule.')
    parser.add_argument('--query_plan_ontology_failure_evidence_min_rule_gt5_precision', type=float, default=0.0,
                        help='Minimum validated new-gold-from-gt5 precision required for a retrieval-failure evidence rule.')
    parser.add_argument('--query_plan_ontology_failure_evidence_min_rule_gold_hits', type=int, default=0,
                        help='Minimum validated gold hits required for a retrieval-failure evidence rule.')
    parser.add_argument('--query_plan_ontology_failure_evidence_min_rule_gt5_hits', type=int, default=0,
                        help='Minimum validated new-gold-from-gt5 hits required for a retrieval-failure evidence rule.')
    parser.add_argument('--enable_evidence_grounded_ontology', type=str, default='false',
                        help='If True, use retrieval-aware evidence-grounded ontology plans as an additional support evidence source.')
    parser.add_argument('--evidence_grounded_ontology_path', type=str, default='',
                        help='Optional explicit evidence_grounded_ontology.json path.')
    parser.add_argument('--evidence_grounded_ontology_root', type=str, default='outputs_evidence_grounded_ontology_v51',
                        help='Root containing <dataset>/evidence_grounded_ontology.json files.')
    parser.add_argument('--evidence_grounded_ontology_dense_top_k', type=int, default=80,
                        help='Dense top-k passages considered by each evidence-grounded ontology probe.')
    parser.add_argument('--evidence_grounded_ontology_top_k_plans', type=int, default=6,
                        help='Maximum evidence-grounded ontology plans matched per query.')
    parser.add_argument('--evidence_grounded_ontology_max_probes', type=int, default=18,
                        help='Maximum evidence-grounded dense probes issued per query.')
    parser.add_argument('--evidence_grounded_ontology_max_candidates', type=int, default=10,
                        help='Maximum evidence-grounded support candidates exposed downstream.')
    parser.add_argument('--evidence_grounded_ontology_min_confidence', type=float, default=0.58,
                        help='Minimum confidence required for evidence-grounded support candidates.')
    parser.add_argument('--evidence_grounded_ontology_preserve_top_k', type=int, default=2,
                        help='Number of top baseline passages protected from evidence-grounded insertion.')
    parser.add_argument('--evidence_grounded_ontology_candidate_only', type=str, default='true',
                        help='If True, append candidates and let evidence-set selection decide final ranking.')
    parser.add_argument('--evidence_grounded_ontology_min_match_score', type=float, default=0.05,
                        help='Minimum query-to-plan match score for evidence-grounded ontology plans.')
    parser.add_argument('--evidence_grounded_leaf_policy_path', type=str, default='',
                        help='Optional learned utility policy JSON for evidence-grounded ontology tree leaves.')
    parser.add_argument('--evidence_grounded_leaf_policy_reject_below', type=float, default=0.0,
                        help='Reject matched evidence-grounded ontology leaves whose utility score is below this value.')
    parser.add_argument('--evidence_grounded_leaf_policy_boost', type=float, default=0.0,
                        help='Boost matched evidence-grounded ontology plan score by this multiplier times leaf utility.')
    parser.add_argument('--query_plan_ontology_enable_llm_evidence_verifier', type=str, default='false',
                        help='If True, use the LLM to verify ontology support evidence candidates before evidence-set selection.')
    parser.add_argument('--query_plan_ontology_llm_evidence_verifier_max_candidates', type=int, default=8,
                        help='Maximum ontology support candidates judged by the LLM evidence verifier per query.')
    parser.add_argument('--query_plan_ontology_llm_evidence_verifier_min_original_rank', type=int, default=3,
                        help='Minimum original retrieval rank for candidates judged by the LLM evidence verifier.')
    parser.add_argument('--query_plan_ontology_llm_evidence_verifier_max_original_rank', type=int, default=30,
                        help='Maximum original retrieval rank for candidates judged by the LLM evidence verifier; <=0 disables the bound.')
    parser.add_argument('--query_plan_ontology_llm_evidence_verifier_min_approval_score', type=float, default=0.70,
                        help='Minimum LLM approval score required to promote an ontology evidence candidate.')
    parser.add_argument('--query_plan_ontology_llm_evidence_verifier_approved_confidence', type=float, default=0.98,
                        help='Confidence assigned to LLM-approved ontology evidence candidates before evidence-set selection.')
    parser.add_argument('--query_plan_ontology_llm_evidence_verifier_rejected_confidence', type=float, default=0.10,
                        help='Confidence cap assigned to LLM-rejected ontology evidence candidates.')
    parser.add_argument('--query_plan_ontology_llm_evidence_verifier_max_passage_chars', type=int, default=850,
                        help='Maximum passage characters shown to the LLM evidence verifier.')
    parser.add_argument('--query_plan_ontology_llm_evidence_verifier_max_completion_tokens', type=int, default=700,
                        help='Maximum completion tokens for each LLM evidence verifier call.')
    parser.add_argument('--query_plan_ontology_enable_supervised_evidence_ranker', type=str, default='false',
                        help='If True, score ontology evidence candidates with a supervised ranker trained from trace labels.')
    parser.add_argument('--query_plan_ontology_supervised_evidence_ranker_model_path', type=str, default='',
                        help='Path to a JSON model for the supervised ontology evidence ranker.')
    parser.add_argument('--query_plan_ontology_supervised_evidence_ranker_min_score', type=float, default=0.55,
                        help='Minimum supervised probability needed to promote an ontology evidence candidate.')
    parser.add_argument('--query_plan_ontology_supervised_evidence_ranker_promote_confidence', type=float, default=0.90,
                        help='Confidence floor for supervised-ranker-promoted ontology evidence candidates.')
    parser.add_argument('--query_plan_ontology_supervised_evidence_ranker_reject_below', type=float, default=0.20,
                        help='Candidates below this supervised probability have their confidence capped.')
    parser.add_argument('--query_plan_ontology_supervised_evidence_ranker_rejected_confidence', type=float, default=0.05,
                        help='Confidence cap for candidates rejected by the supervised ontology evidence ranker.')
    parser.add_argument('--query_plan_ontology_supervised_evidence_ranker_max_candidates', type=int, default=16,
                        help='Maximum candidates promoted by the supervised ontology evidence ranker per query; <=0 disables the cap.')
    parser.add_argument('--query_plan_ontology_enable_metric_aligned_slate_ranker', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_ENABLE_METRIC_ALIGNED_SLATE_RANKER', 'false'),
                        help='If true, apply an OOF-certified gold-free Top-2/Top-5 slate ranker after all evidence operators.')
    parser.add_argument('--query_plan_ontology_metric_aligned_slate_ranker_model_path', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_METRIC_ALIGNED_SLATE_RANKER_MODEL_PATH', ''),
                        help='Path to the metric-aligned slate-ranker manifest.')
    parser.add_argument('--query_plan_ontology_enable_post_physical_delta_optimizer', type=str, default='false',
                        help='If True, optimize one residual evidence action after all base and physical operators.')
    parser.add_argument('--query_plan_ontology_post_physical_delta_optimizer_model_path', type=str, default='',
                        help='Path to the composable post-physical delta model.')
    parser.add_argument('--query_plan_ontology_post_physical_delta_optimizer_candidate_mode', type=str, default='unified',
                        choices=[
                            'unified',
                            'base_only',
                            'content_complete_top2',
                            'content_complete_unified',
                            'rank_partitioned_top2',
                            'rank_partitioned_unified',
                            'rank_partitioned_external_only',
                            'proof_nested_topk',
                            'proof_nested_top5_only',
                            'proof_nested_top2_only',
                            'proof_full_aperture_nested_topk',
                            'executable_proof_semijoin_nested_topk',
                            'clause_complete_lineage_nested_topk',
                        ],
                        help='Candidate relation used by the post-physical optimizer.')
    parser.add_argument('--query_plan_ontology_post_physical_delta_optimizer_enforce_cq', type=str, default='true',
                        help='If True, enforce competency-query constraints on post-physical actions.')
    parser.add_argument('--query_plan_ontology_post_physical_delta_optimizer_shadow_only', type=str, default='false',
                        help='If True, collect aligned final-state actions without changing the ranking.')
    parser.add_argument('--query_plan_ontology_post_physical_delta_optimizer_max_candidates', type=int, default=24,
                        help='Maximum evidence candidates exposed to the post-physical optimizer.')
    parser.add_argument('--query_plan_ontology_enable_complete_plan_optimizer', type=str, default='false',
                        help='If True, choose a complete semantic, ontology, or explicit merge evidence plan.')
    parser.add_argument('--query_plan_ontology_complete_plan_optimizer_model_path', type=str, default='',
                        help='Path to the calibrated complete-plan portfolio CBO model.')
    parser.add_argument('--query_plan_ontology_complete_plan_optimizer_shadow_only', type=str, default='false',
                        help='If True, materialize complete plans without changing the ranking.')
    parser.add_argument('--query_plan_ontology_complete_plan_optimizer_top_k', type=int, default=5,
                        help='Evidence-set size selected atomically by the complete-plan optimizer.')
    parser.add_argument('--query_plan_ontology_complete_plan_optimizer_allowed_plans', type=str, default='',
                        help='Optional comma-separated complete-plan families materialized before model selection.')
    parser.add_argument('--query_plan_ontology_complete_plan_selector', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_COMPLETE_PLAN_SELECTOR', 'full_cbo'),
                        choices=['full_cbo', 'fixed', 'utility_only', 'random', 'offline_oracle'],
                        help='Controlled selector over the same materialized complete-plan portfolio.')
    parser.add_argument('--query_plan_ontology_complete_plan_integrate_semantic_access', type=str, default='false',
                        help='If True, compile semantic/fact candidates into the ontology-guided access window instead of treating their ranking as a separate final plan.')
    parser.add_argument('--query_plan_ontology_complete_plan_immutable_incumbent', type=str, default='false',
                        help='If True, use the frozen pre-ontology semantic/fact ranking as the atomic no-op incumbent while ontology operators contribute sidecar candidates.')
    parser.add_argument('--query_plan_ontology_enable_agentic_evidence_repair', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_ENABLE_AGENTIC_EVIDENCE_REPAIR', 'false'),
                        help='If True, run bounded verifier, repair-planner, and transactional-critic agents over complete evidence plans.')
    parser.add_argument('--query_plan_ontology_agentic_evidence_repair_shadow_only', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_AGENTIC_EVIDENCE_REPAIR_SHADOW_ONLY', 'false'),
                        help='If True, log agentic repair decisions without changing the final slate.')
    parser.add_argument('--query_plan_ontology_agentic_evidence_repair_top_k', type=int,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_AGENTIC_EVIDENCE_REPAIR_TOP_K', '5'),
                        help='Atomic evidence-set size optimized by the bounded repair loop.')
    parser.add_argument('--query_plan_ontology_agentic_evidence_repair_max_iterations', type=int,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_AGENTIC_EVIDENCE_REPAIR_MAX_ITERATIONS', '2'),
                        help='Maximum verify-repair-critic iterations per query.')
    parser.add_argument('--query_plan_ontology_agentic_evidence_repair_enable_top2', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_AGENTIC_EVIDENCE_REPAIR_ENABLE_TOP2', 'true'),
                        help='Allow strictly certified plans to repair rank 1-2 evidence.')
    parser.add_argument('--query_plan_ontology_agentic_evidence_repair_enable_top5', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_AGENTIC_EVIDENCE_REPAIR_ENABLE_TOP5', 'true'),
                        help='Allow proof-complete residual repairs while preserving top-2.')
    parser.add_argument('--query_plan_ontology_agentic_evidence_repair_min_relation_gain', type=float,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_AGENTIC_EVIDENCE_REPAIR_MIN_RELATION_GAIN', '0.10'),
                        help='Minimum structural relation-coverage improvement needed for commit.')
    parser.add_argument('--query_plan_ontology_agentic_evidence_repair_max_top5_displacements', type=int,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_AGENTIC_EVIDENCE_REPAIR_MAX_TOP5_DISPLACEMENTS', '2'),
                        help='Maximum top-5 displacements allowed for a certified top-2 repair.')
    parser.add_argument('--query_plan_ontology_agentic_evidence_repair_max_estimated_cost', type=float,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_AGENTIC_EVIDENCE_REPAIR_MAX_ESTIMATED_COST', '384.0'),
                        help='Maximum physical-plan cost admitted by the transactional critic.')
    parser.add_argument('--query_plan_ontology_agentic_evidence_repair_require_query_local_proof', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_AGENTIC_EVIDENCE_REPAIR_REQUIRE_QUERY_LOCAL_PROOF', 'true'),
                        help='Restrict repairs to query-local proof-carrying plan families.')
    parser.add_argument('--query_plan_ontology_agentic_evidence_repair_action_model_path', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_AGENTIC_EVIDENCE_REPAIR_ACTION_MODEL_PATH', ''),
                        help='Q_ind/Q_cal/Q_val-certified action model used by the agentic Delta-Critic.')
    parser.add_argument('--query_plan_ontology_agentic_evidence_repair_require_calibrated_delta', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_AGENTIC_EVIDENCE_REPAIR_REQUIRE_CALIBRATED_DELTA', 'false'),
                        help='Reject every agentic action not admitted by the certified Delta-Critic.')
    parser.add_argument('--query_plan_ontology_agentic_evidence_repair_skip_after_upstream_commit', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_AGENTIC_EVIDENCE_REPAIR_SKIP_AFTER_UPSTREAM_COMMIT', 'true'),
                        help='Skip agentic repair after an upstream complete-plan transaction commits.')
    parser.add_argument('--query_plan_ontology_enable_agentic_residual_retrieval', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_ENABLE_AGENTIC_RESIDUAL_RETRIEVAL', 'false'),
                        help='Run the native router-verifier-planner-retriever-merger residual access plan.')
    parser.add_argument('--query_plan_ontology_agentic_residual_shadow_only', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_AGENTIC_RESIDUAL_SHADOW_ONLY', 'false'),
                        help='Trace the residual slate without committing the final RRF ranking.')
    parser.add_argument('--query_plan_ontology_agentic_residual_max_rounds', type=int,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_AGENTIC_RESIDUAL_MAX_ROUNDS', '3'))
    parser.add_argument('--query_plan_ontology_agentic_residual_max_queries', type=int,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_AGENTIC_RESIDUAL_MAX_QUERIES', '3'))
    parser.add_argument('--query_plan_ontology_agentic_residual_max_snippets', type=int,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_AGENTIC_RESIDUAL_MAX_SNIPPETS', '3'))
    parser.add_argument('--query_plan_ontology_agentic_residual_max_tokens', type=int,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_AGENTIC_RESIDUAL_MAX_TOKENS', '160'))
    parser.add_argument('--query_plan_ontology_agentic_residual_protect_top_k', type=int,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_AGENTIC_RESIDUAL_PROTECT_TOP_K', '1'))
    parser.add_argument('--query_plan_ontology_agentic_residual_max_promotions', type=int,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_AGENTIC_RESIDUAL_MAX_PROMOTIONS', '2'))
    parser.add_argument('--query_plan_ontology_agentic_residual_plan_top_k', type=int,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_AGENTIC_RESIDUAL_PLAN_TOP_K', '20'))
    parser.add_argument('--query_plan_ontology_agentic_residual_min_plan_score', type=float,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_AGENTIC_RESIDUAL_MIN_PLAN_SCORE', '0.18'))
    parser.add_argument('--query_plan_ontology_agentic_residual_merge_mode', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_AGENTIC_RESIDUAL_MERGE_MODE', 'guarded_promote'))
    parser.add_argument('--query_plan_ontology_agentic_residual_guarded_min_score', type=float,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_AGENTIC_RESIDUAL_GUARDED_MIN_SCORE', '0.82'))
    parser.add_argument('--query_plan_ontology_agentic_residual_guarded_max_original_rank', type=int,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_AGENTIC_RESIDUAL_GUARDED_MAX_ORIGINAL_RANK', '80'))
    parser.add_argument('--query_plan_ontology_agentic_residual_guarded_max_plan_rank', type=int,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_AGENTIC_RESIDUAL_GUARDED_MAX_PLAN_RANK', '8'))
    parser.add_argument('--query_plan_ontology_agentic_residual_guarded_min_query_overlap', type=float,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_AGENTIC_RESIDUAL_GUARDED_MIN_QUERY_OVERLAP', '0.08'))
    parser.add_argument('--query_plan_ontology_agentic_residual_route_mode', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_AGENTIC_RESIDUAL_ROUTE_MODE', 'adaptive'))
    parser.add_argument('--query_plan_ontology_agentic_residual_fusion_depth', type=int,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_AGENTIC_RESIDUAL_FUSION_DEPTH', '5'))
    parser.add_argument('--query_plan_ontology_agentic_residual_rrf_k', type=float,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_AGENTIC_RESIDUAL_RRF_K', '60.0'))
    parser.add_argument('--query_plan_ontology_agentic_residual_base_weight', type=float,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_AGENTIC_RESIDUAL_BASE_WEIGHT', '1.0'))
    parser.add_argument('--query_plan_ontology_agentic_residual_residual_weight', type=float,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_AGENTIC_RESIDUAL_RESIDUAL_WEIGHT', '1.0'))
    parser.add_argument('--query_plan_ontology_enable_certified_direct_execution', type=str, default='false',
                        help='If True, send a certified ontology-only top-k directly to the reader and use the baseline only as atomic fallback.')
    parser.add_argument('--query_plan_ontology_require_certified_schema_mask', type=str, default='false',
                        help='If True, fail closed and restrict the incumbent and compiler-sidecar matchers to a frozen Q_val-certified schema mask.')
    parser.add_argument('--query_plan_ontology_certified_direct_policy_path', type=str, default='',
                        help='Path to a frozen Q_cal/Q_val ontology-direct deployment policy.')
    parser.add_argument('--query_plan_ontology_certified_direct_shadow_only', type=str, default='false',
                        help='If True, log ontology-direct decisions without changing the reader slate.')
    parser.add_argument('--query_plan_ontology_certified_direct_require_policy', type=str, default='true',
                        help='Require a hash-bound Q_val deployment certificate before ontology-direct execution.')
    parser.add_argument('--query_plan_ontology_certified_direct_top_k', type=int, default=5,
                        help='Atomic ontology evidence-set size sent to the reader.')
    parser.add_argument('--query_plan_ontology_certified_direct_min_matched_plans', type=int, default=1,
                        help='Minimum matched ontology schemas required for direct execution.')
    parser.add_argument('--query_plan_ontology_certified_direct_min_strict_paths', type=int, default=1,
                        help='Minimum strictly executable paths required for direct execution.')
    parser.add_argument('--query_plan_ontology_certified_direct_min_relation_coverage', type=float, default=0.35,
                        help='Minimum mean relation coverage of direct ontology candidates.')
    parser.add_argument('--query_plan_ontology_certified_direct_min_answer_type_fraction', type=float, default=0.20,
                        help='Minimum fraction of answer-type-compatible direct ontology candidates.')
    parser.add_argument('--query_plan_ontology_certified_direct_require_complete_proof', type=str, default='true',
                        help='Require a complete proof path before ontology-direct execution.')
    parser.add_argument('--query_plan_ontology_enable_missing_evidence_completer', type=str, default='false',
                        help='If True, issue ontology-guided dense subqueries for fact slots missing from the current top evidence window.')
    parser.add_argument('--query_plan_ontology_missing_evidence_dense_top_k', type=int, default=40,
                        help='Dense top-k passages considered for each missing-evidence subquery.')
    parser.add_argument('--query_plan_ontology_missing_evidence_top_window_k', type=int, default=5,
                        help='Top retrieval window inspected for already-covered query-plan fact slots.')
    parser.add_argument('--query_plan_ontology_missing_evidence_max_missing_slots', type=int, default=6,
                        help='Maximum uncovered query-plan fact slots expanded into dense subqueries.')
    parser.add_argument('--query_plan_ontology_missing_evidence_max_query_variants', type=int, default=8,
                        help='Maximum ontology-derived dense subqueries issued per query.')
    parser.add_argument('--query_plan_ontology_missing_evidence_max_candidates', type=int, default=3,
                        help='Maximum verified missing-evidence candidates exposed to evidence-set selection.')
    parser.add_argument('--query_plan_ontology_missing_evidence_min_confidence', type=float, default=0.70,
                        help='Minimum verifier confidence required for missing-evidence candidates.')
    parser.add_argument('--query_plan_ontology_missing_evidence_promote_to_rank', type=int, default=5,
                        help='Target rank for direct missing-evidence insertion when candidate-only mode is disabled.')
    parser.add_argument('--query_plan_ontology_missing_evidence_preserve_top_k', type=int, default=2,
                        help='Number of top-ranked baseline passages protected from direct missing-evidence insertion.')
    parser.add_argument('--query_plan_ontology_missing_evidence_require_relation_hit', type=str, default='true',
                        help='If True, verified missing-evidence candidates must contain relation clues when available.')
    parser.add_argument('--query_plan_ontology_missing_evidence_require_both_endpoints', type=str, default='false',
                        help='If True, verified missing-evidence candidates must mention both fact endpoints.')
    parser.add_argument('--query_plan_ontology_missing_evidence_candidate_only', type=str, default='true',
                        help='If True, missing-evidence retrieval only adds candidates and lets evidence-set selection decide.')
    parser.add_argument('--query_plan_ontology_missing_evidence_include_covered_slots', type=str, default='false',
                        help='If True, also retrieve alternative support passages for fact slots already covered in the current top window.')
    parser.add_argument('--query_plan_ontology_missing_evidence_max_covered_slots', type=int, default=4,
                        help='Maximum already-covered fact slots expanded as alternative supporting evidence queries.')
    parser.add_argument('--query_plan_ontology_missing_evidence_score_margin', type=float, default=1e-6,
                        help='Small score margin used by direct missing-evidence insertion.')
    parser.add_argument('--query_plan_ontology_enable_evidence_set_retriever', type=str, default='false',
                        help='If True, reselect the top retrieval evidence set by ontology slot coverage.')
    parser.add_argument('--query_plan_ontology_evidence_set_final_top_k', type=int, default=5,
                        help='Top-k evidence window optimized by the ontology evidence-set retriever.')
    parser.add_argument('--query_plan_ontology_evidence_set_preserve_top_k', type=int, default=1,
                        help='Number of strongest baseline passages preserved before evidence-set selection.')
    parser.add_argument('--query_plan_ontology_evidence_set_candidate_pool_top_k', type=int, default=80,
                        help='Maximum original retrieval rank considered by the ontology evidence-set retriever.')
    parser.add_argument('--query_plan_ontology_evidence_set_enable_optimizer_candidate_pool', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_ENABLE_OPTIMIZER_CANDIDATE_POOL', 'false'),
                        help='Expose bounded pre-filter evidence candidates to the learned optimizer.')
    parser.add_argument('--query_plan_ontology_evidence_set_optimizer_candidate_pool_max', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_OPTIMIZER_CANDIDATE_POOL_MAX', 80)),
                        help='Maximum pre-filter evidence candidates exposed to the learned optimizer.')
    parser.add_argument('--query_plan_ontology_evidence_set_enable_source_gap_aperture', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_ENABLE_SOURCE_GAP_APERTURE', 'false'),
                        help='If True, dynamically admits high-utility source-gap candidates beyond the fixed candidate pool.')
    parser.add_argument('--query_plan_ontology_evidence_set_source_gap_aperture_max_original_rank', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_SOURCE_GAP_APERTURE_MAX_ORIGINAL_RANK', 900)),
                        help='Maximum original rank for source-gap aperture candidates; <=0 disables this bound.')
    parser.add_argument('--query_plan_ontology_evidence_set_source_gap_aperture_min_confidence', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_SOURCE_GAP_APERTURE_MIN_CONFIDENCE', 0.34)),
                        help='Minimum confidence required by source-gap aperture admission.')
    parser.add_argument('--query_plan_ontology_evidence_set_source_gap_aperture_min_source_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_SOURCE_GAP_APERTURE_MIN_SOURCE_SCORE', 0.05)),
                        help='Minimum source prior score required by source-gap aperture admission.')
    parser.add_argument('--query_plan_ontology_evidence_set_source_gap_aperture_min_query_pressure', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_SOURCE_GAP_APERTURE_MIN_QUERY_PRESSURE', 0.06)),
                        help='Minimum query recall pressure required by source-gap aperture admission.')
    parser.add_argument('--query_plan_ontology_evidence_set_source_gap_aperture_max_candidates', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_SOURCE_GAP_APERTURE_MAX_CANDIDATES', 18)),
                        help='Maximum source-gap aperture candidates admitted per query; <=0 allows all approved candidates.')
    parser.add_argument('--query_plan_ontology_evidence_set_source_gap_aperture_score_weight', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_SOURCE_GAP_APERTURE_SCORE_WEIGHT', 0.10)),
                        help='Score bonus weight assigned to source-gap aperture candidates.')
    parser.add_argument('--query_plan_ontology_evidence_set_source_gap_aperture_allow_no_new_slot', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_SOURCE_GAP_APERTURE_ALLOW_NO_NEW_SLOT', 'true'),
                        help='If True, approved source-gap aperture candidates bypass strict new-slot rejection.')
    parser.add_argument('--query_plan_ontology_evidence_set_max_ontology_passages', type=int, default=3,
                        help='Maximum ontology-supported passages inserted into the selected evidence set.')
    parser.add_argument('--query_plan_ontology_evidence_set_min_candidate_confidence', type=float, default=0.72,
                        help='Minimum ontology confidence required before a passage can enter the evidence set.')
    parser.add_argument('--query_plan_ontology_evidence_set_top2_min_candidate_confidence', type=float, default=0.92,
                        help='Minimum confidence required before an ontology passage can alter the top-2 evidence window.')
    parser.add_argument('--query_plan_ontology_evidence_set_top2_max_original_rank', type=int, default=8,
                        help='Maximum original rank for ontology passages allowed to enter top-2; <=0 disables.')
    parser.add_argument('--query_plan_ontology_evidence_set_require_new_slot', type=str, default='true',
                        help='If True, inserted ontology passages must add a new plan/role/relation/entity slot.')
    parser.add_argument('--query_plan_ontology_evidence_set_baseline_weight', type=float, default=0.45,
                        help='Baseline rank weight used by ontology evidence-set selection.')
    parser.add_argument('--query_plan_ontology_evidence_set_ontology_weight', type=float, default=0.75,
                        help='Ontology confidence weight used by ontology evidence-set selection.')
    parser.add_argument('--query_plan_ontology_evidence_set_coverage_weight', type=float, default=0.95,
                        help='New slot coverage weight used by ontology evidence-set selection.')
    parser.add_argument('--query_plan_ontology_evidence_set_redundancy_penalty', type=float, default=0.28,
                        help='Penalty for selected passages that repeat already covered ontology slots.')
    parser.add_argument('--query_plan_ontology_evidence_set_coverage_reference_top_k', type=int, default=0,
                        help='Top baseline passages used to decide whether an ontology candidate adds a genuinely missing slot; <=0 uses preserve_top_k.')
    parser.add_argument('--query_plan_ontology_evidence_set_enable_intervention_policy', type=str, default='false',
                        help='If True, ontology evidence-set candidates must pass a strong intervention gate before changing top-k.')
    parser.add_argument('--query_plan_ontology_evidence_set_intervention_min_confidence', type=float, default=0.90,
                        help='Minimum candidate confidence required by the ontology intervention gate.')
    parser.add_argument('--query_plan_ontology_evidence_set_intervention_relation_only_min_confidence', type=float, default=0.995,
                        help='Minimum confidence for relation-only candidates allowed in simple relation queries.')
    parser.add_argument('--query_plan_ontology_evidence_set_intervention_relation_only_max_original_rank', type=int, default=20,
                        help='Maximum original rank for relation-only candidates allowed by the ontology intervention gate; <=0 disables.')
    parser.add_argument('--query_plan_ontology_evidence_set_intervention_allow_relation_only_simple', type=str, default='true',
                        help='If True, allow very high-confidence relation-only candidates for simple relation queries.')
    parser.add_argument('--query_plan_ontology_evidence_set_enable_counterfactual_policy', type=str, default='false',
                        help='If True, ontology evidence-set candidates must beat the baseline passage they would displace.')
    parser.add_argument('--query_plan_ontology_evidence_set_counterfactual_min_candidate_value', type=float, default=0.74,
                        help='Minimum estimated evidence value required by the counterfactual replacement policy.')
    parser.add_argument('--query_plan_ontology_evidence_set_counterfactual_margin', type=float, default=0.03,
                        help='Required estimated value margin over the baseline passage displaced from top-k.')
    parser.add_argument('--query_plan_ontology_evidence_set_policy_model_path', type=str, default='',
                        help='Optional learned top-5 evidence-set intervention policy JSON.')
    parser.add_argument('--query_plan_ontology_evidence_set_policy_mode', type=str, default='off',
                        choices=['off', 'score', 'filter', 'selector'],
                        help='How to use the learned evidence-set policy: off, score, filter, or selector.')
    parser.add_argument('--query_plan_ontology_evidence_set_policy_min_score', type=float, default=0.55,
                        help='Policy score needed to promote an ontology candidate before evidence-set selection.')
    parser.add_argument('--query_plan_ontology_evidence_set_policy_reject_below', type=float, default=0.0,
                        help='In filter mode, reject ontology evidence-set candidates below this policy score.')
    parser.add_argument('--query_plan_ontology_evidence_set_policy_score_weight', type=float, default=0.45,
                        help='Score weight added to evidence-set selection from the learned policy score.')
    parser.add_argument('--query_plan_ontology_evidence_set_policy_promote_confidence', type=float, default=0.90,
                        help='Confidence floor assigned when the learned evidence-set policy promotes a candidate.')
    parser.add_argument('--query_plan_ontology_evidence_set_policy_counterfactual_weight', type=float, default=0.20,
                        help='Policy score contribution to candidate replacement value in counterfactual selection.')
    parser.add_argument('--query_plan_ontology_evidence_set_utility_policy_model_path', type=str, default='',
                        help='Optional learned utility-aware evidence-set policy JSON path.')
    parser.add_argument('--query_plan_ontology_evidence_set_utility_policy_mode', type=str, default='off',
                        choices=['off', 'score', 'filter', 'selector'],
                        help='How to use the learned utility-aware evidence-set policy: off, score, filter, or selector.')
    parser.add_argument('--query_plan_ontology_evidence_set_utility_policy_min_score', type=float, default=0.90,
                        help='Minimum utility-policy score required to promote an ontology candidate.')
    parser.add_argument('--query_plan_ontology_evidence_set_utility_policy_reject_below', type=float, default=0.0,
                        help='Reject ontology evidence-set candidates below this utility-policy score in filter/selector mode.')
    parser.add_argument('--query_plan_ontology_evidence_set_utility_policy_score_weight', type=float, default=0.35,
                        help='Score weight added to evidence-set selection from the utility-policy score.')
    parser.add_argument('--query_plan_ontology_evidence_set_utility_policy_promote_confidence', type=float, default=0.97,
                        help='Confidence floor assigned when the utility policy promotes a candidate.')
    parser.add_argument('--query_plan_ontology_evidence_set_utility_policy_counterfactual_weight', type=float, default=0.40,
                        help='Utility-policy score contribution to candidate replacement value in counterfactual selection.')
    parser.add_argument('--query_plan_ontology_evidence_set_utility_policy_apply_scope', type=str, default='all',
                        choices=['all', 'recall_selector', 'recall_only', 'weak_only'],
                        help='Candidate scope for utility policy: all, recall_selector, recall_only, or weak_only.')
    parser.add_argument('--query_plan_ontology_evidence_set_utility_planner_model_path', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_UTILITY_PLANNER_MODEL_PATH', ''),
                        help='Optional learned cost-aware evidence utility planner JSON path.')
    parser.add_argument('--query_plan_ontology_evidence_set_utility_planner_mode', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_UTILITY_PLANNER_MODE', 'off'),
                        choices=[
                            'off',
                            'score',
                            'planner',
                            'safe_planner',
                            'position_planner',
                            'position_top5_planner',
                            'groupwise_top5_planner',
                            'hard_negative_top5_planner',
                            'gold_aware_text_top5_planner',
                            'gap_aware_topk_planner',
                            'oracle_delta_recall_planner',
                            'counterfactual_topk_benefit_planner',
                            'metric_aligned_action_risk_planner',
                            'set_counterfactual_recall_planner',
                        ],
                        help='How to use the cost-aware utility planner.')
    parser.add_argument('--query_plan_ontology_evidence_set_utility_planner_min_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_UTILITY_PLANNER_MIN_SCORE', 0.78)),
                        help='Minimum planner score required for top-5 ontology insertion.')
    parser.add_argument('--query_plan_ontology_evidence_set_utility_planner_top2_min_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_UTILITY_PLANNER_TOP2_MIN_SCORE', 0.90)),
                        help='Minimum planner score required for rank-2 replacement.')
    parser.add_argument('--query_plan_ontology_evidence_set_utility_planner_score_weight', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_UTILITY_PLANNER_SCORE_WEIGHT', 0.65)),
                        help='Score weight added to evidence-set selection from the cost-aware utility planner.')
    parser.add_argument('--query_plan_ontology_evidence_set_utility_planner_counterfactual_weight', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_UTILITY_PLANNER_COUNTERFACTUAL_WEIGHT', 0.55)),
                        help='Planner score contribution to candidate replacement value.')
    parser.add_argument('--query_plan_ontology_evidence_set_utility_planner_min_value_margin', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_UTILITY_PLANNER_MIN_VALUE_MARGIN', 0.02)),
                        help='Minimum candidate-vs-displaced top-5 utility margin for planner insertion.')
    parser.add_argument('--query_plan_ontology_evidence_set_utility_planner_top2_min_value_margin', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_UTILITY_PLANNER_TOP2_MIN_VALUE_MARGIN', 0.04)),
                        help='Minimum candidate-vs-rank2 utility margin for planner rank-2 replacement.')
    parser.add_argument('--query_plan_ontology_evidence_set_utility_planner_max_original_rank', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_UTILITY_PLANNER_MAX_ORIGINAL_RANK', 120)),
                        help='Maximum original retrieval rank eligible for utility-planner insertion; <=0 disables.')
    parser.add_argument('--query_plan_ontology_evidence_set_utility_planner_require_complex_or_path', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_UTILITY_PLANNER_REQUIRE_COMPLEX_OR_PATH', 'true'),
                        help='If true, planner insertions require complex query or strong path/endpoint evidence.')
    parser.add_argument('--query_plan_ontology_evidence_set_recall_plan_optimizer_mode', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_RECALL_PLAN_OPTIMIZER_MODE', 'off'),
                        choices=['off', 'score', 'recall'],
                        help='How to use the cost-based recall plan optimizer.')
    parser.add_argument('--query_plan_ontology_evidence_set_recall_plan_optimizer_stats_path', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_RECALL_PLAN_OPTIMIZER_STATS_PATH', ''),
                        help='Optional JSON source/operator stats for the cost-based recall plan optimizer.')
    parser.add_argument('--query_plan_ontology_evidence_set_recall_plan_optimizer_min_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_RECALL_PLAN_OPTIMIZER_MIN_SCORE', 0.52)),
                        help='Minimum cost-benefit score required to promote a recall-plan candidate.')
    parser.add_argument('--query_plan_ontology_evidence_set_recall_plan_optimizer_score_weight', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_RECALL_PLAN_OPTIMIZER_SCORE_WEIGHT', 0.40)),
                        help='Score weight added to evidence-set selection from the recall plan optimizer.')
    parser.add_argument('--query_plan_ontology_evidence_set_recall_plan_optimizer_confidence_boost', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_RECALL_PLAN_OPTIMIZER_CONFIDENCE_BOOST', 0.06)),
                        help='Confidence boost scale for promoted recall-plan candidates.')
    parser.add_argument('--query_plan_ontology_evidence_set_recall_plan_optimizer_router_bypass_min_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_RECALL_PLAN_OPTIMIZER_ROUTER_BYPASS_MIN_SCORE', 0.72)),
                        help='Recall-plan score required to override a query-level router skip/precision gate.')
    parser.add_argument('--query_plan_ontology_evidence_set_recall_plan_optimizer_allow_no_new_slot', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_RECALL_PLAN_OPTIMIZER_ALLOW_NO_NEW_SLOT', 'true'),
                        help='If true, high-utility recall-plan candidates may bypass the strict new-slot guard.')
    parser.add_argument('--query_plan_ontology_evidence_set_recall_plan_optimizer_max_original_rank', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_RECALL_PLAN_OPTIMIZER_MAX_ORIGINAL_RANK', 220)),
                        help='Maximum original rank eligible for recall-plan promotion; <=0 disables.')
    parser.add_argument('--query_plan_ontology_evidence_set_recall_plan_optimizer_max_selected', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_RECALL_PLAN_OPTIMIZER_MAX_SELECTED', 2)),
                        help='Maximum no-new-slot recall-plan insertions per query; <=0 disables this limit.')
    parser.add_argument('--query_plan_ontology_evidence_set_recall_plan_optimizer_min_source_precision', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_RECALL_PLAN_OPTIMIZER_MIN_SOURCE_PRECISION', 0.035)),
                        help='Minimum source/operator precision prior required for recall-plan promotion.')
    parser.add_argument('--query_plan_ontology_evidence_set_enable_rank_window_recall', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_ENABLE_RANK_WINDOW_RECALL', 'false'),
                        help='If true, expose dense rank-window passages as a cost-based recall access path.')
    parser.add_argument('--query_plan_ontology_evidence_set_rank_window_model_path', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_RANK_WINDOW_MODEL_PATH', ''),
                        help='Optional JSON scorer for the rank-window recall access path.')
    parser.add_argument('--query_plan_ontology_evidence_set_rank_window_target_datasets', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_RANK_WINDOW_TARGET_DATASETS', ''),
                        help='Comma-separated allowlist of datasets where rank-window recall is allowed; empty enables all datasets.')
    parser.add_argument('--query_plan_ontology_evidence_set_rank_window_start_rank', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_RANK_WINDOW_START_RANK', 3)),
                        help='First original dense rank admitted by the rank-window access path.')
    parser.add_argument('--query_plan_ontology_evidence_set_rank_window_end_rank', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_RANK_WINDOW_END_RANK', 24)),
                        help='Last original dense rank admitted by the rank-window access path.')
    parser.add_argument('--query_plan_ontology_evidence_set_rank_window_min_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_RANK_WINDOW_MIN_SCORE', 0.56)),
                        help='Minimum rank-window benefit score for top-5 recall insertion.')
    parser.add_argument('--query_plan_ontology_evidence_set_rank_window_top2_min_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_RANK_WINDOW_TOP2_MIN_SCORE', 0.80)),
                        help='Minimum rank-window benefit score for rank-2 replacement.')
    parser.add_argument('--query_plan_ontology_evidence_set_rank_window_router_bypass_min_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_RANK_WINDOW_ROUTER_BYPASS_MIN_SCORE', 0.74)),
                        help='Rank-window score required to bypass query-level skip/precision routing.')
    parser.add_argument('--query_plan_ontology_evidence_set_rank_window_score_weight', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_RANK_WINDOW_SCORE_WEIGHT', 0.18)),
                        help='Score weight assigned to rank-window candidates during evidence-set selection.')
    parser.add_argument('--query_plan_ontology_evidence_set_rank_window_confidence_floor', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_RANK_WINDOW_CONFIDENCE_FLOOR', 0.52)),
                        help='Confidence floor assigned to approved rank-window candidates.')
    parser.add_argument('--query_plan_ontology_evidence_set_rank_window_max_selected', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_RANK_WINDOW_MAX_SELECTED', 2)),
                        help='Maximum rank-window candidates inserted per query.')
    parser.add_argument('--query_plan_ontology_evidence_set_rank_window_max_top2_swaps', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_RANK_WINDOW_MAX_TOP2_SWAPS', 1)),
                        help='Maximum rank-window replacements allowed in the top-2 window.')
    parser.add_argument('--query_plan_ontology_evidence_set_rank_window_min_margin', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_RANK_WINDOW_MIN_MARGIN', -0.12)),
                        help='Minimum candidate-value margin for rank-window top-5 replacement.')
    parser.add_argument('--query_plan_ontology_evidence_set_rank_window_top2_min_margin', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_RANK_WINDOW_TOP2_MIN_MARGIN', -0.08)),
                        help='Minimum candidate-value margin for rank-window rank-2 replacement.')
    parser.add_argument('--query_plan_ontology_evidence_set_rank_window_require_text_signal', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_RANK_WINDOW_REQUIRE_TEXT_SIGNAL', 'false'),
                        help='If true, rank-window candidates need non-empty text verifier evidence.')
    parser.add_argument('--query_plan_ontology_evidence_set_rank_window_min_text_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_RANK_WINDOW_MIN_TEXT_SCORE', 0.0)),
                        help='Minimum text verifier score required for rank-window candidates.')
    parser.add_argument('--query_plan_ontology_evidence_set_rank_window_defer_to_planner', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_RANK_WINDOW_DEFER_TO_PLANNER', 'false'),
                        help='If true, pure rank-window candidates are generated as access-path candidates and only inserted by the safe replacement planner.')
    parser.add_argument('--query_plan_ontology_evidence_set_rank_window_protect_original_top_k', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_RANK_WINDOW_PROTECT_ORIGINAL_TOP_K', 0)),
                        help='Protect baseline documents with original rank <= this value from rank-window replacement.')
    parser.add_argument('--query_plan_ontology_evidence_set_rank_window_min_replacement_rank', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_RANK_WINDOW_MIN_REPLACEMENT_RANK', 3)),
                        help='Minimum final rank position that rank-window planner may replace.')
    parser.add_argument('--query_plan_ontology_evidence_set_rank_window_query_router_mode', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_RANK_WINDOW_QUERY_ROUTER_MODE', 'off'),
                        help='Rank-window query router mode: off/global or adaptive.')
    parser.add_argument('--query_plan_ontology_evidence_set_rank_window_complex_datasets', type=str,
                        default=os.environ.get(
                            'QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_RANK_WINDOW_COMPLEX_DATASETS',
                            '2wikimultihopqa,hotpotqa,musique',
                        ),
                        help='Comma-separated datasets where rank-window is treated as a multi-hop access path.')
    parser.add_argument('--query_plan_ontology_evidence_set_rank_window_simple_datasets', type=str,
                        default=os.environ.get(
                            'QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_RANK_WINDOW_SIMPLE_DATASETS',
                            'popqa,nq_rear,narrativeqa_dev_10_doc,lveval',
                        ),
                        help='Comma-separated datasets where rank-window needs stricter answer/entity evidence.')
    parser.add_argument('--query_plan_ontology_evidence_set_rank_window_complex_min_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_RANK_WINDOW_COMPLEX_MIN_SCORE', 0.58)),
                        help='Adaptive-router min score for complex/path queries.')
    parser.add_argument('--query_plan_ontology_evidence_set_rank_window_complex_min_text_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_RANK_WINDOW_COMPLEX_MIN_TEXT_SCORE', 0.04)),
                        help='Adaptive-router min text score for complex/path queries.')
    parser.add_argument('--query_plan_ontology_evidence_set_rank_window_simple_min_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_RANK_WINDOW_SIMPLE_MIN_SCORE', 0.86)),
                        help='Adaptive-router min score for simple/entity queries.')
    parser.add_argument('--query_plan_ontology_evidence_set_rank_window_simple_min_text_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_RANK_WINDOW_SIMPLE_MIN_TEXT_SCORE', 0.16)),
                        help='Adaptive-router min text score for simple/entity queries.')
    parser.add_argument('--query_plan_ontology_evidence_set_rank_window_simple_require_answer_signal', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_RANK_WINDOW_SIMPLE_REQUIRE_ANSWER_SIGNAL', 'true'),
                        help='If true, simple/entity rank-window candidates require terminal/answer-type/query text evidence.')
    parser.add_argument('--query_plan_ontology_evidence_set_rank_window_simple_answer_signal_min_text_score', type=float,
                        default=float(os.environ.get(
                            'QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_RANK_WINDOW_SIMPLE_ANSWER_SIGNAL_MIN_TEXT_SCORE',
                            0.32,
                        )),
                        help='Text verifier score that counts as answer signal for simple/entity rank-window routing.')
    parser.add_argument('--query_plan_ontology_evidence_set_rank_window_enable_net_benefit_gate', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_RANK_WINDOW_ENABLE_NET_BENEFIT_GATE', 'false'),
                        help='If true, rank-window append/replacement must pass a cost-based net-benefit gate.')
    parser.add_argument('--query_plan_ontology_evidence_set_rank_window_allow_no_new_slot_bypass', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_RANK_WINDOW_ALLOW_NO_NEW_SLOT_BYPASS', 'true'),
                        help='If true, rank-window candidates may bypass no-new-slot guard when the net-benefit gate permits it.')
    parser.add_argument('--query_plan_ontology_evidence_set_rank_window_no_new_slot_min_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_RANK_WINDOW_NO_NEW_SLOT_MIN_SCORE', 0.0)),
                        help='Minimum rank-window score for a no-new-slot bypass under the net-benefit gate.')
    parser.add_argument('--query_plan_ontology_evidence_set_rank_window_no_new_slot_min_text_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_RANK_WINDOW_NO_NEW_SLOT_MIN_TEXT_SCORE', 0.0)),
                        help='Minimum text score for a no-new-slot bypass under the net-benefit gate.')
    parser.add_argument('--query_plan_ontology_evidence_set_rank_window_no_new_slot_require_path_or_terminal', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_RANK_WINDOW_NO_NEW_SLOT_REQUIRE_PATH_OR_TERMINAL', 'false'),
                        help='If true, no-new-slot rank-window bypass requires path/terminal/answer-overlap evidence.')
    parser.add_argument('--query_plan_ontology_evidence_set_rank_window_net_benefit_min_margin', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_RANK_WINDOW_NET_BENEFIT_MIN_MARGIN', 0.0)),
                        help='Minimum net-benefit margin for rank-window replacement when the gate is enabled.')
    parser.add_argument('--query_plan_ontology_evidence_set_rank_window_append_min_margin', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_RANK_WINDOW_APPEND_MIN_MARGIN', 0.0)),
                        help='Minimum net-benefit margin for rank-window append when the gate is enabled.')
    parser.add_argument('--query_plan_ontology_evidence_set_rank_window_displacement_cost_weight', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_RANK_WINDOW_DISPLACEMENT_COST_WEIGHT', 0.85)),
                        help='Cost weight for the document displaced by a rank-window candidate.')
    parser.add_argument('--query_plan_ontology_evidence_set_enable_evidence_warehouse_optimizer', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_ENABLE_EVIDENCE_WAREHOUSE_OPTIMIZER', 'false'),
                        help='If true, build a runtime evidence warehouse and apply a cost-based evidence query optimizer to the final top-k window.')
    parser.add_argument('--query_plan_ontology_evidence_set_enable_final_rank_evidence_warehouse', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_ENABLE_FINAL_RANK_EVIDENCE_WAREHOUSE', 'false'),
                        help='If true, apply the evidence warehouse optimizer after the full final rank has been assembled.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_warehouse_target_datasets', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_WAREHOUSE_TARGET_DATASETS', ''),
                        help='Comma-separated dataset allowlist for the evidence warehouse optimizer; empty enables all datasets.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_warehouse_max_candidates', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_WAREHOUSE_MAX_CANDIDATES', 96)),
                        help='Maximum evidence warehouse candidate rows optimized per query.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_warehouse_max_top5_swaps', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_WAREHOUSE_MAX_TOP5_SWAPS', 1)),
                        help='Maximum evidence warehouse replacements allowed inside the top-5 window after protected ranks.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_warehouse_max_top2_swaps', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_WAREHOUSE_MAX_TOP2_SWAPS', 0)),
                        help='Maximum evidence warehouse replacements allowed inside the top-2 window.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_warehouse_protect_top_k', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_WAREHOUSE_PROTECT_TOP_K', 2)),
                        help='Protect baseline documents in final ranks <= this value from evidence warehouse replacement.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_warehouse_max_original_rank', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_WAREHOUSE_MAX_ORIGINAL_RANK', 220)),
                        help='Maximum original rank eligible for evidence warehouse optimization; <=0 disables the bound.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_warehouse_min_net_benefit', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_WAREHOUSE_MIN_NET_BENEFIT', 0.035)),
                        help='Minimum net benefit required for non-top-2 evidence warehouse actions.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_warehouse_top2_min_net_benefit', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_WAREHOUSE_TOP2_MIN_NET_BENEFIT', 0.16)),
                        help='Minimum net benefit required before the evidence warehouse optimizer can change top-2.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_warehouse_min_benefit', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_WAREHOUSE_MIN_BENEFIT', 0.52)),
                        help='Minimum raw utility benefit required before a warehouse row is eligible.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_warehouse_require_answer_signal', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_WAREHOUSE_REQUIRE_ANSWER_SIGNAL', 'true'),
                        help='If true, warehouse rows must have text/path/endpoint evidence related to the expected answer.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_warehouse_allow_relation_only', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_WAREHOUSE_ALLOW_RELATION_ONLY', 'false'),
                        help='If true, relation-only warehouse rows may be selected.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_warehouse_cost_weight', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_WAREHOUSE_COST_WEIGHT', 0.56)),
                        help='Weight for replacement/displacement cost in the warehouse optimizer.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_warehouse_risk_weight', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_WAREHOUSE_RISK_WEIGHT', 0.36)),
                        help='Weight for hard-negative risk in the warehouse optimizer.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_warehouse_text_weight', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_WAREHOUSE_TEXT_WEIGHT', 0.20)),
                        help='Weight for answer-aware text support in warehouse row benefit.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_warehouse_path_weight', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_WAREHOUSE_PATH_WEIGHT', 0.16)),
                        help='Weight for path-aware evidence support in warehouse row benefit.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_warehouse_slot_weight', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_WAREHOUSE_SLOT_WEIGHT', 0.22)),
                        help='Weight for missing/novel schema slot coverage in warehouse row benefit.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_warehouse_source_weight', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_WAREHOUSE_SOURCE_WEIGHT', 0.18)),
                        help='Weight for source/provenance reliability in warehouse row benefit.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_warehouse_selectivity_model_path', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_WAREHOUSE_SELECTIVITY_MODEL_PATH', ''),
                        help='Optional learned selectivity model for final-rank evidence warehouse actions.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_warehouse_selectivity_min_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_WAREHOUSE_SELECTIVITY_MIN_SCORE', 0.54)),
                        help='Minimum learned selectivity score required for top-5 warehouse actions.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_warehouse_selectivity_top2_min_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_WAREHOUSE_SELECTIVITY_TOP2_MIN_SCORE', 0.68)),
                        help='Minimum learned selectivity score required for top-2 warehouse actions.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_warehouse_selectivity_score_weight', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_WAREHOUSE_SELECTIVITY_SCORE_WEIGHT', 0.18)),
                        help='Weight used to add learned selectivity margin into warehouse net-benefit ranking.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_warehouse_selectivity_enable_recall_bypass', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_WAREHOUSE_SELECTIVITY_ENABLE_RECALL_BYPASS', 'false'),
                        help='If true, allow learned selectivity to re-admit high-signal candidates rejected by conservative warehouse row rules.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_warehouse_selectivity_bypass_datasets', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_WAREHOUSE_SELECTIVITY_BYPASS_DATASETS', 'musique,nq_rear'),
                        help='Comma-separated datasets where learned recall bypass is allowed.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_warehouse_selectivity_bypass_reasons', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_WAREHOUSE_SELECTIVITY_BYPASS_REASONS', 'low_benefit,missing_answer_signal,relation_only'),
                        help='Comma-separated warehouse row rejection reasons eligible for learned recall bypass.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_warehouse_selectivity_bypass_min_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_WAREHOUSE_SELECTIVITY_BYPASS_MIN_SCORE', 0.56)),
                        help='Minimum learned score required by the recall bypass gate.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_warehouse_selectivity_bypass_max_original_rank', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_WAREHOUSE_SELECTIVITY_BYPASS_MAX_ORIGINAL_RANK', 18)),
                        help='Maximum original rank eligible for learned recall bypass; <=0 disables the rank cap.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_warehouse_selectivity_bypass_min_confidence', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_WAREHOUSE_SELECTIVITY_BYPASS_MIN_CONFIDENCE', 0.88)),
                        help='Minimum candidate confidence required for learned recall bypass.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_warehouse_selectivity_bypass_min_text_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_WAREHOUSE_SELECTIVITY_BYPASS_MIN_TEXT_SCORE', 0.18)),
                        help='Minimum text support score for weak non-structural learned recall bypass candidates.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_warehouse_selectivity_bypass_max_risk', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_WAREHOUSE_SELECTIVITY_BYPASS_MAX_RISK', 0.64)),
                        help='Maximum warehouse row risk allowed for learned recall bypass.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_warehouse_selectivity_bypass_score_relaxation', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_WAREHOUSE_SELECTIVITY_BYPASS_SCORE_RELAXATION', 0.08)),
                        help='Amount by which recall bypass may relax top-5 learned selectivity thresholds.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_warehouse_selectivity_bypass_net_bonus', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_WAREHOUSE_SELECTIVITY_BYPASS_NET_BONUS', 0.12)),
                        help='Controlled net-benefit bonus for recall-bypass candidates that pass learned selectivity.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_warehouse_selectivity_enable_top2_guard_bypass', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_WAREHOUSE_SELECTIVITY_ENABLE_TOP2_GUARD_BYPASS', 'false'),
                        help='If true, let a high-confidence learned replay model override selected conservative top-2 warehouse guards.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_warehouse_selectivity_top2_guard_bypass_min_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_WAREHOUSE_SELECTIVITY_TOP2_GUARD_BYPASS_MIN_SCORE', 0.86)),
                        help='Minimum learned replay score required to override bypassable top-2 warehouse guards.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_warehouse_selectivity_top2_guard_bypass_max_risk', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_WAREHOUSE_SELECTIVITY_TOP2_GUARD_BYPASS_MAX_RISK', 0.16)),
                        help='Maximum warehouse row risk allowed for learned top-2 guard bypass.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_warehouse_enable_action_counterfactual_gate', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_WAREHOUSE_ENABLE_ACTION_COUNTERFACTUAL_GATE', 'false'),
                        help='If true, apply metric-aligned counterfactual gates to final-rank warehouse swap actions.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_warehouse_action_gate_top2_min_text_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_WAREHOUSE_ACTION_GATE_TOP2_MIN_TEXT_SCORE', 0.52)),
                        help='Minimum text score used by weak top-2 high-replacement warehouse action gates.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_warehouse_action_gate_top2_min_selectivity_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_WAREHOUSE_ACTION_GATE_TOP2_MIN_SELECTIVITY_SCORE', 0.98)),
                        help='Minimum learned selectivity score used by weak top-2 high-replacement warehouse action gates.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_warehouse_action_gate_top5_min_text_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_WAREHOUSE_ACTION_GATE_TOP5_MIN_TEXT_SCORE', 0.43)),
                        help='Minimum text score used by weak top-5 high-replacement warehouse action gates.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_warehouse_action_gate_top5_min_selectivity_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_WAREHOUSE_ACTION_GATE_TOP5_MIN_SELECTIVITY_SCORE', 0.78)),
                        help='Minimum learned selectivity score used by weak top-5 high-replacement warehouse action gates.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_warehouse_action_gate_nq_top5_min_rule_net', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_WAREHOUSE_ACTION_GATE_NQ_TOP5_MIN_RULE_NET', 0.0)),
                        help='Minimum rule net benefit for NQ-Rear top-5 high-replacement warehouse actions.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_warehouse_enable_set_coverage_cbo', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_WAREHOUSE_ENABLE_SET_COVERAGE_CBO', 'false'),
                        help='If true, add evidence-set slot coverage gain into final-rank warehouse CBO top-5 admission.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_warehouse_set_coverage_weight', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_WAREHOUSE_SET_COVERAGE_WEIGHT', 0.20)),
                        help='Net-benefit bonus weight for evidence-set slot coverage gain in final-rank warehouse CBO.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_warehouse_set_coverage_min_gain', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_WAREHOUSE_SET_COVERAGE_MIN_GAIN', 0.08)),
                        help='Minimum evidence-set coverage gain needed for the set-coverage CBO bypass.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_warehouse_set_coverage_max_risk', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_WAREHOUSE_SET_COVERAGE_MAX_RISK', 0.56)),
                        help='Maximum warehouse risk allowed for the set-coverage CBO top-5 bypass.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_warehouse_answer_aware_mode', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_WAREHOUSE_ANSWER_AWARE_MODE', 'off'),
                        help='Answer-aware cost model mode for final-rank warehouse actions: off, prior, model, or auto.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_warehouse_answer_aware_model_path', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_WAREHOUSE_ANSWER_AWARE_MODEL_PATH', ''),
                        help='Optional learned answer-aware cost model path for final-rank warehouse actions.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_warehouse_answer_aware_min_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_WAREHOUSE_ANSWER_AWARE_MIN_SCORE', 0.58)),
                        help='Minimum answer-aware score required for top-5 warehouse actions.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_warehouse_answer_aware_top2_min_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_WAREHOUSE_ANSWER_AWARE_TOP2_MIN_SCORE', 0.72)),
                        help='Minimum answer-aware score required for top-2 warehouse actions.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_warehouse_answer_aware_score_weight', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_WAREHOUSE_ANSWER_AWARE_SCORE_WEIGHT', 0.24)),
                        help='Weight used to add answer-aware score margin into warehouse net-benefit ranking.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_warehouse_answer_aware_require_direct_signal', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_WAREHOUSE_ANSWER_AWARE_REQUIRE_DIRECT_SIGNAL', 'false'),
                        help='If true, answer-aware warehouse actions require direct answer/terminal signal unless path-grounded.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_warehouse_answer_aware_allow_top2_guard_bypass', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_WAREHOUSE_ANSWER_AWARE_ALLOW_TOP2_GUARD_BYPASS', 'false'),
                        help='If true, high answer-aware score may bypass selected conservative top-2 warehouse guards.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_warehouse_answer_aware_top2_guard_bypass_min_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_WAREHOUSE_ANSWER_AWARE_TOP2_GUARD_BYPASS_MIN_SCORE', 0.86)),
                        help='Minimum answer-aware score required to bypass selected top-2 warehouse guards.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_warehouse_answer_aware_top2_guard_bypass_max_risk', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_WAREHOUSE_ANSWER_AWARE_TOP2_GUARD_BYPASS_MAX_RISK', 0.18)),
                        help='Maximum warehouse risk allowed for answer-aware top-2 guard bypass.')
    parser.add_argument('--query_plan_ontology_evidence_set_enable_top5_to_top2_promotion', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_ENABLE_TOP5_TO_TOP2_PROMOTION', 'false'),
                        help='If true, reorder high-confidence ontology-supported top-5 passages into rank 2.')
    parser.add_argument('--query_plan_ontology_evidence_set_top5_to_top2_promotion_target_datasets', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_TOP5_TO_TOP2_PROMOTION_TARGET_DATASETS', ''),
                        help='Comma-separated dataset allowlist for top5-to-top2 promotion; empty enables all datasets.')
    parser.add_argument('--query_plan_ontology_evidence_set_top5_to_top2_promotion_min_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_TOP5_TO_TOP2_PROMOTION_MIN_SCORE', 0.62)),
                        help='Minimum support score required before a top-5 evidence passage can be promoted to rank 2.')
    parser.add_argument('--query_plan_ontology_evidence_set_top5_to_top2_promotion_min_margin', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_TOP5_TO_TOP2_PROMOTION_MIN_MARGIN', 0.055)),
                        help='Minimum support-score margin over the current rank-2 passage before promotion.')
    parser.add_argument('--query_plan_ontology_evidence_set_top5_to_top2_promotion_max_risk', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_TOP5_TO_TOP2_PROMOTION_MAX_RISK', 0.48)),
                        help='Maximum warehouse risk allowed for rank-2 evidence promotion.')
    parser.add_argument('--query_plan_ontology_evidence_set_top5_to_top2_promotion_min_text_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_TOP5_TO_TOP2_PROMOTION_MIN_TEXT_SCORE', 0.06)),
                        help='Minimum text score that can satisfy the promotion support gate.')
    parser.add_argument('--query_plan_ontology_evidence_set_top5_to_top2_promotion_min_path_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_TOP5_TO_TOP2_PROMOTION_MIN_PATH_SCORE', 0.48)),
                        help='Minimum path-aware score that can satisfy the promotion support gate.')
    parser.add_argument('--query_plan_ontology_evidence_set_top5_to_top2_promotion_min_source_precision', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_TOP5_TO_TOP2_PROMOTION_MIN_SOURCE_PRECISION', 0.02)),
                        help='Minimum source precision that can satisfy the promotion support gate.')
    parser.add_argument('--query_plan_ontology_evidence_set_top5_to_top2_promotion_min_slot_value', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_TOP5_TO_TOP2_PROMOTION_MIN_SLOT_VALUE', 0.04)),
                        help='Minimum query-slot value that can satisfy the promotion support gate.')
    parser.add_argument('--query_plan_ontology_evidence_set_top5_to_top2_promotion_require_ontology_signal', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_TOP5_TO_TOP2_PROMOTION_REQUIRE_ONTOLOGY_SIGNAL', 'true'),
                        help='If true, promoted passages must have ontology-derived evidence contexts.')
    parser.add_argument('--query_plan_ontology_evidence_set_top5_to_top2_promotion_max_promotions', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_TOP5_TO_TOP2_PROMOTION_MAX_PROMOTIONS', 1)),
                        help='Maximum number of top5-to-top2 promotions per query.')
    parser.add_argument('--query_plan_ontology_evidence_set_enable_pairwise_top2_join', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_ENABLE_PAIRWISE_TOP2_JOIN', 'false'),
                        help='If true, apply pairwise evidence-join optimization that directly rewrites final top-2.')
    parser.add_argument('--query_plan_ontology_evidence_set_pairwise_top2_join_target_datasets', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_PAIRWISE_TOP2_JOIN_TARGET_DATASETS', ''),
                        help='Comma-separated dataset allowlist for pairwise top-2 evidence join; empty enables all datasets.')
    parser.add_argument('--query_plan_ontology_evidence_set_pairwise_top2_join_candidate_pool', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_PAIRWISE_TOP2_JOIN_CANDIDATE_POOL', 24)),
                        help='Maximum candidate documents considered by pairwise top-2 evidence join.')
    parser.add_argument('--query_plan_ontology_evidence_set_pairwise_top2_join_min_pair_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_PAIRWISE_TOP2_JOIN_MIN_PAIR_SCORE', 0.60)),
                        help='Minimum absolute pair utility score required by pairwise top-2 evidence join.')
    parser.add_argument('--query_plan_ontology_evidence_set_pairwise_top2_join_min_margin', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_PAIRWISE_TOP2_JOIN_MIN_MARGIN', 0.015)),
                        help='Minimum pair utility margin over current top-2.')
    parser.add_argument('--query_plan_ontology_evidence_set_pairwise_top2_join_min_text_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_PAIRWISE_TOP2_JOIN_MIN_TEXT_SCORE', 0.10)),
                        help='Minimum best-document text support score for pairwise top-2 join.')
    parser.add_argument('--query_plan_ontology_evidence_set_pairwise_top2_join_min_second_text_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_PAIRWISE_TOP2_JOIN_MIN_SECOND_TEXT_SCORE', 0.03)),
                        help='Minimum second-document text support score unless structural/path evidence exists.')
    parser.add_argument('--query_plan_ontology_evidence_set_pairwise_top2_join_min_coverage_gain', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_PAIRWISE_TOP2_JOIN_MIN_COVERAGE_GAIN', 0.08)),
                        help='Minimum pair slot coverage gain over current top-2.')
    parser.add_argument('--query_plan_ontology_evidence_set_pairwise_top2_join_max_original_rank', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_PAIRWISE_TOP2_JOIN_MAX_ORIGINAL_RANK', 140)),
                        help='Maximum original rank eligible for pairwise top-2 join; <=0 disables this bound.')
    parser.add_argument('--query_plan_ontology_evidence_set_pairwise_top2_join_protect_top1', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_PAIRWISE_TOP2_JOIN_PROTECT_TOP1', 'false'),
                        help='If true, pairwise top-2 join must keep the original rank-1 document.')
    parser.add_argument('--query_plan_ontology_evidence_set_pairwise_top2_join_require_answer_signal', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_PAIRWISE_TOP2_JOIN_REQUIRE_ANSWER_SIGNAL', 'false'),
                        help='If true, selected pair must have answer/terminal/path evidence.')
    parser.add_argument('--query_plan_ontology_evidence_set_pairwise_top2_join_max_swaps', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_PAIRWISE_TOP2_JOIN_MAX_SWAPS', 1)),
                        help='Maximum number of docs outside current top-2 allowed in a pairwise rewrite.')
    parser.add_argument('--query_plan_ontology_evidence_set_enable_decomposed_chain_join', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_ENABLE_DECOMPOSED_CHAIN_JOIN', 'false'),
                        help='If true, decompose the question into ontology slots and optimize a two-document evidence chain for top-2.')
    parser.add_argument('--query_plan_ontology_evidence_set_decomposed_chain_join_target_datasets', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_DECOMPOSED_CHAIN_JOIN_TARGET_DATASETS', ''),
                        help='Comma-separated dataset allowlist for decomposed chain join; empty enables all datasets.')
    parser.add_argument('--query_plan_ontology_evidence_set_decomposed_chain_join_candidate_pool', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_DECOMPOSED_CHAIN_JOIN_CANDIDATE_POOL', 36)),
                        help='Maximum candidate documents considered by decomposed chain join.')
    parser.add_argument('--query_plan_ontology_evidence_set_decomposed_chain_join_min_chain_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_DECOMPOSED_CHAIN_JOIN_MIN_CHAIN_SCORE', 0.58)),
                        help='Minimum absolute chain utility score required before decomposed chain join rewrites top-2.')
    parser.add_argument('--query_plan_ontology_evidence_set_decomposed_chain_join_min_margin', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_DECOMPOSED_CHAIN_JOIN_MIN_MARGIN', 0.030)),
                        help='Minimum chain-score margin over the current top-2 required by decomposed chain join.')
    parser.add_argument('--query_plan_ontology_evidence_set_decomposed_chain_join_min_coverage_gain', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_DECOMPOSED_CHAIN_JOIN_MIN_COVERAGE_GAIN', 0.06)),
                        help='Minimum ontology slot coverage gain required by decomposed chain join.')
    parser.add_argument('--query_plan_ontology_evidence_set_decomposed_chain_join_max_original_rank', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_DECOMPOSED_CHAIN_JOIN_MAX_ORIGINAL_RANK', 180)),
                        help='Maximum original rank eligible for decomposed chain join; <=0 disables this bound.')
    parser.add_argument('--query_plan_ontology_evidence_set_decomposed_chain_join_max_swaps', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_DECOMPOSED_CHAIN_JOIN_MAX_SWAPS', 2)),
                        help='Maximum number of docs outside current top-2 allowed in a decomposed chain rewrite.')
    parser.add_argument('--query_plan_ontology_evidence_set_decomposed_chain_join_protect_top1', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_DECOMPOSED_CHAIN_JOIN_PROTECT_TOP1', 'false'),
                        help='If true, decomposed chain join must keep the original rank-1 document.')
    parser.add_argument('--query_plan_ontology_evidence_set_decomposed_chain_join_enable_decomposition', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_DECOMPOSED_CHAIN_JOIN_ENABLE_DECOMPOSITION', 'true'),
                        help='Ablation switch: use ontology/query slot decomposition in decomposed chain join.')
    parser.add_argument('--query_plan_ontology_evidence_set_decomposed_chain_join_enable_subquery_access', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_DECOMPOSED_CHAIN_JOIN_ENABLE_SUBQUERY_ACCESS', 'true'),
                        help='Ablation switch: use role-aware subquery access-path scoring in decomposed chain join.')
    parser.add_argument('--query_plan_ontology_evidence_set_decomposed_chain_join_enable_chain_join', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_DECOMPOSED_CHAIN_JOIN_ENABLE_CHAIN_JOIN', 'true'),
                        help='Ablation switch: use pairwise evidence-chain join instead of single-document support.')
    parser.add_argument('--query_plan_ontology_evidence_set_decomposed_chain_join_require_bridge_signal', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_DECOMPOSED_CHAIN_JOIN_REQUIRE_BRIDGE_SIGNAL', 'true'),
                        help='If true, decomposed chain join requires anchor/bridge/answer role complementarity or bridge evidence.')
    parser.add_argument('--query_plan_ontology_evidence_set_decomposed_chain_join_top2_target_datasets', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_DECOMPOSED_CHAIN_JOIN_TOP2_TARGET_DATASETS', ''),
                        help='Comma-separated dataset allowlist where decomposed chain join may rewrite top-2; empty enables all targeted datasets.')
    parser.add_argument('--query_plan_ontology_evidence_set_decomposed_chain_join_enable_top5_fallback', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_DECOMPOSED_CHAIN_JOIN_ENABLE_TOP5_FALLBACK', 'false'),
                        help='If true, safe-router-rejected decomposed chains may enter top-5/context without changing top-2.')
    parser.add_argument('--query_plan_ontology_evidence_set_decomposed_chain_join_top5_fallback_target_datasets', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_DECOMPOSED_CHAIN_JOIN_TOP5_FALLBACK_TARGET_DATASETS', ''),
                        help='Comma-separated dataset allowlist where decomposed chain join may use top-5 fallback; empty enables all targeted datasets.')
    parser.add_argument('--query_plan_ontology_evidence_set_decomposed_chain_join_top5_require_answer_signal', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_DECOMPOSED_CHAIN_JOIN_TOP5_REQUIRE_ANSWER_SIGNAL', 'false'),
                        help='If true, top-5 fallback requires answer/terminal evidence in the selected chain.')
    parser.add_argument('--query_plan_ontology_evidence_set_decomposed_chain_join_top5_min_chain_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_DECOMPOSED_CHAIN_JOIN_TOP5_MIN_CHAIN_SCORE', 0.0)),
                        help='Minimum chain utility required by the top-5 fallback gate.')
    parser.add_argument('--query_plan_ontology_evidence_set_decomposed_chain_join_top5_min_net_margin', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_DECOMPOSED_CHAIN_JOIN_TOP5_MIN_NET_MARGIN', -1.0)),
                        help='Minimum net margin required by the top-5 fallback gate.')
    parser.add_argument('--query_plan_ontology_evidence_set_decomposed_chain_join_top5_min_text_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_DECOMPOSED_CHAIN_JOIN_TOP5_MIN_TEXT_SCORE', 0.0)),
                        help='Minimum best document text score required by the top-5 fallback gate.')
    parser.add_argument('--query_plan_ontology_evidence_set_decomposed_chain_join_top5_max_risk', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_DECOMPOSED_CHAIN_JOIN_TOP5_MAX_RISK', 1.0)),
                        help='Maximum displacement/noise risk allowed by the top-5 fallback gate.')
    parser.add_argument('--query_plan_ontology_evidence_set_decomposed_chain_join_top2_min_chain_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_DECOMPOSED_CHAIN_JOIN_TOP2_MIN_CHAIN_SCORE', 0.58)),
                        help='Minimum chain utility required by the safe top-2 router.')
    parser.add_argument('--query_plan_ontology_evidence_set_decomposed_chain_join_top2_min_margin', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_DECOMPOSED_CHAIN_JOIN_TOP2_MIN_MARGIN', 0.030)),
                        help='Minimum chain margin over current top-2 required by the safe top-2 router.')
    parser.add_argument('--query_plan_ontology_evidence_set_decomposed_chain_join_top2_min_net_margin', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_DECOMPOSED_CHAIN_JOIN_TOP2_MIN_NET_MARGIN', 0.030)),
                        help='Minimum net margin after displacement risk required by the safe top-2 router.')
    parser.add_argument('--query_plan_ontology_evidence_set_decomposed_chain_join_top2_min_text_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_DECOMPOSED_CHAIN_JOIN_TOP2_MIN_TEXT_SCORE', 0.0)),
                        help='Minimum best document text score required by the safe top-2 router.')
    parser.add_argument('--query_plan_ontology_evidence_set_decomposed_chain_join_top2_min_second_text_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_DECOMPOSED_CHAIN_JOIN_TOP2_MIN_SECOND_TEXT_SCORE', 0.0)),
                        help='Minimum weaker document text score required by the safe top-2 router.')
    parser.add_argument('--query_plan_ontology_evidence_set_decomposed_chain_join_top2_min_source_precision', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_DECOMPOSED_CHAIN_JOIN_TOP2_MIN_SOURCE_PRECISION', 0.0)),
                        help='Minimum evidence-source precision required by the safe top-2 router.')
    parser.add_argument('--query_plan_ontology_evidence_set_decomposed_chain_join_top2_max_risk', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_DECOMPOSED_CHAIN_JOIN_TOP2_MAX_RISK', 1.0)),
                        help='Maximum displacement/noise risk allowed by the safe top-2 router.')
    parser.add_argument('--query_plan_ontology_evidence_set_decomposed_chain_join_top2_require_answer_signal', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_DECOMPOSED_CHAIN_JOIN_TOP2_REQUIRE_ANSWER_SIGNAL', 'false'),
                        help='If true, safe top-2 routing requires answer/terminal evidence in the selected chain.')
    parser.add_argument('--query_plan_ontology_evidence_set_decomposed_chain_join_top2_require_non_simple', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_DECOMPOSED_CHAIN_JOIN_TOP2_REQUIRE_NON_SIMPLE', 'false'),
                        help='If true, safe top-2 routing is disabled for simple lookup/entity operators.')
    parser.add_argument('--query_plan_ontology_evidence_set_decomposed_chain_join_enable_slot_completion_promotion', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_DECOMPOSED_CHAIN_JOIN_ENABLE_SLOT_COMPLETION_PROMOTION', 'false'),
                        help='If true, allow slot-completion chains to bypass conservative top-2 margin gates on targeted partial-chain datasets.')
    parser.add_argument('--query_plan_ontology_evidence_set_decomposed_chain_join_slot_completion_target_datasets', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_DECOMPOSED_CHAIN_JOIN_SLOT_COMPLETION_TARGET_DATASETS', 'musique,nq_rear,popqa'),
                        help='Comma-separated dataset allowlist for slot-completion promotion.')
    parser.add_argument('--query_plan_ontology_evidence_set_decomposed_chain_join_slot_completion_min_chain_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_DECOMPOSED_CHAIN_JOIN_SLOT_COMPLETION_MIN_CHAIN_SCORE', 0.46)),
                        help='Minimum chain score required by slot-completion promotion.')
    parser.add_argument('--query_plan_ontology_evidence_set_decomposed_chain_join_slot_completion_min_required_gain', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_DECOMPOSED_CHAIN_JOIN_SLOT_COMPLETION_MIN_REQUIRED_GAIN', 0.12)),
                        help='Minimum required-slot coverage gain required by slot-completion promotion.')
    parser.add_argument('--query_plan_ontology_evidence_set_decomposed_chain_join_slot_completion_min_coverage_gain', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_DECOMPOSED_CHAIN_JOIN_SLOT_COMPLETION_MIN_COVERAGE_GAIN', 0.05)),
                        help='Minimum ontology slot-value coverage gain required by slot-completion promotion.')
    parser.add_argument('--query_plan_ontology_evidence_set_decomposed_chain_join_slot_completion_min_role_gain', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_DECOMPOSED_CHAIN_JOIN_SLOT_COMPLETION_MIN_ROLE_GAIN', 0.0)),
                        help='Minimum anchor/bridge/answer role gain required by slot-completion promotion when positive.')
    parser.add_argument('--query_plan_ontology_evidence_set_decomposed_chain_join_slot_completion_max_current_coverage', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_DECOMPOSED_CHAIN_JOIN_SLOT_COMPLETION_MAX_CURRENT_COVERAGE', 0.82)),
                        help='Maximum current top-2 required-slot coverage eligible for slot-completion promotion.')
    parser.add_argument('--query_plan_ontology_evidence_set_decomposed_chain_join_slot_completion_min_text_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_DECOMPOSED_CHAIN_JOIN_SLOT_COMPLETION_MIN_TEXT_SCORE', 0.03)),
                        help='Minimum best-document text score required by slot-completion promotion.')
    parser.add_argument('--query_plan_ontology_evidence_set_decomposed_chain_join_slot_completion_min_second_text_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_DECOMPOSED_CHAIN_JOIN_SLOT_COMPLETION_MIN_SECOND_TEXT_SCORE', 0.0)),
                        help='Minimum weaker-document text score required by slot-completion promotion.')
    parser.add_argument('--query_plan_ontology_evidence_set_decomposed_chain_join_slot_completion_max_risk', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_DECOMPOSED_CHAIN_JOIN_SLOT_COMPLETION_MAX_RISK', 0.72)),
                        help='Maximum risk allowed by slot-completion promotion.')
    parser.add_argument('--query_plan_ontology_evidence_set_decomposed_chain_join_slot_completion_allow_top1_replacement', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_DECOMPOSED_CHAIN_JOIN_SLOT_COMPLETION_ALLOW_TOP1_REPLACEMENT', 'false'),
                        help='If true, slot-completion promotion may replace the original rank-1 document.')
    parser.add_argument('--query_plan_ontology_evidence_set_decomposed_chain_join_slot_completion_require_bridge_or_answer', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_DECOMPOSED_CHAIN_JOIN_SLOT_COMPLETION_REQUIRE_BRIDGE_OR_ANSWER', 'true'),
                        help='If true, slot-completion promotion requires bridge or answer role evidence.')
    parser.add_argument('--query_plan_ontology_evidence_set_decomposed_chain_join_enable_metric_regret_gate', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_DECOMPOSED_CHAIN_JOIN_ENABLE_METRIC_REGRET_GATE', 'false'),
                        help='If true, metric-aligned regret gate must approve before slot-completion chains can alter top-2.')
    parser.add_argument('--query_plan_ontology_evidence_set_decomposed_chain_join_metric_regret_target_datasets', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_DECOMPOSED_CHAIN_JOIN_METRIC_REGRET_TARGET_DATASETS', 'musique,nq_rear,popqa'),
                        help='Comma-separated dataset allowlist for decomposed-chain metric-regret gating.')
    parser.add_argument('--query_plan_ontology_evidence_set_decomposed_chain_join_metric_regret_protect_top1', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_DECOMPOSED_CHAIN_JOIN_METRIC_REGRET_PROTECT_TOP1', 'true'),
                        help='If true, metric-regret gating protects original rank-1 unless an exceptional gain profile is observed.')
    parser.add_argument('--query_plan_ontology_evidence_set_decomposed_chain_join_metric_regret_allow_top1_min_net_margin', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_DECOMPOSED_CHAIN_JOIN_METRIC_REGRET_ALLOW_TOP1_MIN_NET_MARGIN', 0.55)),
                        help='Minimum net margin required before metric-regret gating allows rank-1 replacement.')
    parser.add_argument('--query_plan_ontology_evidence_set_decomposed_chain_join_metric_regret_allow_top1_min_required_gain', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_DECOMPOSED_CHAIN_JOIN_METRIC_REGRET_ALLOW_TOP1_MIN_REQUIRED_GAIN', 0.85)),
                        help='Minimum required-slot gain required before metric-regret gating allows rank-1 replacement.')
    parser.add_argument('--query_plan_ontology_evidence_set_decomposed_chain_join_metric_regret_allow_top1_min_text_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_DECOMPOSED_CHAIN_JOIN_METRIC_REGRET_ALLOW_TOP1_MIN_TEXT_SCORE', 0.55)),
                        help='Minimum weaker-document text score required before metric-regret gating allows rank-1 replacement.')
    parser.add_argument('--query_plan_ontology_evidence_set_decomposed_chain_join_metric_regret_allow_top1_min_source_precision', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_DECOMPOSED_CHAIN_JOIN_METRIC_REGRET_ALLOW_TOP1_MIN_SOURCE_PRECISION', 0.80)),
                        help='Minimum source precision required before metric-regret gating allows rank-1 replacement.')
    parser.add_argument('--query_plan_ontology_evidence_set_decomposed_chain_join_metric_regret_allow_top1_max_risk', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_DECOMPOSED_CHAIN_JOIN_METRIC_REGRET_ALLOW_TOP1_MAX_RISK', 0.08)),
                        help='Maximum risk allowed before metric-regret gating allows rank-1 replacement.')
    parser.add_argument('--query_plan_ontology_evidence_set_decomposed_chain_join_metric_regret_rank2_min_net_margin', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_DECOMPOSED_CHAIN_JOIN_METRIC_REGRET_RANK2_MIN_NET_MARGIN', 0.08)),
                        help='Minimum net margin required for rank-2-only decomposed-chain promotion under metric-regret gating.')
    parser.add_argument('--query_plan_ontology_evidence_set_decomposed_chain_join_metric_regret_rank2_min_required_gain', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_DECOMPOSED_CHAIN_JOIN_METRIC_REGRET_RANK2_MIN_REQUIRED_GAIN', 0.12)),
                        help='Minimum required-slot gain required for rank-2-only decomposed-chain promotion under metric-regret gating.')
    parser.add_argument('--query_plan_ontology_evidence_set_decomposed_chain_join_metric_regret_rank2_min_text_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_DECOMPOSED_CHAIN_JOIN_METRIC_REGRET_RANK2_MIN_TEXT_SCORE', 0.16)),
                        help='Minimum best-document text score required for rank-2-only decomposed-chain promotion under metric-regret gating.')
    parser.add_argument('--query_plan_ontology_evidence_set_decomposed_chain_join_metric_regret_enable_top5_append', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_DECOMPOSED_CHAIN_JOIN_METRIC_REGRET_ENABLE_TOP5_APPEND', 'true'),
                        help='If true, metric-regret vetoed top-2 actions may still append supporting evidence to top-5/context.')
    parser.add_argument('--query_plan_ontology_evidence_set_enable_regret_calibrated_set_optimizer', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_ENABLE_REGRET_CALIBRATED_SET_OPTIMIZER', 'false'),
                        help='If true, apply a regret-calibrated set optimizer that repairs TopK evidence gaps under displacement cost.')
    parser.add_argument('--query_plan_ontology_evidence_set_regret_calibrated_set_target_datasets', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_REGRET_CALIBRATED_SET_TARGET_DATASETS', ''),
                        help='Comma-separated dataset allowlist for regret-calibrated set optimization; empty enables all datasets.')
    parser.add_argument('--query_plan_ontology_evidence_set_regret_calibrated_set_max_candidates', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_REGRET_CALIBRATED_SET_MAX_CANDIDATES', 160)),
                        help='Maximum candidate rows considered by the regret-calibrated set optimizer per query.')
    parser.add_argument('--query_plan_ontology_evidence_set_regret_calibrated_set_max_top2_swaps', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_REGRET_CALIBRATED_SET_MAX_TOP2_SWAPS', 1)),
                        help='Maximum top-2 replacements allowed by the regret-calibrated set optimizer.')
    parser.add_argument('--query_plan_ontology_evidence_set_regret_calibrated_set_max_top5_swaps', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_REGRET_CALIBRATED_SET_MAX_TOP5_SWAPS', 1)),
                        help='Maximum top-5 replacements allowed by the regret-calibrated set optimizer.')
    parser.add_argument('--query_plan_ontology_evidence_set_regret_calibrated_set_protect_top1', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_REGRET_CALIBRATED_SET_PROTECT_TOP1', 'true'),
                        help='If true, the regret-calibrated optimizer cannot replace the current rank-1 passage.')
    parser.add_argument('--query_plan_ontology_evidence_set_regret_calibrated_set_max_original_rank', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_REGRET_CALIBRATED_SET_MAX_ORIGINAL_RANK', 220)),
                        help='Maximum original rank eligible for regret-calibrated set optimization; <=0 disables this bound.')
    parser.add_argument('--query_plan_ontology_evidence_set_regret_calibrated_set_min_regret', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_REGRET_CALIBRATED_SET_MIN_REGRET', 0.045)),
                        help='Minimum regret margin required for non-top-2 set repair.')
    parser.add_argument('--query_plan_ontology_evidence_set_regret_calibrated_set_top2_min_regret', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_REGRET_CALIBRATED_SET_TOP2_MIN_REGRET', 0.165)),
                        help='Minimum regret margin required before changing top-2.')
    parser.add_argument('--query_plan_ontology_evidence_set_regret_calibrated_set_min_text_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_REGRET_CALIBRATED_SET_MIN_TEXT_SCORE', 0.08)),
                        help='Minimum text score or alternative structural/source support required by regret-calibrated set repair.')
    parser.add_argument('--query_plan_ontology_evidence_set_regret_calibrated_set_top2_min_text_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_REGRET_CALIBRATED_SET_TOP2_MIN_TEXT_SCORE', 0.38)),
                        help='Minimum text score for top-2 set repair unless strong path/source/structural answer support exists.')
    parser.add_argument('--query_plan_ontology_evidence_set_regret_calibrated_set_min_source_precision', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_REGRET_CALIBRATED_SET_MIN_SOURCE_PRECISION', 0.02)),
                        help='Minimum source precision used as an alternative support signal for regret-calibrated set repair.')
    parser.add_argument('--query_plan_ontology_evidence_set_regret_calibrated_set_require_answer_signal', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_REGRET_CALIBRATED_SET_REQUIRE_ANSWER_SIGNAL', 'false'),
                        help='If true, regret-calibrated set repair requires answer/terminal/path evidence.')
    parser.add_argument('--query_plan_ontology_evidence_set_regret_calibrated_set_cost_weight', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_REGRET_CALIBRATED_SET_COST_WEIGHT', 0.50)),
                        help='Weight for displacement cost in regret-calibrated set repair.')
    parser.add_argument('--query_plan_ontology_evidence_set_regret_calibrated_set_risk_weight', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_REGRET_CALIBRATED_SET_RISK_WEIGHT', 0.42)),
                        help='Weight for candidate risk in regret-calibrated set repair.')
    parser.add_argument('--query_plan_ontology_evidence_set_regret_calibrated_set_gap_weight', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_REGRET_CALIBRATED_SET_GAP_WEIGHT', 0.24)),
                        help='Weight for query-level gap pressure in regret-calibrated set repair.')
    parser.add_argument('--query_plan_ontology_evidence_set_regret_calibrated_set_slot_weight', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_REGRET_CALIBRATED_SET_SLOT_WEIGHT', 0.18)),
                        help='Weight for missing-slot repair value in regret-calibrated set repair.')
    parser.add_argument('--query_plan_ontology_evidence_set_regret_calibrated_set_max_risk', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_REGRET_CALIBRATED_SET_MAX_RISK', 0.58)),
                        help='Maximum candidate risk allowed by regret-calibrated set repair.')
    parser.add_argument('--query_plan_ontology_evidence_set_regret_calibrated_set_top2_max_risk', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_REGRET_CALIBRATED_SET_TOP2_MAX_RISK', 0.22)),
                        help='Maximum candidate risk allowed when changing top-2.')
    parser.add_argument('--query_plan_ontology_evidence_set_regret_calibrated_set_min_gap_pressure', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_REGRET_CALIBRATED_SET_MIN_GAP_PRESSURE', 0.04)),
                        help='Minimum query pressure required for weak top-5 repairs without answer signal or slot repair.')
    parser.add_argument('--query_plan_ontology_evidence_set_regret_calibrated_set_enable_top5_safe_bypass', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_REGRET_CALIBRATED_SET_ENABLE_TOP5_SAFE_BYPASS', 'false'),
                        help='If true, allow high-regret top-5 repairs to bypass conservative candidate risk gates when support signals are strong.')
    parser.add_argument('--query_plan_ontology_evidence_set_regret_calibrated_set_top5_safe_bypass_max_risk', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_REGRET_CALIBRATED_SET_TOP5_SAFE_BYPASS_MAX_RISK', 0.72)),
                        help='Maximum candidate risk allowed by the top-5 safe bypass.')
    parser.add_argument('--query_plan_ontology_evidence_set_regret_calibrated_set_top5_safe_bypass_min_support', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_REGRET_CALIBRATED_SET_TOP5_SAFE_BYPASS_MIN_SUPPORT', 0.30)),
                        help='Minimum support signal required by the top-5 safe bypass.')
    parser.add_argument('--query_plan_ontology_evidence_set_regret_calibrated_set_top5_safe_bypass_min_regret', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_REGRET_CALIBRATED_SET_TOP5_SAFE_BYPASS_MIN_REGRET', 0.075)),
                        help='Minimum regret required by the top-5 safe bypass after scoring a replacement option.')
    parser.add_argument('--query_plan_ontology_evidence_set_enable_evidence_sidecar', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_ENABLE_EVIDENCE_SIDECAR', 'false'),
                        help='If true, expose high-confidence ontology candidates as QA-only sidecar evidence without changing top-k retrieval order.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_sidecar_max_items', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_SIDECAR_MAX_ITEMS', 2)),
                        help='Maximum QA-only evidence sidecar items emitted per query.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_sidecar_min_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_SIDECAR_MIN_SCORE', 0.48)),
                        help='Minimum sidecar utility score required before a candidate is exposed to QA.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_sidecar_min_text_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_SIDECAR_MIN_TEXT_SCORE', 0.12)),
                        help='Minimum text support score required for weak non-structural sidecar candidates.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_sidecar_max_original_rank', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_SIDECAR_MAX_ORIGINAL_RANK', 320)),
                        help='Maximum original rank eligible for QA-only evidence sidecar; <=0 disables this bound.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_sidecar_allow_relation_only', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_SIDECAR_ALLOW_RELATION_ONLY', 'false'),
                        help='If true, relation-only candidates can be emitted as QA-only sidecar evidence.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_sidecar_preferred_sources', type=str,
                        default=os.environ.get(
                            'QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_SIDECAR_PREFERRED_SOURCES',
                            'evidence_grounded,support,path_executor,supervised_ranker,schema_slot_recall,rank_window,recall_plan,utility_planner,lifted',
                        ),
                        help='Comma-separated source labels that receive a sidecar utility bonus.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_sidecar_gate_mode', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_SIDECAR_GATE_MODE', 'loose'),
                        help='Sidecar gate mode: loose, strict, or complex_only_strict.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_sidecar_complex_datasets', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_SIDECAR_COMPLEX_DATASETS', '2wikimultihopqa,hotpotqa,musique'),
                        help='Comma-separated datasets routed to strict complex sidecar gating.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_sidecar_simple_datasets', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_SIDECAR_SIMPLE_DATASETS', 'popqa,nq_rear,narrativeqa_dev_10_doc,lveval'),
                        help='Comma-separated datasets routed to conservative simple sidecar gating.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_sidecar_strict_min_text_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_SIDECAR_STRICT_MIN_TEXT_SCORE', 0.45)),
                        help='Minimum text support score required by strict sidecar gating.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_sidecar_strict_min_path_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_SIDECAR_STRICT_MIN_PATH_SCORE', 0.68)),
                        help='Minimum path-aware score required by strict sidecar gating for path-derived sidecar items.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_sidecar_require_anchor_hit', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_SIDECAR_REQUIRE_ANCHOR_HIT', 'false'),
                        help='If true, strict sidecar gating requires an anchor text hit.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_sidecar_require_terminal_hit', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_SIDECAR_REQUIRE_TERMINAL_HIT', 'false'),
                        help='If true, strict sidecar gating requires a terminal or answer text hit.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_sidecar_require_preferred_source', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_SIDECAR_REQUIRE_PREFERRED_SOURCE', 'false'),
                        help='If true, strict sidecar gating only admits candidates from preferred source labels.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_sidecar_disable_simple_without_answer_overlap', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_SIDECAR_DISABLE_SIMPLE_WITHOUT_ANSWER_OVERLAP', 'false'),
                        help='If true, simple-query sidecar items require terminal or answer overlap.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_sidecar_disable_answer_type_conflict', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_SIDECAR_DISABLE_ANSWER_TYPE_CONFLICT', 'false'),
                        help='If true, strict sidecar gating rejects obvious expected-answer-type conflicts.')
    parser.add_argument('--query_plan_ontology_evidence_set_strict_backbone_scope', type=str, default='false',
                        help='If true, bare anchor-relation candidates are treated as weak evidence and must pass utility verification unless feedback-grounded.')
    parser.add_argument('--query_plan_ontology_evidence_set_anchor_relation_min_source_precision', type=float, default=0.30,
                        help='Minimum retrieval-feedback precision for an anchor-relation candidate to be treated as backbone evidence under strict backbone scope.')
    parser.add_argument('--query_plan_ontology_evidence_set_anchor_relation_min_selected_precision', type=float, default=0.05,
                        help='Minimum selected/gold@5 feedback precision for anchor-relation candidates to be treated as backbone evidence under strict backbone scope.')
    parser.add_argument('--query_plan_ontology_evidence_set_recall_selector_model_path', type=str, default='',
                        help='Optional learned recall-oriented evidence selector JSON path.')
    parser.add_argument('--query_plan_ontology_evidence_set_recall_selector_mode', type=str, default='off',
                        choices=['off', 'score', 'selector'],
                        help='How to use the recall-oriented evidence selector: off, score, or selector.')
    parser.add_argument('--query_plan_ontology_evidence_set_recall_selector_min_score', type=float, default=0.62,
                        help='Minimum recall-selector score required to promote an ontology candidate.')
    parser.add_argument('--query_plan_ontology_evidence_set_recall_selector_score_weight', type=float, default=0.55,
                        help='Score weight added to evidence-set selection from the recall selector.')
    parser.add_argument('--query_plan_ontology_evidence_set_recall_selector_promote_confidence', type=float, default=0.97,
                        help='Confidence floor assigned when the recall selector promotes a candidate.')
    parser.add_argument('--query_plan_ontology_evidence_set_recall_selector_counterfactual_weight', type=float, default=0.35,
                        help='Recall-selector score contribution to candidate replacement value in counterfactual selection.')
    parser.add_argument('--query_plan_ontology_evidence_set_recall_selector_bypass_router_min_score', type=float, default=0.80,
                        help='Recall-selector score required to override a query-level router skip decision.')
    parser.add_argument('--query_plan_ontology_evidence_set_recall_swapper_model_path', type=str, default='',
                        help='Optional learned oracle-gap recall swapper JSON path.')
    parser.add_argument('--query_plan_ontology_evidence_set_recall_swapper_mode', type=str, default='off',
                        choices=['off', 'score', 'selector'],
                        help='How to use the learned recall swapper: off, score, or selector.')
    parser.add_argument('--query_plan_ontology_evidence_set_recall_swapper_min_score', type=float, default=0.70,
                        help='Minimum recall-swapper score required to promote an ontology candidate.')
    parser.add_argument('--query_plan_ontology_evidence_set_recall_swapper_top2_min_score', type=float, default=0.88,
                        help='Recall-swapper score required before an ontology candidate can bypass top-2 protection.')
    parser.add_argument('--query_plan_ontology_evidence_set_recall_swapper_score_weight', type=float, default=1.10,
                        help='Score weight added to evidence-set selection from the recall swapper.')
    parser.add_argument('--query_plan_ontology_evidence_set_recall_swapper_promote_confidence', type=float, default=0.98,
                        help='Confidence floor assigned when the recall swapper promotes a candidate.')
    parser.add_argument('--query_plan_ontology_evidence_set_recall_swapper_counterfactual_weight', type=float, default=0.45,
                        help='Recall-swapper score contribution to candidate replacement value in counterfactual selection.')
    parser.add_argument('--query_plan_ontology_evidence_set_recall_swapper_max_top2_swaps', type=int, default=1,
                        help='Maximum recall-swapper bypass insertions allowed in top-2 per query.')
    parser.add_argument('--query_plan_ontology_evidence_set_recall_swapper_allow_relation_only_top2', type=str, default='false',
                        help='If true, allow high-scoring relation-only recall-swapper candidates into top-2.')
    parser.add_argument('--query_plan_ontology_evidence_set_rank2_rescue_model_path', type=str, default='',
                        help='Optional learned high-precision rank-2 rescue policy JSON path.')
    parser.add_argument('--query_plan_ontology_evidence_set_rank2_rescue_mode', type=str, default='off',
                        choices=['off', 'score', 'rescue', 'high_confidence'],
                        help='How to use the rank-2 rescue policy: off, score, rescue, or high_confidence.')
    parser.add_argument('--query_plan_ontology_evidence_set_rank2_rescue_min_score', type=float, default=0.95,
                        help='Minimum rank-2 rescue score required to replace baseline rank 2.')
    parser.add_argument('--query_plan_ontology_evidence_set_rank2_rescue_min_confidence', type=float, default=0.94,
                        help='Minimum candidate confidence required for rank-2 rescue.')
    parser.add_argument('--query_plan_ontology_evidence_set_rank2_rescue_max_original_rank', type=int, default=80,
                        help='Maximum original rank allowed for rank-2 rescue; <=0 disables this guard.')
    parser.add_argument('--query_plan_ontology_evidence_set_rank2_rescue_min_source_precision', type=float, default=0.20,
                        help='Minimum retrieval-feedback source precision required for rank-2 rescue.')
    parser.add_argument('--query_plan_ontology_evidence_set_rank2_rescue_min_selected_precision', type=float, default=0.05,
                        help='Minimum historical selected precision required for rank-2 rescue.')
    parser.add_argument('--query_plan_ontology_evidence_set_rank2_rescue_require_terminal_or_strict', type=str, default='true',
                        help='If true, require terminal/strict/multihop/path evidence for rank-2 rescue.')
    parser.add_argument('--query_plan_ontology_evidence_set_rank2_rescue_max_swaps', type=int, default=1,
                        help='Maximum baseline rank-2 replacements allowed per query.')
    parser.add_argument('--query_plan_ontology_evidence_set_optimizer_mode', type=str, default='off',
                        choices=['off', 'top2'],
                        help='How to use the provenance-aware evidence-set optimizer: off or top2.')
    parser.add_argument('--query_plan_ontology_evidence_set_optimizer_min_confidence', type=float, default=0.90,
                        help='Minimum ontology candidate confidence for evidence-set optimizer replacement.')
    parser.add_argument('--query_plan_ontology_evidence_set_optimizer_min_source_precision', type=float, default=0.30,
                        help='Minimum retrieval-feedback source precision required by evidence-set optimizer.')
    parser.add_argument('--query_plan_ontology_evidence_set_optimizer_min_path_source_precision', type=float, default=0.20,
                        help='Lower source-precision threshold allowed when path-aware evidence approves a candidate.')
    parser.add_argument('--query_plan_ontology_evidence_set_optimizer_max_original_rank', type=int, default=20,
                        help='Maximum original retrieval rank eligible for optimizer top-2 replacement; <=0 disables this guard.')
    parser.add_argument('--query_plan_ontology_evidence_set_optimizer_rank2_margin', type=float, default=0.03,
                        help='Minimum estimated utility margin over current rank 2 for optimizer replacement.')
    parser.add_argument('--query_plan_ontology_evidence_set_optimizer_require_complex_or_path', type=str, default='true',
                        help='If true, optimizer swaps require complex query, path-approved evidence, or strict/multihop evidence.')
    parser.add_argument('--query_plan_ontology_evidence_set_optimizer_max_swaps', type=int, default=1,
                        help='Maximum optimizer top-2 replacements allowed per query.')
    parser.add_argument('--query_plan_ontology_evidence_set_rank2_pairwise_optimizer_model_path', type=str, default='',
                        help='Optional learned pairwise optimizer JSON for replacing the current rank-2 evidence passage.')
    parser.add_argument('--query_plan_ontology_evidence_set_rank2_pairwise_optimizer_mode', type=str, default='off',
                        choices=['off', 'top2'],
                        help='How to use the learned rank-2 pairwise optimizer: off or top2.')
    parser.add_argument('--query_plan_ontology_evidence_set_rank2_pairwise_optimizer_min_score', type=float, default=0.85,
                        help='Minimum pairwise optimizer score required to replace baseline rank 2.')
    parser.add_argument('--query_plan_ontology_evidence_set_rank2_pairwise_optimizer_min_confidence', type=float, default=0.0,
                        help='Minimum ontology candidate confidence required by the pairwise rank-2 optimizer.')
    parser.add_argument('--query_plan_ontology_evidence_set_rank2_pairwise_optimizer_max_original_rank', type=int, default=120,
                        help='Maximum original rank eligible for learned pairwise rank-2 replacement; <=0 disables this guard.')
    parser.add_argument('--query_plan_ontology_evidence_set_rank2_pairwise_optimizer_max_swaps', type=int, default=1,
                        help='Maximum learned pairwise rank-2 replacements allowed per query.')
    parser.add_argument('--query_plan_ontology_evidence_set_rank2_pairwise_optimizer_min_value_margin', type=float, default=-1.0,
                        help='Minimum estimated candidate-vs-current-rank2 utility margin required by the learned pairwise optimizer; < -0.999 disables this cost guard.')
    parser.add_argument('--query_plan_ontology_evidence_set_rank2_pairwise_optimizer_path_override_min_value_margin', type=float, default=-1.0,
                        help='Relaxed value-margin threshold for strong path/terminal candidates; < -0.999 disables path override.')
    parser.add_argument('--query_plan_ontology_evidence_set_rank2_pairwise_optimizer_require_complex_or_path', type=str, default='false',
                        help='If true, learned pairwise top-2 swaps require a complex query or strong path/strict/multihop evidence.')
    parser.add_argument('--query_plan_ontology_evidence_set_rank2_pairwise_optimizer_after_evidence_optimizer', type=str, default='false',
                        help='If true, run the learned pairwise rank-2 optimizer after the provenance evidence-set optimizer.')
    parser.add_argument('--query_plan_ontology_evidence_set_window_selector_model_path', type=str, default='',
                        help='Optional learned final evidence-window selector JSON path.')
    parser.add_argument('--query_plan_ontology_evidence_set_window_selector_mode', type=str, default='off',
                        choices=['off', 'score', 'selector', 'veto_rescue'],
                        help='How to use the learned final evidence-window selector: off, score, selector, or veto_rescue.')
    parser.add_argument('--query_plan_ontology_evidence_set_window_selector_min_score', type=float, default=0.60,
                        help='Minimum window-selector score required before an ontology candidate can enter the final evidence window.')
    parser.add_argument('--query_plan_ontology_evidence_set_window_selector_top2_min_score', type=float, default=0.90,
                        help='Window-selector score required before an ontology candidate can alter the top-2 evidence window.')
    parser.add_argument('--query_plan_ontology_evidence_set_window_selector_score_weight', type=float, default=0.85,
                        help='Score weight added to evidence-set selection from the final evidence-window selector.')
    parser.add_argument('--query_plan_ontology_evidence_set_window_selector_max_ontology_passages', type=int, default=0,
                        help='Maximum ontology passages selected when window selector mode is active; <=0 keeps the global evidence-set limit.')
    parser.add_argument('--query_plan_ontology_evidence_set_window_selector_veto_below', type=float, default=0.0,
                        help='In veto_rescue mode, veto non-backbone ontology candidates below this window-selector score.')
    parser.add_argument('--query_plan_ontology_evidence_set_window_selector_backbone_veto_below', type=float, default=0.0,
                        help='In veto_rescue mode, veto backbone ontology candidates below this window-selector score.')
    parser.add_argument('--query_plan_ontology_evidence_set_window_selector_rescue_min_score', type=float, default=0.85,
                        help='In veto_rescue mode, minimum window-selector score for one extra rescue candidate.')
    parser.add_argument('--query_plan_ontology_evidence_set_window_selector_rescue_max_passages', type=int, default=0,
                        help='In veto_rescue mode, maximum rescue candidates allowed beyond source-aware evidence-grounded guard.')
    parser.add_argument('--query_plan_ontology_evidence_set_router_model_path', type=str, default='',
                        help='Optional learned query-level router JSON for ontology evidence-set intervention.')
    parser.add_argument('--query_plan_ontology_evidence_set_router_mode', type=str, default='off',
                        choices=['off', 'filter', 'action'],
                        help='How to use the query-level evidence-set router: off, filter, or action.')
    parser.add_argument('--query_plan_ontology_evidence_set_router_min_score', type=float, default=0.50,
                        help='Minimum router score required before ontology evidence-set intervention can alter top-k.')
    parser.add_argument('--query_plan_ontology_evidence_set_router_precision_min_score', type=float, default=0.50,
                        help='In action mode, minimum router score for high-precision low-budget ontology intervention.')
    parser.add_argument('--query_plan_ontology_evidence_set_router_precision_min_confidence', type=float, default=0.90,
                        help='In action mode, minimum candidate confidence in high-precision intervention mode.')
    parser.add_argument('--query_plan_ontology_evidence_set_router_precision_max_ontology_passages', type=int, default=1,
                        help='In action mode, maximum ontology passages selected for high-precision intervention.')
    parser.add_argument('--query_plan_ontology_evidence_set_schema_route_adapter_mode', type=str, default='off',
                        choices=['off', 'score', 'filter', 'selector'],
                        help='Youtu-style schema route adapter mode for ontology evidence candidates.')
    parser.add_argument('--query_plan_ontology_evidence_set_schema_route_adapter_score_weight', type=float, default=0.18,
                        help='Score weight added from schema-guided entity/triple/path/community route coverage.')
    parser.add_argument('--query_plan_ontology_evidence_set_schema_route_adapter_confidence_boost', type=float, default=0.04,
                        help='Candidate confidence boost scale when schema-route coverage is strong.')
    parser.add_argument('--query_plan_ontology_evidence_set_schema_route_adapter_min_score', type=float, default=0.30,
                        help='Schema-route score threshold used for promotion-style confidence boosts.')
    parser.add_argument('--query_plan_ontology_evidence_set_schema_route_adapter_reject_below', type=float, default=0.0,
                        help='In filter/selector mode, reject schema-route candidates below this score.')
    parser.add_argument('--query_plan_ontology_evidence_set_schema_route_adapter_max_subqueries', type=int, default=6,
                        help='Maximum schema-guided atomic route probes recorded per query.')
    parser.add_argument('--query_plan_ontology_evidence_set_enable_schema_slot_utility', type=str, default='false',
                        help='If True, score schema-slot recall candidates with a utility gate before final evidence-set ranking.')
    parser.add_argument('--query_plan_ontology_evidence_set_schema_slot_utility_min_score', type=float, default=0.70,
                        help='Minimum schema-slot utility score required before a schema-slot candidate is trusted.')
    parser.add_argument('--query_plan_ontology_evidence_set_schema_slot_utility_top2_min_score', type=float, default=0.86,
                        help='Minimum schema-slot utility score required before a schema-slot candidate can replace rank 2.')
    parser.add_argument('--query_plan_ontology_evidence_set_schema_slot_utility_score_weight', type=float, default=0.45,
                        help='Score weight added to evidence-set ranking from schema-slot utility.')
    parser.add_argument('--query_plan_ontology_evidence_set_schema_slot_utility_confidence_boost', type=float, default=0.08,
                        help='Confidence boost applied to high-utility schema-slot candidates.')
    parser.add_argument('--query_plan_ontology_evidence_set_schema_slot_utility_top2_max_swaps', type=int, default=1,
                        help='Maximum number of rank-2 replacements allowed by schema-slot utility per query.')
    parser.add_argument('--query_plan_ontology_evidence_set_enable_provenance_groupwise_selector', type=str, default='false',
                        help='If True, use provenance/text/source-aware groupwise replacement for evidence-set top-5 repair.')
    parser.add_argument('--query_plan_ontology_evidence_set_provenance_groupwise_min_score', type=float, default=0.68,
                        help='Minimum provenance-groupwise score required before a candidate can repair top-5.')
    parser.add_argument('--query_plan_ontology_evidence_set_provenance_groupwise_top2_min_score', type=float, default=0.92,
                        help='Minimum provenance-groupwise score required before a candidate can replace rank 2.')
    parser.add_argument('--query_plan_ontology_evidence_set_provenance_groupwise_score_weight', type=float, default=0.45,
                        help='Score weight added from provenance-groupwise candidate utility.')
    parser.add_argument('--query_plan_ontology_evidence_set_provenance_groupwise_max_top5_swaps', type=int, default=2,
                        help='Maximum number of top-5 replacements allowed by provenance-groupwise selector per query.')
    parser.add_argument('--query_plan_ontology_evidence_set_provenance_groupwise_max_top2_swaps', type=int, default=1,
                        help='Maximum number of rank-2 replacements allowed by provenance-groupwise selector per query.')
    parser.add_argument('--query_plan_ontology_evidence_set_provenance_groupwise_min_source_precision', type=float, default=0.06,
                        help='Minimum source-level precision needed by provenance-groupwise selector unless structure is strong.')
    parser.add_argument('--query_plan_ontology_evidence_set_provenance_groupwise_require_text_signal', type=str, default='true',
                        help='If True, require passage text signal for provenance-groupwise candidates unless structure is strong.')
    parser.add_argument('--query_plan_ontology_evidence_set_enable_evidence_join_planner', type=str, default='false',
                        help='If True, use cost-based evidence join optimization for ontology evidence-set top-k selection.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_join_planner_min_score', type=float, default=0.58,
                        help='Minimum static evidence-join profile score before a candidate is eligible.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_join_planner_top2_min_score', type=float, default=0.82,
                        help='Minimum evidence-join profile score before a candidate can replace rank 2.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_join_planner_min_marginal_gain', type=float, default=0.30,
                        help='Minimum marginal slot gain required by evidence-join planner.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_join_planner_min_value_margin', type=float, default=-0.02,
                        help='Minimum utility margin over the replaced passage for evidence-join planner.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_join_planner_score_weight', type=float, default=0.50,
                        help='Selection score weight from evidence-join profile score.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_join_planner_marginal_gain_weight', type=float, default=0.45,
                        help='Selection score weight from evidence-join marginal slot gain.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_join_planner_max_top5_swaps', type=int, default=2,
                        help='Maximum top-5 replacements allowed by evidence-join planner.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_join_planner_max_top2_swaps', type=int, default=1,
                        help='Maximum rank-2 replacements allowed by evidence-join planner.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_join_planner_router_bypass_min_score', type=float, default=0.68,
                        help='Evidence-join profile score needed to bypass the old intervention policy.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_join_planner_require_text_or_path', type=str, default='true',
                        help='If True, evidence-join candidates need text, path, or structural support.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_join_planner_enable_answer_slot_rescue', type=str, default='false',
                        help='If True, strong answer-slot evidence can rescue evidence-join candidates.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_join_planner_answer_slot_score_weight', type=float, default=0.22,
                        help='Selection score weight from answer-slot compatibility in evidence-join planner.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_join_planner_answer_slot_min_score', type=float, default=0.52,
                        help='Minimum answer-slot score for evidence-join top-5 rescue.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_join_planner_answer_slot_top2_min_score', type=float, default=0.70,
                        help='Minimum answer-slot score for evidence-join rank-2 rescue.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_join_planner_answer_slot_router_bypass_min_score', type=float, default=0.76,
                        help='Answer-slot score needed to bypass the intervention router for evidence-join candidates.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_join_planner_enable_operator_aware', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_JOIN_PLANNER_ENABLE_OPERATOR_AWARE', 'false'),
                        help='If True, make evidence join scoring and replacement gates aware of query operator and missing join slots.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_join_planner_operator_score_weight', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_JOIN_PLANNER_OPERATOR_SCORE_WEIGHT', 0.18)),
                        help='Selection score weight from operator-aware evidence join compatibility.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_join_planner_operator_min_gain', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_JOIN_PLANNER_OPERATOR_MIN_GAIN', 0.14)),
                        help='Minimum operator-slot marginal gain for top-5 evidence join replacements.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_join_planner_operator_top2_min_gain', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_JOIN_PLANNER_OPERATOR_TOP2_MIN_GAIN', 0.24)),
                        help='Minimum operator-slot marginal gain for rank-2 evidence join replacements.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_join_planner_operator_chain_pressure_weight', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_EVIDENCE_JOIN_PLANNER_OPERATOR_CHAIN_PRESSURE_WEIGHT', 0.10)),
                        help='Extra score weight for complex operator chain pressure when candidate operator evidence matches.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_join_planner_enable_risk_gate', type=str, default='false',
                        help='If True, apply counterfactual risk gates before evidence-join top-k replacements.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_join_planner_risk_gate_top2_min_source_precision', type=float, default=0.18,
                        help='Minimum source precision for evidence-join rank-2 replacement unless answer-slot rescue fires.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_join_planner_risk_gate_top5_min_source_precision', type=float, default=0.06,
                        help='Minimum source precision for evidence-join top-5 replacement without strong answer signal.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_join_planner_risk_gate_top2_min_margin', type=float, default=0.10,
                        help='Minimum utility margin for evidence-join rank-2 replacement when risk gate is enabled.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_join_planner_risk_gate_top5_min_margin', type=float, default=0.03,
                        help='Minimum utility margin for evidence-join top-5 replacement when risk gate is enabled.')
    parser.add_argument('--query_plan_ontology_evidence_set_evidence_join_planner_risk_gate_max_lost_value', type=float, default=0.36,
                        help='Maximum lost query-slot value allowed before evidence-join replacement is vetoed.')
    parser.add_argument('--query_plan_ontology_evidence_set_enable_answer_pairwise_verifier', type=str, default='false',
                        help='If True, use answer-aware pairwise verification before ontology evidence-set replacement.')
    parser.add_argument('--query_plan_ontology_evidence_set_answer_pairwise_min_score', type=float, default=0.56,
                        help='Minimum answer-pairwise score before a candidate can replace top-5.')
    parser.add_argument('--query_plan_ontology_evidence_set_answer_pairwise_top2_min_score', type=float, default=0.64,
                        help='Minimum answer-pairwise score before a candidate can replace rank 2.')
    parser.add_argument('--query_plan_ontology_evidence_set_answer_pairwise_min_margin', type=float, default=0.04,
                        help='Minimum pairwise utility margin for top-5 answer-aware replacement.')
    parser.add_argument('--query_plan_ontology_evidence_set_answer_pairwise_top2_min_margin', type=float, default=0.08,
                        help='Minimum pairwise utility margin for rank-2 answer-aware replacement.')
    parser.add_argument('--query_plan_ontology_evidence_set_answer_pairwise_max_top5_swaps', type=int, default=2,
                        help='Maximum top-5 replacements allowed by answer-aware pairwise verifier.')
    parser.add_argument('--query_plan_ontology_evidence_set_answer_pairwise_max_top2_swaps', type=int, default=1,
                        help='Maximum rank-2 replacements allowed by answer-aware pairwise verifier.')
    parser.add_argument('--query_plan_ontology_evidence_set_answer_pairwise_require_answer_signal', type=str, default='true',
                        help='If True, require answer/terminal/text/path signal before answer-aware pairwise replacement.')
    parser.add_argument('--query_plan_ontology_evidence_set_answer_pairwise_score_weight', type=float, default=0.30,
                        help='Selection score weight from answer-aware pairwise verifier score.')
    parser.add_argument('--query_plan_ontology_evidence_set_enable_counterfactual_text_verifier', type=str, default='false',
                        help='If True, compare candidate passage text against the exact passage it would replace before final evidence-set swaps.')
    parser.add_argument('--query_plan_ontology_evidence_set_counterfactual_text_verifier_min_score', type=float, default=0.62,
                        help='Minimum counterfactual text verifier score before a candidate can replace a top-5 passage.')
    parser.add_argument('--query_plan_ontology_evidence_set_counterfactual_text_verifier_top2_min_score', type=float, default=0.74,
                        help='Minimum counterfactual text verifier score before a candidate can replace rank 2.')
    parser.add_argument('--query_plan_ontology_evidence_set_counterfactual_text_verifier_min_margin', type=float, default=0.10,
                        help='Minimum candidate-vs-replacement utility margin for top-5 counterfactual text verification.')
    parser.add_argument('--query_plan_ontology_evidence_set_counterfactual_text_verifier_top2_min_margin', type=float, default=0.16,
                        help='Minimum candidate-vs-current-rank2 utility margin for top-2 counterfactual text verification.')
    parser.add_argument('--query_plan_ontology_evidence_set_counterfactual_text_verifier_max_top5_swaps', type=int, default=1,
                        help='Maximum top-5 swaps allowed by the counterfactual text verifier per query.')
    parser.add_argument('--query_plan_ontology_evidence_set_counterfactual_text_verifier_max_top2_swaps', type=int, default=1,
                        help='Maximum rank-2 swaps allowed by the counterfactual text verifier per query.')
    parser.add_argument('--query_plan_ontology_evidence_set_counterfactual_text_verifier_max_original_rank', type=int, default=80,
                        help='Maximum original candidate rank eligible for counterfactual text verification; <=0 disables this guard.')
    parser.add_argument('--query_plan_ontology_evidence_set_counterfactual_text_verifier_min_candidate_text_score', type=float, default=0.38,
                        help='Minimum passage text-verifier score unless answer/path evidence is strong.')
    parser.add_argument('--query_plan_ontology_evidence_set_counterfactual_text_verifier_score_weight', type=float, default=0.0,
                        help='Optional soft ranking weight from counterfactual text verifier score before final swap verification.')
    parser.add_argument('--query_plan_ontology_evidence_set_counterfactual_text_verifier_require_answer_signal', type=str, default='true',
                        help='If True, require answer/terminal/path/text evidence before counterfactual replacement.')
    parser.add_argument('--query_plan_ontology_evidence_set_enable_decomposition_slots', type=str, default='false',
                        help='If True, use schema-decomposed sub-question slots as an evidence-set coverage signal.')
    parser.add_argument('--query_plan_ontology_evidence_set_decomposition_slot_weight', type=float, default=0.35,
                        help='Score weight added when an ontology candidate covers a new schema-decomposition slot.')
    parser.add_argument('--query_plan_ontology_evidence_set_decomposition_slot_max_slots', type=int, default=6,
                        help='Maximum schema-decomposition slots extracted per candidate passage.')
    parser.add_argument('--query_plan_ontology_evidence_set_enable_path_aware_selector', type=str, default='false',
                        help='If True, use verified ontology path support as an explicit evidence-set selection signal.')
    parser.add_argument('--query_plan_ontology_evidence_set_path_aware_score_weight', type=float, default=0.0,
                        help='Score weight added to evidence-set selection from verified ontology path support.')
    parser.add_argument('--query_plan_ontology_evidence_set_path_aware_confidence_boost', type=float, default=0.0,
                        help='Confidence boost for candidates with approved verified ontology path support.')
    parser.add_argument('--query_plan_ontology_evidence_set_path_aware_min_score', type=float, default=0.65,
                        help='Minimum path-aware score required before path-supported candidate promotion.')
    parser.add_argument('--query_plan_ontology_evidence_set_path_aware_min_terminal_type_score', type=float, default=0.45,
                        help='Minimum terminal answer-type score required by the path-aware selector.')
    parser.add_argument('--query_plan_ontology_evidence_set_path_aware_require_query_relation_coverage', type=str, default='true',
                        help='If True, path-aware promotion requires full strong query-relation coverage.')
    parser.add_argument('--query_plan_ontology_evidence_set_text_verifier_mode', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_TEXT_VERIFIER_MODE', 'off'),
                        choices=['off', 'score', 'filter'],
                        help='Use passage text to verify ontology evidence candidates: off, score, or filter.')
    parser.add_argument('--query_plan_ontology_evidence_set_text_verifier_min_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_TEXT_VERIFIER_MIN_SCORE', 0.55)),
                        help='Text-verifier score needed to promote an ontology evidence candidate.')
    parser.add_argument('--query_plan_ontology_evidence_set_text_verifier_reject_below', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_TEXT_VERIFIER_REJECT_BELOW', 0.0)),
                        help='In filter mode, reject weak non-strict candidates below this text-verifier score.')
    parser.add_argument('--query_plan_ontology_evidence_set_text_verifier_score_weight', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_TEXT_VERIFIER_SCORE_WEIGHT', 0.20)),
                        help='Score weight added to evidence-set selection from passage-text verification.')
    parser.add_argument('--query_plan_ontology_evidence_set_text_verifier_confidence_boost', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_TEXT_VERIFIER_CONFIDENCE_BOOST', 0.04)),
                        help='Small confidence boost for candidates whose passage text matches query/schema evidence.')
    parser.add_argument('--query_plan_ontology_evidence_set_text_verifier_max_chars', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_SET_TEXT_VERIFIER_MAX_CHARS', 1600)),
                        help='Maximum passage characters used by the lightweight text verifier.')
    parser.add_argument('--query_plan_ontology_lift_min_original_rank', type=int, default=1,
                        help='Minimum original passage rank eligible for query-plan lift; <=0 disables lower bound.')
    parser.add_argument('--query_plan_ontology_lift_max_original_rank', type=int, default=20,
                        help='Maximum original passage rank eligible for query-plan lift; <=0 disables upper bound.')
    parser.add_argument('--query_plan_ontology_soft_fallback_lift_max_original_rank', type=int, default=5,
                        help='Maximum original passage rank eligible when all evidence came from soft fallback; <=0 disables this bound.')
    parser.add_argument('--query_plan_ontology_soft_fallback_weight_scale', type=float, default=0.35,
                        help='Multiplier for passage lift evidence that only came from soft fallback paths.')
    parser.add_argument('--query_plan_ontology_validated_soft_max_paths', type=int, default=4,
                        help='Maximum provenance-validated soft paths added beside strict paths.')
    parser.add_argument('--query_plan_ontology_validated_soft_min_answer_path_score', type=float, default=0.50,
                        help='Minimum query/answer-path compatibility score for a validated soft path.')
    parser.add_argument('--query_plan_ontology_gate_require_fact_endpoints', type=str, default='true',
                        help='If True, lift only passages containing both endpoints of the lifted fact.')
    parser.add_argument('--query_plan_ontology_gate_require_anchor_or_terminal', type=str, default='true',
                        help='If True, require first-hop anchor and last-hop terminal mentions in lifted passages.')
    parser.add_argument('--query_plan_ontology_gate_min_relation_token_overlap', type=int, default=0,
                        help='Minimum relation/signature token overlap required in a passage before query-plan lift; 0 disables this gate.')
    parser.add_argument('--query_plan_ontology_enable_simple_evidence_validator', type=str, default='false',
                        help='If True, route simple attribute queries to candidate-passage evidence validation instead of source-passage lift.')
    parser.add_argument('--query_plan_ontology_simple_validator_candidate_top_k', type=int, default=30,
                        help='Number of current candidate passages considered by the simple query evidence validator.')
    parser.add_argument('--query_plan_ontology_simple_validator_max_evidence_paths', type=int, default=16,
                        help='Maximum executable query-plan paths used as terminal-object evidence for simple queries.')
    parser.add_argument('--query_plan_ontology_simple_validator_weight', type=float, default=0.10,
                        help='Boost weight for simple query candidate-passage evidence validation.')
    parser.add_argument('--query_plan_ontology_simple_validator_fallback_to_path_lift', type=str, default='false',
                        help='If True, simple queries with no validator evidence can still fall back to source-passage path lift.')
    parser.add_argument('--query_plan_ontology_enable_qa_evidence', type=str, default='false',
                        help='If True, add query-plan ontology evidence bullets to the QA reader prompt.')
    parser.add_argument('--query_plan_ontology_enable_adapter_policy', type=str, default='false',
                        help='If True, route query-plan ontology PPR, QA evidence style, and answer-format hints through the adapter policy.')
    parser.add_argument('--query_plan_ontology_qa_evidence_max_items', type=int, default=6,
                        help='Maximum query-plan ontology evidence bullets added to each QA prompt.')
    parser.add_argument('--query_plan_ontology_qa_evidence_min_score', type=float, default=0.0,
                        help='Minimum query-plan evidence score required before adding a QA evidence bullet.')
    parser.add_argument('--query_plan_ontology_qa_evidence_style', type=str, default='structured_chain',
                        choices=['structured_chain', 'facts_only', 'bridge_cues', 'auto', 'disabled'],
                        help='How query-plan ontology evidence is rendered in QA prompts; disabled suppresses concrete graph evidence while retaining independently enabled format hints.')
    parser.add_argument('--query_plan_ontology_enable_evidence_chain_materializer', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_ENABLE_EVIDENCE_CHAIN_MATERIALIZER', 'false'),
                        help='If True, materialize top-window evidence snippets missing from the QA-only context.')
    parser.add_argument('--query_plan_ontology_evidence_chain_materializer_max_items', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_CHAIN_MATERIALIZER_MAX_ITEMS', 3)),
                        help='Maximum evidence-chain materializer snippets added to each QA prompt.')
    parser.add_argument('--query_plan_ontology_evidence_chain_materializer_source_top_k', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_CHAIN_MATERIALIZER_SOURCE_TOP_K', 5)),
                        help='Number of original retrieved passages inspected by the evidence-chain materializer.')
    parser.add_argument('--query_plan_ontology_evidence_chain_materializer_max_doc_chars', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_CHAIN_MATERIALIZER_MAX_DOC_CHARS', 1200)),
                        help='Maximum passage characters used when extracting materialized evidence snippets.')
    parser.add_argument('--query_plan_ontology_evidence_chain_materializer_min_query_overlap', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_CHAIN_MATERIALIZER_MIN_QUERY_OVERLAP', 2)),
                        help='Minimum query-token overlap required unless an answer-like span is present.')
    parser.add_argument('--query_plan_ontology_evidence_chain_materializer_max_spans', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_CHAIN_MATERIALIZER_MAX_SPANS', 4)),
                        help='Maximum answer-like spans shown per materialized evidence snippet.')
    parser.add_argument('--query_plan_ontology_evidence_chain_materializer_require_context_gap', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_CHAIN_MATERIALIZER_REQUIRE_CONTEXT_GAP', 'true'),
                        help='If True, materialize only original top-window passages absent from the QA-only context.')
    parser.add_argument('--query_plan_ontology_evidence_chain_materializer_add_docs_to_context', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_CHAIN_MATERIALIZER_ADD_DOCS_TO_CONTEXT', 'true'),
                        help='If True, add materialized passages back to the QA-only context.')
    parser.add_argument('--query_plan_ontology_evidence_chain_materializer_preserve_context_top_k', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_CHAIN_MATERIALIZER_PRESERVE_CONTEXT_TOP_K', 1)),
                        help='Number of existing QA-context passages preserved before materialized companion passages.')
    parser.add_argument('--query_plan_ontology_evidence_chain_materializer_enable_chain_complete', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_CHAIN_MATERIALIZER_ENABLE_CHAIN_COMPLETE', 'false'),
                        help='If True, greedily materialize a chain-complete QA context covering distinct ontology roles.')
    parser.add_argument('--query_plan_ontology_evidence_chain_materializer_chain_complete_target_datasets', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_CHAIN_MATERIALIZER_CHAIN_COMPLETE_TARGET_DATASETS', ''),
                        help='Comma-separated dataset allowlist for chain-complete materialization; empty enables all datasets.')
    parser.add_argument('--query_plan_ontology_evidence_chain_materializer_chain_complete_min_roles', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_CHAIN_MATERIALIZER_CHAIN_COMPLETE_MIN_ROLES', 2)),
                        help='Minimum distinct ontology role categories targeted by chain-complete materialization.')
    parser.add_argument('--query_plan_ontology_evidence_chain_materializer_chain_complete_role_weight', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_CHAIN_MATERIALIZER_CHAIN_COMPLETE_ROLE_WEIGHT', 0.24)),
                        help='Score bonus per distinct ontology role covered by a materialized passage.')
    parser.add_argument('--query_plan_ontology_evidence_chain_materializer_require_chain_complete_route', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_CHAIN_MATERIALIZER_REQUIRE_CHAIN_COMPLETE_ROUTE', 'false'),
                        help='If True, skip evidence-chain materialization unless the chain-complete router approves the query.')
    parser.add_argument('--query_plan_ontology_evidence_chain_materializer_strict_admission', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_CHAIN_MATERIALIZER_STRICT_ADMISSION', 'false'),
                        help='If True, admit only materialized passages that cover concrete ontology roles.')
    parser.add_argument('--query_plan_ontology_evidence_chain_materializer_min_real_roles', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_CHAIN_MATERIALIZER_MIN_REAL_ROLES', 1)),
                        help='Minimum concrete ontology roles required under strict materializer admission.')
    parser.add_argument('--query_plan_ontology_evidence_chain_materializer_context_gap_use_qa_context', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_CHAIN_MATERIALIZER_CONTEXT_GAP_USE_QA_CONTEXT', 'false'),
                        help='If True, use the current QA context as the context-gap baseline.')
    parser.add_argument('--query_plan_ontology_evidence_chain_materializer_emit_answer_spans', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_CHAIN_MATERIALIZER_EMIT_ANSWER_SPANS', 'true'),
                        help='If False, do not expose answer_like_spans in materialized evidence.')
    parser.add_argument('--query_plan_ontology_evidence_chain_materializer_suppress_answer_spans_for_slots', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_EVIDENCE_CHAIN_MATERIALIZER_SUPPRESS_ANSWER_SPANS_FOR_SLOTS', ''),
                        help='Comma-separated answer slots for which materialized evidence suppresses answer_like_spans.')
    parser.add_argument('--query_plan_ontology_enable_answer_candidate_operator', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_ENABLE_ANSWER_CANDIDATE_OPERATOR', 'false'),
                        help='If True, add a provenance-aware answer-candidate table to the QA prompt.')
    parser.add_argument('--query_plan_ontology_answer_candidate_max_items', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_ANSWER_CANDIDATE_MAX_ITEMS', 3)),
                        help='Maximum answer candidates shown in the QA prompt candidate table.')
    parser.add_argument('--query_plan_ontology_answer_candidate_min_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_ANSWER_CANDIDATE_MIN_SCORE', 0.62)),
                        help='Minimum provenance-aware score required for answer candidates.')
    parser.add_argument('--query_plan_ontology_answer_candidate_max_doc_chars', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_ANSWER_CANDIDATE_MAX_DOC_CHARS', 1400)),
                        help='Maximum passage characters used when verifying answer-candidate support.')
    parser.add_argument('--query_plan_ontology_answer_candidate_require_passage_support', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_ANSWER_CANDIDATE_REQUIRE_PASSAGE_SUPPORT', 'true'),
                        help='If True, answer candidates must occur in retrieved passages before QA exposure.')
    parser.add_argument('--query_plan_ontology_answer_candidate_use_doc_spans', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_ANSWER_CANDIDATE_USE_DOC_SPANS', 'false'),
                        help='If True, date/number spans may be proposed directly from top passages.')
    parser.add_argument('--query_plan_ontology_answer_candidate_gate_mode', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_ANSWER_CANDIDATE_GATE_MODE', 'loose'),
                        choices=['loose', 'strict', 'strict_unique', 'unique'],
                        help='Answer-candidate exposure gate. strict_unique only shows one high-margin candidate.')
    parser.add_argument('--query_plan_ontology_answer_candidate_min_margin', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_ANSWER_CANDIDATE_MIN_MARGIN', 0.0)),
                        help='Minimum score margin between top answer candidates under strict gates.')
    parser.add_argument('--query_plan_ontology_answer_candidate_disable_ambiguous_slots', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_ANSWER_CANDIDATE_DISABLE_AMBIGUOUS_SLOTS', 'false'),
                        help='If True, suppress candidate tables for ambiguous slots when multiple candidates survive.')
    parser.add_argument('--query_plan_ontology_answer_candidate_enable_slot_hint', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_ANSWER_CANDIDATE_ENABLE_SLOT_HINT', 'false'),
                        help='If True, add slot-specific answer extraction constraints to QA evidence.')
    parser.add_argument('--query_plan_ontology_enable_answer_canonicalizer', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_ENABLE_ANSWER_CANONICALIZER', 'false'),
                        help='If True, conservatively post-process QA answers using supported answer candidates.')
    parser.add_argument('--query_plan_ontology_answer_canonicalizer_min_candidate_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_ANSWER_CANONICALIZER_MIN_CANDIDATE_SCORE', 0.82)),
                        help='Minimum supported answer-candidate score before answer canonicalization can adopt it.')
    parser.add_argument('--query_plan_ontology_answer_canonicalizer_require_candidate_support', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_ANSWER_CANONICALIZER_REQUIRE_CANDIDATE_SUPPORT', 'true'),
                        help='If True, answer canonicalization only adopts candidates occurring in retrieved passages.')
    parser.add_argument('--query_plan_ontology_answer_canonicalizer_fix_year_only', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_ANSWER_CANONICALIZER_FIX_YEAR_ONLY', 'true'),
                        help='If True, shorten temporal answers to a year when the question explicitly asks for a year.')
    parser.add_argument('--query_plan_ontology_answer_canonicalizer_replace_unknown', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_ANSWER_CANONICALIZER_REPLACE_UNKNOWN', 'false'),
                        help='If True, replace unknown-like answers with a supported high-confidence candidate.')
    parser.add_argument('--query_plan_ontology_answer_canonicalizer_allow_expansion', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_ANSWER_CANONICALIZER_ALLOW_EXPANSION', 'false'),
                        help='If True, allow canonicalizer to expand shorter reader answers into longer candidates.')
    parser.add_argument('--query_plan_ontology_answer_canonicalizer_enable_slot_shrink', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_ANSWER_CANONICALIZER_ENABLE_SLOT_SHRINK', 'true'),
                        help='If True, shrink over-specific answers for known slots such as city, year, and occupation.')
    parser.add_argument('--query_plan_ontology_answer_canonicalizer_allow_candidate_shrink', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_ANSWER_CANONICALIZER_ALLOW_CANDIDATE_SHRINK', 'false'),
                        help='If True, allow generic supported-candidate span shrink; disabled by default to avoid shortening exact names/dates.')
    parser.add_argument('--query_plan_ontology_enable_grounded_answer_verifier', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_ENABLE_GROUNDED_ANSWER_VERIFIER', 'false'),
                        help='If true, conservatively verify the reader answer against the submitted passages.')
    parser.add_argument('--query_plan_ontology_grounded_answer_verifier_min_confidence', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_GROUNDED_ANSWER_VERIFIER_MIN_CONFIDENCE', 0.93)),
                        help='Minimum confidence for a passage-grounded answer replacement.')
    parser.add_argument('--query_plan_ontology_grounded_answer_verifier_max_passage_chars', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_GROUNDED_ANSWER_VERIFIER_MAX_PASSAGE_CHARS', 1800)),
                        help='Maximum characters per passage shown to the grounded answer verifier.')
    parser.add_argument('--query_plan_ontology_enable_answer_operator_calibrator', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_ENABLE_ANSWER_OPERATOR_CALIBRATOR', 'false'),
                        help='If True, apply operator-aware final answer constraints for boolean, comparison, and single-slot questions.')
    parser.add_argument('--query_plan_ontology_answer_operator_calibrator_min_candidate_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_ANSWER_OPERATOR_CALIBRATOR_MIN_CANDIDATE_SCORE', 0.82)),
                        help='Minimum supported candidate score before operator-aware calibration can override the reader answer.')
    parser.add_argument('--query_plan_ontology_answer_operator_calibrator_min_margin', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_ANSWER_OPERATOR_CALIBRATOR_MIN_MARGIN', 0.08)),
                        help='Minimum margin over the second candidate/current answer for operator-aware calibration.')
    parser.add_argument('--query_plan_ontology_answer_operator_calibrator_single_value_slots', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_ANSWER_OPERATOR_CALIBRATOR_SINGLE_VALUE_SLOTS', 'nationality,occupation,type'),
                        help='Comma-separated slots where compound reader answers may be shrunk to the first requested value.')
    parser.add_argument('--query_plan_ontology_enable_typed_answer_projection', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_ENABLE_TYPED_ANSWER_PROJECTION', 'false'),
                        help='If True, use a typed final-answer projection scorer to override reader answers when high-confidence evidence exists.')
    parser.add_argument('--query_plan_ontology_typed_answer_projection_min_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_TYPED_ANSWER_PROJECTION_MIN_SCORE', 0.82)),
                        help='Minimum score for typed answer projection to override the reader answer.')
    parser.add_argument('--query_plan_ontology_typed_answer_projection_min_margin', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_TYPED_ANSWER_PROJECTION_MIN_MARGIN', 0.06)),
                        help='Minimum score margin for typed answer projection override.')
    parser.add_argument('--query_plan_ontology_typed_answer_projection_allowed_slots', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_TYPED_ANSWER_PROJECTION_ALLOWED_SLOTS', 'occupation,birth_place,death_place,location,date'),
                        help='Comma-separated answer slots where typed answer projection may override the reader answer.')
    parser.add_argument('--query_plan_ontology_typed_answer_projection_require_support', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_TYPED_ANSWER_PROJECTION_REQUIRE_SUPPORT', 'true'),
                        help='If True, typed projection candidates must occur in retrieved passages.')
    parser.add_argument('--query_plan_ontology_typed_answer_projection_prefer_chain_order', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_TYPED_ANSWER_PROJECTION_PREFER_CHAIN_ORDER', 'true'),
                        help='If True, earlier ontology-chain candidates receive a small projection tie-break bonus.')
    parser.add_argument('--query_plan_ontology_typed_answer_projection_operator_gate', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_TYPED_ANSWER_PROJECTION_OPERATOR_GATE', 'true'),
                        help='If True, only allow typed answer projection for configured query operators.')
    parser.add_argument('--query_plan_ontology_typed_answer_projection_allowed_operators', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_TYPED_ANSWER_PROJECTION_ALLOWED_OPERATORS', 'select_attr'),
                        help='Comma-separated query operators where typed answer projection may override the reader answer.')
    parser.add_argument('--query_plan_ontology_typed_answer_projection_require_relation_or_slot_source', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_TYPED_ANSWER_PROJECTION_REQUIRE_RELATION_OR_SLOT_SOURCE', 'true'),
                        help='If True, projection candidates must have a relation match or slot-specific provenance.')
    parser.add_argument('--query_plan_ontology_typed_answer_projection_protect_supported_reader_answer', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_TYPED_ANSWER_PROJECTION_PROTECT_SUPPORTED_READER_ANSWER', 'true'),
                        help='If True, keep a non-unknown reader answer that is directly supported by retrieved passages.')
    parser.add_argument('--query_plan_ontology_enable_span_constrained_answer_resolver', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_ENABLE_SPAN_CONSTRAINED_ANSWER_RESOLVER', 'false'),
                        help='If True, resolve unsupported reader answers to high-confidence spans from QA-context evidence.')
    parser.add_argument('--query_plan_ontology_span_resolver_min_score', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_SPAN_RESOLVER_MIN_SCORE', 0.86)),
                        help='Minimum span score required before span-constrained answer resolution.')
    parser.add_argument('--query_plan_ontology_span_resolver_min_margin', type=float,
                        default=float(os.environ.get('QUERY_PLAN_ONTOLOGY_SPAN_RESOLVER_MIN_MARGIN', 0.08)),
                        help='Minimum score margin required by span-constrained answer resolution.')
    parser.add_argument('--query_plan_ontology_span_resolver_allowed_slots', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_SPAN_RESOLVER_ALLOWED_SLOTS', 'date,number,occupation'),
                        help='Comma-separated answer slots where span-constrained resolution may override the reader.')
    parser.add_argument('--query_plan_ontology_span_resolver_allowed_operators', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_SPAN_RESOLVER_ALLOWED_OPERATORS', 'select_attr,bridge'),
                        help='Comma-separated query operators where span-constrained resolution may override the reader.')
    parser.add_argument('--query_plan_ontology_span_resolver_max_doc_spans', type=int,
                        default=int(os.environ.get('QUERY_PLAN_ONTOLOGY_SPAN_RESOLVER_MAX_DOC_SPANS', 8)),
                        help='Maximum answer-like spans collected from QA-context passages.')
    parser.add_argument('--query_plan_ontology_span_resolver_replace_unsupported_only', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_SPAN_RESOLVER_REPLACE_UNSUPPORTED_ONLY', 'true'),
                        help='If True, only replace unknown or unsupported reader answers unless the span is a safe shrink.')
    parser.add_argument('--query_plan_ontology_span_resolver_protect_supported_reader_answer', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_SPAN_RESOLVER_PROTECT_SUPPORTED_READER_ANSWER', 'true'),
                        help='If True, preserve non-unknown reader answers that are directly supported by the QA context.')
    parser.add_argument('--query_plan_ontology_enable_quality_gate', type=str, default='false',
                        help='If True, suppress noisy query-plan QA evidence using plan trace quality and query fan-out.')
    parser.add_argument('--query_plan_ontology_quality_gate_min_plan_precision', type=float, default=0.0,
                        help='Minimum effective plan precision required before non-simple QA evidence can be added; <=0 disables.')
    parser.add_argument('--query_plan_ontology_quality_gate_min_plan_scale', type=float, default=0.0,
                        help='Minimum trace lift_weight_scale required before non-simple QA evidence can be added; <=0 disables.')
    parser.add_argument('--query_plan_ontology_quality_gate_max_matched_plans_for_qa', type=int, default=0,
                        help='Disable non-simple QA evidence when a query matches more plans than this; <=0 disables.')
    parser.add_argument('--query_plan_ontology_quality_gate_max_executable_paths_for_qa', type=int, default=0,
                        help='Disable non-simple QA evidence when a query has more executable paths than this; <=0 disables.')
    parser.add_argument('--query_plan_ontology_quality_gate_max_avg_paths_per_plan_for_qa', type=float, default=0.0,
                        help='Disable non-simple QA evidence when executable paths per matched plan exceed this; <=0 disables.')
    parser.add_argument('--query_plan_ontology_quality_gate_allow_simple_validator', type=str, default='true',
                        help='If True, simple validator evidence can bypass fan-out quality gates.')
    parser.add_argument('--query_plan_ontology_enable_qa_format_hint', type=str, default='false',
                        help='If True, add a short answer-format hint to QA prompts even without safe graph evidence.')
    parser.add_argument('--query_plan_ontology_qa_format_hint_mode', type=str, default='typed',
                        choices=['typed', 'generic', 'auto'],
                        help='Answer-format hint mode. typed uses query-form/type-specific rules; generic only asks for the shortest supported answer span; auto routes generic hints by query type and evidence quality.')
    parser.add_argument('--query_plan_ontology_enable_answer_path_verifier', type=str, default='false',
                        help='If True, score query-plan paths as answer paths before complex passage lift or bridge-cue QA evidence.')
    parser.add_argument('--query_plan_ontology_answer_path_min_score', type=float, default=0.50,
                        help='Minimum answer-path verifier score required for non-simple query-plan paths.')
    parser.add_argument('--query_plan_ontology_answer_path_score_weight', type=float, default=0.0,
                        help='Soft multiplicative bonus weight for verified answer paths; 0 disables score adjustment.')
    parser.add_argument('--query_plan_ontology_answer_path_filter_lift', type=str, default='true',
                        help='If True with answer-path verifier enabled, only verified non-simple paths can lift passages.')
    parser.add_argument('--query_plan_ontology_answer_path_filter_qa_evidence', type=str, default='true',
                        help='If True with answer-path verifier enabled, bridge-cue QA evidence is built only from verified paths.')
    parser.add_argument('--query_plan_ontology_enable_answer_object_gate', type=str, default='true',
                        help='If True, use multi-agent terminal answer/object constraints to filter broad generic query-plan paths.')
    parser.add_argument('--query_plan_ontology_enable_query_conditioning', type=str, default='false',
                        help='If True, refine matched query-plan schemas with current query answer type and relation intent.')
    parser.add_argument('--query_plan_ontology_condition_min_score', type=float, default=0.0,
                        help='Minimum conditioned query-plan score after query-aware refinement.')
    parser.add_argument('--query_plan_ontology_condition_require_relation_overlap', type=str, default='false',
                        help='If True, drop conditioned plans with no inferred query relation overlap when query relation hints exist.')
    parser.add_argument('--query_plan_ontology_enable_simple_fallback', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_ENABLE_SIMPLE_FALLBACK', 'false'),
                        help='If True, create a conservative one-hop query-plan from inferred query relations when no learned plan covers the query.')
    parser.add_argument('--query_plan_ontology_simple_fallback_only_when_unmatched', type=str, default='true',
                        help='If True, add simple fallback plans only when learned query-plan matching returned no usable plan.')
    parser.add_argument('--query_plan_ontology_simple_fallback_precision', type=float, default=0.62,
                        help='Synthetic precision assigned to simple one-hop query-plan fallback templates.')
    parser.add_argument('--query_plan_ontology_simple_fallback_weight_scale', type=float, default=0.65,
                        help='Passage lift weight multiplier for simple one-hop query-plan fallback evidence.')
    parser.add_argument('--query_plan_ontology_enable_plan_repair', type=str, default='false',
                        help='If True, add low-weight repaired query plans from inferred query slots and relation-realization aliases.')
    parser.add_argument('--query_plan_ontology_plan_repair_max_repairs', type=int, default=4,
                        help='Maximum repaired query-plan templates appended per query.')
    parser.add_argument('--query_plan_ontology_plan_repair_min_query_token_overlap', type=int, default=1,
                        help='Minimum relation/query token overlap required before relaxing a matched plan step.')
    parser.add_argument('--query_plan_ontology_plan_repair_min_plan_precision', type=float, default=0.30,
                        help='Minimum effective precision for a matched plan to receive a relaxed repair copy.')
    parser.add_argument('--query_plan_ontology_plan_repair_score_floor', type=float, default=0.46,
                        help='Minimum match score assigned to condition-fallback repaired plans.')
    parser.add_argument('--query_plan_ontology_plan_repair_enable_query_condition', type=str, default='true',
                        help='If True, use inferred query-condition relation slots to synthesize repair plans.')
    parser.add_argument('--query_plan_ontology_plan_repair_enable_relation_realization', type=str, default='true',
                        help='If True, create soft relation-realization repair copies of matched plans.')
    parser.add_argument('--query_plan_ontology_anchor_mode', type=str, default='coverage',
                        choices=['coverage', 'exact'],
                        help='Anchor matching mode for query-plan execution. Exact only uses explicitly mentioned query entities.')
    parser.add_argument('--query_plan_ontology_anchor_require_concrete', type=str, default='false',
                        help='If True, drop role/schema anchors and execute query plans only from concrete query entities.')
    parser.add_argument('--query_plan_ontology_anchor_prefer_longest_exact', type=str, default='false',
                        help='If True with exact anchor mode, drop exact anchors contained in a longer exact query entity mention.')
    parser.add_argument('--enable_query_plan_ontology_ppr_injection', type=str, default='false',
                        help='If True, instantiate query-plan paths before graph PPR and inject selected path facts as PPR seeds.')
    parser.add_argument('--query_plan_ontology_ppr_enable_query_router', type=str, default='true',
                        help='If True, skip query-plan PPR injection for query forms where ontology path injection is likely noisy.')
    parser.add_argument('--query_plan_ontology_ppr_enable_path_quality_gate', type=str, default='false',
                        help='If True, skip query-plan PPR injection when executable paths are too diffuse or low-quality.')
    parser.add_argument('--query_plan_ontology_ppr_max_executable_paths_for_quality_gate', type=int, default=0,
                        help='With path quality gate enabled, skip PPR injection when executable path count exceeds this value; <=0 disables.')
    parser.add_argument('--query_plan_ontology_ppr_max_unique_terminals_for_quality_gate', type=int, default=0,
                        help='With path quality gate enabled, skip PPR injection when unique terminal count exceeds this value; <=0 disables.')
    parser.add_argument('--query_plan_ontology_ppr_min_plan_precision', type=float, default=0.0,
                        help='Minimum trace/plan precision required before a query-plan can inject facts into PPR.')
    parser.add_argument('--query_plan_ontology_ppr_require_anchor_fact', type=str, default='true',
                        help='If True, injected PPR facts must contain a query anchor unless their source passage is dense-top-k.')
    parser.add_argument('--query_plan_ontology_ppr_source_top_k', type=int, default=0,
                        help='If >0, PPR injection requires a fact source passage in dense top-k or a fact endpoint query anchor match.')
    parser.add_argument('--query_plan_ontology_ppr_top_k', type=int, default=6,
                        help='Maximum query-plan path facts injected into graph PPR per query.')
    parser.add_argument('--query_plan_ontology_ppr_score_weight', type=float, default=0.55,
                        help='Weight used to raise query-plan injected PPR fact scores toward the current anchor fact score.')
    parser.add_argument('--query_plan_ontology_ppr_max_paths', type=int, default=40,
                        help='Maximum executable query-plan paths considered for PPR fact injection.')
    parser.add_argument('--query_plan_ontology_ppr_relation_match_mode', type=str, default='strict',
                        choices=['strict', 'soft', 'family', 'strict_then_soft'],
                        help='Relation matching mode used for query-plan PPR injection.')
    parser.add_argument('--query_plan_ontology_enable_ppr_passage_seeds', type=str, default='false',
                        help='If True, source passages of high-quality query-plan PPR facts are added as graph PPR reset seeds.')
    parser.add_argument('--query_plan_ontology_ppr_passage_seed_weight', type=float, default=0.08,
                        help='Maximum reset weight assigned to ontology-guided source passage seeds.')
    parser.add_argument('--query_plan_ontology_ppr_passage_seed_top_k', type=int, default=4,
                        help='Maximum ontology-guided source passages added as graph PPR reset seeds per query.')
    parser.add_argument('--query_plan_ontology_ppr_passage_seed_min_plan_precision', type=float, default=0.65,
                        help='Minimum query-plan effective precision required before a source passage can become a PPR passage seed.')
    parser.add_argument('--query_plan_ontology_ppr_passage_seed_require_query_anchor', type=str, default='true',
                        help='If True, source passage PPR seeds must come from facts containing a query anchor.')
    parser.add_argument('--query_plan_ontology_ppr_passage_seed_dense_bonus', type=float, default=0.25,
                        help='Additional score multiplier for ontology passage seeds whose source passage appears in dense top-k.')
    parser.add_argument('--enable_query_plan_ontology_graph_scorer', type=str, default='false',
                        help='If True, boost graph-search fact seeds with executable query-plan path evidence before PPR.')
    parser.add_argument('--query_plan_ontology_graph_scorer_score_weight', type=float, default=0.28,
                        help='Weight used to raise query-plan graph-scorer fact scores toward the current anchor fact score.')
    parser.add_argument('--query_plan_ontology_graph_scorer_top_k', type=int, default=6,
                        help='Maximum new query-plan graph-scorer facts appended to graph-search seeds per query.')
    parser.add_argument('--query_plan_ontology_graph_scorer_min_confidence', type=float, default=0.45,
                        help='Minimum confidence required for query-plan graph-scorer fact boosting.')
    parser.add_argument('--query_plan_ontology_graph_scorer_require_anchor_or_dense_source', type=str, default='true',
                        help='If True, graph-scorer facts must contain a query anchor or originate from dense-top-k source passages.')
    parser.add_argument('--query_plan_ontology_graph_scorer_boost_existing', type=str, default='true',
                        help='If True, boost already-selected top facts when they are part of executable query-plan paths.')
    parser.add_argument('--query_plan_ontology_graph_scorer_reorder_facts', type=str, default='true',
                        help='If True, reorder graph-search fact seeds by graph-scorer-adjusted fact scores.')
    parser.add_argument('--query_plan_ontology_enable_structural_candidate_retriever', type=str, default='false',
                        help='If True, scan retrieved passages with query-plan entity/relation clues to add ontology-supported candidates.')
    parser.add_argument('--query_plan_ontology_structural_candidate_retriever_scan_top_k', type=int, default=80,
                        help='Top retrieved passages scanned by the structural query-plan candidate retriever.')
    parser.add_argument('--query_plan_ontology_structural_candidate_retriever_max_original_rank', type=int, default=120,
                        help='Maximum original rank eligible for structural candidate retrieval; <=0 disables.')
    parser.add_argument('--query_plan_ontology_structural_candidate_retriever_max_added_candidates', type=int, default=8,
                        help='Maximum structural candidates added before candidate promotion.')
    parser.add_argument('--query_plan_ontology_structural_candidate_retriever_min_score', type=float, default=0.56,
                        help='Minimum structural candidate score required before adding a passage candidate.')
    parser.add_argument('--query_plan_ontology_structural_candidate_retriever_require_anchor', type=str, default='true',
                        help='If True, structural candidates must mention a concrete query-plan anchor.')
    parser.add_argument('--query_plan_ontology_structural_candidate_retriever_require_terminal_or_relation', type=str, default='true',
                        help='If True, structural candidates must mention a terminal entity or relation clue.')
    parser.add_argument('--query_plan_ontology_structural_candidate_retriever_min_relation_token_overlap', type=int, default=1,
                        help='Minimum relation-token overlap for structural candidates when no terminal entity is matched.')
    parser.add_argument('--query_plan_ontology_structural_candidate_retriever_reject_generic_anchors', type=str, default='true',
                        help='If True, reject generic/directional anchors for structural candidates.')
    parser.add_argument('--query_plan_ontology_structural_candidate_retriever_score_weight', type=float, default=0.85,
                        help='Score scale used when structural candidates are merged with source-linked scores.')
    parser.add_argument('--query_plan_ontology_structural_candidate_retriever_require_existing_source', type=str, default='false',
                        help='If True, structural scan only enriches source-linked passages instead of adding structural-only candidates.')
    parser.add_argument('--query_plan_ontology_enable_candidate_retriever', type=str, default='false',
                        help='If True, directly promote high-confidence query-plan source passages into retrieval rankings.')
    parser.add_argument('--query_plan_ontology_candidate_retriever_top_k', type=int, default=2,
                        help='Maximum ontology-supported passages promoted by the query-plan candidate retriever per query.')
    parser.add_argument('--query_plan_ontology_candidate_retriever_promote_to_rank', type=int, default=5,
                        help='Target rank for query-plan candidate-retriever promotions.')
    parser.add_argument('--query_plan_ontology_candidate_retriever_preserve_top_k', type=int, default=2,
                        help='Number of baseline top passages protected from candidate-retriever insertion.')
    parser.add_argument('--query_plan_ontology_candidate_retriever_max_original_rank', type=int, default=120,
                        help='Maximum original retrieval rank eligible for candidate-retriever promotion; <=0 disables.')
    parser.add_argument('--query_plan_ontology_candidate_retriever_min_confidence', type=float, default=0.72,
                        help='Minimum structural confidence required before candidate-retriever promotion.')
    parser.add_argument('--query_plan_ontology_candidate_retriever_score_margin', type=float, default=1e-6,
                        help='Small score margin used to place promoted candidates at the requested rank.')
    parser.add_argument('--query_plan_ontology_candidate_retriever_aggressive_promote_to_rank', type=int, default=2,
                        help='Target rank for very high-confidence candidate-retriever promotions.')
    parser.add_argument('--query_plan_ontology_candidate_retriever_aggressive_min_confidence', type=float, default=0.84,
                        help='Minimum confidence required before using the aggressive candidate promotion rank.')
    parser.add_argument('--query_plan_ontology_candidate_retriever_aggressive_require_strict', type=str, default='true',
                        help='If True, aggressive candidate promotions require at least one strict query-plan context.')
    parser.add_argument('--query_plan_ontology_candidate_retriever_aggressive_require_multihop_or_multi_context', type=str, default='true',
                        help='If True, aggressive candidate promotions require a multi-hop path or multiple evidence contexts.')
    parser.add_argument('--query_plan_ontology_candidate_retriever_enable_anchor_guard', type=str, default='true',
                        help='If True, candidate promotions require a concrete anchor or terminal entity.')
    parser.add_argument('--query_plan_ontology_candidate_retriever_allow_structural_only', type=str, default='true',
                        help='If False, passages supported only by structural lexical matching cannot be directly promoted.')
    parser.add_argument('--query_plan_ontology_candidate_retriever_structural_only_min_confidence', type=float, default=0.92,
                        help='Minimum confidence required when structural-only candidate promotion is allowed.')
    parser.add_argument('--query_plan_ontology_enable_terminal_evidence_retriever', type=str, default='false',
                        help='If True, issue query-plan terminal evidence dense subqueries and promote high-confidence answer-terminal passages.')
    parser.add_argument('--query_plan_ontology_terminal_evidence_dense_top_k', type=int, default=40,
                        help='Dense top-k passages considered for each terminal evidence subquery.')
    parser.add_argument('--query_plan_ontology_terminal_evidence_max_query_variants', type=int, default=4,
                        help='Maximum ontology-derived dense subqueries issued per query.')
    parser.add_argument('--query_plan_ontology_terminal_evidence_max_selected', type=int, default=1,
                        help='Maximum terminal evidence passages inserted into the retrieval ranking per query.')
    parser.add_argument('--query_plan_ontology_terminal_evidence_promote_to_rank', type=int, default=5,
                        help='Target rank slot for terminal evidence passage insertion.')
    parser.add_argument('--query_plan_ontology_terminal_evidence_preserve_top_k', type=int, default=2,
                        help='Number of highest baseline-ranked passages protected from terminal evidence insertion.')
    parser.add_argument('--query_plan_ontology_terminal_evidence_min_confidence', type=float, default=0.58,
                        help='Minimum terminal evidence confidence required before inserting a dense-retrieved passage.')
    parser.add_argument('--query_plan_ontology_terminal_evidence_require_relation_hint', type=str, default='true',
                        help='If True, terminal evidence candidates must mention query-plan relation clues.')
    parser.add_argument('--query_plan_ontology_terminal_evidence_require_entity_hint', type=str, default='true',
                        help='If True, terminal evidence candidates must mention query-plan anchor or terminal entities.')
    parser.add_argument('--query_plan_ontology_terminal_evidence_max_entity_terms', type=int, default=6,
                        help='Maximum anchor/terminal entity phrases used to create terminal evidence subqueries.')
    parser.add_argument('--query_plan_ontology_terminal_evidence_score_margin', type=float, default=1e-6,
                        help='Small score margin used to place terminal evidence passages at the requested rank.')
    parser.add_argument('--query_plan_ontology_terminal_evidence_candidate_only', type=str, default='false',
                        help='If True, terminal evidence retrieval only produces candidates and does not directly alter passage ranks.')
    parser.add_argument('--query_plan_ontology_enable_terminal_evidence_verifier', type=str, default='false',
                        help='If True, verify terminal evidence candidates before inserting them into passage ranking.')
    parser.add_argument('--query_plan_ontology_terminal_evidence_verifier_min_confidence', type=float, default=0.74,
                        help='Minimum terminal evidence confidence required by the verifier.')
    parser.add_argument('--query_plan_ontology_terminal_evidence_verifier_min_relation_hits', type=int, default=1,
                        help='Minimum relation-token hits required by the terminal evidence verifier.')
    parser.add_argument('--query_plan_ontology_terminal_evidence_verifier_max_original_rank', type=int, default=120,
                        help='Farthest original passage rank eligible for terminal evidence verifier insertion.')
    parser.add_argument('--query_plan_ontology_terminal_evidence_verifier_max_matched_plans', type=int, default=3,
                        help='Maximum matched query-plan count eligible for terminal evidence verifier insertion.')
    parser.add_argument('--query_plan_ontology_terminal_evidence_verifier_max_executable_paths', type=int, default=6,
                        help='Maximum executable path count eligible for terminal evidence verifier insertion.')
    parser.add_argument('--query_plan_ontology_terminal_evidence_verifier_require_new_coverage', type=str, default='true',
                        help='If True, verifier requires the candidate to add entity/relation coverage absent from the current top window.')
    parser.add_argument('--query_plan_ontology_terminal_evidence_verifier_promote_to_rank', type=int, default=5,
                        help='Target rank for verifier-approved terminal evidence insertion.')
    parser.add_argument('--query_plan_ontology_terminal_evidence_verifier_preserve_top_k', type=int, default=2,
                        help='Number of top-ranked passages protected from verifier-approved terminal evidence insertion.')
    parser.add_argument('--query_plan_ontology_terminal_evidence_verifier_max_inserted', type=int, default=1,
                        help='Maximum verifier-approved terminal evidence passages inserted per query.')
    parser.add_argument('--query_plan_ontology_terminal_evidence_verifier_score_margin', type=float, default=1e-6,
                        help='Small score margin used by verifier-approved terminal evidence insertion.')
    parser.add_argument('--query_plan_ontology_trace_path', type=str, default=None,
                        help='Optional JSONL path for per-query query-plan ontology diagnostics.')
    parser.add_argument('--query_plan_ontology_trace_compact_physical_shadow', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_TRACE_COMPACT_PHYSICAL_SHADOW', 'false'),
                        help='Write only query identity and factorized physical-shadow data to the trace.')
    parser.add_argument('--query_plan_ontology_trace_compact_end_to_end_slate', type=str,
                        default=os.environ.get('QUERY_PLAN_ONTOLOGY_TRACE_COMPACT_END_TO_END_SLATE', 'false'),
                        help='Write a compact trace containing the unified evidence slate and exact training labels.')
    parser.add_argument('--ontology_include_concepts', type=str, default='false',
                        help='If True, append coarse ontology concepts to expanded queries.')
    parser.add_argument('--ontology_include_related_terms', type=str, default='false',
                        help='If True, append ontology neighbor/schema related terms to expanded queries.')
    parser.add_argument('--ontology_min_concept_score', type=int, default=3,
                        help='Minimum concept match score required before a concept can be appended.')
    parser.add_argument('--ontology_require_entity_match', type=str, default='false',
                        help='If True, only apply ontology query expansion when at least one concrete entity is matched.')
    parser.add_argument('--ontology_max_entities_per_query', type=int, default=3,
                        help='Maximum matched ontology entities appended to each query.')
    parser.add_argument('--ontology_max_relations_per_query', type=int, default=3,
                        help='Maximum matched ontology relations appended to each query.')
    parser.add_argument('--ontology_max_entity_terms', type=int, default=5000,
                        help='Maximum entity profiles stored in the graph ontology.')
    parser.add_argument('--ontology_max_relation_schemas', type=int, default=300,
                        help='Maximum relation schemas stored in the graph ontology.')
    parser.add_argument('--rerank_dspy_file_path', type=str,
                        default=None,
                        help='DSPy prompt file used by fact reranking.')
    parser.add_argument('--skip_qa', type=str, default='false',
                        help='If True, run retrieval and retrieval evaluation only, without QA API calls.')
    parser.add_argument('--save_dir', type=str, default='outputs', help='Save directory')
    args = parser.parse_args()

    dataset_name = args.dataset
    save_dir = args.save_dir
    llm_base_url = args.llm_base_url
    llm_name = args.llm_name
    if save_dir == 'outputs':
        save_dir = save_dir + '/' + dataset_name
    else:
        save_dir = save_dir + '_' + dataset_name

    corpus_path = args.corpus_path or f"reproduce/dataset/{dataset_name}_corpus.json"
    logging.info("Using corpus file: %s", corpus_path)
    with open(corpus_path, "r") as f:
        corpus = json.load(f)

    docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]

    force_index_from_scratch = string_to_bool(args.force_index_from_scratch)
    force_openie_from_scratch = string_to_bool(args.force_openie_from_scratch)
    reuse_index_read_only = string_to_bool(args.reuse_index_read_only)
    if reuse_index_read_only and (force_index_from_scratch or force_openie_from_scratch):
        raise ValueError(
            "reuse_index_read_only=true is incompatible with force_index_from_scratch "
            "or force_openie_from_scratch"
        )
    enable_ontology = string_to_bool(args.enable_ontology)
    enable_adaptive_base_access_router = string_to_bool(
        args.enable_adaptive_base_access_router
    )
    adaptive_base_access_collect_only = string_to_bool(
        args.adaptive_base_access_collect_only
    )
    force_ontology_from_scratch = string_to_bool(args.force_ontology_from_scratch)
    ontology_include_concepts = string_to_bool(args.ontology_include_concepts)
    ontology_include_related_terms = string_to_bool(args.ontology_include_related_terms)
    ontology_require_entity_match = string_to_bool(args.ontology_require_entity_match)
    enable_query_aware_fact_expansion = string_to_bool(args.enable_query_aware_fact_expansion)
    enable_query_local_ontology = string_to_bool(args.enable_query_local_ontology)
    enable_query_ontology_prior = string_to_bool(args.enable_query_ontology_prior)
    query_ontology_require_query_anchor = string_to_bool(args.query_ontology_require_query_anchor)
    enable_query_ontology_prior_ppr_injection = string_to_bool(args.enable_query_ontology_prior_ppr_injection)
    query_ontology_prior_ppr_expand_graph_paths = string_to_bool(args.query_ontology_prior_ppr_expand_graph_paths)
    query_ontology_prior_ppr_intent_gate = string_to_bool(args.query_ontology_prior_ppr_intent_gate)
    enable_query_ontology_prior_path_instantiation = string_to_bool(args.enable_query_ontology_prior_path_instantiation)
    enable_query_ontology_prior_passage_lift = string_to_bool(args.enable_query_ontology_prior_passage_lift)
    query_ontology_prior_passage_lift_use_entity_fallback = string_to_bool(args.query_ontology_prior_passage_lift_use_entity_fallback)
    enable_query_plan_ontology = string_to_bool(args.enable_query_plan_ontology)
    query_plan_use_relation_realization = string_to_bool(args.query_plan_use_relation_realization)
    query_plan_relation_realization_allow_family_fallback = string_to_bool(args.query_plan_relation_realization_allow_family_fallback)
    query_plan_relation_realization_require_ontology_hash_match = string_to_bool(
        args.query_plan_relation_realization_require_ontology_hash_match
    )
    query_plan_ontology_enable_schema_linked_router = string_to_bool(
        args.query_plan_ontology_enable_schema_linked_router
    )
    query_plan_ontology_enable_schema_linked_compiler_sidecar = string_to_bool(
        args.query_plan_ontology_enable_schema_linked_compiler_sidecar
    )
    query_plan_ontology_gate_require_fact_endpoints = string_to_bool(args.query_plan_ontology_gate_require_fact_endpoints)
    query_plan_ontology_gate_require_anchor_or_terminal = string_to_bool(args.query_plan_ontology_gate_require_anchor_or_terminal)
    query_plan_ontology_enable_simple_evidence_validator = string_to_bool(args.query_plan_ontology_enable_simple_evidence_validator)
    query_plan_ontology_simple_validator_fallback_to_path_lift = string_to_bool(args.query_plan_ontology_simple_validator_fallback_to_path_lift)
    query_plan_ontology_enable_qa_evidence = string_to_bool(args.query_plan_ontology_enable_qa_evidence)
    query_plan_ontology_enable_evidence_chain_materializer = string_to_bool(
        args.query_plan_ontology_enable_evidence_chain_materializer
    )
    query_plan_ontology_evidence_chain_materializer_require_context_gap = string_to_bool(
        args.query_plan_ontology_evidence_chain_materializer_require_context_gap
    )
    query_plan_ontology_evidence_chain_materializer_add_docs_to_context = string_to_bool(
        args.query_plan_ontology_evidence_chain_materializer_add_docs_to_context
    )
    query_plan_ontology_evidence_chain_materializer_enable_chain_complete = string_to_bool(
        args.query_plan_ontology_evidence_chain_materializer_enable_chain_complete
    )
    query_plan_ontology_evidence_chain_materializer_require_chain_complete_route = string_to_bool(
        args.query_plan_ontology_evidence_chain_materializer_require_chain_complete_route
    )
    query_plan_ontology_evidence_chain_materializer_strict_admission = string_to_bool(
        args.query_plan_ontology_evidence_chain_materializer_strict_admission
    )
    query_plan_ontology_evidence_chain_materializer_context_gap_use_qa_context = string_to_bool(
        args.query_plan_ontology_evidence_chain_materializer_context_gap_use_qa_context
    )
    query_plan_ontology_evidence_chain_materializer_emit_answer_spans = string_to_bool(
        args.query_plan_ontology_evidence_chain_materializer_emit_answer_spans
    )
    query_plan_ontology_enable_answer_candidate_operator = string_to_bool(
        args.query_plan_ontology_enable_answer_candidate_operator
    )
    query_plan_ontology_answer_candidate_require_passage_support = string_to_bool(
        args.query_plan_ontology_answer_candidate_require_passage_support
    )
    query_plan_ontology_answer_candidate_use_doc_spans = string_to_bool(
        args.query_plan_ontology_answer_candidate_use_doc_spans
    )
    query_plan_ontology_answer_candidate_disable_ambiguous_slots = string_to_bool(
        args.query_plan_ontology_answer_candidate_disable_ambiguous_slots
    )
    query_plan_ontology_answer_candidate_enable_slot_hint = string_to_bool(
        args.query_plan_ontology_answer_candidate_enable_slot_hint
    )
    query_plan_ontology_enable_answer_canonicalizer = string_to_bool(
        args.query_plan_ontology_enable_answer_canonicalizer
    )
    query_plan_ontology_answer_canonicalizer_require_candidate_support = string_to_bool(
        args.query_plan_ontology_answer_canonicalizer_require_candidate_support
    )
    query_plan_ontology_answer_canonicalizer_fix_year_only = string_to_bool(
        args.query_plan_ontology_answer_canonicalizer_fix_year_only
    )
    query_plan_ontology_answer_canonicalizer_replace_unknown = string_to_bool(
        args.query_plan_ontology_answer_canonicalizer_replace_unknown
    )
    query_plan_ontology_answer_canonicalizer_allow_expansion = string_to_bool(
        args.query_plan_ontology_answer_canonicalizer_allow_expansion
    )
    query_plan_ontology_answer_canonicalizer_enable_slot_shrink = string_to_bool(
        args.query_plan_ontology_answer_canonicalizer_enable_slot_shrink
    )
    query_plan_ontology_answer_canonicalizer_allow_candidate_shrink = string_to_bool(
        args.query_plan_ontology_answer_canonicalizer_allow_candidate_shrink
    )
    query_plan_ontology_enable_grounded_answer_verifier = string_to_bool(
        args.query_plan_ontology_enable_grounded_answer_verifier
    )
    query_plan_ontology_enable_answer_operator_calibrator = string_to_bool(
        args.query_plan_ontology_enable_answer_operator_calibrator
    )
    query_plan_ontology_enable_typed_answer_projection = string_to_bool(
        args.query_plan_ontology_enable_typed_answer_projection
    )
    query_plan_ontology_typed_answer_projection_require_support = string_to_bool(
        args.query_plan_ontology_typed_answer_projection_require_support
    )
    query_plan_ontology_typed_answer_projection_prefer_chain_order = string_to_bool(
        args.query_plan_ontology_typed_answer_projection_prefer_chain_order
    )
    query_plan_ontology_typed_answer_projection_operator_gate = string_to_bool(
        args.query_plan_ontology_typed_answer_projection_operator_gate
    )
    query_plan_ontology_typed_answer_projection_require_relation_or_slot_source = string_to_bool(
        args.query_plan_ontology_typed_answer_projection_require_relation_or_slot_source
    )
    query_plan_ontology_typed_answer_projection_protect_supported_reader_answer = string_to_bool(
        args.query_plan_ontology_typed_answer_projection_protect_supported_reader_answer
    )
    query_plan_ontology_enable_span_constrained_answer_resolver = string_to_bool(
        args.query_plan_ontology_enable_span_constrained_answer_resolver
    )
    query_plan_ontology_span_resolver_replace_unsupported_only = string_to_bool(
        args.query_plan_ontology_span_resolver_replace_unsupported_only
    )
    query_plan_ontology_span_resolver_protect_supported_reader_answer = string_to_bool(
        args.query_plan_ontology_span_resolver_protect_supported_reader_answer
    )
    query_plan_ontology_enable_adapter_policy = string_to_bool(args.query_plan_ontology_enable_adapter_policy)
    query_plan_ontology_passage_reranker_enable_promotion_verifier = string_to_bool(
        args.query_plan_ontology_passage_reranker_enable_promotion_verifier
    )
    query_plan_ontology_passage_promotion_policy_enable_coverage_gate = string_to_bool(
        args.query_plan_ontology_passage_promotion_policy_enable_coverage_gate
    )
    query_plan_ontology_enable_evidence_context_builder = string_to_bool(
        args.query_plan_ontology_enable_evidence_context_builder
    )
    query_plan_ontology_evidence_context_builder_require_new_evidence_role = string_to_bool(
        args.query_plan_ontology_evidence_context_builder_require_new_evidence_role
    )
    query_plan_ontology_reader_intervention_require_atomic_commit = string_to_bool(
        args.query_plan_ontology_reader_intervention_require_atomic_commit
    )
    query_plan_ontology_evidence_context_builder_enable_chain_guidance = string_to_bool(
        args.query_plan_ontology_evidence_context_builder_enable_chain_guidance
    )
    query_plan_ontology_evidence_context_builder_enable_logical_plan_guidance = string_to_bool(
        args.query_plan_ontology_evidence_context_builder_enable_logical_plan_guidance
    )
    query_plan_ontology_evidence_context_builder_logical_plan_require_commit = string_to_bool(
        args.query_plan_ontology_evidence_context_builder_logical_plan_require_commit
    )
    query_plan_ontology_evidence_context_builder_allow_hint_only = string_to_bool(
        args.query_plan_ontology_evidence_context_builder_allow_hint_only
    )
    query_plan_ontology_evidence_context_builder_enable_comparative_branches = string_to_bool(
        args.query_plan_ontology_evidence_context_builder_enable_comparative_branches
    )
    query_plan_ontology_enable_state_retriever = string_to_bool(
        args.query_plan_ontology_enable_state_retriever
    )
    query_plan_ontology_state_retriever_require_new_slot = string_to_bool(
        args.query_plan_ontology_state_retriever_require_new_slot
    )
    query_plan_ontology_state_retriever_require_policy_approval = string_to_bool(
        args.query_plan_ontology_state_retriever_require_policy_approval
    )
    query_plan_ontology_state_retriever_require_structural_slot = string_to_bool(
        args.query_plan_ontology_state_retriever_require_structural_slot
    )
    query_plan_ontology_enable_subgraph_retriever = string_to_bool(
        args.query_plan_ontology_enable_subgraph_retriever
    )
    query_plan_ontology_subgraph_use_entity_fallback = string_to_bool(
        args.query_plan_ontology_subgraph_use_entity_fallback
    )
    query_plan_ontology_subgraph_require_strong_fact_gate = string_to_bool(
        args.query_plan_ontology_subgraph_require_strong_fact_gate
    )
    query_plan_ontology_enable_support_evidence_retriever = string_to_bool(
        args.query_plan_ontology_enable_support_evidence_retriever
    )
    query_plan_ontology_support_evidence_require_anchor_hit = string_to_bool(
        args.query_plan_ontology_support_evidence_require_anchor_hit
    )
    query_plan_ontology_support_evidence_require_relation_or_answer_hit = string_to_bool(
        args.query_plan_ontology_support_evidence_require_relation_or_answer_hit
    )
    query_plan_ontology_support_evidence_candidate_only = string_to_bool(
        args.query_plan_ontology_support_evidence_candidate_only
    )
    query_plan_ontology_enable_schema_route_retriever = string_to_bool(
        args.query_plan_ontology_enable_schema_route_retriever
    )
    query_plan_ontology_schema_route_candidate_only = string_to_bool(
        args.query_plan_ontology_schema_route_candidate_only
    )
    query_plan_ontology_schema_route_require_anchor_hit = string_to_bool(
        args.query_plan_ontology_schema_route_require_anchor_hit
    )
    query_plan_ontology_schema_route_require_route_signal = string_to_bool(
        args.query_plan_ontology_schema_route_require_route_signal
    )
    query_plan_ontology_schema_route_enable_community_route = string_to_bool(
        args.query_plan_ontology_schema_route_enable_community_route
    )
    query_plan_ontology_schema_route_enable_dfs_route = string_to_bool(
        args.query_plan_ontology_schema_route_enable_dfs_route
    )
    query_plan_ontology_enable_schema_slot_recall_expander = string_to_bool(
        args.query_plan_ontology_enable_schema_slot_recall_expander
    )
    query_plan_ontology_schema_slot_recall_require_anchor_or_relation = string_to_bool(
        args.query_plan_ontology_schema_slot_recall_require_anchor_or_relation
    )
    query_plan_ontology_schema_slot_recall_include_original_query = string_to_bool(
        args.query_plan_ontology_schema_slot_recall_include_original_query
    )
    query_plan_ontology_schema_slot_recall_enable_multi_probe = string_to_bool(
        args.query_plan_ontology_schema_slot_recall_enable_multi_probe
    )
    query_plan_ontology_enable_bind_probe_join = string_to_bool(
        args.query_plan_ontology_enable_bind_probe_join
    )
    query_plan_ontology_bind_probe_join_require_relation_or_answer = string_to_bool(
        args.query_plan_ontology_bind_probe_join_require_relation_or_answer
    )
    query_plan_ontology_bind_probe_join_enable_relation_role_constraints = string_to_bool(
        args.query_plan_ontology_bind_probe_join_enable_relation_role_constraints
    )
    query_plan_ontology_bind_probe_join_require_role_consistent_candidate = string_to_bool(
        args.query_plan_ontology_bind_probe_join_require_role_consistent_candidate
    )
    query_plan_ontology_enable_bind_probe_transaction_optimizer = string_to_bool(
        args.query_plan_ontology_enable_bind_probe_transaction_optimizer
    )
    query_plan_ontology_bind_probe_transaction_require_source_top1 = string_to_bool(
        args.query_plan_ontology_bind_probe_transaction_require_source_top1
    )
    query_plan_ontology_enable_entity_keyed_top5_optimizer = string_to_bool(
        args.query_plan_ontology_enable_entity_keyed_top5_optimizer
    )
    query_plan_ontology_enable_factorized_query_executor = string_to_bool(
        args.query_plan_ontology_enable_factorized_query_executor
    )
    query_plan_ontology_factorized_enable_entity_relation_index_scan = string_to_bool(
        args.query_plan_ontology_factorized_enable_entity_relation_index_scan
    )
    query_plan_ontology_factorized_enable_cascaded_bind_join = string_to_bool(
        args.query_plan_ontology_factorized_enable_cascaded_bind_join
    )
    query_plan_ontology_factorized_enable_dual_branch_comparison_join = string_to_bool(
        args.query_plan_ontology_factorized_enable_dual_branch_comparison_join
    )
    query_plan_ontology_factorized_enable_top2_admission = string_to_bool(
        args.query_plan_ontology_factorized_enable_top2_admission
    )
    query_plan_ontology_factorized_enable_top5_admission = string_to_bool(
        args.query_plan_ontology_factorized_enable_top5_admission
    )
    query_plan_ontology_factorized_shadow_only = string_to_bool(
        args.query_plan_ontology_factorized_shadow_only
    )
    query_plan_ontology_factorized_enable_typed_semantic_compiler = string_to_bool(
        args.query_plan_ontology_factorized_enable_typed_semantic_compiler
    )
    query_plan_ontology_factorized_enable_lossless_typed_plan_ir = string_to_bool(
        args.query_plan_ontology_factorized_enable_lossless_typed_plan_ir
    )
    query_plan_ontology_factorized_require_typed_variable_unification = string_to_bool(
        args.query_plan_ontology_factorized_require_typed_variable_unification
    )
    query_plan_ontology_factorized_enable_certificate_aware_plan_portfolio = string_to_bool(
        args.query_plan_ontology_factorized_enable_certificate_aware_plan_portfolio
    )
    query_plan_ontology_factorized_enable_certificate_aware_access_path_reoptimization = string_to_bool(
        args.query_plan_ontology_factorized_enable_certificate_aware_access_path_reoptimization
    )
    query_plan_ontology_factorized_enable_typed_certificate_top2 = string_to_bool(
        args.query_plan_ontology_factorized_enable_typed_certificate_top2
    )
    query_plan_ontology_factorized_require_exact_entity_keys = string_to_bool(
        args.query_plan_ontology_factorized_require_exact_entity_keys
    )
    query_plan_ontology_factorized_require_typed_join_keys = string_to_bool(
        args.query_plan_ontology_factorized_require_typed_join_keys
    )
    query_plan_ontology_factorized_enable_simple_top2_admission = string_to_bool(
        args.query_plan_ontology_factorized_enable_simple_top2_admission
    )
    query_plan_ontology_factorized_require_chain_source_top2 = string_to_bool(
        args.query_plan_ontology_factorized_require_chain_source_top2
    )
    query_plan_ontology_factorized_enable_branch_complete_compiler = string_to_bool(
        args.query_plan_ontology_factorized_enable_branch_complete_compiler
    )
    query_plan_ontology_factorized_require_branch_qualifier_evidence = string_to_bool(
        args.query_plan_ontology_factorized_require_branch_qualifier_evidence
    )
    query_plan_ontology_factorized_enable_evidence_query_compiler = string_to_bool(
        args.query_plan_ontology_factorized_enable_evidence_query_compiler
    )
    query_plan_ontology_factorized_enable_executable_ontology_compiler = string_to_bool(
        args.query_plan_ontology_factorized_enable_executable_ontology_compiler
    )
    query_plan_ontology_factorized_enable_relation_realization = string_to_bool(
        args.query_plan_ontology_factorized_enable_relation_realization
    )
    query_plan_ontology_factorized_enable_relation_keyed_catalog = string_to_bool(
        args.query_plan_ontology_factorized_enable_relation_keyed_catalog
    )
    query_plan_ontology_factorized_relation_keyed_allow_matched_schema_fallback = string_to_bool(
        args.query_plan_ontology_factorized_relation_keyed_allow_matched_schema_fallback
    )
    query_plan_ontology_factorized_enable_query_relation_alternative_recovery = string_to_bool(
        args.query_plan_ontology_factorized_enable_query_relation_alternative_recovery
    )
    query_plan_ontology_factorized_enable_execution_aware_contract = string_to_bool(
        args.query_plan_ontology_factorized_enable_execution_aware_contract
    )
    query_plan_ontology_factorized_enable_provenance_plan_portfolio = string_to_bool(
        args.query_plan_ontology_factorized_enable_provenance_plan_portfolio
    )
    query_plan_ontology_factorized_enable_candidate_window_path_compiler = string_to_bool(
        args.query_plan_ontology_factorized_enable_candidate_window_path_compiler
    )
    query_plan_ontology_factorized_candidate_window_path_allow_contextual_terminal_edges = string_to_bool(
        args.query_plan_ontology_factorized_candidate_window_path_allow_contextual_terminal_edges
    )
    query_plan_ontology_factorized_candidate_window_path_enable_entity_index_expansion = string_to_bool(
        args.query_plan_ontology_factorized_candidate_window_path_enable_entity_index_expansion
    )
    query_plan_ontology_factorized_candidate_window_path_enable_bound_dense_expansion = string_to_bool(
        args.query_plan_ontology_factorized_candidate_window_path_enable_bound_dense_expansion
    )
    query_plan_ontology_factorized_candidate_window_path_enable_relation_agnostic_bridge = string_to_bool(
        args.query_plan_ontology_factorized_candidate_window_path_enable_relation_agnostic_bridge
    )
    query_plan_ontology_factorized_enable_inverse_direction_fallback = string_to_bool(
        args.query_plan_ontology_factorized_enable_inverse_direction_fallback
    )
    query_plan_ontology_factorized_enable_whole_warehouse_probe = string_to_bool(
        args.query_plan_ontology_factorized_enable_whole_warehouse_probe
    )
    query_plan_ontology_factorized_enable_cost_based_physical_planner = string_to_bool(
        args.query_plan_ontology_factorized_enable_cost_based_physical_planner
    )
    query_plan_ontology_factorized_enable_physical_shadow_collection = string_to_bool(
        args.query_plan_ontology_factorized_enable_physical_shadow_collection
    )
    query_plan_ontology_trace_compact_physical_shadow = string_to_bool(
        args.query_plan_ontology_trace_compact_physical_shadow
    )
    query_plan_ontology_trace_compact_end_to_end_slate = string_to_bool(
        args.query_plan_ontology_trace_compact_end_to_end_slate
    )
    query_plan_ontology_factorized_enable_missing_slot_executor = string_to_bool(
        args.query_plan_ontology_factorized_enable_missing_slot_executor
    )
    query_plan_ontology_factorized_missing_slot_enable_top2_admission = string_to_bool(
        args.query_plan_ontology_factorized_missing_slot_enable_top2_admission
    )
    query_plan_ontology_factorized_enable_typed_proof_join = string_to_bool(
        args.query_plan_ontology_factorized_enable_typed_proof_join
    )
    query_plan_ontology_factorized_enable_proof_gap_access_path = string_to_bool(
        args.query_plan_ontology_factorized_enable_proof_gap_access_path
    )
    query_plan_ontology_factorized_proof_gap_require_external_member = string_to_bool(
        args.query_plan_ontology_factorized_proof_gap_require_external_member
    )
    query_plan_ontology_factorized_proof_gap_merge_candidate_pool = string_to_bool(
        args.query_plan_ontology_factorized_proof_gap_merge_candidate_pool
    )
    query_plan_ontology_factorized_require_branch_complete_proof = string_to_bool(
        args.query_plan_ontology_factorized_require_branch_complete_proof
    )
    query_plan_ontology_factorized_require_provenance_validation = string_to_bool(
        args.query_plan_ontology_factorized_require_provenance_validation
    )
    query_plan_ontology_factorized_require_strict_branch_identity = string_to_bool(
        args.query_plan_ontology_factorized_require_strict_branch_identity
    )
    query_plan_ontology_factorized_project_full_proof_set = string_to_bool(
        args.query_plan_ontology_factorized_project_full_proof_set
    )
    query_plan_ontology_factorized_enable_proof_residual_commit = string_to_bool(
        args.query_plan_ontology_factorized_enable_proof_residual_commit
    )
    query_plan_ontology_factorized_proof_residual_direct_commit = string_to_bool(
        args.query_plan_ontology_factorized_proof_residual_direct_commit
    )
    query_plan_ontology_factorized_proof_residual_post_optimizer_commit = string_to_bool(
        args.query_plan_ontology_factorized_proof_residual_post_optimizer_commit
    )
    query_plan_ontology_factorized_proof_residual_enable_top2 = string_to_bool(
        args.query_plan_ontology_factorized_proof_residual_enable_top2
    )
    query_plan_ontology_factorized_enable_proof_materialized_view = string_to_bool(
        args.query_plan_ontology_factorized_enable_proof_materialized_view
    )
    query_plan_ontology_factorized_enable_learned_delta_admission = string_to_bool(
        args.query_plan_ontology_factorized_enable_learned_delta_admission
    )
    query_plan_ontology_factorized_delta_admission_fail_open = string_to_bool(
        args.query_plan_ontology_factorized_delta_admission_fail_open
    )
    query_plan_ontology_factorized_delta_admission_collect_only = string_to_bool(
        args.query_plan_ontology_factorized_delta_admission_collect_only
    )
    query_plan_ontology_factorized_enable_set_level_optimizer = string_to_bool(
        args.query_plan_ontology_factorized_enable_set_level_optimizer
    )
    query_plan_ontology_factorized_set_level_optimizer_fail_open = string_to_bool(
        args.query_plan_ontology_factorized_set_level_optimizer_fail_open
    )
    query_plan_ontology_factorized_enable_incumbent_reoptimizer = string_to_bool(
        args.query_plan_ontology_factorized_enable_incumbent_reoptimizer
    )
    query_plan_ontology_factorized_enable_counterfactual_action_replay = string_to_bool(
        args.query_plan_ontology_factorized_enable_counterfactual_action_replay
    )
    query_plan_ontology_factorized_action_replay_include_evidence_warehouse = string_to_bool(
        args.query_plan_ontology_factorized_action_replay_include_evidence_warehouse
    )
    query_plan_ontology_factorized_action_replay_enable_atomic_chain_top2 = string_to_bool(
        args.query_plan_ontology_factorized_action_replay_enable_atomic_chain_top2
    )
    query_plan_ontology_factorized_enable_evidence_set_compiler = string_to_bool(
        args.query_plan_ontology_factorized_enable_evidence_set_compiler
    )
    query_plan_ontology_enable_top5_listwise_optimizer = string_to_bool(
        args.query_plan_ontology_enable_top5_listwise_optimizer
    )
    query_plan_ontology_top5_listwise_require_ontology_candidate = string_to_bool(
        args.query_plan_ontology_top5_listwise_require_ontology_candidate
    )
    query_plan_ontology_enable_projected_graph_retriever = string_to_bool(
        args.query_plan_ontology_enable_projected_graph_retriever
    )
    query_plan_ontology_projected_graph_candidate_only = string_to_bool(
        args.query_plan_ontology_projected_graph_candidate_only
    )
    query_plan_ontology_projected_graph_require_anchor_overlap = string_to_bool(
        args.query_plan_ontology_projected_graph_require_anchor_overlap
    )
    query_plan_ontology_projected_graph_enable_bridge_paths = string_to_bool(
        args.query_plan_ontology_projected_graph_enable_bridge_paths
    )
    query_plan_ontology_enable_evidence_index_retriever = string_to_bool(
        args.query_plan_ontology_enable_evidence_index_retriever
    )
    query_plan_ontology_evidence_index_candidate_only = string_to_bool(
        args.query_plan_ontology_evidence_index_candidate_only
    )
    query_plan_ontology_evidence_index_require_anchor_overlap = string_to_bool(
        args.query_plan_ontology_evidence_index_require_anchor_overlap
    )
    query_plan_ontology_evidence_index_top2_require_terminal = string_to_bool(
        args.query_plan_ontology_evidence_index_top2_require_terminal
    )
    query_plan_ontology_enable_path_executor_retriever = string_to_bool(
        args.query_plan_ontology_enable_path_executor_retriever
    )
    query_plan_ontology_path_executor_candidate_only = string_to_bool(
        args.query_plan_ontology_path_executor_candidate_only
    )
    query_plan_ontology_path_executor_enable_answer_type_completion = string_to_bool(
        args.query_plan_ontology_path_executor_enable_answer_type_completion
    )
    query_plan_ontology_path_executor_enable_terminal_type_filter = string_to_bool(
        args.query_plan_ontology_path_executor_enable_terminal_type_filter
    )
    query_plan_ontology_path_executor_allow_query_condition_single = string_to_bool(
        args.query_plan_ontology_path_executor_allow_query_condition_single
    )
    query_plan_ontology_path_executor_require_terminal_type_for_single_step = string_to_bool(
        args.query_plan_ontology_path_executor_require_terminal_type_for_single_step
    )
    query_plan_ontology_path_executor_strict_query_relation_binding = string_to_bool(
        args.query_plan_ontology_path_executor_strict_query_relation_binding
    )
    query_plan_ontology_enable_failure_evidence_retriever = string_to_bool(
        args.query_plan_ontology_enable_failure_evidence_retriever
    )
    query_plan_ontology_failure_evidence_require_anchor_hit = string_to_bool(
        args.query_plan_ontology_failure_evidence_require_anchor_hit
    )
    query_plan_ontology_failure_evidence_require_rule_term_hit = string_to_bool(
        args.query_plan_ontology_failure_evidence_require_rule_term_hit
    )
    query_plan_ontology_failure_evidence_candidate_only = string_to_bool(
        args.query_plan_ontology_failure_evidence_candidate_only
    )
    enable_evidence_grounded_ontology = string_to_bool(args.enable_evidence_grounded_ontology)
    evidence_grounded_ontology_candidate_only = string_to_bool(args.evidence_grounded_ontology_candidate_only)
    query_plan_ontology_enable_llm_evidence_verifier = string_to_bool(
        args.query_plan_ontology_enable_llm_evidence_verifier
    )
    query_plan_ontology_enable_supervised_evidence_ranker = string_to_bool(
        args.query_plan_ontology_enable_supervised_evidence_ranker
    )
    query_plan_ontology_enable_metric_aligned_slate_ranker = string_to_bool(
        args.query_plan_ontology_enable_metric_aligned_slate_ranker
    )
    query_plan_ontology_enable_post_physical_delta_optimizer = string_to_bool(
        args.query_plan_ontology_enable_post_physical_delta_optimizer
    )
    query_plan_ontology_post_physical_delta_optimizer_enforce_cq = string_to_bool(
        args.query_plan_ontology_post_physical_delta_optimizer_enforce_cq
    )
    query_plan_ontology_post_physical_delta_optimizer_shadow_only = string_to_bool(
        args.query_plan_ontology_post_physical_delta_optimizer_shadow_only
    )
    query_plan_ontology_enable_complete_plan_optimizer = string_to_bool(
        args.query_plan_ontology_enable_complete_plan_optimizer
    )
    query_plan_ontology_complete_plan_optimizer_shadow_only = string_to_bool(
        args.query_plan_ontology_complete_plan_optimizer_shadow_only
    )
    query_plan_ontology_complete_plan_integrate_semantic_access = string_to_bool(
        args.query_plan_ontology_complete_plan_integrate_semantic_access
    )
    query_plan_ontology_complete_plan_immutable_incumbent = string_to_bool(
        args.query_plan_ontology_complete_plan_immutable_incumbent
    )
    query_plan_ontology_enable_agentic_evidence_repair = string_to_bool(
        args.query_plan_ontology_enable_agentic_evidence_repair
    )
    query_plan_ontology_agentic_evidence_repair_shadow_only = string_to_bool(
        args.query_plan_ontology_agentic_evidence_repair_shadow_only
    )
    query_plan_ontology_agentic_evidence_repair_enable_top2 = string_to_bool(
        args.query_plan_ontology_agentic_evidence_repair_enable_top2
    )
    query_plan_ontology_agentic_evidence_repair_enable_top5 = string_to_bool(
        args.query_plan_ontology_agentic_evidence_repair_enable_top5
    )
    query_plan_ontology_agentic_evidence_repair_require_query_local_proof = string_to_bool(
        args.query_plan_ontology_agentic_evidence_repair_require_query_local_proof
    )
    query_plan_ontology_agentic_evidence_repair_require_calibrated_delta = string_to_bool(
        args.query_plan_ontology_agentic_evidence_repair_require_calibrated_delta
    )
    query_plan_ontology_agentic_evidence_repair_skip_after_upstream_commit = string_to_bool(
        args.query_plan_ontology_agentic_evidence_repair_skip_after_upstream_commit
    )
    query_plan_ontology_enable_agentic_residual_retrieval = string_to_bool(
        args.query_plan_ontology_enable_agentic_residual_retrieval
    )
    query_plan_ontology_agentic_residual_shadow_only = string_to_bool(
        args.query_plan_ontology_agentic_residual_shadow_only
    )
    query_plan_ontology_enable_certified_direct_execution = string_to_bool(
        args.query_plan_ontology_enable_certified_direct_execution
    )
    query_plan_ontology_require_certified_schema_mask = string_to_bool(
        args.query_plan_ontology_require_certified_schema_mask
    )
    query_plan_ontology_certified_direct_shadow_only = string_to_bool(
        args.query_plan_ontology_certified_direct_shadow_only
    )
    query_plan_ontology_certified_direct_require_policy = string_to_bool(
        args.query_plan_ontology_certified_direct_require_policy
    )
    query_plan_ontology_certified_direct_require_complete_proof = string_to_bool(
        args.query_plan_ontology_certified_direct_require_complete_proof
    )
    query_plan_ontology_enable_missing_evidence_completer = string_to_bool(
        args.query_plan_ontology_enable_missing_evidence_completer
    )
    query_plan_ontology_missing_evidence_require_relation_hit = string_to_bool(
        args.query_plan_ontology_missing_evidence_require_relation_hit
    )
    query_plan_ontology_missing_evidence_require_both_endpoints = string_to_bool(
        args.query_plan_ontology_missing_evidence_require_both_endpoints
    )
    query_plan_ontology_missing_evidence_candidate_only = string_to_bool(
        args.query_plan_ontology_missing_evidence_candidate_only
    )
    query_plan_ontology_missing_evidence_include_covered_slots = string_to_bool(
        args.query_plan_ontology_missing_evidence_include_covered_slots
    )
    query_plan_ontology_enable_evidence_set_retriever = string_to_bool(
        args.query_plan_ontology_enable_evidence_set_retriever
    )
    query_plan_ontology_evidence_set_enable_optimizer_candidate_pool = string_to_bool(
        args.query_plan_ontology_evidence_set_enable_optimizer_candidate_pool
    )
    query_plan_ontology_evidence_set_require_new_slot = string_to_bool(
        args.query_plan_ontology_evidence_set_require_new_slot
    )
    query_plan_ontology_evidence_set_enable_source_gap_aperture = string_to_bool(
        args.query_plan_ontology_evidence_set_enable_source_gap_aperture
    )
    query_plan_ontology_evidence_set_source_gap_aperture_allow_no_new_slot = string_to_bool(
        args.query_plan_ontology_evidence_set_source_gap_aperture_allow_no_new_slot
    )
    query_plan_ontology_evidence_set_enable_intervention_policy = string_to_bool(
        args.query_plan_ontology_evidence_set_enable_intervention_policy
    )
    query_plan_ontology_evidence_set_intervention_allow_relation_only_simple = string_to_bool(
        args.query_plan_ontology_evidence_set_intervention_allow_relation_only_simple
    )
    query_plan_ontology_evidence_set_enable_counterfactual_policy = string_to_bool(
        args.query_plan_ontology_evidence_set_enable_counterfactual_policy
    )
    query_plan_ontology_evidence_set_utility_planner_require_complex_or_path = string_to_bool(
        args.query_plan_ontology_evidence_set_utility_planner_require_complex_or_path
    )
    query_plan_ontology_evidence_set_recall_plan_optimizer_allow_no_new_slot = string_to_bool(
        args.query_plan_ontology_evidence_set_recall_plan_optimizer_allow_no_new_slot
    )
    query_plan_ontology_evidence_set_enable_rank_window_recall = string_to_bool(
        args.query_plan_ontology_evidence_set_enable_rank_window_recall
    )
    query_plan_ontology_evidence_set_rank_window_require_text_signal = string_to_bool(
        args.query_plan_ontology_evidence_set_rank_window_require_text_signal
    )
    query_plan_ontology_evidence_set_rank_window_defer_to_planner = string_to_bool(
        args.query_plan_ontology_evidence_set_rank_window_defer_to_planner
    )
    query_plan_ontology_evidence_set_rank_window_simple_require_answer_signal = string_to_bool(
        args.query_plan_ontology_evidence_set_rank_window_simple_require_answer_signal
    )
    query_plan_ontology_evidence_set_rank_window_enable_net_benefit_gate = string_to_bool(
        args.query_plan_ontology_evidence_set_rank_window_enable_net_benefit_gate
    )
    query_plan_ontology_evidence_set_rank_window_allow_no_new_slot_bypass = string_to_bool(
        args.query_plan_ontology_evidence_set_rank_window_allow_no_new_slot_bypass
    )
    query_plan_ontology_evidence_set_rank_window_no_new_slot_require_path_or_terminal = string_to_bool(
        args.query_plan_ontology_evidence_set_rank_window_no_new_slot_require_path_or_terminal
    )
    query_plan_ontology_evidence_set_enable_evidence_warehouse_optimizer = string_to_bool(
        args.query_plan_ontology_evidence_set_enable_evidence_warehouse_optimizer
    )
    query_plan_ontology_evidence_set_enable_final_rank_evidence_warehouse = string_to_bool(
        args.query_plan_ontology_evidence_set_enable_final_rank_evidence_warehouse
    )
    query_plan_ontology_evidence_set_evidence_warehouse_require_answer_signal = string_to_bool(
        args.query_plan_ontology_evidence_set_evidence_warehouse_require_answer_signal
    )
    query_plan_ontology_evidence_set_evidence_warehouse_allow_relation_only = string_to_bool(
        args.query_plan_ontology_evidence_set_evidence_warehouse_allow_relation_only
    )
    query_plan_ontology_evidence_set_evidence_warehouse_selectivity_enable_recall_bypass = string_to_bool(
        args.query_plan_ontology_evidence_set_evidence_warehouse_selectivity_enable_recall_bypass
    )
    query_plan_ontology_evidence_set_evidence_warehouse_selectivity_enable_top2_guard_bypass = string_to_bool(
        args.query_plan_ontology_evidence_set_evidence_warehouse_selectivity_enable_top2_guard_bypass
    )
    query_plan_ontology_evidence_set_evidence_warehouse_enable_action_counterfactual_gate = string_to_bool(
        args.query_plan_ontology_evidence_set_evidence_warehouse_enable_action_counterfactual_gate
    )
    query_plan_ontology_evidence_set_evidence_warehouse_enable_set_coverage_cbo = string_to_bool(
        args.query_plan_ontology_evidence_set_evidence_warehouse_enable_set_coverage_cbo
    )
    query_plan_ontology_evidence_set_evidence_warehouse_answer_aware_require_direct_signal = string_to_bool(
        args.query_plan_ontology_evidence_set_evidence_warehouse_answer_aware_require_direct_signal
    )
    query_plan_ontology_evidence_set_evidence_warehouse_answer_aware_allow_top2_guard_bypass = string_to_bool(
        args.query_plan_ontology_evidence_set_evidence_warehouse_answer_aware_allow_top2_guard_bypass
    )
    query_plan_ontology_evidence_set_enable_top5_to_top2_promotion = string_to_bool(
        args.query_plan_ontology_evidence_set_enable_top5_to_top2_promotion
    )
    query_plan_ontology_evidence_set_top5_to_top2_promotion_require_ontology_signal = string_to_bool(
        args.query_plan_ontology_evidence_set_top5_to_top2_promotion_require_ontology_signal
    )
    query_plan_ontology_evidence_set_enable_pairwise_top2_join = string_to_bool(
        args.query_plan_ontology_evidence_set_enable_pairwise_top2_join
    )
    query_plan_ontology_evidence_set_pairwise_top2_join_protect_top1 = string_to_bool(
        args.query_plan_ontology_evidence_set_pairwise_top2_join_protect_top1
    )
    query_plan_ontology_evidence_set_pairwise_top2_join_require_answer_signal = string_to_bool(
        args.query_plan_ontology_evidence_set_pairwise_top2_join_require_answer_signal
    )
    query_plan_ontology_evidence_set_enable_decomposed_chain_join = string_to_bool(
        args.query_plan_ontology_evidence_set_enable_decomposed_chain_join
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_protect_top1 = string_to_bool(
        args.query_plan_ontology_evidence_set_decomposed_chain_join_protect_top1
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_enable_decomposition = string_to_bool(
        args.query_plan_ontology_evidence_set_decomposed_chain_join_enable_decomposition
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_enable_subquery_access = string_to_bool(
        args.query_plan_ontology_evidence_set_decomposed_chain_join_enable_subquery_access
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_enable_chain_join = string_to_bool(
        args.query_plan_ontology_evidence_set_decomposed_chain_join_enable_chain_join
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_require_bridge_signal = string_to_bool(
        args.query_plan_ontology_evidence_set_decomposed_chain_join_require_bridge_signal
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_enable_top5_fallback = string_to_bool(
        args.query_plan_ontology_evidence_set_decomposed_chain_join_enable_top5_fallback
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_top5_require_answer_signal = string_to_bool(
        args.query_plan_ontology_evidence_set_decomposed_chain_join_top5_require_answer_signal
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_top2_require_answer_signal = string_to_bool(
        args.query_plan_ontology_evidence_set_decomposed_chain_join_top2_require_answer_signal
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_top2_require_non_simple = string_to_bool(
        args.query_plan_ontology_evidence_set_decomposed_chain_join_top2_require_non_simple
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_enable_slot_completion_promotion = string_to_bool(
        args.query_plan_ontology_evidence_set_decomposed_chain_join_enable_slot_completion_promotion
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_slot_completion_allow_top1_replacement = string_to_bool(
        args.query_plan_ontology_evidence_set_decomposed_chain_join_slot_completion_allow_top1_replacement
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_slot_completion_require_bridge_or_answer = string_to_bool(
        args.query_plan_ontology_evidence_set_decomposed_chain_join_slot_completion_require_bridge_or_answer
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_enable_metric_regret_gate = string_to_bool(
        args.query_plan_ontology_evidence_set_decomposed_chain_join_enable_metric_regret_gate
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_metric_regret_protect_top1 = string_to_bool(
        args.query_plan_ontology_evidence_set_decomposed_chain_join_metric_regret_protect_top1
    )
    query_plan_ontology_evidence_set_decomposed_chain_join_metric_regret_enable_top5_append = string_to_bool(
        args.query_plan_ontology_evidence_set_decomposed_chain_join_metric_regret_enable_top5_append
    )
    query_plan_ontology_evidence_set_enable_regret_calibrated_set_optimizer = string_to_bool(
        args.query_plan_ontology_evidence_set_enable_regret_calibrated_set_optimizer
    )
    query_plan_ontology_evidence_set_regret_calibrated_set_protect_top1 = string_to_bool(
        args.query_plan_ontology_evidence_set_regret_calibrated_set_protect_top1
    )
    query_plan_ontology_evidence_set_regret_calibrated_set_require_answer_signal = string_to_bool(
        args.query_plan_ontology_evidence_set_regret_calibrated_set_require_answer_signal
    )
    query_plan_ontology_evidence_set_regret_calibrated_set_enable_top5_safe_bypass = string_to_bool(
        args.query_plan_ontology_evidence_set_regret_calibrated_set_enable_top5_safe_bypass
    )
    query_plan_ontology_evidence_set_enable_evidence_sidecar = string_to_bool(
        args.query_plan_ontology_evidence_set_enable_evidence_sidecar
    )
    query_plan_ontology_evidence_set_evidence_sidecar_allow_relation_only = string_to_bool(
        args.query_plan_ontology_evidence_set_evidence_sidecar_allow_relation_only
    )
    query_plan_ontology_evidence_set_evidence_sidecar_require_anchor_hit = string_to_bool(
        args.query_plan_ontology_evidence_set_evidence_sidecar_require_anchor_hit
    )
    query_plan_ontology_evidence_set_evidence_sidecar_require_terminal_hit = string_to_bool(
        args.query_plan_ontology_evidence_set_evidence_sidecar_require_terminal_hit
    )
    query_plan_ontology_evidence_set_evidence_sidecar_require_preferred_source = string_to_bool(
        args.query_plan_ontology_evidence_set_evidence_sidecar_require_preferred_source
    )
    query_plan_ontology_evidence_set_evidence_sidecar_disable_simple_without_answer_overlap = string_to_bool(
        args.query_plan_ontology_evidence_set_evidence_sidecar_disable_simple_without_answer_overlap
    )
    query_plan_ontology_evidence_set_evidence_sidecar_disable_answer_type_conflict = string_to_bool(
        args.query_plan_ontology_evidence_set_evidence_sidecar_disable_answer_type_conflict
    )
    query_plan_ontology_evidence_set_strict_backbone_scope = string_to_bool(
        args.query_plan_ontology_evidence_set_strict_backbone_scope
    )
    query_plan_ontology_evidence_set_recall_swapper_allow_relation_only_top2 = string_to_bool(
        args.query_plan_ontology_evidence_set_recall_swapper_allow_relation_only_top2
    )
    query_plan_ontology_evidence_set_rank2_rescue_require_terminal_or_strict = string_to_bool(
        args.query_plan_ontology_evidence_set_rank2_rescue_require_terminal_or_strict
    )
    query_plan_ontology_evidence_set_optimizer_require_complex_or_path = string_to_bool(
        args.query_plan_ontology_evidence_set_optimizer_require_complex_or_path
    )
    query_plan_ontology_evidence_set_rank2_pairwise_optimizer_require_complex_or_path = string_to_bool(
        args.query_plan_ontology_evidence_set_rank2_pairwise_optimizer_require_complex_or_path
    )
    query_plan_ontology_evidence_set_rank2_pairwise_optimizer_after_evidence_optimizer = string_to_bool(
        args.query_plan_ontology_evidence_set_rank2_pairwise_optimizer_after_evidence_optimizer
    )
    query_plan_ontology_evidence_set_enable_schema_slot_utility = string_to_bool(
        args.query_plan_ontology_evidence_set_enable_schema_slot_utility
    )
    query_plan_ontology_evidence_set_enable_provenance_groupwise_selector = string_to_bool(
        args.query_plan_ontology_evidence_set_enable_provenance_groupwise_selector
    )
    query_plan_ontology_evidence_set_provenance_groupwise_require_text_signal = string_to_bool(
        args.query_plan_ontology_evidence_set_provenance_groupwise_require_text_signal
    )
    query_plan_ontology_evidence_set_enable_evidence_join_planner = string_to_bool(
        args.query_plan_ontology_evidence_set_enable_evidence_join_planner
    )
    query_plan_ontology_evidence_set_evidence_join_planner_require_text_or_path = string_to_bool(
        args.query_plan_ontology_evidence_set_evidence_join_planner_require_text_or_path
    )
    query_plan_ontology_evidence_set_evidence_join_planner_enable_answer_slot_rescue = string_to_bool(
        args.query_plan_ontology_evidence_set_evidence_join_planner_enable_answer_slot_rescue
    )
    query_plan_ontology_evidence_set_evidence_join_planner_enable_operator_aware = string_to_bool(
        args.query_plan_ontology_evidence_set_evidence_join_planner_enable_operator_aware
    )
    query_plan_ontology_evidence_set_evidence_join_planner_enable_risk_gate = string_to_bool(
        args.query_plan_ontology_evidence_set_evidence_join_planner_enable_risk_gate
    )
    query_plan_ontology_evidence_set_enable_answer_pairwise_verifier = string_to_bool(
        args.query_plan_ontology_evidence_set_enable_answer_pairwise_verifier
    )
    query_plan_ontology_evidence_set_answer_pairwise_require_answer_signal = string_to_bool(
        args.query_plan_ontology_evidence_set_answer_pairwise_require_answer_signal
    )
    query_plan_ontology_evidence_set_enable_counterfactual_text_verifier = string_to_bool(
        args.query_plan_ontology_evidence_set_enable_counterfactual_text_verifier
    )
    query_plan_ontology_evidence_set_counterfactual_text_verifier_require_answer_signal = string_to_bool(
        args.query_plan_ontology_evidence_set_counterfactual_text_verifier_require_answer_signal
    )
    query_plan_ontology_evidence_set_enable_decomposition_slots = string_to_bool(
        args.query_plan_ontology_evidence_set_enable_decomposition_slots
    )
    query_plan_ontology_evidence_set_enable_path_aware_selector = string_to_bool(
        args.query_plan_ontology_evidence_set_enable_path_aware_selector
    )
    query_plan_ontology_evidence_set_path_aware_require_query_relation_coverage = string_to_bool(
        args.query_plan_ontology_evidence_set_path_aware_require_query_relation_coverage
    )
    query_plan_ontology_enable_quality_gate = string_to_bool(args.query_plan_ontology_enable_quality_gate)
    query_plan_ontology_quality_gate_allow_simple_validator = string_to_bool(args.query_plan_ontology_quality_gate_allow_simple_validator)
    query_plan_ontology_enable_qa_format_hint = string_to_bool(args.query_plan_ontology_enable_qa_format_hint)
    query_plan_ontology_enable_answer_path_verifier = string_to_bool(args.query_plan_ontology_enable_answer_path_verifier)
    query_plan_ontology_answer_path_filter_lift = string_to_bool(args.query_plan_ontology_answer_path_filter_lift)
    query_plan_ontology_answer_path_filter_qa_evidence = string_to_bool(args.query_plan_ontology_answer_path_filter_qa_evidence)
    query_plan_ontology_enable_answer_object_gate = string_to_bool(args.query_plan_ontology_enable_answer_object_gate)
    query_plan_ontology_enable_query_conditioning = string_to_bool(args.query_plan_ontology_enable_query_conditioning)
    query_plan_ontology_condition_require_relation_overlap = string_to_bool(args.query_plan_ontology_condition_require_relation_overlap)
    query_plan_ontology_enable_simple_fallback = string_to_bool(args.query_plan_ontology_enable_simple_fallback)
    query_plan_ontology_simple_fallback_only_when_unmatched = string_to_bool(args.query_plan_ontology_simple_fallback_only_when_unmatched)
    query_plan_ontology_enable_plan_repair = string_to_bool(args.query_plan_ontology_enable_plan_repair)
    query_plan_ontology_plan_repair_enable_query_condition = string_to_bool(
        args.query_plan_ontology_plan_repair_enable_query_condition
    )
    query_plan_ontology_plan_repair_enable_relation_realization = string_to_bool(
        args.query_plan_ontology_plan_repair_enable_relation_realization
    )
    query_plan_ontology_anchor_require_concrete = string_to_bool(args.query_plan_ontology_anchor_require_concrete)
    query_plan_ontology_anchor_prefer_longest_exact = string_to_bool(args.query_plan_ontology_anchor_prefer_longest_exact)
    enable_query_plan_ontology_ppr_injection = string_to_bool(args.enable_query_plan_ontology_ppr_injection)
    query_plan_ontology_ppr_enable_query_router = string_to_bool(args.query_plan_ontology_ppr_enable_query_router)
    query_plan_ontology_ppr_enable_path_quality_gate = string_to_bool(args.query_plan_ontology_ppr_enable_path_quality_gate)
    query_plan_ontology_ppr_require_anchor_fact = string_to_bool(args.query_plan_ontology_ppr_require_anchor_fact)
    query_plan_ontology_enable_ppr_passage_seeds = string_to_bool(args.query_plan_ontology_enable_ppr_passage_seeds)
    query_plan_ontology_ppr_passage_seed_require_query_anchor = string_to_bool(
        args.query_plan_ontology_ppr_passage_seed_require_query_anchor
    )
    enable_query_plan_ontology_graph_scorer = string_to_bool(args.enable_query_plan_ontology_graph_scorer)
    query_plan_ontology_graph_scorer_require_anchor_or_dense_source = string_to_bool(
        args.query_plan_ontology_graph_scorer_require_anchor_or_dense_source
    )
    query_plan_ontology_graph_scorer_boost_existing = string_to_bool(
        args.query_plan_ontology_graph_scorer_boost_existing
    )
    query_plan_ontology_graph_scorer_reorder_facts = string_to_bool(
        args.query_plan_ontology_graph_scorer_reorder_facts
    )
    query_plan_ontology_enable_structural_candidate_retriever = string_to_bool(
        args.query_plan_ontology_enable_structural_candidate_retriever
    )
    query_plan_ontology_structural_candidate_retriever_require_anchor = string_to_bool(
        args.query_plan_ontology_structural_candidate_retriever_require_anchor
    )
    query_plan_ontology_structural_candidate_retriever_require_terminal_or_relation = string_to_bool(
        args.query_plan_ontology_structural_candidate_retriever_require_terminal_or_relation
    )
    query_plan_ontology_structural_candidate_retriever_reject_generic_anchors = string_to_bool(
        args.query_plan_ontology_structural_candidate_retriever_reject_generic_anchors
    )
    query_plan_ontology_structural_candidate_retriever_require_existing_source = string_to_bool(
        args.query_plan_ontology_structural_candidate_retriever_require_existing_source
    )
    query_plan_ontology_enable_candidate_retriever = string_to_bool(
        args.query_plan_ontology_enable_candidate_retriever
    )
    query_plan_ontology_candidate_retriever_aggressive_require_strict = string_to_bool(
        args.query_plan_ontology_candidate_retriever_aggressive_require_strict
    )
    query_plan_ontology_candidate_retriever_aggressive_require_multihop_or_multi_context = string_to_bool(
        args.query_plan_ontology_candidate_retriever_aggressive_require_multihop_or_multi_context
    )
    query_plan_ontology_candidate_retriever_enable_anchor_guard = string_to_bool(
        args.query_plan_ontology_candidate_retriever_enable_anchor_guard
    )
    query_plan_ontology_candidate_retriever_allow_structural_only = string_to_bool(
        args.query_plan_ontology_candidate_retriever_allow_structural_only
    )
    query_plan_ontology_enable_terminal_evidence_retriever = string_to_bool(
        args.query_plan_ontology_enable_terminal_evidence_retriever
    )
    query_plan_ontology_terminal_evidence_require_relation_hint = string_to_bool(
        args.query_plan_ontology_terminal_evidence_require_relation_hint
    )
    query_plan_ontology_terminal_evidence_require_entity_hint = string_to_bool(
        args.query_plan_ontology_terminal_evidence_require_entity_hint
    )
    query_plan_ontology_terminal_evidence_candidate_only = string_to_bool(
        args.query_plan_ontology_terminal_evidence_candidate_only
    )
    query_plan_ontology_enable_terminal_evidence_verifier = string_to_bool(
        args.query_plan_ontology_enable_terminal_evidence_verifier
    )
    query_plan_ontology_terminal_evidence_verifier_require_new_coverage = string_to_bool(
        args.query_plan_ontology_terminal_evidence_verifier_require_new_coverage
    )
    skip_qa = string_to_bool(args.skip_qa)


    query_path = args.query_path or f"reproduce/dataset/{dataset_name}.json"
    logging.info("Using query/eval file: %s", query_path)
    samples = json.load(open(query_path, "r"))
    all_queries = [s['question'] for s in samples]

    gold_answers = get_gold_answers(samples)
    try:
        gold_docs = get_gold_docs(samples, dataset_name)
        assert len(all_queries) == len(gold_docs) == len(gold_answers), "Length of queries, gold_docs, and gold_answers should be the same."
    except:
        gold_docs = None

    config = BaseConfig(
        save_dir=save_dir,
        llm_base_url=llm_base_url,
        llm_name=llm_name,
        dataset=dataset_name,
        embedding_model_name=args.embedding_name,
        force_index_from_scratch=force_index_from_scratch,
        force_openie_from_scratch=force_openie_from_scratch,
        rerank_dspy_file_path=args.rerank_dspy_file_path,
        retrieval_top_k=200,
        linking_top_k=5,
        enable_adaptive_base_access_router=enable_adaptive_base_access_router,
        adaptive_base_access_collect_only=adaptive_base_access_collect_only,
        adaptive_base_access_model_path=args.adaptive_base_access_model_path,
        adaptive_base_access_sparse_top_k=max(
            int(args.adaptive_base_access_sparse_top_k), 5
        ),
        max_qa_steps=3,
        qa_top_k=5,
        qa_api_concurrency=max(int(args.qa_api_concurrency), 1),
        graph_type="facts_and_sim_passage_node_unidirectional",
        embedding_batch_size=max(int(args.embedding_batch_size), 1),
        max_new_tokens=max(int(args.max_new_tokens), 1),
        corpus_len=len(corpus),
        openie_mode=args.openie_mode,
        enable_ontology=enable_ontology,
        force_ontology_from_scratch=force_ontology_from_scratch,
        ontology_query_top_k=args.ontology_query_top_k,
        ontology_max_query_terms=args.ontology_max_query_terms,
        ontology_embedding_mode=args.ontology_embedding_mode,
        ontology_embedding_weight=args.ontology_embedding_weight,
        ontology_fact_boost_weight=args.ontology_fact_boost_weight,
        ontology_fact_boost_max_facts_per_hint=args.ontology_fact_boost_max_facts_per_hint,
        enable_query_aware_fact_expansion=enable_query_aware_fact_expansion,
        query_aware_candidate_top_k=args.query_aware_candidate_top_k,
        query_aware_max_added_facts=args.query_aware_max_added_facts,
        query_aware_max_facts_per_entity=args.query_aware_max_facts_per_entity,
        query_aware_score_weight=args.query_aware_score_weight,
        enable_query_local_ontology=enable_query_local_ontology,
        query_local_ontology_candidate_top_k=args.query_local_ontology_candidate_top_k,
        query_local_ontology_top_passages=args.query_local_ontology_top_passages,
        query_local_ontology_passage_chars=args.query_local_ontology_passage_chars,
        query_local_ontology_max_added_facts=args.query_local_ontology_max_added_facts,
        query_local_ontology_score_weight=args.query_local_ontology_score_weight,
        query_local_ontology_max_completion_tokens=args.query_local_ontology_max_completion_tokens,
        enable_query_ontology_prior=enable_query_ontology_prior,
        query_ontology_prior_path=args.query_ontology_prior_path,
        query_ontology_prior_boost_weight=args.query_ontology_prior_boost_weight,
        query_ontology_prior_top_k=args.query_ontology_prior_top_k,
        query_ontology_prior_max_boost_facts=args.query_ontology_prior_max_boost_facts,
        query_ontology_schema_mode=args.query_ontology_schema_mode,
        query_ontology_template_candidate_top_k=args.query_ontology_template_candidate_top_k,
        query_ontology_template_max_boost_facts=args.query_ontology_template_max_boost_facts,
        query_ontology_hint_candidate_top_k=args.query_ontology_hint_candidate_top_k,
        query_ontology_require_query_anchor=query_ontology_require_query_anchor,
        enable_query_ontology_prior_ppr_injection=enable_query_ontology_prior_ppr_injection,
        query_ontology_prior_ppr_top_k=args.query_ontology_prior_ppr_top_k,
        query_ontology_prior_ppr_score_weight=args.query_ontology_prior_ppr_score_weight,
        query_ontology_prior_ppr_expand_graph_paths=query_ontology_prior_ppr_expand_graph_paths,
        query_ontology_prior_ppr_max_facts_per_entity=args.query_ontology_prior_ppr_max_facts_per_entity,
        query_ontology_prior_ppr_intent_gate=query_ontology_prior_ppr_intent_gate,
        enable_query_ontology_prior_path_instantiation=enable_query_ontology_prior_path_instantiation,
        query_ontology_prior_path_instantiation_top_k=args.query_ontology_prior_path_instantiation_top_k,
        query_ontology_prior_path_instantiation_max_anchor_entities=args.query_ontology_prior_path_instantiation_max_anchor_entities,
        query_ontology_prior_path_instantiation_max_paths=args.query_ontology_prior_path_instantiation_max_paths,
        query_ontology_prior_path_instantiation_min_template_precision=args.query_ontology_prior_path_instantiation_min_template_precision,
        query_ontology_prior_path_instantiation_max_facts_per_entity=args.query_ontology_prior_path_instantiation_max_facts_per_entity,
        enable_query_ontology_prior_passage_lift=enable_query_ontology_prior_passage_lift,
        query_ontology_prior_passage_lift_weight=args.query_ontology_prior_passage_lift_weight,
        query_ontology_prior_passage_lift_top_k=args.query_ontology_prior_passage_lift_top_k,
        query_ontology_prior_passage_lift_use_entity_fallback=query_ontology_prior_passage_lift_use_entity_fallback,
        query_ontology_prior_trace_path=args.query_ontology_prior_trace_path,
        enable_query_plan_ontology=enable_query_plan_ontology,
        query_plan_ontology_path=args.query_plan_ontology_path,
        query_plan_ontology_compiler_sidecar_path=args.query_plan_ontology_compiler_sidecar_path,
        query_plan_relation_realization_path=args.query_plan_relation_realization_path,
        query_plan_ontology_compiler_sidecar_relation_realization_path=(
            args.query_plan_ontology_compiler_sidecar_relation_realization_path
        ),
        query_plan_use_relation_realization=query_plan_use_relation_realization,
        query_plan_relation_realization_allow_family_fallback=query_plan_relation_realization_allow_family_fallback,
        query_plan_relation_realization_require_ontology_hash_match=query_plan_relation_realization_require_ontology_hash_match,
        query_plan_ontology_enable_schema_linked_router=query_plan_ontology_enable_schema_linked_router,
        query_plan_ontology_enable_schema_linked_compiler_sidecar=(
            query_plan_ontology_enable_schema_linked_compiler_sidecar
        ),
        query_plan_ontology_matching_mode=args.query_plan_ontology_matching_mode,
        query_plan_ontology_top_k=args.query_plan_ontology_top_k,
        query_plan_ontology_min_confidence=args.query_plan_ontology_min_confidence,
        query_plan_ontology_max_paths=args.query_plan_ontology_max_paths,
        query_plan_ontology_max_branch_facts=args.query_plan_ontology_max_branch_facts,
        query_plan_ontology_signature_match_mode=args.query_plan_ontology_signature_match_mode,
        query_plan_ontology_passage_lift_weight=args.query_plan_ontology_passage_lift_weight,
        query_plan_ontology_passage_rerank_mode=args.query_plan_ontology_passage_rerank_mode,
        query_plan_ontology_passage_top_k=args.query_plan_ontology_passage_top_k,
        query_plan_ontology_passage_reranker_high_confidence_threshold=args.query_plan_ontology_passage_reranker_high_confidence_threshold,
        query_plan_ontology_passage_reranker_promote_top_k=args.query_plan_ontology_passage_reranker_promote_top_k,
        query_plan_ontology_passage_reranker_promote_to_rank=args.query_plan_ontology_passage_reranker_promote_to_rank,
        query_plan_ontology_passage_reranker_preserve_top_k=args.query_plan_ontology_passage_reranker_preserve_top_k,
        query_plan_ontology_passage_reranker_enable_promotion_verifier=query_plan_ontology_passage_reranker_enable_promotion_verifier,
        query_plan_ontology_passage_reranker_verifier_min_confidence=args.query_plan_ontology_passage_reranker_verifier_min_confidence,
        query_plan_ontology_passage_reranker_verifier_max_original_rank=args.query_plan_ontology_passage_reranker_verifier_max_original_rank,
        query_plan_ontology_passage_promotion_policy_path=args.query_plan_ontology_passage_promotion_policy_path,
        query_plan_ontology_passage_promotion_policy_mode=args.query_plan_ontology_passage_promotion_policy_mode,
        query_plan_ontology_passage_promotion_policy_enable_coverage_gate=query_plan_ontology_passage_promotion_policy_enable_coverage_gate,
        query_plan_ontology_passage_promotion_policy_coverage_top_k=args.query_plan_ontology_passage_promotion_policy_coverage_top_k,
        query_plan_ontology_enable_evidence_context_builder=query_plan_ontology_enable_evidence_context_builder,
        query_plan_ontology_evidence_context_builder_mode=args.query_plan_ontology_evidence_context_builder_mode,
        query_plan_ontology_evidence_context_builder_preserve_top_k=args.query_plan_ontology_evidence_context_builder_preserve_top_k,
        query_plan_ontology_evidence_context_builder_max_evidence_passages=args.query_plan_ontology_evidence_context_builder_max_evidence_passages,
        query_plan_ontology_evidence_context_builder_max_original_rank=args.query_plan_ontology_evidence_context_builder_max_original_rank,
        query_plan_ontology_evidence_context_builder_min_candidate_confidence=args.query_plan_ontology_evidence_context_builder_min_candidate_confidence,
        query_plan_ontology_evidence_context_builder_require_new_evidence_role=query_plan_ontology_evidence_context_builder_require_new_evidence_role,
        query_plan_ontology_reader_intervention_require_atomic_commit=query_plan_ontology_reader_intervention_require_atomic_commit,
        query_plan_ontology_evidence_context_builder_enable_chain_guidance=query_plan_ontology_evidence_context_builder_enable_chain_guidance,
        query_plan_ontology_evidence_context_builder_enable_logical_plan_guidance=query_plan_ontology_evidence_context_builder_enable_logical_plan_guidance,
        query_plan_ontology_evidence_context_builder_logical_plan_require_commit=query_plan_ontology_evidence_context_builder_logical_plan_require_commit,
        query_plan_ontology_evidence_context_builder_max_chain_items=args.query_plan_ontology_evidence_context_builder_max_chain_items,
        query_plan_ontology_evidence_context_builder_chain_guidance_mode=args.query_plan_ontology_evidence_context_builder_chain_guidance_mode,
        query_plan_ontology_evidence_context_builder_allow_hint_only=query_plan_ontology_evidence_context_builder_allow_hint_only,
        query_plan_ontology_evidence_context_builder_enable_comparative_branches=query_plan_ontology_evidence_context_builder_enable_comparative_branches,
        query_plan_ontology_enable_state_retriever=query_plan_ontology_enable_state_retriever,
        query_plan_ontology_state_retriever_top_k=args.query_plan_ontology_state_retriever_top_k,
        query_plan_ontology_state_retriever_candidate_top_k=args.query_plan_ontology_state_retriever_candidate_top_k,
        query_plan_ontology_state_retriever_preserve_top_k=args.query_plan_ontology_state_retriever_preserve_top_k,
        query_plan_ontology_state_retriever_max_original_rank=args.query_plan_ontology_state_retriever_max_original_rank,
        query_plan_ontology_state_retriever_baseline_weight=args.query_plan_ontology_state_retriever_baseline_weight,
        query_plan_ontology_state_retriever_coverage_weight=args.query_plan_ontology_state_retriever_coverage_weight,
        query_plan_ontology_state_retriever_evidence_weight=args.query_plan_ontology_state_retriever_evidence_weight,
        query_plan_ontology_state_retriever_redundancy_penalty=args.query_plan_ontology_state_retriever_redundancy_penalty,
        query_plan_ontology_state_retriever_min_evidence_score=args.query_plan_ontology_state_retriever_min_evidence_score,
        query_plan_ontology_state_retriever_require_new_slot=query_plan_ontology_state_retriever_require_new_slot,
        query_plan_ontology_state_retriever_max_ontology_passages=args.query_plan_ontology_state_retriever_max_ontology_passages,
        query_plan_ontology_state_retriever_require_policy_approval=query_plan_ontology_state_retriever_require_policy_approval,
        query_plan_ontology_state_retriever_policy_fallback_min_confidence=args.query_plan_ontology_state_retriever_policy_fallback_min_confidence,
        query_plan_ontology_state_retriever_policy_fallback_max_original_rank=args.query_plan_ontology_state_retriever_policy_fallback_max_original_rank,
        query_plan_ontology_state_retriever_require_structural_slot=query_plan_ontology_state_retriever_require_structural_slot,
        query_plan_ontology_state_policy_path=args.query_plan_ontology_state_policy_path,
        query_plan_ontology_state_policy_mode=args.query_plan_ontology_state_policy_mode,
        query_plan_ontology_enable_subgraph_retriever=query_plan_ontology_enable_subgraph_retriever,
        query_plan_ontology_subgraph_candidate_fact_top_k=args.query_plan_ontology_subgraph_candidate_fact_top_k,
        query_plan_ontology_subgraph_max_facts_per_anchor=args.query_plan_ontology_subgraph_max_facts_per_anchor,
        query_plan_ontology_subgraph_max_facts_per_relation=args.query_plan_ontology_subgraph_max_facts_per_relation,
        query_plan_ontology_subgraph_max_source_passages_per_fact=args.query_plan_ontology_subgraph_max_source_passages_per_fact,
        query_plan_ontology_subgraph_max_added_passages=args.query_plan_ontology_subgraph_max_added_passages,
        query_plan_ontology_subgraph_max_candidate_pool_rank=args.query_plan_ontology_subgraph_max_candidate_pool_rank,
        query_plan_ontology_subgraph_min_fact_score=args.query_plan_ontology_subgraph_min_fact_score,
        query_plan_ontology_subgraph_min_passage_score=args.query_plan_ontology_subgraph_min_passage_score,
        query_plan_ontology_subgraph_relation_weight=args.query_plan_ontology_subgraph_relation_weight,
        query_plan_ontology_subgraph_anchor_weight=args.query_plan_ontology_subgraph_anchor_weight,
        query_plan_ontology_subgraph_path_neighbor_weight=args.query_plan_ontology_subgraph_path_neighbor_weight,
        query_plan_ontology_subgraph_query_fact_weight=args.query_plan_ontology_subgraph_query_fact_weight,
        query_plan_ontology_subgraph_use_entity_fallback=query_plan_ontology_subgraph_use_entity_fallback,
        query_plan_ontology_subgraph_append_score_margin=args.query_plan_ontology_subgraph_append_score_margin,
        query_plan_ontology_subgraph_min_original_rank=args.query_plan_ontology_subgraph_min_original_rank,
        query_plan_ontology_subgraph_max_original_rank=args.query_plan_ontology_subgraph_max_original_rank,
        query_plan_ontology_subgraph_require_strong_fact_gate=query_plan_ontology_subgraph_require_strong_fact_gate,
        query_plan_ontology_subgraph_relation_only_min_query_fact_score=args.query_plan_ontology_subgraph_relation_only_min_query_fact_score,
        query_plan_ontology_enable_support_evidence_retriever=query_plan_ontology_enable_support_evidence_retriever,
        query_plan_ontology_support_evidence_dense_top_k=args.query_plan_ontology_support_evidence_dense_top_k,
        query_plan_ontology_support_evidence_max_probes=args.query_plan_ontology_support_evidence_max_probes,
        query_plan_ontology_support_evidence_max_candidates=args.query_plan_ontology_support_evidence_max_candidates,
        query_plan_ontology_support_evidence_min_confidence=args.query_plan_ontology_support_evidence_min_confidence,
        query_plan_ontology_support_evidence_preserve_top_k=args.query_plan_ontology_support_evidence_preserve_top_k,
        query_plan_ontology_support_evidence_synthetic_original_rank=args.query_plan_ontology_support_evidence_synthetic_original_rank,
        query_plan_ontology_support_evidence_require_anchor_hit=query_plan_ontology_support_evidence_require_anchor_hit,
        query_plan_ontology_support_evidence_require_relation_or_answer_hit=query_plan_ontology_support_evidence_require_relation_or_answer_hit,
        query_plan_ontology_support_evidence_candidate_only=query_plan_ontology_support_evidence_candidate_only,
        query_plan_ontology_support_evidence_score_margin=args.query_plan_ontology_support_evidence_score_margin,
        query_plan_ontology_support_evidence_probe_policy_path=args.query_plan_ontology_support_evidence_probe_policy_path,
        query_plan_ontology_support_evidence_probe_policy_mode=args.query_plan_ontology_support_evidence_probe_policy_mode,
        query_plan_ontology_enable_schema_route_retriever=query_plan_ontology_enable_schema_route_retriever,
        query_plan_ontology_schema_route_dense_top_k=args.query_plan_ontology_schema_route_dense_top_k,
        query_plan_ontology_schema_route_max_subqueries=args.query_plan_ontology_schema_route_max_subqueries,
        query_plan_ontology_schema_route_max_candidates=args.query_plan_ontology_schema_route_max_candidates,
        query_plan_ontology_schema_route_min_confidence=args.query_plan_ontology_schema_route_min_confidence,
        query_plan_ontology_schema_route_preserve_top_k=args.query_plan_ontology_schema_route_preserve_top_k,
        query_plan_ontology_schema_route_synthetic_original_rank=args.query_plan_ontology_schema_route_synthetic_original_rank,
        query_plan_ontology_schema_route_candidate_only=query_plan_ontology_schema_route_candidate_only,
        query_plan_ontology_schema_route_score_margin=args.query_plan_ontology_schema_route_score_margin,
        query_plan_ontology_schema_route_require_anchor_hit=query_plan_ontology_schema_route_require_anchor_hit,
        query_plan_ontology_schema_route_require_route_signal=query_plan_ontology_schema_route_require_route_signal,
        query_plan_ontology_schema_route_enable_community_route=query_plan_ontology_schema_route_enable_community_route,
        query_plan_ontology_schema_route_enable_dfs_route=query_plan_ontology_schema_route_enable_dfs_route,
        query_plan_ontology_schema_route_policy_model_path=args.query_plan_ontology_schema_route_policy_model_path,
        query_plan_ontology_schema_route_policy_mode=args.query_plan_ontology_schema_route_policy_mode,
        query_plan_ontology_schema_route_policy_min_score=args.query_plan_ontology_schema_route_policy_min_score,
        query_plan_ontology_schema_route_policy_reject_below=args.query_plan_ontology_schema_route_policy_reject_below,
        query_plan_ontology_schema_route_policy_score_weight=args.query_plan_ontology_schema_route_policy_score_weight,
        query_plan_ontology_enable_schema_slot_recall_expander=query_plan_ontology_enable_schema_slot_recall_expander,
        query_plan_ontology_schema_slot_recall_dense_top_k=args.query_plan_ontology_schema_slot_recall_dense_top_k,
        query_plan_ontology_schema_slot_recall_max_probes=args.query_plan_ontology_schema_slot_recall_max_probes,
        query_plan_ontology_schema_slot_recall_max_candidates=args.query_plan_ontology_schema_slot_recall_max_candidates,
        query_plan_ontology_schema_slot_recall_min_confidence=args.query_plan_ontology_schema_slot_recall_min_confidence,
        query_plan_ontology_schema_slot_recall_synthetic_original_rank=args.query_plan_ontology_schema_slot_recall_synthetic_original_rank,
        query_plan_ontology_schema_slot_recall_require_anchor_or_relation=query_plan_ontology_schema_slot_recall_require_anchor_or_relation,
        query_plan_ontology_schema_slot_recall_include_original_query=query_plan_ontology_schema_slot_recall_include_original_query,
        query_plan_ontology_schema_slot_recall_enable_multi_probe=query_plan_ontology_schema_slot_recall_enable_multi_probe,
        query_plan_ontology_schema_slot_recall_target_datasets=args.query_plan_ontology_schema_slot_recall_target_datasets,
        query_plan_ontology_schema_slot_recall_max_missing_slot_terms=args.query_plan_ontology_schema_slot_recall_max_missing_slot_terms,
        query_plan_ontology_schema_slot_recall_min_relation_overlap=args.query_plan_ontology_schema_slot_recall_min_relation_overlap,
        query_plan_ontology_schema_slot_recall_answer_type_boost=args.query_plan_ontology_schema_slot_recall_answer_type_boost,
        query_plan_ontology_enable_bind_probe_join=query_plan_ontology_enable_bind_probe_join,
        query_plan_ontology_bind_probe_join_target_datasets=args.query_plan_ontology_bind_probe_join_target_datasets,
        query_plan_ontology_bind_probe_join_anchor_top_k=args.query_plan_ontology_bind_probe_join_anchor_top_k,
        query_plan_ontology_bind_probe_join_max_bindings=args.query_plan_ontology_bind_probe_join_max_bindings,
        query_plan_ontology_bind_probe_join_max_probes=args.query_plan_ontology_bind_probe_join_max_probes,
        query_plan_ontology_bind_probe_join_dense_top_k=args.query_plan_ontology_bind_probe_join_dense_top_k,
        query_plan_ontology_bind_probe_join_max_candidates=args.query_plan_ontology_bind_probe_join_max_candidates,
        query_plan_ontology_bind_probe_join_min_binding_score=args.query_plan_ontology_bind_probe_join_min_binding_score,
        query_plan_ontology_bind_probe_join_min_candidate_score=args.query_plan_ontology_bind_probe_join_min_candidate_score,
        query_plan_ontology_bind_probe_join_min_relation_overlap=args.query_plan_ontology_bind_probe_join_min_relation_overlap,
        query_plan_ontology_bind_probe_join_synthetic_original_rank=args.query_plan_ontology_bind_probe_join_synthetic_original_rank,
        query_plan_ontology_bind_probe_join_require_relation_or_answer=query_plan_ontology_bind_probe_join_require_relation_or_answer,
        query_plan_ontology_bind_probe_join_enable_relation_role_constraints=query_plan_ontology_bind_probe_join_enable_relation_role_constraints,
        query_plan_ontology_bind_probe_join_require_role_consistent_candidate=query_plan_ontology_bind_probe_join_require_role_consistent_candidate,
        query_plan_ontology_bind_probe_join_min_terminal_relation_overlap=args.query_plan_ontology_bind_probe_join_min_terminal_relation_overlap,
        query_plan_ontology_bind_probe_join_min_terminal_type_score=args.query_plan_ontology_bind_probe_join_min_terminal_type_score,
        query_plan_ontology_bind_probe_join_max_role_candidate_dense_rank=args.query_plan_ontology_bind_probe_join_max_role_candidate_dense_rank,
        query_plan_ontology_enable_bind_probe_transaction_optimizer=query_plan_ontology_enable_bind_probe_transaction_optimizer,
        query_plan_ontology_bind_probe_transaction_target_datasets=args.query_plan_ontology_bind_probe_transaction_target_datasets,
        query_plan_ontology_bind_probe_transaction_top2_target_datasets=args.query_plan_ontology_bind_probe_transaction_top2_target_datasets,
        query_plan_ontology_bind_probe_transaction_min_action_utility=args.query_plan_ontology_bind_probe_transaction_min_action_utility,
        query_plan_ontology_bind_probe_transaction_min_net_benefit=args.query_plan_ontology_bind_probe_transaction_min_net_benefit,
        query_plan_ontology_bind_probe_transaction_max_terminal_rank=args.query_plan_ontology_bind_probe_transaction_max_terminal_rank,
        query_plan_ontology_bind_probe_transaction_require_source_top1=query_plan_ontology_bind_probe_transaction_require_source_top1,
        query_plan_ontology_bind_probe_transaction_displacement_cost=args.query_plan_ontology_bind_probe_transaction_displacement_cost,
        query_plan_ontology_bind_probe_transaction_probe_cost_weight=args.query_plan_ontology_bind_probe_transaction_probe_cost_weight,
        query_plan_ontology_enable_entity_keyed_top5_optimizer=query_plan_ontology_enable_entity_keyed_top5_optimizer,
        query_plan_ontology_entity_keyed_top5_target_datasets=args.query_plan_ontology_entity_keyed_top5_target_datasets,
        query_plan_ontology_entity_keyed_top5_min_confidence=args.query_plan_ontology_entity_keyed_top5_min_confidence,
        query_plan_ontology_entity_keyed_top5_min_relation_score=args.query_plan_ontology_entity_keyed_top5_min_relation_score,
        query_plan_ontology_entity_keyed_top5_min_type_score=args.query_plan_ontology_entity_keyed_top5_min_type_score,
        query_plan_ontology_entity_keyed_top5_min_binding_score=args.query_plan_ontology_entity_keyed_top5_min_binding_score,
        query_plan_ontology_entity_keyed_top5_min_action_utility=args.query_plan_ontology_entity_keyed_top5_min_action_utility,
        query_plan_ontology_entity_keyed_top5_max_source_rank=args.query_plan_ontology_entity_keyed_top5_max_source_rank,
        query_plan_ontology_enable_factorized_query_executor=query_plan_ontology_enable_factorized_query_executor,
        query_plan_ontology_factorized_target_datasets=args.query_plan_ontology_factorized_target_datasets,
        query_plan_ontology_factorized_enable_entity_relation_index_scan=query_plan_ontology_factorized_enable_entity_relation_index_scan,
        query_plan_ontology_factorized_enable_cascaded_bind_join=query_plan_ontology_factorized_enable_cascaded_bind_join,
        query_plan_ontology_factorized_enable_dual_branch_comparison_join=query_plan_ontology_factorized_enable_dual_branch_comparison_join,
        query_plan_ontology_factorized_dense_top_k=args.query_plan_ontology_factorized_dense_top_k,
        query_plan_ontology_factorized_max_probes=args.query_plan_ontology_factorized_max_probes,
        query_plan_ontology_factorized_max_candidates=args.query_plan_ontology_factorized_max_candidates,
        query_plan_ontology_factorized_max_hops=args.query_plan_ontology_factorized_max_hops,
        query_plan_ontology_factorized_beam_width=args.query_plan_ontology_factorized_beam_width,
        query_plan_ontology_factorized_max_index_rows=args.query_plan_ontology_factorized_max_index_rows,
        query_plan_ontology_factorized_index_max_rows_per_entity=args.query_plan_ontology_factorized_index_max_rows_per_entity,
        query_plan_ontology_factorized_min_relation_score=args.query_plan_ontology_factorized_min_relation_score,
        query_plan_ontology_factorized_min_type_score=args.query_plan_ontology_factorized_min_type_score,
        query_plan_ontology_factorized_min_candidate_utility=args.query_plan_ontology_factorized_min_candidate_utility,
        query_plan_ontology_factorized_cost_weight=args.query_plan_ontology_factorized_cost_weight,
        query_plan_ontology_factorized_enable_top2_admission=query_plan_ontology_factorized_enable_top2_admission,
        query_plan_ontology_factorized_enable_top5_admission=query_plan_ontology_factorized_enable_top5_admission,
        query_plan_ontology_factorized_shadow_only=query_plan_ontology_factorized_shadow_only,
        query_plan_ontology_factorized_min_top2_utility=args.query_plan_ontology_factorized_min_top2_utility,
        query_plan_ontology_factorized_min_top5_utility=args.query_plan_ontology_factorized_min_top5_utility,
        query_plan_ontology_factorized_enable_typed_semantic_compiler=query_plan_ontology_factorized_enable_typed_semantic_compiler,
        query_plan_ontology_factorized_enable_lossless_typed_plan_ir=(
            query_plan_ontology_factorized_enable_lossless_typed_plan_ir
        ),
        query_plan_ontology_factorized_require_typed_variable_unification=(
            query_plan_ontology_factorized_require_typed_variable_unification
        ),
        query_plan_ontology_factorized_enable_certificate_aware_plan_portfolio=(
            query_plan_ontology_factorized_enable_certificate_aware_plan_portfolio
        ),
        query_plan_ontology_factorized_enable_certificate_aware_access_path_reoptimization=(
            query_plan_ontology_factorized_enable_certificate_aware_access_path_reoptimization
        ),
        query_plan_ontology_factorized_enable_typed_certificate_top2=(
            query_plan_ontology_factorized_enable_typed_certificate_top2
        ),
        query_plan_ontology_factorized_typed_certificate_top2_min_utility=(
            args.query_plan_ontology_factorized_typed_certificate_top2_min_utility
        ),
        query_plan_ontology_factorized_typed_certificate_top2_min_semantic_score=(
            args.query_plan_ontology_factorized_typed_certificate_top2_min_semantic_score
        ),
        query_plan_ontology_factorized_require_exact_entity_keys=query_plan_ontology_factorized_require_exact_entity_keys,
        query_plan_ontology_factorized_require_typed_join_keys=query_plan_ontology_factorized_require_typed_join_keys,
        query_plan_ontology_factorized_min_query_predicate_coverage=args.query_plan_ontology_factorized_min_query_predicate_coverage,
        query_plan_ontology_factorized_enable_simple_top2_admission=query_plan_ontology_factorized_enable_simple_top2_admission,
        query_plan_ontology_factorized_require_chain_source_top2=query_plan_ontology_factorized_require_chain_source_top2,
        query_plan_ontology_factorized_min_counterfactual_net_benefit=args.query_plan_ontology_factorized_min_counterfactual_net_benefit,
        query_plan_ontology_factorized_enable_branch_complete_compiler=(
            query_plan_ontology_factorized_enable_branch_complete_compiler
        ),
        query_plan_ontology_factorized_require_branch_qualifier_evidence=(
            query_plan_ontology_factorized_require_branch_qualifier_evidence
        ),
        query_plan_ontology_factorized_enable_evidence_query_compiler=(
            query_plan_ontology_factorized_enable_evidence_query_compiler
        ),
        query_plan_ontology_factorized_enable_executable_ontology_compiler=(
            query_plan_ontology_factorized_enable_executable_ontology_compiler
        ),
        query_plan_ontology_factorized_enable_relation_realization=(
            query_plan_ontology_factorized_enable_relation_realization
        ),
        query_plan_ontology_factorized_enable_relation_keyed_catalog=(
            query_plan_ontology_factorized_enable_relation_keyed_catalog
        ),
        query_plan_ontology_factorized_relation_keyed_allow_matched_schema_fallback=(
            query_plan_ontology_factorized_relation_keyed_allow_matched_schema_fallback
        ),
        query_plan_ontology_factorized_enable_query_relation_alternative_recovery=(
            query_plan_ontology_factorized_enable_query_relation_alternative_recovery
        ),
        query_plan_ontology_factorized_max_query_relation_alternatives=(
            args.query_plan_ontology_factorized_max_query_relation_alternatives
        ),
        query_plan_ontology_factorized_enable_execution_aware_contract=(
            query_plan_ontology_factorized_enable_execution_aware_contract
        ),
        query_plan_ontology_factorized_enable_provenance_plan_portfolio=(
            query_plan_ontology_factorized_enable_provenance_plan_portfolio
        ),
        query_plan_ontology_factorized_enable_candidate_window_path_compiler=(
            query_plan_ontology_factorized_enable_candidate_window_path_compiler
        ),
        query_plan_ontology_factorized_candidate_window_path_size=(
            args.query_plan_ontology_factorized_candidate_window_path_size
        ),
        query_plan_ontology_factorized_candidate_window_path_beam_width=(
            args.query_plan_ontology_factorized_candidate_window_path_beam_width
        ),
        query_plan_ontology_factorized_candidate_window_path_max_paths=(
            args.query_plan_ontology_factorized_candidate_window_path_max_paths
        ),
        query_plan_ontology_factorized_candidate_window_path_min_anchor_score=(
            args.query_plan_ontology_factorized_candidate_window_path_min_anchor_score
        ),
        query_plan_ontology_factorized_candidate_window_path_min_relation_score=(
            args.query_plan_ontology_factorized_candidate_window_path_min_relation_score
        ),
        query_plan_ontology_factorized_candidate_window_path_min_path_score=(
            args.query_plan_ontology_factorized_candidate_window_path_min_path_score
        ),
        query_plan_ontology_factorized_candidate_window_path_allow_contextual_terminal_edges=(
            query_plan_ontology_factorized_candidate_window_path_allow_contextual_terminal_edges
        ),
        query_plan_ontology_factorized_candidate_window_path_enable_entity_index_expansion=(
            query_plan_ontology_factorized_candidate_window_path_enable_entity_index_expansion
        ),
        query_plan_ontology_factorized_candidate_window_path_enable_bound_dense_expansion=(
            query_plan_ontology_factorized_candidate_window_path_enable_bound_dense_expansion
        ),
        query_plan_ontology_factorized_candidate_window_path_entity_index_top_k=(
            args.query_plan_ontology_factorized_candidate_window_path_entity_index_top_k
        ),
        query_plan_ontology_factorized_candidate_window_path_bound_dense_top_k=(
            args.query_plan_ontology_factorized_candidate_window_path_bound_dense_top_k
        ),
        query_plan_ontology_factorized_candidate_window_path_max_access_probes=(
            args.query_plan_ontology_factorized_candidate_window_path_max_access_probes
        ),
        query_plan_ontology_factorized_candidate_window_path_max_external_passages=(
            args.query_plan_ontology_factorized_candidate_window_path_max_external_passages
        ),
        query_plan_ontology_factorized_candidate_window_path_enable_relation_agnostic_bridge=(
            query_plan_ontology_factorized_candidate_window_path_enable_relation_agnostic_bridge
        ),
        query_plan_ontology_factorized_candidate_window_path_bridge_max_paths=(
            args.query_plan_ontology_factorized_candidate_window_path_bridge_max_paths
        ),
        query_plan_ontology_factorized_candidate_window_path_bridge_max_edges_per_state=(
            args.query_plan_ontology_factorized_candidate_window_path_bridge_max_edges_per_state
        ),
        query_plan_ontology_factorized_candidate_window_path_bridge_min_path_score=(
            args.query_plan_ontology_factorized_candidate_window_path_bridge_min_path_score
        ),
        query_plan_ontology_factorized_enable_inverse_direction_fallback=(
            query_plan_ontology_factorized_enable_inverse_direction_fallback
        ),
        query_plan_ontology_factorized_inverse_direction_penalty=(
            args.query_plan_ontology_factorized_inverse_direction_penalty
        ),
        query_plan_ontology_factorized_enable_whole_warehouse_probe=(
            query_plan_ontology_factorized_enable_whole_warehouse_probe
        ),
        query_plan_ontology_factorized_warehouse_max_entity_keys=(
            args.query_plan_ontology_factorized_warehouse_max_entity_keys
        ),
        query_plan_ontology_factorized_enable_cost_based_physical_planner=(
            query_plan_ontology_factorized_enable_cost_based_physical_planner
        ),
        query_plan_ontology_factorized_evidence_catalog_path=(
            args.query_plan_ontology_factorized_evidence_catalog_path
        ),
        query_plan_ontology_factorized_physical_planner_quality_weight=(
            args.query_plan_ontology_factorized_physical_planner_quality_weight
        ),
        query_plan_ontology_factorized_physical_planner_recall_weight=(
            args.query_plan_ontology_factorized_physical_planner_recall_weight
        ),
        query_plan_ontology_factorized_physical_planner_proof_weight=(
            args.query_plan_ontology_factorized_physical_planner_proof_weight
        ),
        query_plan_ontology_factorized_physical_planner_cost_weight=(
            args.query_plan_ontology_factorized_physical_planner_cost_weight
        ),
        query_plan_ontology_factorized_physical_planner_risk_weight=(
            args.query_plan_ontology_factorized_physical_planner_risk_weight
        ),
        query_plan_ontology_factorized_physical_planner_min_proof_coverage=(
            args.query_plan_ontology_factorized_physical_planner_min_proof_coverage
        ),
        query_plan_ontology_factorized_physical_planner_max_estimated_cost=(
            args.query_plan_ontology_factorized_physical_planner_max_estimated_cost
        ),
        query_plan_ontology_factorized_physical_planner_no_op_utility=(
            args.query_plan_ontology_factorized_physical_planner_no_op_utility
        ),
        query_plan_ontology_factorized_physical_planner_selection_mode=(
            args.query_plan_ontology_factorized_physical_planner_selection_mode
        ),
        query_plan_ontology_factorized_physical_planner_max_loss_probability=(
            args.query_plan_ontology_factorized_physical_planner_max_loss_probability
        ),
        query_plan_ontology_factorized_physical_planner_confidence_delta=(
            args.query_plan_ontology_factorized_physical_planner_confidence_delta
        ),
        query_plan_ontology_factorized_physical_planner_min_catalog_samples=(
            args.query_plan_ontology_factorized_physical_planner_min_catalog_samples
        ),
        query_plan_ontology_factorized_physical_access_strategy=(
            args.query_plan_ontology_factorized_physical_access_strategy
        ),
        query_plan_ontology_factorized_enable_physical_shadow_collection=(
            query_plan_ontology_factorized_enable_physical_shadow_collection
        ),
        query_plan_ontology_factorized_physical_shadow_strategies=(
            args.query_plan_ontology_factorized_physical_shadow_strategies
        ),
        query_plan_ontology_factorized_enable_missing_slot_executor=(
            query_plan_ontology_factorized_enable_missing_slot_executor
        ),
        query_plan_ontology_factorized_missing_slot_trigger_reasons=(
            args.query_plan_ontology_factorized_missing_slot_trigger_reasons
        ),
        query_plan_ontology_factorized_missing_slot_dense_top_k=(
            args.query_plan_ontology_factorized_missing_slot_dense_top_k
        ),
        query_plan_ontology_factorized_missing_slot_max_probes=(
            args.query_plan_ontology_factorized_missing_slot_max_probes
        ),
        query_plan_ontology_factorized_missing_slot_max_bridge_entities=(
            args.query_plan_ontology_factorized_missing_slot_max_bridge_entities
        ),
        query_plan_ontology_factorized_missing_slot_max_candidates=(
            args.query_plan_ontology_factorized_missing_slot_max_candidates
        ),
        query_plan_ontology_factorized_missing_slot_max_hops=(
            args.query_plan_ontology_factorized_missing_slot_max_hops
        ),
        query_plan_ontology_factorized_missing_slot_min_relation_score=(
            args.query_plan_ontology_factorized_missing_slot_min_relation_score
        ),
        query_plan_ontology_factorized_missing_slot_min_terminal_type_score=(
            args.query_plan_ontology_factorized_missing_slot_min_terminal_type_score
        ),
        query_plan_ontology_factorized_missing_slot_min_candidate_utility=(
            args.query_plan_ontology_factorized_missing_slot_min_candidate_utility
        ),
        query_plan_ontology_factorized_missing_slot_enable_top2_admission=(
            query_plan_ontology_factorized_missing_slot_enable_top2_admission
        ),
        query_plan_ontology_factorized_enable_typed_proof_join=(
            query_plan_ontology_factorized_enable_typed_proof_join
        ),
        query_plan_ontology_factorized_require_branch_complete_proof=(
            query_plan_ontology_factorized_require_branch_complete_proof
        ),
        query_plan_ontology_factorized_require_provenance_validation=(
            query_plan_ontology_factorized_require_provenance_validation
        ),
        query_plan_ontology_factorized_require_strict_branch_identity=(
            query_plan_ontology_factorized_require_strict_branch_identity
        ),
        query_plan_ontology_factorized_project_full_proof_set=(
            query_plan_ontology_factorized_project_full_proof_set
        ),
        query_plan_ontology_factorized_min_proof_confidence=(
            args.query_plan_ontology_factorized_min_proof_confidence
        ),
        query_plan_ontology_factorized_enable_proof_gap_access_path=(
            query_plan_ontology_factorized_enable_proof_gap_access_path
        ),
        query_plan_ontology_factorized_proof_gap_top_window_k=(
            args.query_plan_ontology_factorized_proof_gap_top_window_k
        ),
        query_plan_ontology_factorized_proof_gap_max_paths=(
            args.query_plan_ontology_factorized_proof_gap_max_paths
        ),
        query_plan_ontology_factorized_proof_gap_min_missing_slot_fraction=(
            args.query_plan_ontology_factorized_proof_gap_min_missing_slot_fraction
        ),
        query_plan_ontology_factorized_proof_gap_min_binding_score=(
            args.query_plan_ontology_factorized_proof_gap_min_binding_score
        ),
        query_plan_ontology_factorized_proof_gap_min_proof_confidence=(
            args.query_plan_ontology_factorized_proof_gap_min_proof_confidence
        ),
        query_plan_ontology_factorized_proof_gap_require_external_member=(
            query_plan_ontology_factorized_proof_gap_require_external_member
        ),
        query_plan_ontology_factorized_proof_gap_merge_candidate_pool=(
            query_plan_ontology_factorized_proof_gap_merge_candidate_pool
        ),
        query_plan_ontology_factorized_enable_proof_residual_commit=(
            query_plan_ontology_factorized_enable_proof_residual_commit
        ),
        query_plan_ontology_factorized_proof_residual_direct_commit=(
            query_plan_ontology_factorized_proof_residual_direct_commit
        ),
        query_plan_ontology_factorized_proof_residual_post_optimizer_commit=(
            query_plan_ontology_factorized_proof_residual_post_optimizer_commit
        ),
        query_plan_ontology_factorized_proof_residual_target_datasets=(
            args.query_plan_ontology_factorized_proof_residual_target_datasets
        ),
        query_plan_ontology_factorized_proof_residual_min_certificate_confidence=(
            args.query_plan_ontology_factorized_proof_residual_min_certificate_confidence
        ),
        query_plan_ontology_factorized_proof_residual_max_top5_insertions=(
            args.query_plan_ontology_factorized_proof_residual_max_top5_insertions
        ),
        query_plan_ontology_factorized_proof_residual_enable_top2=(
            query_plan_ontology_factorized_proof_residual_enable_top2
        ),
        query_plan_ontology_factorized_enable_proof_materialized_view=(
            query_plan_ontology_factorized_enable_proof_materialized_view
        ),
        query_plan_ontology_factorized_proof_materialized_view_target_datasets=(
            args.query_plan_ontology_factorized_proof_materialized_view_target_datasets
        ),
        query_plan_ontology_factorized_proof_materialized_view_max_edges=(
            args.query_plan_ontology_factorized_proof_materialized_view_max_edges
        ),
        query_plan_ontology_factorized_enable_learned_delta_admission=(
            query_plan_ontology_factorized_enable_learned_delta_admission
        ),
        query_plan_ontology_factorized_delta_admission_model_path=(
            args.query_plan_ontology_factorized_delta_admission_model_path
        ),
        query_plan_ontology_factorized_delta_admission_min_expected_delta=(
            args.query_plan_ontology_factorized_delta_admission_min_expected_delta
        ),
        query_plan_ontology_factorized_delta_admission_max_loss_probability=(
            args.query_plan_ontology_factorized_delta_admission_max_loss_probability
        ),
        query_plan_ontology_factorized_delta_admission_fail_open=(
            query_plan_ontology_factorized_delta_admission_fail_open
        ),
        query_plan_ontology_factorized_delta_admission_collect_only=(
            query_plan_ontology_factorized_delta_admission_collect_only
        ),
        query_plan_ontology_factorized_enable_set_level_optimizer=(
            query_plan_ontology_factorized_enable_set_level_optimizer
        ),
        query_plan_ontology_factorized_set_level_optimizer_model_path=(
            args.query_plan_ontology_factorized_set_level_optimizer_model_path
        ),
        query_plan_ontology_factorized_set_level_optimizer_min_expected_utility=(
            args.query_plan_ontology_factorized_set_level_optimizer_min_expected_utility
        ),
        query_plan_ontology_factorized_set_level_optimizer_max_loss_probability=(
            args.query_plan_ontology_factorized_set_level_optimizer_max_loss_probability
        ),
        query_plan_ontology_factorized_set_level_optimizer_fail_open=(
            query_plan_ontology_factorized_set_level_optimizer_fail_open
        ),
        query_plan_ontology_factorized_enable_incumbent_reoptimizer=(
            query_plan_ontology_factorized_enable_incumbent_reoptimizer
        ),
        query_plan_ontology_factorized_incumbent_reoptimizer_model_path=(
            args.query_plan_ontology_factorized_incumbent_reoptimizer_model_path
        ),
        query_plan_ontology_factorized_incumbent_reoptimizer_min_expected_delta=(
            args.query_plan_ontology_factorized_incumbent_reoptimizer_min_expected_delta
        ),
        query_plan_ontology_factorized_incumbent_reoptimizer_max_loss_probability=(
            args.query_plan_ontology_factorized_incumbent_reoptimizer_max_loss_probability
        ),
        query_plan_ontology_factorized_incumbent_reoptimizer_set_trigger_mode=(
            args.query_plan_ontology_factorized_incumbent_reoptimizer_set_trigger_mode
        ),
        query_plan_ontology_factorized_entity_scan_mutation_target_datasets=(
            args.query_plan_ontology_factorized_entity_scan_mutation_target_datasets
        ),
        query_plan_ontology_factorized_enable_counterfactual_action_replay=(
            query_plan_ontology_factorized_enable_counterfactual_action_replay
        ),
        query_plan_ontology_factorized_counterfactual_action_replay_max_actions=(
            args.query_plan_ontology_factorized_counterfactual_action_replay_max_actions
        ),
        query_plan_ontology_factorized_action_replay_include_evidence_warehouse=(
            query_plan_ontology_factorized_action_replay_include_evidence_warehouse
        ),
        query_plan_ontology_factorized_action_replay_enable_atomic_chain_top2=(
            query_plan_ontology_factorized_action_replay_enable_atomic_chain_top2
        ),
        query_plan_ontology_factorized_enable_evidence_set_compiler=(
            query_plan_ontology_factorized_enable_evidence_set_compiler
        ),
        query_plan_ontology_factorized_evidence_set_max_candidates=(
            args.query_plan_ontology_factorized_evidence_set_max_candidates
        ),
        query_plan_ontology_factorized_evidence_set_beam_width=(
            args.query_plan_ontology_factorized_evidence_set_beam_width
        ),
        query_plan_ontology_factorized_evidence_set_max_top2_actions=(
            args.query_plan_ontology_factorized_evidence_set_max_top2_actions
        ),
        query_plan_ontology_factorized_evidence_set_max_top5_actions=(
            args.query_plan_ontology_factorized_evidence_set_max_top5_actions
        ),
        query_plan_ontology_enable_top5_listwise_optimizer=query_plan_ontology_enable_top5_listwise_optimizer,
        query_plan_ontology_top5_listwise_target_datasets=args.query_plan_ontology_top5_listwise_target_datasets,
        query_plan_ontology_top5_listwise_require_ontology_candidate=query_plan_ontology_top5_listwise_require_ontology_candidate,
        query_plan_ontology_top5_listwise_max_passage_chars=args.query_plan_ontology_top5_listwise_max_passage_chars,
        query_plan_ontology_top5_listwise_max_completion_tokens=args.query_plan_ontology_top5_listwise_max_completion_tokens,
        query_plan_ontology_top5_listwise_max_external_candidates=args.query_plan_ontology_top5_listwise_max_external_candidates,
        query_plan_ontology_top5_listwise_min_external_access_score=args.query_plan_ontology_top5_listwise_min_external_access_score,
        query_plan_ontology_top5_listwise_min_candidate_score=args.query_plan_ontology_top5_listwise_min_candidate_score,
        query_plan_ontology_top5_listwise_min_score_margin=args.query_plan_ontology_top5_listwise_min_score_margin,
        query_plan_ontology_top5_listwise_min_judge_confidence=args.query_plan_ontology_top5_listwise_min_judge_confidence,
        query_plan_ontology_top5_listwise_min_evidence_span_chars=args.query_plan_ontology_top5_listwise_min_evidence_span_chars,
        query_plan_ontology_enable_projected_graph_retriever=query_plan_ontology_enable_projected_graph_retriever,
        query_plan_ontology_projected_graph_max_schema_relations=args.query_plan_ontology_projected_graph_max_schema_relations,
        query_plan_ontology_projected_graph_max_facts_per_schema=args.query_plan_ontology_projected_graph_max_facts_per_schema,
        query_plan_ontology_projected_graph_max_facts_per_step_query=args.query_plan_ontology_projected_graph_max_facts_per_step_query,
        query_plan_ontology_projected_graph_max_candidates=args.query_plan_ontology_projected_graph_max_candidates,
        query_plan_ontology_projected_graph_min_score=args.query_plan_ontology_projected_graph_min_score,
        query_plan_ontology_projected_graph_preserve_top_k=args.query_plan_ontology_projected_graph_preserve_top_k,
        query_plan_ontology_projected_graph_candidate_only=query_plan_ontology_projected_graph_candidate_only,
        query_plan_ontology_projected_graph_score_margin=args.query_plan_ontology_projected_graph_score_margin,
        query_plan_ontology_projected_graph_require_anchor_overlap=query_plan_ontology_projected_graph_require_anchor_overlap,
        query_plan_ontology_projected_graph_min_relation_token_overlap=args.query_plan_ontology_projected_graph_min_relation_token_overlap,
        query_plan_ontology_projected_graph_enable_bridge_paths=query_plan_ontology_projected_graph_enable_bridge_paths,
        query_plan_ontology_enable_evidence_index_retriever=query_plan_ontology_enable_evidence_index_retriever,
        query_plan_ontology_evidence_index_max_schema_slots=args.query_plan_ontology_evidence_index_max_schema_slots,
        query_plan_ontology_evidence_index_max_facts_per_slot=args.query_plan_ontology_evidence_index_max_facts_per_slot,
        query_plan_ontology_evidence_index_max_facts_per_query_slot=args.query_plan_ontology_evidence_index_max_facts_per_query_slot,
        query_plan_ontology_evidence_index_max_candidates=args.query_plan_ontology_evidence_index_max_candidates,
        query_plan_ontology_evidence_index_min_score=args.query_plan_ontology_evidence_index_min_score,
        query_plan_ontology_evidence_index_preserve_top_k=args.query_plan_ontology_evidence_index_preserve_top_k,
        query_plan_ontology_evidence_index_candidate_only=query_plan_ontology_evidence_index_candidate_only,
        query_plan_ontology_evidence_index_score_margin=args.query_plan_ontology_evidence_index_score_margin,
        query_plan_ontology_evidence_index_require_anchor_overlap=query_plan_ontology_evidence_index_require_anchor_overlap,
        query_plan_ontology_evidence_index_min_relation_token_overlap=args.query_plan_ontology_evidence_index_min_relation_token_overlap,
        query_plan_ontology_evidence_index_synthetic_original_rank=args.query_plan_ontology_evidence_index_synthetic_original_rank,
        query_plan_ontology_evidence_index_top2_min_score=args.query_plan_ontology_evidence_index_top2_min_score,
        query_plan_ontology_evidence_index_top2_max_original_rank=args.query_plan_ontology_evidence_index_top2_max_original_rank,
        query_plan_ontology_evidence_index_top2_require_terminal=query_plan_ontology_evidence_index_top2_require_terminal,
        query_plan_ontology_evidence_index_recall_utility_model_path=args.query_plan_ontology_evidence_index_recall_utility_model_path,
        query_plan_ontology_evidence_index_recall_utility_mode=args.query_plan_ontology_evidence_index_recall_utility_mode,
        query_plan_ontology_evidence_index_recall_utility_min_score=args.query_plan_ontology_evidence_index_recall_utility_min_score,
        query_plan_ontology_evidence_index_recall_utility_score_weight=args.query_plan_ontology_evidence_index_recall_utility_score_weight,
        query_plan_ontology_evidence_index_recall_utility_promote_confidence=args.query_plan_ontology_evidence_index_recall_utility_promote_confidence,
        query_plan_ontology_evidence_index_recall_utility_top2_min_score=args.query_plan_ontology_evidence_index_recall_utility_top2_min_score,
        query_plan_ontology_enable_path_executor_retriever=query_plan_ontology_enable_path_executor_retriever,
        query_plan_ontology_path_executor_max_schema_slots=args.query_plan_ontology_path_executor_max_schema_slots,
        query_plan_ontology_path_executor_max_facts_per_slot=args.query_plan_ontology_path_executor_max_facts_per_slot,
        query_plan_ontology_path_executor_max_facts_per_step=args.query_plan_ontology_path_executor_max_facts_per_step,
        query_plan_ontology_path_executor_max_runtime_paths=args.query_plan_ontology_path_executor_max_runtime_paths,
        query_plan_ontology_path_executor_max_candidates=args.query_plan_ontology_path_executor_max_candidates,
        query_plan_ontology_path_executor_min_score=args.query_plan_ontology_path_executor_min_score,
        query_plan_ontology_path_executor_preserve_top_k=args.query_plan_ontology_path_executor_preserve_top_k,
        query_plan_ontology_path_executor_candidate_only=query_plan_ontology_path_executor_candidate_only,
        query_plan_ontology_path_executor_score_margin=args.query_plan_ontology_path_executor_score_margin,
        query_plan_ontology_path_executor_synthetic_original_rank=args.query_plan_ontology_path_executor_synthetic_original_rank,
        query_plan_ontology_path_executor_beam_size=args.query_plan_ontology_path_executor_beam_size,
        query_plan_ontology_path_executor_max_steps=args.query_plan_ontology_path_executor_max_steps,
        query_plan_ontology_path_executor_min_relation_token_overlap=args.query_plan_ontology_path_executor_min_relation_token_overlap,
        query_plan_ontology_path_executor_enable_answer_type_completion=query_plan_ontology_path_executor_enable_answer_type_completion,
        query_plan_ontology_path_executor_enable_terminal_type_filter=query_plan_ontology_path_executor_enable_terminal_type_filter,
        query_plan_ontology_path_executor_allow_query_condition_single=query_plan_ontology_path_executor_allow_query_condition_single,
        query_plan_ontology_path_executor_require_terminal_type_for_single_step=query_plan_ontology_path_executor_require_terminal_type_for_single_step,
        query_plan_ontology_path_executor_min_terminal_type_score=args.query_plan_ontology_path_executor_min_terminal_type_score,
        query_plan_ontology_path_executor_direction_penalty=args.query_plan_ontology_path_executor_direction_penalty,
        query_plan_ontology_path_executor_max_relation_repeats=args.query_plan_ontology_path_executor_max_relation_repeats,
        query_plan_ontology_path_executor_strict_query_relation_binding=query_plan_ontology_path_executor_strict_query_relation_binding,
        query_plan_ontology_path_executor_min_query_relation_coverage=args.query_plan_ontology_path_executor_min_query_relation_coverage,
        query_plan_ontology_enable_failure_evidence_retriever=query_plan_ontology_enable_failure_evidence_retriever,
        query_plan_ontology_failure_evidence_path=args.query_plan_ontology_failure_evidence_path,
        query_plan_ontology_failure_evidence_root=args.query_plan_ontology_failure_evidence_root,
        query_plan_ontology_failure_evidence_dense_top_k=args.query_plan_ontology_failure_evidence_dense_top_k,
        query_plan_ontology_failure_evidence_max_rules=args.query_plan_ontology_failure_evidence_max_rules,
        query_plan_ontology_failure_evidence_max_probes=args.query_plan_ontology_failure_evidence_max_probes,
        query_plan_ontology_failure_evidence_max_candidates=args.query_plan_ontology_failure_evidence_max_candidates,
        query_plan_ontology_failure_evidence_min_confidence=args.query_plan_ontology_failure_evidence_min_confidence,
        query_plan_ontology_failure_evidence_preserve_top_k=args.query_plan_ontology_failure_evidence_preserve_top_k,
        query_plan_ontology_failure_evidence_synthetic_original_rank=args.query_plan_ontology_failure_evidence_synthetic_original_rank,
        query_plan_ontology_failure_evidence_require_anchor_hit=query_plan_ontology_failure_evidence_require_anchor_hit,
        query_plan_ontology_failure_evidence_require_rule_term_hit=query_plan_ontology_failure_evidence_require_rule_term_hit,
        query_plan_ontology_failure_evidence_candidate_only=query_plan_ontology_failure_evidence_candidate_only,
        query_plan_ontology_failure_evidence_score_margin=args.query_plan_ontology_failure_evidence_score_margin,
        query_plan_ontology_failure_evidence_min_rule_gold_precision=args.query_plan_ontology_failure_evidence_min_rule_gold_precision,
        query_plan_ontology_failure_evidence_min_rule_gt5_precision=args.query_plan_ontology_failure_evidence_min_rule_gt5_precision,
        query_plan_ontology_failure_evidence_min_rule_gold_hits=args.query_plan_ontology_failure_evidence_min_rule_gold_hits,
        query_plan_ontology_failure_evidence_min_rule_gt5_hits=args.query_plan_ontology_failure_evidence_min_rule_gt5_hits,
        enable_evidence_grounded_ontology=enable_evidence_grounded_ontology,
        evidence_grounded_ontology_path=args.evidence_grounded_ontology_path,
        evidence_grounded_ontology_root=args.evidence_grounded_ontology_root,
        evidence_grounded_ontology_dense_top_k=args.evidence_grounded_ontology_dense_top_k,
        evidence_grounded_ontology_top_k_plans=args.evidence_grounded_ontology_top_k_plans,
        evidence_grounded_ontology_max_probes=args.evidence_grounded_ontology_max_probes,
        evidence_grounded_ontology_max_candidates=args.evidence_grounded_ontology_max_candidates,
        evidence_grounded_ontology_min_confidence=args.evidence_grounded_ontology_min_confidence,
        evidence_grounded_ontology_preserve_top_k=args.evidence_grounded_ontology_preserve_top_k,
        evidence_grounded_ontology_candidate_only=evidence_grounded_ontology_candidate_only,
        evidence_grounded_ontology_min_match_score=args.evidence_grounded_ontology_min_match_score,
        evidence_grounded_leaf_policy_path=args.evidence_grounded_leaf_policy_path,
        evidence_grounded_leaf_policy_reject_below=args.evidence_grounded_leaf_policy_reject_below,
        evidence_grounded_leaf_policy_boost=args.evidence_grounded_leaf_policy_boost,
        query_plan_ontology_enable_llm_evidence_verifier=query_plan_ontology_enable_llm_evidence_verifier,
        query_plan_ontology_llm_evidence_verifier_max_candidates=args.query_plan_ontology_llm_evidence_verifier_max_candidates,
        query_plan_ontology_llm_evidence_verifier_min_original_rank=args.query_plan_ontology_llm_evidence_verifier_min_original_rank,
        query_plan_ontology_llm_evidence_verifier_max_original_rank=args.query_plan_ontology_llm_evidence_verifier_max_original_rank,
        query_plan_ontology_llm_evidence_verifier_min_approval_score=args.query_plan_ontology_llm_evidence_verifier_min_approval_score,
        query_plan_ontology_llm_evidence_verifier_approved_confidence=args.query_plan_ontology_llm_evidence_verifier_approved_confidence,
        query_plan_ontology_llm_evidence_verifier_rejected_confidence=args.query_plan_ontology_llm_evidence_verifier_rejected_confidence,
        query_plan_ontology_llm_evidence_verifier_max_passage_chars=args.query_plan_ontology_llm_evidence_verifier_max_passage_chars,
        query_plan_ontology_llm_evidence_verifier_max_completion_tokens=args.query_plan_ontology_llm_evidence_verifier_max_completion_tokens,
        query_plan_ontology_enable_supervised_evidence_ranker=query_plan_ontology_enable_supervised_evidence_ranker,
        query_plan_ontology_supervised_evidence_ranker_model_path=args.query_plan_ontology_supervised_evidence_ranker_model_path,
        query_plan_ontology_supervised_evidence_ranker_min_score=args.query_plan_ontology_supervised_evidence_ranker_min_score,
        query_plan_ontology_supervised_evidence_ranker_promote_confidence=args.query_plan_ontology_supervised_evidence_ranker_promote_confidence,
        query_plan_ontology_supervised_evidence_ranker_reject_below=args.query_plan_ontology_supervised_evidence_ranker_reject_below,
        query_plan_ontology_supervised_evidence_ranker_rejected_confidence=args.query_plan_ontology_supervised_evidence_ranker_rejected_confidence,
        query_plan_ontology_supervised_evidence_ranker_max_candidates=args.query_plan_ontology_supervised_evidence_ranker_max_candidates,
        query_plan_ontology_enable_metric_aligned_slate_ranker=(
            query_plan_ontology_enable_metric_aligned_slate_ranker
        ),
        query_plan_ontology_metric_aligned_slate_ranker_model_path=(
            args.query_plan_ontology_metric_aligned_slate_ranker_model_path
        ),
        query_plan_ontology_enable_post_physical_delta_optimizer=query_plan_ontology_enable_post_physical_delta_optimizer,
        query_plan_ontology_post_physical_delta_optimizer_model_path=args.query_plan_ontology_post_physical_delta_optimizer_model_path,
        query_plan_ontology_post_physical_delta_optimizer_candidate_mode=args.query_plan_ontology_post_physical_delta_optimizer_candidate_mode,
        query_plan_ontology_post_physical_delta_optimizer_enforce_cq=query_plan_ontology_post_physical_delta_optimizer_enforce_cq,
        query_plan_ontology_post_physical_delta_optimizer_shadow_only=query_plan_ontology_post_physical_delta_optimizer_shadow_only,
        query_plan_ontology_post_physical_delta_optimizer_max_candidates=args.query_plan_ontology_post_physical_delta_optimizer_max_candidates,
        query_plan_ontology_enable_complete_plan_optimizer=query_plan_ontology_enable_complete_plan_optimizer,
        query_plan_ontology_complete_plan_optimizer_model_path=args.query_plan_ontology_complete_plan_optimizer_model_path,
        query_plan_ontology_complete_plan_optimizer_shadow_only=query_plan_ontology_complete_plan_optimizer_shadow_only,
        query_plan_ontology_complete_plan_optimizer_top_k=args.query_plan_ontology_complete_plan_optimizer_top_k,
        query_plan_ontology_complete_plan_optimizer_allowed_plans=args.query_plan_ontology_complete_plan_optimizer_allowed_plans,
        query_plan_ontology_complete_plan_selector=args.query_plan_ontology_complete_plan_selector,
        query_plan_ontology_complete_plan_integrate_semantic_access=query_plan_ontology_complete_plan_integrate_semantic_access,
        query_plan_ontology_complete_plan_immutable_incumbent=query_plan_ontology_complete_plan_immutable_incumbent,
        query_plan_ontology_enable_agentic_evidence_repair=query_plan_ontology_enable_agentic_evidence_repair,
        query_plan_ontology_agentic_evidence_repair_shadow_only=query_plan_ontology_agentic_evidence_repair_shadow_only,
        query_plan_ontology_agentic_evidence_repair_top_k=args.query_plan_ontology_agentic_evidence_repair_top_k,
        query_plan_ontology_agentic_evidence_repair_max_iterations=args.query_plan_ontology_agentic_evidence_repair_max_iterations,
        query_plan_ontology_agentic_evidence_repair_enable_top2=query_plan_ontology_agentic_evidence_repair_enable_top2,
        query_plan_ontology_agentic_evidence_repair_enable_top5=query_plan_ontology_agentic_evidence_repair_enable_top5,
        query_plan_ontology_agentic_evidence_repair_min_relation_gain=args.query_plan_ontology_agentic_evidence_repair_min_relation_gain,
        query_plan_ontology_agentic_evidence_repair_max_top5_displacements=args.query_plan_ontology_agentic_evidence_repair_max_top5_displacements,
        query_plan_ontology_agentic_evidence_repair_max_estimated_cost=args.query_plan_ontology_agentic_evidence_repair_max_estimated_cost,
        query_plan_ontology_agentic_evidence_repair_require_query_local_proof=query_plan_ontology_agentic_evidence_repair_require_query_local_proof,
        query_plan_ontology_agentic_evidence_repair_action_model_path=args.query_plan_ontology_agentic_evidence_repair_action_model_path,
        query_plan_ontology_agentic_evidence_repair_require_calibrated_delta=query_plan_ontology_agentic_evidence_repair_require_calibrated_delta,
        query_plan_ontology_agentic_evidence_repair_skip_after_upstream_commit=query_plan_ontology_agentic_evidence_repair_skip_after_upstream_commit,
        query_plan_ontology_enable_agentic_residual_retrieval=query_plan_ontology_enable_agentic_residual_retrieval,
        query_plan_ontology_agentic_residual_shadow_only=query_plan_ontology_agentic_residual_shadow_only,
        query_plan_ontology_agentic_residual_max_rounds=args.query_plan_ontology_agentic_residual_max_rounds,
        query_plan_ontology_agentic_residual_max_queries=args.query_plan_ontology_agentic_residual_max_queries,
        query_plan_ontology_agentic_residual_max_snippets=args.query_plan_ontology_agentic_residual_max_snippets,
        query_plan_ontology_agentic_residual_max_tokens=args.query_plan_ontology_agentic_residual_max_tokens,
        query_plan_ontology_agentic_residual_protect_top_k=args.query_plan_ontology_agentic_residual_protect_top_k,
        query_plan_ontology_agentic_residual_max_promotions=args.query_plan_ontology_agentic_residual_max_promotions,
        query_plan_ontology_agentic_residual_plan_top_k=args.query_plan_ontology_agentic_residual_plan_top_k,
        query_plan_ontology_agentic_residual_min_plan_score=args.query_plan_ontology_agentic_residual_min_plan_score,
        query_plan_ontology_agentic_residual_merge_mode=args.query_plan_ontology_agentic_residual_merge_mode,
        query_plan_ontology_agentic_residual_guarded_min_score=args.query_plan_ontology_agentic_residual_guarded_min_score,
        query_plan_ontology_agentic_residual_guarded_max_original_rank=args.query_plan_ontology_agentic_residual_guarded_max_original_rank,
        query_plan_ontology_agentic_residual_guarded_max_plan_rank=args.query_plan_ontology_agentic_residual_guarded_max_plan_rank,
        query_plan_ontology_agentic_residual_guarded_min_query_overlap=args.query_plan_ontology_agentic_residual_guarded_min_query_overlap,
        query_plan_ontology_agentic_residual_route_mode=args.query_plan_ontology_agentic_residual_route_mode,
        query_plan_ontology_agentic_residual_fusion_depth=args.query_plan_ontology_agentic_residual_fusion_depth,
        query_plan_ontology_agentic_residual_rrf_k=args.query_plan_ontology_agentic_residual_rrf_k,
        query_plan_ontology_agentic_residual_base_weight=args.query_plan_ontology_agentic_residual_base_weight,
        query_plan_ontology_agentic_residual_residual_weight=args.query_plan_ontology_agentic_residual_residual_weight,
        query_plan_ontology_enable_certified_direct_execution=query_plan_ontology_enable_certified_direct_execution,
        query_plan_ontology_require_certified_schema_mask=query_plan_ontology_require_certified_schema_mask,
        query_plan_ontology_certified_direct_policy_path=args.query_plan_ontology_certified_direct_policy_path,
        query_plan_ontology_certified_direct_shadow_only=query_plan_ontology_certified_direct_shadow_only,
        query_plan_ontology_certified_direct_require_policy=query_plan_ontology_certified_direct_require_policy,
        query_plan_ontology_certified_direct_top_k=args.query_plan_ontology_certified_direct_top_k,
        query_plan_ontology_certified_direct_min_matched_plans=args.query_plan_ontology_certified_direct_min_matched_plans,
        query_plan_ontology_certified_direct_min_strict_paths=args.query_plan_ontology_certified_direct_min_strict_paths,
        query_plan_ontology_certified_direct_min_relation_coverage=args.query_plan_ontology_certified_direct_min_relation_coverage,
        query_plan_ontology_certified_direct_min_answer_type_fraction=args.query_plan_ontology_certified_direct_min_answer_type_fraction,
        query_plan_ontology_certified_direct_require_complete_proof=query_plan_ontology_certified_direct_require_complete_proof,
        query_plan_ontology_enable_missing_evidence_completer=query_plan_ontology_enable_missing_evidence_completer,
        query_plan_ontology_missing_evidence_dense_top_k=args.query_plan_ontology_missing_evidence_dense_top_k,
        query_plan_ontology_missing_evidence_top_window_k=args.query_plan_ontology_missing_evidence_top_window_k,
        query_plan_ontology_missing_evidence_max_missing_slots=args.query_plan_ontology_missing_evidence_max_missing_slots,
        query_plan_ontology_missing_evidence_max_query_variants=args.query_plan_ontology_missing_evidence_max_query_variants,
        query_plan_ontology_missing_evidence_max_candidates=args.query_plan_ontology_missing_evidence_max_candidates,
        query_plan_ontology_missing_evidence_min_confidence=args.query_plan_ontology_missing_evidence_min_confidence,
        query_plan_ontology_missing_evidence_promote_to_rank=args.query_plan_ontology_missing_evidence_promote_to_rank,
        query_plan_ontology_missing_evidence_preserve_top_k=args.query_plan_ontology_missing_evidence_preserve_top_k,
        query_plan_ontology_missing_evidence_require_relation_hit=query_plan_ontology_missing_evidence_require_relation_hit,
        query_plan_ontology_missing_evidence_require_both_endpoints=query_plan_ontology_missing_evidence_require_both_endpoints,
        query_plan_ontology_missing_evidence_candidate_only=query_plan_ontology_missing_evidence_candidate_only,
        query_plan_ontology_missing_evidence_include_covered_slots=query_plan_ontology_missing_evidence_include_covered_slots,
        query_plan_ontology_missing_evidence_max_covered_slots=args.query_plan_ontology_missing_evidence_max_covered_slots,
        query_plan_ontology_missing_evidence_score_margin=args.query_plan_ontology_missing_evidence_score_margin,
        query_plan_ontology_enable_evidence_set_retriever=query_plan_ontology_enable_evidence_set_retriever,
        query_plan_ontology_evidence_set_final_top_k=args.query_plan_ontology_evidence_set_final_top_k,
        query_plan_ontology_evidence_set_preserve_top_k=args.query_plan_ontology_evidence_set_preserve_top_k,
        query_plan_ontology_evidence_set_candidate_pool_top_k=args.query_plan_ontology_evidence_set_candidate_pool_top_k,
        query_plan_ontology_evidence_set_enable_optimizer_candidate_pool=(
            query_plan_ontology_evidence_set_enable_optimizer_candidate_pool
        ),
        query_plan_ontology_evidence_set_optimizer_candidate_pool_max=(
            args.query_plan_ontology_evidence_set_optimizer_candidate_pool_max
        ),
        query_plan_ontology_evidence_set_enable_source_gap_aperture=query_plan_ontology_evidence_set_enable_source_gap_aperture,
        query_plan_ontology_evidence_set_source_gap_aperture_max_original_rank=args.query_plan_ontology_evidence_set_source_gap_aperture_max_original_rank,
        query_plan_ontology_evidence_set_source_gap_aperture_min_confidence=args.query_plan_ontology_evidence_set_source_gap_aperture_min_confidence,
        query_plan_ontology_evidence_set_source_gap_aperture_min_source_score=args.query_plan_ontology_evidence_set_source_gap_aperture_min_source_score,
        query_plan_ontology_evidence_set_source_gap_aperture_min_query_pressure=args.query_plan_ontology_evidence_set_source_gap_aperture_min_query_pressure,
        query_plan_ontology_evidence_set_source_gap_aperture_max_candidates=args.query_plan_ontology_evidence_set_source_gap_aperture_max_candidates,
        query_plan_ontology_evidence_set_source_gap_aperture_score_weight=args.query_plan_ontology_evidence_set_source_gap_aperture_score_weight,
        query_plan_ontology_evidence_set_source_gap_aperture_allow_no_new_slot=query_plan_ontology_evidence_set_source_gap_aperture_allow_no_new_slot,
        query_plan_ontology_evidence_set_max_ontology_passages=args.query_plan_ontology_evidence_set_max_ontology_passages,
        query_plan_ontology_evidence_set_min_candidate_confidence=args.query_plan_ontology_evidence_set_min_candidate_confidence,
        query_plan_ontology_evidence_set_top2_min_candidate_confidence=args.query_plan_ontology_evidence_set_top2_min_candidate_confidence,
        query_plan_ontology_evidence_set_top2_max_original_rank=args.query_plan_ontology_evidence_set_top2_max_original_rank,
        query_plan_ontology_evidence_set_require_new_slot=query_plan_ontology_evidence_set_require_new_slot,
        query_plan_ontology_evidence_set_baseline_weight=args.query_plan_ontology_evidence_set_baseline_weight,
        query_plan_ontology_evidence_set_ontology_weight=args.query_plan_ontology_evidence_set_ontology_weight,
        query_plan_ontology_evidence_set_coverage_weight=args.query_plan_ontology_evidence_set_coverage_weight,
        query_plan_ontology_evidence_set_redundancy_penalty=args.query_plan_ontology_evidence_set_redundancy_penalty,
        query_plan_ontology_evidence_set_coverage_reference_top_k=args.query_plan_ontology_evidence_set_coverage_reference_top_k,
        query_plan_ontology_evidence_set_enable_intervention_policy=query_plan_ontology_evidence_set_enable_intervention_policy,
        query_plan_ontology_evidence_set_intervention_min_confidence=args.query_plan_ontology_evidence_set_intervention_min_confidence,
        query_plan_ontology_evidence_set_intervention_relation_only_min_confidence=args.query_plan_ontology_evidence_set_intervention_relation_only_min_confidence,
        query_plan_ontology_evidence_set_intervention_relation_only_max_original_rank=args.query_plan_ontology_evidence_set_intervention_relation_only_max_original_rank,
        query_plan_ontology_evidence_set_intervention_allow_relation_only_simple=query_plan_ontology_evidence_set_intervention_allow_relation_only_simple,
        query_plan_ontology_evidence_set_enable_counterfactual_policy=query_plan_ontology_evidence_set_enable_counterfactual_policy,
        query_plan_ontology_evidence_set_counterfactual_min_candidate_value=args.query_plan_ontology_evidence_set_counterfactual_min_candidate_value,
        query_plan_ontology_evidence_set_counterfactual_margin=args.query_plan_ontology_evidence_set_counterfactual_margin,
        query_plan_ontology_evidence_set_policy_model_path=args.query_plan_ontology_evidence_set_policy_model_path,
        query_plan_ontology_evidence_set_policy_mode=args.query_plan_ontology_evidence_set_policy_mode,
        query_plan_ontology_evidence_set_policy_min_score=args.query_plan_ontology_evidence_set_policy_min_score,
        query_plan_ontology_evidence_set_policy_reject_below=args.query_plan_ontology_evidence_set_policy_reject_below,
        query_plan_ontology_evidence_set_policy_score_weight=args.query_plan_ontology_evidence_set_policy_score_weight,
        query_plan_ontology_evidence_set_policy_promote_confidence=args.query_plan_ontology_evidence_set_policy_promote_confidence,
        query_plan_ontology_evidence_set_policy_counterfactual_weight=args.query_plan_ontology_evidence_set_policy_counterfactual_weight,
        query_plan_ontology_evidence_set_utility_policy_model_path=args.query_plan_ontology_evidence_set_utility_policy_model_path,
        query_plan_ontology_evidence_set_utility_policy_mode=args.query_plan_ontology_evidence_set_utility_policy_mode,
        query_plan_ontology_evidence_set_utility_policy_min_score=args.query_plan_ontology_evidence_set_utility_policy_min_score,
        query_plan_ontology_evidence_set_utility_policy_reject_below=args.query_plan_ontology_evidence_set_utility_policy_reject_below,
        query_plan_ontology_evidence_set_utility_policy_score_weight=args.query_plan_ontology_evidence_set_utility_policy_score_weight,
        query_plan_ontology_evidence_set_utility_policy_promote_confidence=args.query_plan_ontology_evidence_set_utility_policy_promote_confidence,
        query_plan_ontology_evidence_set_utility_policy_counterfactual_weight=args.query_plan_ontology_evidence_set_utility_policy_counterfactual_weight,
        query_plan_ontology_evidence_set_utility_policy_apply_scope=args.query_plan_ontology_evidence_set_utility_policy_apply_scope,
        query_plan_ontology_evidence_set_utility_planner_model_path=args.query_plan_ontology_evidence_set_utility_planner_model_path,
        query_plan_ontology_evidence_set_utility_planner_mode=args.query_plan_ontology_evidence_set_utility_planner_mode,
        query_plan_ontology_evidence_set_utility_planner_min_score=args.query_plan_ontology_evidence_set_utility_planner_min_score,
        query_plan_ontology_evidence_set_utility_planner_top2_min_score=args.query_plan_ontology_evidence_set_utility_planner_top2_min_score,
        query_plan_ontology_evidence_set_utility_planner_score_weight=args.query_plan_ontology_evidence_set_utility_planner_score_weight,
        query_plan_ontology_evidence_set_utility_planner_counterfactual_weight=args.query_plan_ontology_evidence_set_utility_planner_counterfactual_weight,
        query_plan_ontology_evidence_set_utility_planner_min_value_margin=args.query_plan_ontology_evidence_set_utility_planner_min_value_margin,
        query_plan_ontology_evidence_set_utility_planner_top2_min_value_margin=args.query_plan_ontology_evidence_set_utility_planner_top2_min_value_margin,
        query_plan_ontology_evidence_set_utility_planner_max_original_rank=args.query_plan_ontology_evidence_set_utility_planner_max_original_rank,
        query_plan_ontology_evidence_set_utility_planner_require_complex_or_path=query_plan_ontology_evidence_set_utility_planner_require_complex_or_path,
        query_plan_ontology_evidence_set_recall_plan_optimizer_mode=args.query_plan_ontology_evidence_set_recall_plan_optimizer_mode,
        query_plan_ontology_evidence_set_recall_plan_optimizer_stats_path=args.query_plan_ontology_evidence_set_recall_plan_optimizer_stats_path,
        query_plan_ontology_evidence_set_recall_plan_optimizer_min_score=args.query_plan_ontology_evidence_set_recall_plan_optimizer_min_score,
        query_plan_ontology_evidence_set_recall_plan_optimizer_score_weight=args.query_plan_ontology_evidence_set_recall_plan_optimizer_score_weight,
        query_plan_ontology_evidence_set_recall_plan_optimizer_confidence_boost=args.query_plan_ontology_evidence_set_recall_plan_optimizer_confidence_boost,
        query_plan_ontology_evidence_set_recall_plan_optimizer_router_bypass_min_score=args.query_plan_ontology_evidence_set_recall_plan_optimizer_router_bypass_min_score,
        query_plan_ontology_evidence_set_recall_plan_optimizer_allow_no_new_slot=query_plan_ontology_evidence_set_recall_plan_optimizer_allow_no_new_slot,
        query_plan_ontology_evidence_set_recall_plan_optimizer_max_original_rank=args.query_plan_ontology_evidence_set_recall_plan_optimizer_max_original_rank,
        query_plan_ontology_evidence_set_recall_plan_optimizer_max_selected=args.query_plan_ontology_evidence_set_recall_plan_optimizer_max_selected,
        query_plan_ontology_evidence_set_recall_plan_optimizer_min_source_precision=args.query_plan_ontology_evidence_set_recall_plan_optimizer_min_source_precision,
        query_plan_ontology_evidence_set_enable_rank_window_recall=query_plan_ontology_evidence_set_enable_rank_window_recall,
        query_plan_ontology_evidence_set_rank_window_model_path=args.query_plan_ontology_evidence_set_rank_window_model_path,
        query_plan_ontology_evidence_set_rank_window_target_datasets=args.query_plan_ontology_evidence_set_rank_window_target_datasets,
        query_plan_ontology_evidence_set_rank_window_start_rank=args.query_plan_ontology_evidence_set_rank_window_start_rank,
        query_plan_ontology_evidence_set_rank_window_end_rank=args.query_plan_ontology_evidence_set_rank_window_end_rank,
        query_plan_ontology_evidence_set_rank_window_min_score=args.query_plan_ontology_evidence_set_rank_window_min_score,
        query_plan_ontology_evidence_set_rank_window_top2_min_score=args.query_plan_ontology_evidence_set_rank_window_top2_min_score,
        query_plan_ontology_evidence_set_rank_window_router_bypass_min_score=args.query_plan_ontology_evidence_set_rank_window_router_bypass_min_score,
        query_plan_ontology_evidence_set_rank_window_score_weight=args.query_plan_ontology_evidence_set_rank_window_score_weight,
        query_plan_ontology_evidence_set_rank_window_confidence_floor=args.query_plan_ontology_evidence_set_rank_window_confidence_floor,
        query_plan_ontology_evidence_set_rank_window_max_selected=args.query_plan_ontology_evidence_set_rank_window_max_selected,
        query_plan_ontology_evidence_set_rank_window_max_top2_swaps=args.query_plan_ontology_evidence_set_rank_window_max_top2_swaps,
        query_plan_ontology_evidence_set_rank_window_min_margin=args.query_plan_ontology_evidence_set_rank_window_min_margin,
        query_plan_ontology_evidence_set_rank_window_top2_min_margin=args.query_plan_ontology_evidence_set_rank_window_top2_min_margin,
        query_plan_ontology_evidence_set_rank_window_require_text_signal=query_plan_ontology_evidence_set_rank_window_require_text_signal,
        query_plan_ontology_evidence_set_rank_window_min_text_score=args.query_plan_ontology_evidence_set_rank_window_min_text_score,
        query_plan_ontology_evidence_set_rank_window_defer_to_planner=query_plan_ontology_evidence_set_rank_window_defer_to_planner,
        query_plan_ontology_evidence_set_rank_window_protect_original_top_k=args.query_plan_ontology_evidence_set_rank_window_protect_original_top_k,
        query_plan_ontology_evidence_set_rank_window_min_replacement_rank=args.query_plan_ontology_evidence_set_rank_window_min_replacement_rank,
        query_plan_ontology_evidence_set_rank_window_query_router_mode=args.query_plan_ontology_evidence_set_rank_window_query_router_mode,
        query_plan_ontology_evidence_set_rank_window_complex_datasets=args.query_plan_ontology_evidence_set_rank_window_complex_datasets,
        query_plan_ontology_evidence_set_rank_window_simple_datasets=args.query_plan_ontology_evidence_set_rank_window_simple_datasets,
        query_plan_ontology_evidence_set_rank_window_complex_min_score=args.query_plan_ontology_evidence_set_rank_window_complex_min_score,
        query_plan_ontology_evidence_set_rank_window_complex_min_text_score=args.query_plan_ontology_evidence_set_rank_window_complex_min_text_score,
        query_plan_ontology_evidence_set_rank_window_simple_min_score=args.query_plan_ontology_evidence_set_rank_window_simple_min_score,
        query_plan_ontology_evidence_set_rank_window_simple_min_text_score=args.query_plan_ontology_evidence_set_rank_window_simple_min_text_score,
        query_plan_ontology_evidence_set_rank_window_simple_require_answer_signal=(
            query_plan_ontology_evidence_set_rank_window_simple_require_answer_signal
        ),
        query_plan_ontology_evidence_set_rank_window_simple_answer_signal_min_text_score=(
            args.query_plan_ontology_evidence_set_rank_window_simple_answer_signal_min_text_score
        ),
        query_plan_ontology_evidence_set_rank_window_enable_net_benefit_gate=(
            query_plan_ontology_evidence_set_rank_window_enable_net_benefit_gate
        ),
        query_plan_ontology_evidence_set_rank_window_allow_no_new_slot_bypass=(
            query_plan_ontology_evidence_set_rank_window_allow_no_new_slot_bypass
        ),
        query_plan_ontology_evidence_set_rank_window_no_new_slot_min_score=(
            args.query_plan_ontology_evidence_set_rank_window_no_new_slot_min_score
        ),
        query_plan_ontology_evidence_set_rank_window_no_new_slot_min_text_score=(
            args.query_plan_ontology_evidence_set_rank_window_no_new_slot_min_text_score
        ),
        query_plan_ontology_evidence_set_rank_window_no_new_slot_require_path_or_terminal=(
            query_plan_ontology_evidence_set_rank_window_no_new_slot_require_path_or_terminal
        ),
        query_plan_ontology_evidence_set_rank_window_net_benefit_min_margin=(
            args.query_plan_ontology_evidence_set_rank_window_net_benefit_min_margin
        ),
        query_plan_ontology_evidence_set_rank_window_append_min_margin=(
            args.query_plan_ontology_evidence_set_rank_window_append_min_margin
        ),
        query_plan_ontology_evidence_set_rank_window_displacement_cost_weight=(
            args.query_plan_ontology_evidence_set_rank_window_displacement_cost_weight
        ),
        query_plan_ontology_evidence_set_enable_evidence_warehouse_optimizer=(
            query_plan_ontology_evidence_set_enable_evidence_warehouse_optimizer
        ),
        query_plan_ontology_evidence_set_enable_final_rank_evidence_warehouse=(
            query_plan_ontology_evidence_set_enable_final_rank_evidence_warehouse
        ),
        query_plan_ontology_evidence_set_evidence_warehouse_target_datasets=(
            args.query_plan_ontology_evidence_set_evidence_warehouse_target_datasets
        ),
        query_plan_ontology_evidence_set_evidence_warehouse_max_candidates=(
            args.query_plan_ontology_evidence_set_evidence_warehouse_max_candidates
        ),
        query_plan_ontology_evidence_set_evidence_warehouse_max_top5_swaps=(
            args.query_plan_ontology_evidence_set_evidence_warehouse_max_top5_swaps
        ),
        query_plan_ontology_evidence_set_evidence_warehouse_max_top2_swaps=(
            args.query_plan_ontology_evidence_set_evidence_warehouse_max_top2_swaps
        ),
        query_plan_ontology_evidence_set_evidence_warehouse_protect_top_k=(
            args.query_plan_ontology_evidence_set_evidence_warehouse_protect_top_k
        ),
        query_plan_ontology_evidence_set_evidence_warehouse_max_original_rank=(
            args.query_plan_ontology_evidence_set_evidence_warehouse_max_original_rank
        ),
        query_plan_ontology_evidence_set_evidence_warehouse_min_net_benefit=(
            args.query_plan_ontology_evidence_set_evidence_warehouse_min_net_benefit
        ),
        query_plan_ontology_evidence_set_evidence_warehouse_top2_min_net_benefit=(
            args.query_plan_ontology_evidence_set_evidence_warehouse_top2_min_net_benefit
        ),
        query_plan_ontology_evidence_set_evidence_warehouse_min_benefit=(
            args.query_plan_ontology_evidence_set_evidence_warehouse_min_benefit
        ),
        query_plan_ontology_evidence_set_evidence_warehouse_require_answer_signal=(
            query_plan_ontology_evidence_set_evidence_warehouse_require_answer_signal
        ),
        query_plan_ontology_evidence_set_evidence_warehouse_allow_relation_only=(
            query_plan_ontology_evidence_set_evidence_warehouse_allow_relation_only
        ),
        query_plan_ontology_evidence_set_evidence_warehouse_cost_weight=(
            args.query_plan_ontology_evidence_set_evidence_warehouse_cost_weight
        ),
        query_plan_ontology_evidence_set_evidence_warehouse_risk_weight=(
            args.query_plan_ontology_evidence_set_evidence_warehouse_risk_weight
        ),
        query_plan_ontology_evidence_set_evidence_warehouse_text_weight=(
            args.query_plan_ontology_evidence_set_evidence_warehouse_text_weight
        ),
        query_plan_ontology_evidence_set_evidence_warehouse_path_weight=(
            args.query_plan_ontology_evidence_set_evidence_warehouse_path_weight
        ),
        query_plan_ontology_evidence_set_evidence_warehouse_slot_weight=(
            args.query_plan_ontology_evidence_set_evidence_warehouse_slot_weight
        ),
        query_plan_ontology_evidence_set_evidence_warehouse_source_weight=(
            args.query_plan_ontology_evidence_set_evidence_warehouse_source_weight
        ),
        query_plan_ontology_evidence_set_evidence_warehouse_selectivity_model_path=(
            args.query_plan_ontology_evidence_set_evidence_warehouse_selectivity_model_path
        ),
        query_plan_ontology_evidence_set_evidence_warehouse_selectivity_min_score=(
            args.query_plan_ontology_evidence_set_evidence_warehouse_selectivity_min_score
        ),
        query_plan_ontology_evidence_set_evidence_warehouse_selectivity_top2_min_score=(
            args.query_plan_ontology_evidence_set_evidence_warehouse_selectivity_top2_min_score
        ),
        query_plan_ontology_evidence_set_evidence_warehouse_selectivity_score_weight=(
            args.query_plan_ontology_evidence_set_evidence_warehouse_selectivity_score_weight
        ),
        query_plan_ontology_evidence_set_evidence_warehouse_selectivity_enable_recall_bypass=(
            query_plan_ontology_evidence_set_evidence_warehouse_selectivity_enable_recall_bypass
        ),
        query_plan_ontology_evidence_set_evidence_warehouse_selectivity_bypass_datasets=(
            args.query_plan_ontology_evidence_set_evidence_warehouse_selectivity_bypass_datasets
        ),
        query_plan_ontology_evidence_set_evidence_warehouse_selectivity_bypass_reasons=(
            args.query_plan_ontology_evidence_set_evidence_warehouse_selectivity_bypass_reasons
        ),
        query_plan_ontology_evidence_set_evidence_warehouse_selectivity_bypass_min_score=(
            args.query_plan_ontology_evidence_set_evidence_warehouse_selectivity_bypass_min_score
        ),
        query_plan_ontology_evidence_set_evidence_warehouse_selectivity_bypass_max_original_rank=(
            args.query_plan_ontology_evidence_set_evidence_warehouse_selectivity_bypass_max_original_rank
        ),
        query_plan_ontology_evidence_set_evidence_warehouse_selectivity_bypass_min_confidence=(
            args.query_plan_ontology_evidence_set_evidence_warehouse_selectivity_bypass_min_confidence
        ),
        query_plan_ontology_evidence_set_evidence_warehouse_selectivity_bypass_min_text_score=(
            args.query_plan_ontology_evidence_set_evidence_warehouse_selectivity_bypass_min_text_score
        ),
        query_plan_ontology_evidence_set_evidence_warehouse_selectivity_bypass_max_risk=(
            args.query_plan_ontology_evidence_set_evidence_warehouse_selectivity_bypass_max_risk
        ),
        query_plan_ontology_evidence_set_evidence_warehouse_selectivity_bypass_score_relaxation=(
            args.query_plan_ontology_evidence_set_evidence_warehouse_selectivity_bypass_score_relaxation
        ),
        query_plan_ontology_evidence_set_evidence_warehouse_selectivity_bypass_net_bonus=(
            args.query_plan_ontology_evidence_set_evidence_warehouse_selectivity_bypass_net_bonus
        ),
        query_plan_ontology_evidence_set_evidence_warehouse_selectivity_enable_top2_guard_bypass=(
            query_plan_ontology_evidence_set_evidence_warehouse_selectivity_enable_top2_guard_bypass
        ),
        query_plan_ontology_evidence_set_evidence_warehouse_selectivity_top2_guard_bypass_min_score=(
            args.query_plan_ontology_evidence_set_evidence_warehouse_selectivity_top2_guard_bypass_min_score
        ),
        query_plan_ontology_evidence_set_evidence_warehouse_selectivity_top2_guard_bypass_max_risk=(
            args.query_plan_ontology_evidence_set_evidence_warehouse_selectivity_top2_guard_bypass_max_risk
        ),
        query_plan_ontology_evidence_set_evidence_warehouse_enable_action_counterfactual_gate=(
            query_plan_ontology_evidence_set_evidence_warehouse_enable_action_counterfactual_gate
        ),
        query_plan_ontology_evidence_set_evidence_warehouse_action_gate_top2_min_text_score=(
            args.query_plan_ontology_evidence_set_evidence_warehouse_action_gate_top2_min_text_score
        ),
        query_plan_ontology_evidence_set_evidence_warehouse_action_gate_top2_min_selectivity_score=(
            args.query_plan_ontology_evidence_set_evidence_warehouse_action_gate_top2_min_selectivity_score
        ),
        query_plan_ontology_evidence_set_evidence_warehouse_action_gate_top5_min_text_score=(
            args.query_plan_ontology_evidence_set_evidence_warehouse_action_gate_top5_min_text_score
        ),
        query_plan_ontology_evidence_set_evidence_warehouse_action_gate_top5_min_selectivity_score=(
            args.query_plan_ontology_evidence_set_evidence_warehouse_action_gate_top5_min_selectivity_score
        ),
        query_plan_ontology_evidence_set_evidence_warehouse_action_gate_nq_top5_min_rule_net=(
            args.query_plan_ontology_evidence_set_evidence_warehouse_action_gate_nq_top5_min_rule_net
        ),
        query_plan_ontology_evidence_set_evidence_warehouse_enable_set_coverage_cbo=(
            query_plan_ontology_evidence_set_evidence_warehouse_enable_set_coverage_cbo
        ),
        query_plan_ontology_evidence_set_evidence_warehouse_set_coverage_weight=(
            args.query_plan_ontology_evidence_set_evidence_warehouse_set_coverage_weight
        ),
        query_plan_ontology_evidence_set_evidence_warehouse_set_coverage_min_gain=(
            args.query_plan_ontology_evidence_set_evidence_warehouse_set_coverage_min_gain
        ),
        query_plan_ontology_evidence_set_evidence_warehouse_set_coverage_max_risk=(
            args.query_plan_ontology_evidence_set_evidence_warehouse_set_coverage_max_risk
        ),
        query_plan_ontology_evidence_set_evidence_warehouse_answer_aware_mode=(
            args.query_plan_ontology_evidence_set_evidence_warehouse_answer_aware_mode
        ),
        query_plan_ontology_evidence_set_evidence_warehouse_answer_aware_model_path=(
            args.query_plan_ontology_evidence_set_evidence_warehouse_answer_aware_model_path
        ),
        query_plan_ontology_evidence_set_evidence_warehouse_answer_aware_min_score=(
            args.query_plan_ontology_evidence_set_evidence_warehouse_answer_aware_min_score
        ),
        query_plan_ontology_evidence_set_evidence_warehouse_answer_aware_top2_min_score=(
            args.query_plan_ontology_evidence_set_evidence_warehouse_answer_aware_top2_min_score
        ),
        query_plan_ontology_evidence_set_evidence_warehouse_answer_aware_score_weight=(
            args.query_plan_ontology_evidence_set_evidence_warehouse_answer_aware_score_weight
        ),
        query_plan_ontology_evidence_set_evidence_warehouse_answer_aware_require_direct_signal=(
            query_plan_ontology_evidence_set_evidence_warehouse_answer_aware_require_direct_signal
        ),
        query_plan_ontology_evidence_set_evidence_warehouse_answer_aware_allow_top2_guard_bypass=(
            query_plan_ontology_evidence_set_evidence_warehouse_answer_aware_allow_top2_guard_bypass
        ),
        query_plan_ontology_evidence_set_evidence_warehouse_answer_aware_top2_guard_bypass_min_score=(
            args.query_plan_ontology_evidence_set_evidence_warehouse_answer_aware_top2_guard_bypass_min_score
        ),
        query_plan_ontology_evidence_set_evidence_warehouse_answer_aware_top2_guard_bypass_max_risk=(
            args.query_plan_ontology_evidence_set_evidence_warehouse_answer_aware_top2_guard_bypass_max_risk
        ),
        query_plan_ontology_evidence_set_enable_top5_to_top2_promotion=(
            query_plan_ontology_evidence_set_enable_top5_to_top2_promotion
        ),
        query_plan_ontology_evidence_set_top5_to_top2_promotion_target_datasets=(
            args.query_plan_ontology_evidence_set_top5_to_top2_promotion_target_datasets
        ),
        query_plan_ontology_evidence_set_top5_to_top2_promotion_min_score=(
            args.query_plan_ontology_evidence_set_top5_to_top2_promotion_min_score
        ),
        query_plan_ontology_evidence_set_top5_to_top2_promotion_min_margin=(
            args.query_plan_ontology_evidence_set_top5_to_top2_promotion_min_margin
        ),
        query_plan_ontology_evidence_set_top5_to_top2_promotion_max_risk=(
            args.query_plan_ontology_evidence_set_top5_to_top2_promotion_max_risk
        ),
        query_plan_ontology_evidence_set_top5_to_top2_promotion_min_text_score=(
            args.query_plan_ontology_evidence_set_top5_to_top2_promotion_min_text_score
        ),
        query_plan_ontology_evidence_set_top5_to_top2_promotion_min_path_score=(
            args.query_plan_ontology_evidence_set_top5_to_top2_promotion_min_path_score
        ),
        query_plan_ontology_evidence_set_top5_to_top2_promotion_min_source_precision=(
            args.query_plan_ontology_evidence_set_top5_to_top2_promotion_min_source_precision
        ),
        query_plan_ontology_evidence_set_top5_to_top2_promotion_min_slot_value=(
            args.query_plan_ontology_evidence_set_top5_to_top2_promotion_min_slot_value
        ),
        query_plan_ontology_evidence_set_top5_to_top2_promotion_require_ontology_signal=(
            query_plan_ontology_evidence_set_top5_to_top2_promotion_require_ontology_signal
        ),
        query_plan_ontology_evidence_set_top5_to_top2_promotion_max_promotions=(
            args.query_plan_ontology_evidence_set_top5_to_top2_promotion_max_promotions
        ),
        query_plan_ontology_evidence_set_enable_pairwise_top2_join=(
            query_plan_ontology_evidence_set_enable_pairwise_top2_join
        ),
        query_plan_ontology_evidence_set_pairwise_top2_join_target_datasets=(
            args.query_plan_ontology_evidence_set_pairwise_top2_join_target_datasets
        ),
        query_plan_ontology_evidence_set_pairwise_top2_join_candidate_pool=(
            args.query_plan_ontology_evidence_set_pairwise_top2_join_candidate_pool
        ),
        query_plan_ontology_evidence_set_pairwise_top2_join_min_pair_score=(
            args.query_plan_ontology_evidence_set_pairwise_top2_join_min_pair_score
        ),
        query_plan_ontology_evidence_set_pairwise_top2_join_min_margin=(
            args.query_plan_ontology_evidence_set_pairwise_top2_join_min_margin
        ),
        query_plan_ontology_evidence_set_pairwise_top2_join_min_text_score=(
            args.query_plan_ontology_evidence_set_pairwise_top2_join_min_text_score
        ),
        query_plan_ontology_evidence_set_pairwise_top2_join_min_second_text_score=(
            args.query_plan_ontology_evidence_set_pairwise_top2_join_min_second_text_score
        ),
        query_plan_ontology_evidence_set_pairwise_top2_join_min_coverage_gain=(
            args.query_plan_ontology_evidence_set_pairwise_top2_join_min_coverage_gain
        ),
        query_plan_ontology_evidence_set_pairwise_top2_join_max_original_rank=(
            args.query_plan_ontology_evidence_set_pairwise_top2_join_max_original_rank
        ),
        query_plan_ontology_evidence_set_pairwise_top2_join_protect_top1=(
            query_plan_ontology_evidence_set_pairwise_top2_join_protect_top1
        ),
        query_plan_ontology_evidence_set_pairwise_top2_join_require_answer_signal=(
            query_plan_ontology_evidence_set_pairwise_top2_join_require_answer_signal
        ),
        query_plan_ontology_evidence_set_pairwise_top2_join_max_swaps=(
            args.query_plan_ontology_evidence_set_pairwise_top2_join_max_swaps
        ),
        query_plan_ontology_evidence_set_enable_decomposed_chain_join=(
            query_plan_ontology_evidence_set_enable_decomposed_chain_join
        ),
        query_plan_ontology_evidence_set_decomposed_chain_join_target_datasets=(
            args.query_plan_ontology_evidence_set_decomposed_chain_join_target_datasets
        ),
        query_plan_ontology_evidence_set_decomposed_chain_join_candidate_pool=(
            args.query_plan_ontology_evidence_set_decomposed_chain_join_candidate_pool
        ),
        query_plan_ontology_evidence_set_decomposed_chain_join_min_chain_score=(
            args.query_plan_ontology_evidence_set_decomposed_chain_join_min_chain_score
        ),
        query_plan_ontology_evidence_set_decomposed_chain_join_min_margin=(
            args.query_plan_ontology_evidence_set_decomposed_chain_join_min_margin
        ),
        query_plan_ontology_evidence_set_decomposed_chain_join_min_coverage_gain=(
            args.query_plan_ontology_evidence_set_decomposed_chain_join_min_coverage_gain
        ),
        query_plan_ontology_evidence_set_decomposed_chain_join_max_original_rank=(
            args.query_plan_ontology_evidence_set_decomposed_chain_join_max_original_rank
        ),
        query_plan_ontology_evidence_set_decomposed_chain_join_max_swaps=(
            args.query_plan_ontology_evidence_set_decomposed_chain_join_max_swaps
        ),
        query_plan_ontology_evidence_set_decomposed_chain_join_protect_top1=(
            query_plan_ontology_evidence_set_decomposed_chain_join_protect_top1
        ),
        query_plan_ontology_evidence_set_decomposed_chain_join_enable_decomposition=(
            query_plan_ontology_evidence_set_decomposed_chain_join_enable_decomposition
        ),
        query_plan_ontology_evidence_set_decomposed_chain_join_enable_subquery_access=(
            query_plan_ontology_evidence_set_decomposed_chain_join_enable_subquery_access
        ),
        query_plan_ontology_evidence_set_decomposed_chain_join_enable_chain_join=(
            query_plan_ontology_evidence_set_decomposed_chain_join_enable_chain_join
        ),
        query_plan_ontology_evidence_set_decomposed_chain_join_require_bridge_signal=(
            query_plan_ontology_evidence_set_decomposed_chain_join_require_bridge_signal
        ),
        query_plan_ontology_evidence_set_decomposed_chain_join_top2_target_datasets=(
            args.query_plan_ontology_evidence_set_decomposed_chain_join_top2_target_datasets
        ),
        query_plan_ontology_evidence_set_decomposed_chain_join_enable_top5_fallback=(
            query_plan_ontology_evidence_set_decomposed_chain_join_enable_top5_fallback
        ),
        query_plan_ontology_evidence_set_decomposed_chain_join_top5_fallback_target_datasets=(
            args.query_plan_ontology_evidence_set_decomposed_chain_join_top5_fallback_target_datasets
        ),
        query_plan_ontology_evidence_set_decomposed_chain_join_top5_require_answer_signal=(
            query_plan_ontology_evidence_set_decomposed_chain_join_top5_require_answer_signal
        ),
        query_plan_ontology_evidence_set_decomposed_chain_join_top5_min_chain_score=(
            args.query_plan_ontology_evidence_set_decomposed_chain_join_top5_min_chain_score
        ),
        query_plan_ontology_evidence_set_decomposed_chain_join_top5_min_net_margin=(
            args.query_plan_ontology_evidence_set_decomposed_chain_join_top5_min_net_margin
        ),
        query_plan_ontology_evidence_set_decomposed_chain_join_top5_min_text_score=(
            args.query_plan_ontology_evidence_set_decomposed_chain_join_top5_min_text_score
        ),
        query_plan_ontology_evidence_set_decomposed_chain_join_top5_max_risk=(
            args.query_plan_ontology_evidence_set_decomposed_chain_join_top5_max_risk
        ),
        query_plan_ontology_evidence_set_decomposed_chain_join_top2_min_chain_score=(
            args.query_plan_ontology_evidence_set_decomposed_chain_join_top2_min_chain_score
        ),
        query_plan_ontology_evidence_set_decomposed_chain_join_top2_min_margin=(
            args.query_plan_ontology_evidence_set_decomposed_chain_join_top2_min_margin
        ),
        query_plan_ontology_evidence_set_decomposed_chain_join_top2_min_net_margin=(
            args.query_plan_ontology_evidence_set_decomposed_chain_join_top2_min_net_margin
        ),
        query_plan_ontology_evidence_set_decomposed_chain_join_top2_min_text_score=(
            args.query_plan_ontology_evidence_set_decomposed_chain_join_top2_min_text_score
        ),
        query_plan_ontology_evidence_set_decomposed_chain_join_top2_min_second_text_score=(
            args.query_plan_ontology_evidence_set_decomposed_chain_join_top2_min_second_text_score
        ),
        query_plan_ontology_evidence_set_decomposed_chain_join_top2_min_source_precision=(
            args.query_plan_ontology_evidence_set_decomposed_chain_join_top2_min_source_precision
        ),
        query_plan_ontology_evidence_set_decomposed_chain_join_top2_max_risk=(
            args.query_plan_ontology_evidence_set_decomposed_chain_join_top2_max_risk
        ),
        query_plan_ontology_evidence_set_decomposed_chain_join_top2_require_answer_signal=(
            query_plan_ontology_evidence_set_decomposed_chain_join_top2_require_answer_signal
        ),
        query_plan_ontology_evidence_set_decomposed_chain_join_top2_require_non_simple=(
            query_plan_ontology_evidence_set_decomposed_chain_join_top2_require_non_simple
        ),
        query_plan_ontology_evidence_set_decomposed_chain_join_enable_slot_completion_promotion=(
            query_plan_ontology_evidence_set_decomposed_chain_join_enable_slot_completion_promotion
        ),
        query_plan_ontology_evidence_set_decomposed_chain_join_slot_completion_target_datasets=(
            args.query_plan_ontology_evidence_set_decomposed_chain_join_slot_completion_target_datasets
        ),
        query_plan_ontology_evidence_set_decomposed_chain_join_slot_completion_min_chain_score=(
            args.query_plan_ontology_evidence_set_decomposed_chain_join_slot_completion_min_chain_score
        ),
        query_plan_ontology_evidence_set_decomposed_chain_join_slot_completion_min_required_gain=(
            args.query_plan_ontology_evidence_set_decomposed_chain_join_slot_completion_min_required_gain
        ),
        query_plan_ontology_evidence_set_decomposed_chain_join_slot_completion_min_coverage_gain=(
            args.query_plan_ontology_evidence_set_decomposed_chain_join_slot_completion_min_coverage_gain
        ),
        query_plan_ontology_evidence_set_decomposed_chain_join_slot_completion_min_role_gain=(
            args.query_plan_ontology_evidence_set_decomposed_chain_join_slot_completion_min_role_gain
        ),
        query_plan_ontology_evidence_set_decomposed_chain_join_slot_completion_max_current_coverage=(
            args.query_plan_ontology_evidence_set_decomposed_chain_join_slot_completion_max_current_coverage
        ),
        query_plan_ontology_evidence_set_decomposed_chain_join_slot_completion_min_text_score=(
            args.query_plan_ontology_evidence_set_decomposed_chain_join_slot_completion_min_text_score
        ),
        query_plan_ontology_evidence_set_decomposed_chain_join_slot_completion_min_second_text_score=(
            args.query_plan_ontology_evidence_set_decomposed_chain_join_slot_completion_min_second_text_score
        ),
        query_plan_ontology_evidence_set_decomposed_chain_join_slot_completion_max_risk=(
            args.query_plan_ontology_evidence_set_decomposed_chain_join_slot_completion_max_risk
        ),
        query_plan_ontology_evidence_set_decomposed_chain_join_slot_completion_allow_top1_replacement=(
            query_plan_ontology_evidence_set_decomposed_chain_join_slot_completion_allow_top1_replacement
        ),
        query_plan_ontology_evidence_set_decomposed_chain_join_slot_completion_require_bridge_or_answer=(
            query_plan_ontology_evidence_set_decomposed_chain_join_slot_completion_require_bridge_or_answer
        ),
        query_plan_ontology_evidence_set_decomposed_chain_join_enable_metric_regret_gate=(
            query_plan_ontology_evidence_set_decomposed_chain_join_enable_metric_regret_gate
        ),
        query_plan_ontology_evidence_set_decomposed_chain_join_metric_regret_target_datasets=(
            args.query_plan_ontology_evidence_set_decomposed_chain_join_metric_regret_target_datasets
        ),
        query_plan_ontology_evidence_set_decomposed_chain_join_metric_regret_protect_top1=(
            query_plan_ontology_evidence_set_decomposed_chain_join_metric_regret_protect_top1
        ),
        query_plan_ontology_evidence_set_decomposed_chain_join_metric_regret_allow_top1_min_net_margin=(
            args.query_plan_ontology_evidence_set_decomposed_chain_join_metric_regret_allow_top1_min_net_margin
        ),
        query_plan_ontology_evidence_set_decomposed_chain_join_metric_regret_allow_top1_min_required_gain=(
            args.query_plan_ontology_evidence_set_decomposed_chain_join_metric_regret_allow_top1_min_required_gain
        ),
        query_plan_ontology_evidence_set_decomposed_chain_join_metric_regret_allow_top1_min_text_score=(
            args.query_plan_ontology_evidence_set_decomposed_chain_join_metric_regret_allow_top1_min_text_score
        ),
        query_plan_ontology_evidence_set_decomposed_chain_join_metric_regret_allow_top1_min_source_precision=(
            args.query_plan_ontology_evidence_set_decomposed_chain_join_metric_regret_allow_top1_min_source_precision
        ),
        query_plan_ontology_evidence_set_decomposed_chain_join_metric_regret_allow_top1_max_risk=(
            args.query_plan_ontology_evidence_set_decomposed_chain_join_metric_regret_allow_top1_max_risk
        ),
        query_plan_ontology_evidence_set_decomposed_chain_join_metric_regret_rank2_min_net_margin=(
            args.query_plan_ontology_evidence_set_decomposed_chain_join_metric_regret_rank2_min_net_margin
        ),
        query_plan_ontology_evidence_set_decomposed_chain_join_metric_regret_rank2_min_required_gain=(
            args.query_plan_ontology_evidence_set_decomposed_chain_join_metric_regret_rank2_min_required_gain
        ),
        query_plan_ontology_evidence_set_decomposed_chain_join_metric_regret_rank2_min_text_score=(
            args.query_plan_ontology_evidence_set_decomposed_chain_join_metric_regret_rank2_min_text_score
        ),
        query_plan_ontology_evidence_set_decomposed_chain_join_metric_regret_enable_top5_append=(
            query_plan_ontology_evidence_set_decomposed_chain_join_metric_regret_enable_top5_append
        ),
        query_plan_ontology_evidence_set_enable_regret_calibrated_set_optimizer=(
            query_plan_ontology_evidence_set_enable_regret_calibrated_set_optimizer
        ),
        query_plan_ontology_evidence_set_regret_calibrated_set_target_datasets=(
            args.query_plan_ontology_evidence_set_regret_calibrated_set_target_datasets
        ),
        query_plan_ontology_evidence_set_regret_calibrated_set_max_candidates=(
            args.query_plan_ontology_evidence_set_regret_calibrated_set_max_candidates
        ),
        query_plan_ontology_evidence_set_regret_calibrated_set_max_top2_swaps=(
            args.query_plan_ontology_evidence_set_regret_calibrated_set_max_top2_swaps
        ),
        query_plan_ontology_evidence_set_regret_calibrated_set_max_top5_swaps=(
            args.query_plan_ontology_evidence_set_regret_calibrated_set_max_top5_swaps
        ),
        query_plan_ontology_evidence_set_regret_calibrated_set_protect_top1=(
            query_plan_ontology_evidence_set_regret_calibrated_set_protect_top1
        ),
        query_plan_ontology_evidence_set_regret_calibrated_set_max_original_rank=(
            args.query_plan_ontology_evidence_set_regret_calibrated_set_max_original_rank
        ),
        query_plan_ontology_evidence_set_regret_calibrated_set_min_regret=(
            args.query_plan_ontology_evidence_set_regret_calibrated_set_min_regret
        ),
        query_plan_ontology_evidence_set_regret_calibrated_set_top2_min_regret=(
            args.query_plan_ontology_evidence_set_regret_calibrated_set_top2_min_regret
        ),
        query_plan_ontology_evidence_set_regret_calibrated_set_min_text_score=(
            args.query_plan_ontology_evidence_set_regret_calibrated_set_min_text_score
        ),
        query_plan_ontology_evidence_set_regret_calibrated_set_top2_min_text_score=(
            args.query_plan_ontology_evidence_set_regret_calibrated_set_top2_min_text_score
        ),
        query_plan_ontology_evidence_set_regret_calibrated_set_min_source_precision=(
            args.query_plan_ontology_evidence_set_regret_calibrated_set_min_source_precision
        ),
        query_plan_ontology_evidence_set_regret_calibrated_set_require_answer_signal=(
            query_plan_ontology_evidence_set_regret_calibrated_set_require_answer_signal
        ),
        query_plan_ontology_evidence_set_regret_calibrated_set_cost_weight=(
            args.query_plan_ontology_evidence_set_regret_calibrated_set_cost_weight
        ),
        query_plan_ontology_evidence_set_regret_calibrated_set_risk_weight=(
            args.query_plan_ontology_evidence_set_regret_calibrated_set_risk_weight
        ),
        query_plan_ontology_evidence_set_regret_calibrated_set_gap_weight=(
            args.query_plan_ontology_evidence_set_regret_calibrated_set_gap_weight
        ),
        query_plan_ontology_evidence_set_regret_calibrated_set_slot_weight=(
            args.query_plan_ontology_evidence_set_regret_calibrated_set_slot_weight
        ),
        query_plan_ontology_evidence_set_regret_calibrated_set_max_risk=(
            args.query_plan_ontology_evidence_set_regret_calibrated_set_max_risk
        ),
        query_plan_ontology_evidence_set_regret_calibrated_set_top2_max_risk=(
            args.query_plan_ontology_evidence_set_regret_calibrated_set_top2_max_risk
        ),
        query_plan_ontology_evidence_set_regret_calibrated_set_min_gap_pressure=(
            args.query_plan_ontology_evidence_set_regret_calibrated_set_min_gap_pressure
        ),
        query_plan_ontology_evidence_set_regret_calibrated_set_enable_top5_safe_bypass=(
            query_plan_ontology_evidence_set_regret_calibrated_set_enable_top5_safe_bypass
        ),
        query_plan_ontology_evidence_set_regret_calibrated_set_top5_safe_bypass_max_risk=(
            args.query_plan_ontology_evidence_set_regret_calibrated_set_top5_safe_bypass_max_risk
        ),
        query_plan_ontology_evidence_set_regret_calibrated_set_top5_safe_bypass_min_support=(
            args.query_plan_ontology_evidence_set_regret_calibrated_set_top5_safe_bypass_min_support
        ),
        query_plan_ontology_evidence_set_regret_calibrated_set_top5_safe_bypass_min_regret=(
            args.query_plan_ontology_evidence_set_regret_calibrated_set_top5_safe_bypass_min_regret
        ),
        query_plan_ontology_evidence_set_enable_evidence_sidecar=(
            query_plan_ontology_evidence_set_enable_evidence_sidecar
        ),
        query_plan_ontology_evidence_set_evidence_sidecar_max_items=(
            args.query_plan_ontology_evidence_set_evidence_sidecar_max_items
        ),
        query_plan_ontology_evidence_set_evidence_sidecar_min_score=(
            args.query_plan_ontology_evidence_set_evidence_sidecar_min_score
        ),
        query_plan_ontology_evidence_set_evidence_sidecar_min_text_score=(
            args.query_plan_ontology_evidence_set_evidence_sidecar_min_text_score
        ),
        query_plan_ontology_evidence_set_evidence_sidecar_max_original_rank=(
            args.query_plan_ontology_evidence_set_evidence_sidecar_max_original_rank
        ),
        query_plan_ontology_evidence_set_evidence_sidecar_allow_relation_only=(
            query_plan_ontology_evidence_set_evidence_sidecar_allow_relation_only
        ),
        query_plan_ontology_evidence_set_evidence_sidecar_preferred_sources=(
            args.query_plan_ontology_evidence_set_evidence_sidecar_preferred_sources
        ),
        query_plan_ontology_evidence_set_evidence_sidecar_gate_mode=(
            args.query_plan_ontology_evidence_set_evidence_sidecar_gate_mode
        ),
        query_plan_ontology_evidence_set_evidence_sidecar_complex_datasets=(
            args.query_plan_ontology_evidence_set_evidence_sidecar_complex_datasets
        ),
        query_plan_ontology_evidence_set_evidence_sidecar_simple_datasets=(
            args.query_plan_ontology_evidence_set_evidence_sidecar_simple_datasets
        ),
        query_plan_ontology_evidence_set_evidence_sidecar_strict_min_text_score=(
            args.query_plan_ontology_evidence_set_evidence_sidecar_strict_min_text_score
        ),
        query_plan_ontology_evidence_set_evidence_sidecar_strict_min_path_score=(
            args.query_plan_ontology_evidence_set_evidence_sidecar_strict_min_path_score
        ),
        query_plan_ontology_evidence_set_evidence_sidecar_require_anchor_hit=(
            query_plan_ontology_evidence_set_evidence_sidecar_require_anchor_hit
        ),
        query_plan_ontology_evidence_set_evidence_sidecar_require_terminal_hit=(
            query_plan_ontology_evidence_set_evidence_sidecar_require_terminal_hit
        ),
        query_plan_ontology_evidence_set_evidence_sidecar_require_preferred_source=(
            query_plan_ontology_evidence_set_evidence_sidecar_require_preferred_source
        ),
        query_plan_ontology_evidence_set_evidence_sidecar_disable_simple_without_answer_overlap=(
            query_plan_ontology_evidence_set_evidence_sidecar_disable_simple_without_answer_overlap
        ),
        query_plan_ontology_evidence_set_evidence_sidecar_disable_answer_type_conflict=(
            query_plan_ontology_evidence_set_evidence_sidecar_disable_answer_type_conflict
        ),
        query_plan_ontology_evidence_set_strict_backbone_scope=query_plan_ontology_evidence_set_strict_backbone_scope,
        query_plan_ontology_evidence_set_anchor_relation_min_source_precision=args.query_plan_ontology_evidence_set_anchor_relation_min_source_precision,
        query_plan_ontology_evidence_set_anchor_relation_min_selected_precision=args.query_plan_ontology_evidence_set_anchor_relation_min_selected_precision,
        query_plan_ontology_evidence_set_recall_selector_model_path=args.query_plan_ontology_evidence_set_recall_selector_model_path,
        query_plan_ontology_evidence_set_recall_selector_mode=args.query_plan_ontology_evidence_set_recall_selector_mode,
        query_plan_ontology_evidence_set_recall_selector_min_score=args.query_plan_ontology_evidence_set_recall_selector_min_score,
        query_plan_ontology_evidence_set_recall_selector_score_weight=args.query_plan_ontology_evidence_set_recall_selector_score_weight,
        query_plan_ontology_evidence_set_recall_selector_promote_confidence=args.query_plan_ontology_evidence_set_recall_selector_promote_confidence,
        query_plan_ontology_evidence_set_recall_selector_counterfactual_weight=args.query_plan_ontology_evidence_set_recall_selector_counterfactual_weight,
        query_plan_ontology_evidence_set_recall_selector_bypass_router_min_score=args.query_plan_ontology_evidence_set_recall_selector_bypass_router_min_score,
        query_plan_ontology_evidence_set_recall_swapper_model_path=args.query_plan_ontology_evidence_set_recall_swapper_model_path,
        query_plan_ontology_evidence_set_recall_swapper_mode=args.query_plan_ontology_evidence_set_recall_swapper_mode,
        query_plan_ontology_evidence_set_recall_swapper_min_score=args.query_plan_ontology_evidence_set_recall_swapper_min_score,
        query_plan_ontology_evidence_set_recall_swapper_top2_min_score=args.query_plan_ontology_evidence_set_recall_swapper_top2_min_score,
        query_plan_ontology_evidence_set_recall_swapper_score_weight=args.query_plan_ontology_evidence_set_recall_swapper_score_weight,
        query_plan_ontology_evidence_set_recall_swapper_promote_confidence=args.query_plan_ontology_evidence_set_recall_swapper_promote_confidence,
        query_plan_ontology_evidence_set_recall_swapper_counterfactual_weight=args.query_plan_ontology_evidence_set_recall_swapper_counterfactual_weight,
        query_plan_ontology_evidence_set_recall_swapper_max_top2_swaps=args.query_plan_ontology_evidence_set_recall_swapper_max_top2_swaps,
        query_plan_ontology_evidence_set_recall_swapper_allow_relation_only_top2=query_plan_ontology_evidence_set_recall_swapper_allow_relation_only_top2,
        query_plan_ontology_evidence_set_rank2_rescue_model_path=args.query_plan_ontology_evidence_set_rank2_rescue_model_path,
        query_plan_ontology_evidence_set_rank2_rescue_mode=args.query_plan_ontology_evidence_set_rank2_rescue_mode,
        query_plan_ontology_evidence_set_rank2_rescue_min_score=args.query_plan_ontology_evidence_set_rank2_rescue_min_score,
        query_plan_ontology_evidence_set_rank2_rescue_min_confidence=args.query_plan_ontology_evidence_set_rank2_rescue_min_confidence,
        query_plan_ontology_evidence_set_rank2_rescue_max_original_rank=args.query_plan_ontology_evidence_set_rank2_rescue_max_original_rank,
        query_plan_ontology_evidence_set_rank2_rescue_min_source_precision=args.query_plan_ontology_evidence_set_rank2_rescue_min_source_precision,
        query_plan_ontology_evidence_set_rank2_rescue_min_selected_precision=args.query_plan_ontology_evidence_set_rank2_rescue_min_selected_precision,
        query_plan_ontology_evidence_set_rank2_rescue_require_terminal_or_strict=query_plan_ontology_evidence_set_rank2_rescue_require_terminal_or_strict,
        query_plan_ontology_evidence_set_rank2_rescue_max_swaps=args.query_plan_ontology_evidence_set_rank2_rescue_max_swaps,
        query_plan_ontology_evidence_set_optimizer_mode=args.query_plan_ontology_evidence_set_optimizer_mode,
        query_plan_ontology_evidence_set_optimizer_min_confidence=args.query_plan_ontology_evidence_set_optimizer_min_confidence,
        query_plan_ontology_evidence_set_optimizer_min_source_precision=args.query_plan_ontology_evidence_set_optimizer_min_source_precision,
        query_plan_ontology_evidence_set_optimizer_min_path_source_precision=args.query_plan_ontology_evidence_set_optimizer_min_path_source_precision,
        query_plan_ontology_evidence_set_optimizer_max_original_rank=args.query_plan_ontology_evidence_set_optimizer_max_original_rank,
        query_plan_ontology_evidence_set_optimizer_rank2_margin=args.query_plan_ontology_evidence_set_optimizer_rank2_margin,
        query_plan_ontology_evidence_set_optimizer_require_complex_or_path=query_plan_ontology_evidence_set_optimizer_require_complex_or_path,
        query_plan_ontology_evidence_set_optimizer_max_swaps=args.query_plan_ontology_evidence_set_optimizer_max_swaps,
        query_plan_ontology_evidence_set_rank2_pairwise_optimizer_model_path=args.query_plan_ontology_evidence_set_rank2_pairwise_optimizer_model_path,
        query_plan_ontology_evidence_set_rank2_pairwise_optimizer_mode=args.query_plan_ontology_evidence_set_rank2_pairwise_optimizer_mode,
        query_plan_ontology_evidence_set_rank2_pairwise_optimizer_min_score=args.query_plan_ontology_evidence_set_rank2_pairwise_optimizer_min_score,
        query_plan_ontology_evidence_set_rank2_pairwise_optimizer_min_confidence=args.query_plan_ontology_evidence_set_rank2_pairwise_optimizer_min_confidence,
        query_plan_ontology_evidence_set_rank2_pairwise_optimizer_max_original_rank=args.query_plan_ontology_evidence_set_rank2_pairwise_optimizer_max_original_rank,
        query_plan_ontology_evidence_set_rank2_pairwise_optimizer_max_swaps=args.query_plan_ontology_evidence_set_rank2_pairwise_optimizer_max_swaps,
        query_plan_ontology_evidence_set_rank2_pairwise_optimizer_min_value_margin=args.query_plan_ontology_evidence_set_rank2_pairwise_optimizer_min_value_margin,
        query_plan_ontology_evidence_set_rank2_pairwise_optimizer_path_override_min_value_margin=args.query_plan_ontology_evidence_set_rank2_pairwise_optimizer_path_override_min_value_margin,
        query_plan_ontology_evidence_set_rank2_pairwise_optimizer_require_complex_or_path=query_plan_ontology_evidence_set_rank2_pairwise_optimizer_require_complex_or_path,
        query_plan_ontology_evidence_set_rank2_pairwise_optimizer_after_evidence_optimizer=query_plan_ontology_evidence_set_rank2_pairwise_optimizer_after_evidence_optimizer,
        query_plan_ontology_evidence_set_window_selector_model_path=args.query_plan_ontology_evidence_set_window_selector_model_path,
        query_plan_ontology_evidence_set_window_selector_mode=args.query_plan_ontology_evidence_set_window_selector_mode,
        query_plan_ontology_evidence_set_window_selector_min_score=args.query_plan_ontology_evidence_set_window_selector_min_score,
        query_plan_ontology_evidence_set_window_selector_top2_min_score=args.query_plan_ontology_evidence_set_window_selector_top2_min_score,
        query_plan_ontology_evidence_set_window_selector_score_weight=args.query_plan_ontology_evidence_set_window_selector_score_weight,
        query_plan_ontology_evidence_set_window_selector_max_ontology_passages=args.query_plan_ontology_evidence_set_window_selector_max_ontology_passages,
        query_plan_ontology_evidence_set_window_selector_veto_below=args.query_plan_ontology_evidence_set_window_selector_veto_below,
        query_plan_ontology_evidence_set_window_selector_backbone_veto_below=args.query_plan_ontology_evidence_set_window_selector_backbone_veto_below,
        query_plan_ontology_evidence_set_window_selector_rescue_min_score=args.query_plan_ontology_evidence_set_window_selector_rescue_min_score,
        query_plan_ontology_evidence_set_window_selector_rescue_max_passages=args.query_plan_ontology_evidence_set_window_selector_rescue_max_passages,
        query_plan_ontology_evidence_set_router_model_path=args.query_plan_ontology_evidence_set_router_model_path,
        query_plan_ontology_evidence_set_router_mode=args.query_plan_ontology_evidence_set_router_mode,
        query_plan_ontology_evidence_set_router_min_score=args.query_plan_ontology_evidence_set_router_min_score,
        query_plan_ontology_evidence_set_router_precision_min_score=args.query_plan_ontology_evidence_set_router_precision_min_score,
        query_plan_ontology_evidence_set_router_precision_min_confidence=args.query_plan_ontology_evidence_set_router_precision_min_confidence,
        query_plan_ontology_evidence_set_router_precision_max_ontology_passages=args.query_plan_ontology_evidence_set_router_precision_max_ontology_passages,
        query_plan_ontology_evidence_set_schema_route_adapter_mode=args.query_plan_ontology_evidence_set_schema_route_adapter_mode,
        query_plan_ontology_evidence_set_schema_route_adapter_score_weight=args.query_plan_ontology_evidence_set_schema_route_adapter_score_weight,
        query_plan_ontology_evidence_set_schema_route_adapter_confidence_boost=args.query_plan_ontology_evidence_set_schema_route_adapter_confidence_boost,
        query_plan_ontology_evidence_set_schema_route_adapter_min_score=args.query_plan_ontology_evidence_set_schema_route_adapter_min_score,
        query_plan_ontology_evidence_set_schema_route_adapter_reject_below=args.query_plan_ontology_evidence_set_schema_route_adapter_reject_below,
        query_plan_ontology_evidence_set_schema_route_adapter_max_subqueries=args.query_plan_ontology_evidence_set_schema_route_adapter_max_subqueries,
        query_plan_ontology_evidence_set_enable_schema_slot_utility=query_plan_ontology_evidence_set_enable_schema_slot_utility,
        query_plan_ontology_evidence_set_schema_slot_utility_min_score=args.query_plan_ontology_evidence_set_schema_slot_utility_min_score,
        query_plan_ontology_evidence_set_schema_slot_utility_top2_min_score=args.query_plan_ontology_evidence_set_schema_slot_utility_top2_min_score,
        query_plan_ontology_evidence_set_schema_slot_utility_score_weight=args.query_plan_ontology_evidence_set_schema_slot_utility_score_weight,
        query_plan_ontology_evidence_set_schema_slot_utility_confidence_boost=args.query_plan_ontology_evidence_set_schema_slot_utility_confidence_boost,
        query_plan_ontology_evidence_set_schema_slot_utility_top2_max_swaps=args.query_plan_ontology_evidence_set_schema_slot_utility_top2_max_swaps,
        query_plan_ontology_evidence_set_enable_provenance_groupwise_selector=query_plan_ontology_evidence_set_enable_provenance_groupwise_selector,
        query_plan_ontology_evidence_set_provenance_groupwise_min_score=args.query_plan_ontology_evidence_set_provenance_groupwise_min_score,
        query_plan_ontology_evidence_set_provenance_groupwise_top2_min_score=args.query_plan_ontology_evidence_set_provenance_groupwise_top2_min_score,
        query_plan_ontology_evidence_set_provenance_groupwise_score_weight=args.query_plan_ontology_evidence_set_provenance_groupwise_score_weight,
        query_plan_ontology_evidence_set_provenance_groupwise_max_top5_swaps=args.query_plan_ontology_evidence_set_provenance_groupwise_max_top5_swaps,
        query_plan_ontology_evidence_set_provenance_groupwise_max_top2_swaps=args.query_plan_ontology_evidence_set_provenance_groupwise_max_top2_swaps,
        query_plan_ontology_evidence_set_provenance_groupwise_min_source_precision=args.query_plan_ontology_evidence_set_provenance_groupwise_min_source_precision,
        query_plan_ontology_evidence_set_provenance_groupwise_require_text_signal=query_plan_ontology_evidence_set_provenance_groupwise_require_text_signal,
        query_plan_ontology_evidence_set_enable_evidence_join_planner=query_plan_ontology_evidence_set_enable_evidence_join_planner,
        query_plan_ontology_evidence_set_evidence_join_planner_min_score=args.query_plan_ontology_evidence_set_evidence_join_planner_min_score,
        query_plan_ontology_evidence_set_evidence_join_planner_top2_min_score=args.query_plan_ontology_evidence_set_evidence_join_planner_top2_min_score,
        query_plan_ontology_evidence_set_evidence_join_planner_min_marginal_gain=args.query_plan_ontology_evidence_set_evidence_join_planner_min_marginal_gain,
        query_plan_ontology_evidence_set_evidence_join_planner_min_value_margin=args.query_plan_ontology_evidence_set_evidence_join_planner_min_value_margin,
        query_plan_ontology_evidence_set_evidence_join_planner_score_weight=args.query_plan_ontology_evidence_set_evidence_join_planner_score_weight,
        query_plan_ontology_evidence_set_evidence_join_planner_marginal_gain_weight=args.query_plan_ontology_evidence_set_evidence_join_planner_marginal_gain_weight,
        query_plan_ontology_evidence_set_evidence_join_planner_max_top5_swaps=args.query_plan_ontology_evidence_set_evidence_join_planner_max_top5_swaps,
        query_plan_ontology_evidence_set_evidence_join_planner_max_top2_swaps=args.query_plan_ontology_evidence_set_evidence_join_planner_max_top2_swaps,
        query_plan_ontology_evidence_set_evidence_join_planner_router_bypass_min_score=args.query_plan_ontology_evidence_set_evidence_join_planner_router_bypass_min_score,
        query_plan_ontology_evidence_set_evidence_join_planner_require_text_or_path=query_plan_ontology_evidence_set_evidence_join_planner_require_text_or_path,
        query_plan_ontology_evidence_set_evidence_join_planner_enable_answer_slot_rescue=query_plan_ontology_evidence_set_evidence_join_planner_enable_answer_slot_rescue,
        query_plan_ontology_evidence_set_evidence_join_planner_answer_slot_score_weight=args.query_plan_ontology_evidence_set_evidence_join_planner_answer_slot_score_weight,
        query_plan_ontology_evidence_set_evidence_join_planner_answer_slot_min_score=args.query_plan_ontology_evidence_set_evidence_join_planner_answer_slot_min_score,
        query_plan_ontology_evidence_set_evidence_join_planner_answer_slot_top2_min_score=args.query_plan_ontology_evidence_set_evidence_join_planner_answer_slot_top2_min_score,
        query_plan_ontology_evidence_set_evidence_join_planner_answer_slot_router_bypass_min_score=args.query_plan_ontology_evidence_set_evidence_join_planner_answer_slot_router_bypass_min_score,
        query_plan_ontology_evidence_set_evidence_join_planner_enable_operator_aware=query_plan_ontology_evidence_set_evidence_join_planner_enable_operator_aware,
        query_plan_ontology_evidence_set_evidence_join_planner_operator_score_weight=args.query_plan_ontology_evidence_set_evidence_join_planner_operator_score_weight,
        query_plan_ontology_evidence_set_evidence_join_planner_operator_min_gain=args.query_plan_ontology_evidence_set_evidence_join_planner_operator_min_gain,
        query_plan_ontology_evidence_set_evidence_join_planner_operator_top2_min_gain=args.query_plan_ontology_evidence_set_evidence_join_planner_operator_top2_min_gain,
        query_plan_ontology_evidence_set_evidence_join_planner_operator_chain_pressure_weight=args.query_plan_ontology_evidence_set_evidence_join_planner_operator_chain_pressure_weight,
        query_plan_ontology_evidence_set_evidence_join_planner_enable_risk_gate=query_plan_ontology_evidence_set_evidence_join_planner_enable_risk_gate,
        query_plan_ontology_evidence_set_evidence_join_planner_risk_gate_top2_min_source_precision=args.query_plan_ontology_evidence_set_evidence_join_planner_risk_gate_top2_min_source_precision,
        query_plan_ontology_evidence_set_evidence_join_planner_risk_gate_top5_min_source_precision=args.query_plan_ontology_evidence_set_evidence_join_planner_risk_gate_top5_min_source_precision,
        query_plan_ontology_evidence_set_evidence_join_planner_risk_gate_top2_min_margin=args.query_plan_ontology_evidence_set_evidence_join_planner_risk_gate_top2_min_margin,
        query_plan_ontology_evidence_set_evidence_join_planner_risk_gate_top5_min_margin=args.query_plan_ontology_evidence_set_evidence_join_planner_risk_gate_top5_min_margin,
        query_plan_ontology_evidence_set_evidence_join_planner_risk_gate_max_lost_value=args.query_plan_ontology_evidence_set_evidence_join_planner_risk_gate_max_lost_value,
        query_plan_ontology_evidence_set_enable_answer_pairwise_verifier=query_plan_ontology_evidence_set_enable_answer_pairwise_verifier,
        query_plan_ontology_evidence_set_answer_pairwise_min_score=args.query_plan_ontology_evidence_set_answer_pairwise_min_score,
        query_plan_ontology_evidence_set_answer_pairwise_top2_min_score=args.query_plan_ontology_evidence_set_answer_pairwise_top2_min_score,
        query_plan_ontology_evidence_set_answer_pairwise_min_margin=args.query_plan_ontology_evidence_set_answer_pairwise_min_margin,
        query_plan_ontology_evidence_set_answer_pairwise_top2_min_margin=args.query_plan_ontology_evidence_set_answer_pairwise_top2_min_margin,
        query_plan_ontology_evidence_set_answer_pairwise_max_top5_swaps=args.query_plan_ontology_evidence_set_answer_pairwise_max_top5_swaps,
        query_plan_ontology_evidence_set_answer_pairwise_max_top2_swaps=args.query_plan_ontology_evidence_set_answer_pairwise_max_top2_swaps,
        query_plan_ontology_evidence_set_answer_pairwise_require_answer_signal=query_plan_ontology_evidence_set_answer_pairwise_require_answer_signal,
        query_plan_ontology_evidence_set_answer_pairwise_score_weight=args.query_plan_ontology_evidence_set_answer_pairwise_score_weight,
        query_plan_ontology_evidence_set_enable_counterfactual_text_verifier=query_plan_ontology_evidence_set_enable_counterfactual_text_verifier,
        query_plan_ontology_evidence_set_counterfactual_text_verifier_min_score=args.query_plan_ontology_evidence_set_counterfactual_text_verifier_min_score,
        query_plan_ontology_evidence_set_counterfactual_text_verifier_top2_min_score=args.query_plan_ontology_evidence_set_counterfactual_text_verifier_top2_min_score,
        query_plan_ontology_evidence_set_counterfactual_text_verifier_min_margin=args.query_plan_ontology_evidence_set_counterfactual_text_verifier_min_margin,
        query_plan_ontology_evidence_set_counterfactual_text_verifier_top2_min_margin=args.query_plan_ontology_evidence_set_counterfactual_text_verifier_top2_min_margin,
        query_plan_ontology_evidence_set_counterfactual_text_verifier_max_top5_swaps=args.query_plan_ontology_evidence_set_counterfactual_text_verifier_max_top5_swaps,
        query_plan_ontology_evidence_set_counterfactual_text_verifier_max_top2_swaps=args.query_plan_ontology_evidence_set_counterfactual_text_verifier_max_top2_swaps,
        query_plan_ontology_evidence_set_counterfactual_text_verifier_max_original_rank=args.query_plan_ontology_evidence_set_counterfactual_text_verifier_max_original_rank,
        query_plan_ontology_evidence_set_counterfactual_text_verifier_min_candidate_text_score=args.query_plan_ontology_evidence_set_counterfactual_text_verifier_min_candidate_text_score,
        query_plan_ontology_evidence_set_counterfactual_text_verifier_score_weight=args.query_plan_ontology_evidence_set_counterfactual_text_verifier_score_weight,
        query_plan_ontology_evidence_set_counterfactual_text_verifier_require_answer_signal=query_plan_ontology_evidence_set_counterfactual_text_verifier_require_answer_signal,
        query_plan_ontology_evidence_set_enable_decomposition_slots=query_plan_ontology_evidence_set_enable_decomposition_slots,
        query_plan_ontology_evidence_set_decomposition_slot_weight=args.query_plan_ontology_evidence_set_decomposition_slot_weight,
        query_plan_ontology_evidence_set_decomposition_slot_max_slots=args.query_plan_ontology_evidence_set_decomposition_slot_max_slots,
        query_plan_ontology_evidence_set_enable_path_aware_selector=query_plan_ontology_evidence_set_enable_path_aware_selector,
        query_plan_ontology_evidence_set_path_aware_score_weight=args.query_plan_ontology_evidence_set_path_aware_score_weight,
        query_plan_ontology_evidence_set_path_aware_confidence_boost=args.query_plan_ontology_evidence_set_path_aware_confidence_boost,
        query_plan_ontology_evidence_set_path_aware_min_score=args.query_plan_ontology_evidence_set_path_aware_min_score,
        query_plan_ontology_evidence_set_path_aware_min_terminal_type_score=args.query_plan_ontology_evidence_set_path_aware_min_terminal_type_score,
        query_plan_ontology_evidence_set_path_aware_require_query_relation_coverage=query_plan_ontology_evidence_set_path_aware_require_query_relation_coverage,
        query_plan_ontology_evidence_set_text_verifier_mode=args.query_plan_ontology_evidence_set_text_verifier_mode,
        query_plan_ontology_evidence_set_text_verifier_min_score=args.query_plan_ontology_evidence_set_text_verifier_min_score,
        query_plan_ontology_evidence_set_text_verifier_reject_below=args.query_plan_ontology_evidence_set_text_verifier_reject_below,
        query_plan_ontology_evidence_set_text_verifier_score_weight=args.query_plan_ontology_evidence_set_text_verifier_score_weight,
        query_plan_ontology_evidence_set_text_verifier_confidence_boost=args.query_plan_ontology_evidence_set_text_verifier_confidence_boost,
        query_plan_ontology_evidence_set_text_verifier_max_chars=args.query_plan_ontology_evidence_set_text_verifier_max_chars,
        query_plan_ontology_lift_min_original_rank=args.query_plan_ontology_lift_min_original_rank,
        query_plan_ontology_lift_max_original_rank=args.query_plan_ontology_lift_max_original_rank,
        query_plan_ontology_soft_fallback_lift_max_original_rank=args.query_plan_ontology_soft_fallback_lift_max_original_rank,
        query_plan_ontology_soft_fallback_weight_scale=args.query_plan_ontology_soft_fallback_weight_scale,
        query_plan_ontology_validated_soft_max_paths=args.query_plan_ontology_validated_soft_max_paths,
        query_plan_ontology_validated_soft_min_answer_path_score=args.query_plan_ontology_validated_soft_min_answer_path_score,
        query_plan_ontology_gate_require_fact_endpoints=query_plan_ontology_gate_require_fact_endpoints,
        query_plan_ontology_gate_require_anchor_or_terminal=query_plan_ontology_gate_require_anchor_or_terminal,
        query_plan_ontology_gate_min_relation_token_overlap=args.query_plan_ontology_gate_min_relation_token_overlap,
        query_plan_ontology_enable_simple_evidence_validator=query_plan_ontology_enable_simple_evidence_validator,
        query_plan_ontology_simple_validator_candidate_top_k=args.query_plan_ontology_simple_validator_candidate_top_k,
        query_plan_ontology_simple_validator_max_evidence_paths=args.query_plan_ontology_simple_validator_max_evidence_paths,
        query_plan_ontology_simple_validator_weight=args.query_plan_ontology_simple_validator_weight,
        query_plan_ontology_simple_validator_fallback_to_path_lift=query_plan_ontology_simple_validator_fallback_to_path_lift,
        query_plan_ontology_enable_qa_evidence=query_plan_ontology_enable_qa_evidence,
        query_plan_ontology_enable_adapter_policy=query_plan_ontology_enable_adapter_policy,
        query_plan_ontology_qa_evidence_max_items=args.query_plan_ontology_qa_evidence_max_items,
        query_plan_ontology_qa_evidence_min_score=args.query_plan_ontology_qa_evidence_min_score,
        query_plan_ontology_qa_evidence_style=args.query_plan_ontology_qa_evidence_style,
        query_plan_ontology_enable_evidence_chain_materializer=(
            query_plan_ontology_enable_evidence_chain_materializer
        ),
        query_plan_ontology_evidence_chain_materializer_max_items=(
            args.query_plan_ontology_evidence_chain_materializer_max_items
        ),
        query_plan_ontology_evidence_chain_materializer_source_top_k=(
            args.query_plan_ontology_evidence_chain_materializer_source_top_k
        ),
        query_plan_ontology_evidence_chain_materializer_max_doc_chars=(
            args.query_plan_ontology_evidence_chain_materializer_max_doc_chars
        ),
        query_plan_ontology_evidence_chain_materializer_min_query_overlap=(
            args.query_plan_ontology_evidence_chain_materializer_min_query_overlap
        ),
        query_plan_ontology_evidence_chain_materializer_max_spans=(
            args.query_plan_ontology_evidence_chain_materializer_max_spans
        ),
        query_plan_ontology_evidence_chain_materializer_require_context_gap=(
            query_plan_ontology_evidence_chain_materializer_require_context_gap
        ),
        query_plan_ontology_evidence_chain_materializer_add_docs_to_context=(
            query_plan_ontology_evidence_chain_materializer_add_docs_to_context
        ),
        query_plan_ontology_evidence_chain_materializer_preserve_context_top_k=(
            args.query_plan_ontology_evidence_chain_materializer_preserve_context_top_k
        ),
        query_plan_ontology_evidence_chain_materializer_enable_chain_complete=(
            query_plan_ontology_evidence_chain_materializer_enable_chain_complete
        ),
        query_plan_ontology_evidence_chain_materializer_chain_complete_target_datasets=(
            args.query_plan_ontology_evidence_chain_materializer_chain_complete_target_datasets
        ),
        query_plan_ontology_evidence_chain_materializer_chain_complete_min_roles=(
            args.query_plan_ontology_evidence_chain_materializer_chain_complete_min_roles
        ),
        query_plan_ontology_evidence_chain_materializer_chain_complete_role_weight=(
            args.query_plan_ontology_evidence_chain_materializer_chain_complete_role_weight
        ),
        query_plan_ontology_evidence_chain_materializer_require_chain_complete_route=(
            query_plan_ontology_evidence_chain_materializer_require_chain_complete_route
        ),
        query_plan_ontology_evidence_chain_materializer_strict_admission=(
            query_plan_ontology_evidence_chain_materializer_strict_admission
        ),
        query_plan_ontology_evidence_chain_materializer_min_real_roles=(
            args.query_plan_ontology_evidence_chain_materializer_min_real_roles
        ),
        query_plan_ontology_evidence_chain_materializer_context_gap_use_qa_context=(
            query_plan_ontology_evidence_chain_materializer_context_gap_use_qa_context
        ),
        query_plan_ontology_evidence_chain_materializer_emit_answer_spans=(
            query_plan_ontology_evidence_chain_materializer_emit_answer_spans
        ),
        query_plan_ontology_evidence_chain_materializer_suppress_answer_spans_for_slots=(
            args.query_plan_ontology_evidence_chain_materializer_suppress_answer_spans_for_slots
        ),
        query_plan_ontology_enable_answer_candidate_operator=query_plan_ontology_enable_answer_candidate_operator,
        query_plan_ontology_answer_candidate_max_items=args.query_plan_ontology_answer_candidate_max_items,
        query_plan_ontology_answer_candidate_min_score=args.query_plan_ontology_answer_candidate_min_score,
        query_plan_ontology_answer_candidate_max_doc_chars=args.query_plan_ontology_answer_candidate_max_doc_chars,
        query_plan_ontology_answer_candidate_require_passage_support=(
            query_plan_ontology_answer_candidate_require_passage_support
        ),
        query_plan_ontology_answer_candidate_use_doc_spans=query_plan_ontology_answer_candidate_use_doc_spans,
        query_plan_ontology_answer_candidate_gate_mode=args.query_plan_ontology_answer_candidate_gate_mode,
        query_plan_ontology_answer_candidate_min_margin=args.query_plan_ontology_answer_candidate_min_margin,
        query_plan_ontology_answer_candidate_disable_ambiguous_slots=(
            query_plan_ontology_answer_candidate_disable_ambiguous_slots
        ),
        query_plan_ontology_answer_candidate_enable_slot_hint=(
            query_plan_ontology_answer_candidate_enable_slot_hint
        ),
        query_plan_ontology_enable_answer_canonicalizer=query_plan_ontology_enable_answer_canonicalizer,
        query_plan_ontology_answer_canonicalizer_min_candidate_score=(
            args.query_plan_ontology_answer_canonicalizer_min_candidate_score
        ),
        query_plan_ontology_answer_canonicalizer_require_candidate_support=(
            query_plan_ontology_answer_canonicalizer_require_candidate_support
        ),
        query_plan_ontology_answer_canonicalizer_fix_year_only=(
            query_plan_ontology_answer_canonicalizer_fix_year_only
        ),
        query_plan_ontology_answer_canonicalizer_replace_unknown=(
            query_plan_ontology_answer_canonicalizer_replace_unknown
        ),
        query_plan_ontology_answer_canonicalizer_allow_expansion=(
            query_plan_ontology_answer_canonicalizer_allow_expansion
        ),
        query_plan_ontology_answer_canonicalizer_enable_slot_shrink=(
            query_plan_ontology_answer_canonicalizer_enable_slot_shrink
        ),
        query_plan_ontology_answer_canonicalizer_allow_candidate_shrink=(
            query_plan_ontology_answer_canonicalizer_allow_candidate_shrink
        ),
        query_plan_ontology_enable_grounded_answer_verifier=(
            query_plan_ontology_enable_grounded_answer_verifier
        ),
        query_plan_ontology_grounded_answer_verifier_min_confidence=(
            args.query_plan_ontology_grounded_answer_verifier_min_confidence
        ),
        query_plan_ontology_grounded_answer_verifier_max_passage_chars=(
            args.query_plan_ontology_grounded_answer_verifier_max_passage_chars
        ),
        query_plan_ontology_enable_answer_operator_calibrator=(
            query_plan_ontology_enable_answer_operator_calibrator
        ),
        query_plan_ontology_answer_operator_calibrator_min_candidate_score=(
            args.query_plan_ontology_answer_operator_calibrator_min_candidate_score
        ),
        query_plan_ontology_answer_operator_calibrator_min_margin=(
            args.query_plan_ontology_answer_operator_calibrator_min_margin
        ),
        query_plan_ontology_answer_operator_calibrator_single_value_slots=(
            args.query_plan_ontology_answer_operator_calibrator_single_value_slots
        ),
        query_plan_ontology_enable_typed_answer_projection=(
            query_plan_ontology_enable_typed_answer_projection
        ),
        query_plan_ontology_typed_answer_projection_min_score=(
            args.query_plan_ontology_typed_answer_projection_min_score
        ),
        query_plan_ontology_typed_answer_projection_min_margin=(
            args.query_plan_ontology_typed_answer_projection_min_margin
        ),
        query_plan_ontology_typed_answer_projection_allowed_slots=(
            args.query_plan_ontology_typed_answer_projection_allowed_slots
        ),
        query_plan_ontology_typed_answer_projection_require_support=(
            query_plan_ontology_typed_answer_projection_require_support
        ),
        query_plan_ontology_typed_answer_projection_prefer_chain_order=(
            query_plan_ontology_typed_answer_projection_prefer_chain_order
        ),
        query_plan_ontology_typed_answer_projection_operator_gate=(
            query_plan_ontology_typed_answer_projection_operator_gate
        ),
        query_plan_ontology_typed_answer_projection_allowed_operators=(
            args.query_plan_ontology_typed_answer_projection_allowed_operators
        ),
        query_plan_ontology_typed_answer_projection_require_relation_or_slot_source=(
            query_plan_ontology_typed_answer_projection_require_relation_or_slot_source
        ),
        query_plan_ontology_typed_answer_projection_protect_supported_reader_answer=(
            query_plan_ontology_typed_answer_projection_protect_supported_reader_answer
        ),
        query_plan_ontology_enable_span_constrained_answer_resolver=(
            query_plan_ontology_enable_span_constrained_answer_resolver
        ),
        query_plan_ontology_span_resolver_min_score=args.query_plan_ontology_span_resolver_min_score,
        query_plan_ontology_span_resolver_min_margin=args.query_plan_ontology_span_resolver_min_margin,
        query_plan_ontology_span_resolver_allowed_slots=args.query_plan_ontology_span_resolver_allowed_slots,
        query_plan_ontology_span_resolver_allowed_operators=args.query_plan_ontology_span_resolver_allowed_operators,
        query_plan_ontology_span_resolver_max_doc_spans=args.query_plan_ontology_span_resolver_max_doc_spans,
        query_plan_ontology_span_resolver_replace_unsupported_only=(
            query_plan_ontology_span_resolver_replace_unsupported_only
        ),
        query_plan_ontology_span_resolver_protect_supported_reader_answer=(
            query_plan_ontology_span_resolver_protect_supported_reader_answer
        ),
        query_plan_ontology_enable_quality_gate=query_plan_ontology_enable_quality_gate,
        query_plan_ontology_quality_gate_min_plan_precision=args.query_plan_ontology_quality_gate_min_plan_precision,
        query_plan_ontology_quality_gate_min_plan_scale=args.query_plan_ontology_quality_gate_min_plan_scale,
        query_plan_ontology_quality_gate_max_matched_plans_for_qa=args.query_plan_ontology_quality_gate_max_matched_plans_for_qa,
        query_plan_ontology_quality_gate_max_executable_paths_for_qa=args.query_plan_ontology_quality_gate_max_executable_paths_for_qa,
        query_plan_ontology_quality_gate_max_avg_paths_per_plan_for_qa=args.query_plan_ontology_quality_gate_max_avg_paths_per_plan_for_qa,
        query_plan_ontology_quality_gate_allow_simple_validator=query_plan_ontology_quality_gate_allow_simple_validator,
        query_plan_ontology_enable_qa_format_hint=query_plan_ontology_enable_qa_format_hint,
        query_plan_ontology_qa_format_hint_mode=args.query_plan_ontology_qa_format_hint_mode,
        query_plan_ontology_enable_answer_path_verifier=query_plan_ontology_enable_answer_path_verifier,
        query_plan_ontology_answer_path_min_score=args.query_plan_ontology_answer_path_min_score,
        query_plan_ontology_answer_path_score_weight=args.query_plan_ontology_answer_path_score_weight,
        query_plan_ontology_answer_path_filter_lift=query_plan_ontology_answer_path_filter_lift,
        query_plan_ontology_answer_path_filter_qa_evidence=query_plan_ontology_answer_path_filter_qa_evidence,
        query_plan_ontology_enable_answer_object_gate=query_plan_ontology_enable_answer_object_gate,
        query_plan_ontology_enable_query_conditioning=query_plan_ontology_enable_query_conditioning,
        query_plan_ontology_condition_min_score=args.query_plan_ontology_condition_min_score,
        query_plan_ontology_condition_require_relation_overlap=query_plan_ontology_condition_require_relation_overlap,
        query_plan_ontology_enable_simple_fallback=query_plan_ontology_enable_simple_fallback,
        query_plan_ontology_simple_fallback_only_when_unmatched=query_plan_ontology_simple_fallback_only_when_unmatched,
        query_plan_ontology_simple_fallback_precision=args.query_plan_ontology_simple_fallback_precision,
        query_plan_ontology_simple_fallback_weight_scale=args.query_plan_ontology_simple_fallback_weight_scale,
        query_plan_ontology_enable_plan_repair=query_plan_ontology_enable_plan_repair,
        query_plan_ontology_plan_repair_max_repairs=args.query_plan_ontology_plan_repair_max_repairs,
        query_plan_ontology_plan_repair_min_query_token_overlap=args.query_plan_ontology_plan_repair_min_query_token_overlap,
        query_plan_ontology_plan_repair_min_plan_precision=args.query_plan_ontology_plan_repair_min_plan_precision,
        query_plan_ontology_plan_repair_score_floor=args.query_plan_ontology_plan_repair_score_floor,
        query_plan_ontology_plan_repair_enable_query_condition=query_plan_ontology_plan_repair_enable_query_condition,
        query_plan_ontology_plan_repair_enable_relation_realization=query_plan_ontology_plan_repair_enable_relation_realization,
        query_plan_ontology_anchor_mode=args.query_plan_ontology_anchor_mode,
        query_plan_ontology_anchor_require_concrete=query_plan_ontology_anchor_require_concrete,
        query_plan_ontology_anchor_prefer_longest_exact=query_plan_ontology_anchor_prefer_longest_exact,
        enable_query_plan_ontology_ppr_injection=enable_query_plan_ontology_ppr_injection,
        query_plan_ontology_ppr_enable_query_router=query_plan_ontology_ppr_enable_query_router,
        query_plan_ontology_ppr_enable_path_quality_gate=query_plan_ontology_ppr_enable_path_quality_gate,
        query_plan_ontology_ppr_max_executable_paths_for_quality_gate=args.query_plan_ontology_ppr_max_executable_paths_for_quality_gate,
        query_plan_ontology_ppr_max_unique_terminals_for_quality_gate=args.query_plan_ontology_ppr_max_unique_terminals_for_quality_gate,
        query_plan_ontology_ppr_min_plan_precision=args.query_plan_ontology_ppr_min_plan_precision,
        query_plan_ontology_ppr_require_anchor_fact=query_plan_ontology_ppr_require_anchor_fact,
        query_plan_ontology_ppr_source_top_k=args.query_plan_ontology_ppr_source_top_k,
        query_plan_ontology_ppr_top_k=args.query_plan_ontology_ppr_top_k,
        query_plan_ontology_ppr_score_weight=args.query_plan_ontology_ppr_score_weight,
        query_plan_ontology_ppr_max_paths=args.query_plan_ontology_ppr_max_paths,
        query_plan_ontology_ppr_relation_match_mode=args.query_plan_ontology_ppr_relation_match_mode,
        query_plan_ontology_enable_ppr_passage_seeds=query_plan_ontology_enable_ppr_passage_seeds,
        query_plan_ontology_ppr_passage_seed_weight=args.query_plan_ontology_ppr_passage_seed_weight,
        query_plan_ontology_ppr_passage_seed_top_k=args.query_plan_ontology_ppr_passage_seed_top_k,
        query_plan_ontology_ppr_passage_seed_min_plan_precision=args.query_plan_ontology_ppr_passage_seed_min_plan_precision,
        query_plan_ontology_ppr_passage_seed_require_query_anchor=query_plan_ontology_ppr_passage_seed_require_query_anchor,
        query_plan_ontology_ppr_passage_seed_dense_bonus=args.query_plan_ontology_ppr_passage_seed_dense_bonus,
        enable_query_plan_ontology_graph_scorer=enable_query_plan_ontology_graph_scorer,
        query_plan_ontology_graph_scorer_score_weight=args.query_plan_ontology_graph_scorer_score_weight,
        query_plan_ontology_graph_scorer_top_k=args.query_plan_ontology_graph_scorer_top_k,
        query_plan_ontology_graph_scorer_min_confidence=args.query_plan_ontology_graph_scorer_min_confidence,
        query_plan_ontology_graph_scorer_require_anchor_or_dense_source=query_plan_ontology_graph_scorer_require_anchor_or_dense_source,
        query_plan_ontology_graph_scorer_boost_existing=query_plan_ontology_graph_scorer_boost_existing,
        query_plan_ontology_graph_scorer_reorder_facts=query_plan_ontology_graph_scorer_reorder_facts,
        query_plan_ontology_enable_structural_candidate_retriever=query_plan_ontology_enable_structural_candidate_retriever,
        query_plan_ontology_structural_candidate_retriever_scan_top_k=args.query_plan_ontology_structural_candidate_retriever_scan_top_k,
        query_plan_ontology_structural_candidate_retriever_max_original_rank=args.query_plan_ontology_structural_candidate_retriever_max_original_rank,
        query_plan_ontology_structural_candidate_retriever_max_added_candidates=args.query_plan_ontology_structural_candidate_retriever_max_added_candidates,
        query_plan_ontology_structural_candidate_retriever_min_score=args.query_plan_ontology_structural_candidate_retriever_min_score,
        query_plan_ontology_structural_candidate_retriever_require_anchor=query_plan_ontology_structural_candidate_retriever_require_anchor,
        query_plan_ontology_structural_candidate_retriever_require_terminal_or_relation=query_plan_ontology_structural_candidate_retriever_require_terminal_or_relation,
        query_plan_ontology_structural_candidate_retriever_min_relation_token_overlap=args.query_plan_ontology_structural_candidate_retriever_min_relation_token_overlap,
        query_plan_ontology_structural_candidate_retriever_reject_generic_anchors=query_plan_ontology_structural_candidate_retriever_reject_generic_anchors,
        query_plan_ontology_structural_candidate_retriever_score_weight=args.query_plan_ontology_structural_candidate_retriever_score_weight,
        query_plan_ontology_structural_candidate_retriever_require_existing_source=query_plan_ontology_structural_candidate_retriever_require_existing_source,
        query_plan_ontology_enable_candidate_retriever=query_plan_ontology_enable_candidate_retriever,
        query_plan_ontology_candidate_retriever_top_k=args.query_plan_ontology_candidate_retriever_top_k,
        query_plan_ontology_candidate_retriever_promote_to_rank=args.query_plan_ontology_candidate_retriever_promote_to_rank,
        query_plan_ontology_candidate_retriever_preserve_top_k=args.query_plan_ontology_candidate_retriever_preserve_top_k,
        query_plan_ontology_candidate_retriever_max_original_rank=args.query_plan_ontology_candidate_retriever_max_original_rank,
        query_plan_ontology_candidate_retriever_min_confidence=args.query_plan_ontology_candidate_retriever_min_confidence,
        query_plan_ontology_candidate_retriever_score_margin=args.query_plan_ontology_candidate_retriever_score_margin,
        query_plan_ontology_candidate_retriever_aggressive_promote_to_rank=args.query_plan_ontology_candidate_retriever_aggressive_promote_to_rank,
        query_plan_ontology_candidate_retriever_aggressive_min_confidence=args.query_plan_ontology_candidate_retriever_aggressive_min_confidence,
        query_plan_ontology_candidate_retriever_aggressive_require_strict=query_plan_ontology_candidate_retriever_aggressive_require_strict,
        query_plan_ontology_candidate_retriever_aggressive_require_multihop_or_multi_context=query_plan_ontology_candidate_retriever_aggressive_require_multihop_or_multi_context,
        query_plan_ontology_candidate_retriever_enable_anchor_guard=query_plan_ontology_candidate_retriever_enable_anchor_guard,
        query_plan_ontology_candidate_retriever_allow_structural_only=query_plan_ontology_candidate_retriever_allow_structural_only,
        query_plan_ontology_candidate_retriever_structural_only_min_confidence=args.query_plan_ontology_candidate_retriever_structural_only_min_confidence,
        query_plan_ontology_enable_terminal_evidence_retriever=query_plan_ontology_enable_terminal_evidence_retriever,
        query_plan_ontology_terminal_evidence_dense_top_k=args.query_plan_ontology_terminal_evidence_dense_top_k,
        query_plan_ontology_terminal_evidence_max_query_variants=args.query_plan_ontology_terminal_evidence_max_query_variants,
        query_plan_ontology_terminal_evidence_max_selected=args.query_plan_ontology_terminal_evidence_max_selected,
        query_plan_ontology_terminal_evidence_promote_to_rank=args.query_plan_ontology_terminal_evidence_promote_to_rank,
        query_plan_ontology_terminal_evidence_preserve_top_k=args.query_plan_ontology_terminal_evidence_preserve_top_k,
        query_plan_ontology_terminal_evidence_min_confidence=args.query_plan_ontology_terminal_evidence_min_confidence,
        query_plan_ontology_terminal_evidence_require_relation_hint=query_plan_ontology_terminal_evidence_require_relation_hint,
        query_plan_ontology_terminal_evidence_require_entity_hint=query_plan_ontology_terminal_evidence_require_entity_hint,
        query_plan_ontology_terminal_evidence_max_entity_terms=args.query_plan_ontology_terminal_evidence_max_entity_terms,
        query_plan_ontology_terminal_evidence_score_margin=args.query_plan_ontology_terminal_evidence_score_margin,
        query_plan_ontology_terminal_evidence_candidate_only=query_plan_ontology_terminal_evidence_candidate_only,
        query_plan_ontology_enable_terminal_evidence_verifier=query_plan_ontology_enable_terminal_evidence_verifier,
        query_plan_ontology_terminal_evidence_verifier_min_confidence=args.query_plan_ontology_terminal_evidence_verifier_min_confidence,
        query_plan_ontology_terminal_evidence_verifier_min_relation_hits=args.query_plan_ontology_terminal_evidence_verifier_min_relation_hits,
        query_plan_ontology_terminal_evidence_verifier_max_original_rank=args.query_plan_ontology_terminal_evidence_verifier_max_original_rank,
        query_plan_ontology_terminal_evidence_verifier_max_matched_plans=args.query_plan_ontology_terminal_evidence_verifier_max_matched_plans,
        query_plan_ontology_terminal_evidence_verifier_max_executable_paths=args.query_plan_ontology_terminal_evidence_verifier_max_executable_paths,
        query_plan_ontology_terminal_evidence_verifier_require_new_coverage=query_plan_ontology_terminal_evidence_verifier_require_new_coverage,
        query_plan_ontology_terminal_evidence_verifier_promote_to_rank=args.query_plan_ontology_terminal_evidence_verifier_promote_to_rank,
        query_plan_ontology_terminal_evidence_verifier_preserve_top_k=args.query_plan_ontology_terminal_evidence_verifier_preserve_top_k,
        query_plan_ontology_terminal_evidence_verifier_max_inserted=args.query_plan_ontology_terminal_evidence_verifier_max_inserted,
        query_plan_ontology_terminal_evidence_verifier_score_margin=args.query_plan_ontology_terminal_evidence_verifier_score_margin,
        query_plan_ontology_trace_path=args.query_plan_ontology_trace_path,
        query_plan_ontology_trace_compact_physical_shadow=(
            query_plan_ontology_trace_compact_physical_shadow
        ),
        query_plan_ontology_trace_compact_end_to_end_slate=(
            query_plan_ontology_trace_compact_end_to_end_slate
        ),
        ontology_include_concepts=ontology_include_concepts,
        ontology_include_related_terms=ontology_include_related_terms,
        ontology_min_concept_score=args.ontology_min_concept_score,
        ontology_require_entity_match=ontology_require_entity_match,
        ontology_max_entities_per_query=args.ontology_max_entities_per_query,
        ontology_max_relations_per_query=args.ontology_max_relations_per_query,
        ontology_max_entity_terms=args.ontology_max_entity_terms,
        ontology_max_relation_schemas=args.ontology_max_relation_schemas
    )

    logging.basicConfig(level=logging.INFO)

    compassrag = CompassRAG(global_config=config)

    read_only_snapshot_contract = None
    if reuse_index_read_only:
        read_only_snapshot_contract = validate_read_only_retrieval_snapshot(compassrag, docs)
    else:
        compassrag.index(docs)


    if skip_qa:
        retrieval_result = compassrag.retrieve(queries=all_queries, gold_docs=gold_docs)
        if isinstance(retrieval_result, tuple) and len(retrieval_result) >= 2:
            _, overall_retrieval_result = retrieval_result[:2]
            if overall_retrieval_result is not None:
                print(f"Evaluation results for retrieval: {overall_retrieval_result}", flush=True)
        if read_only_snapshot_contract is not None:
            assert_read_only_retrieval_snapshot_unchanged(
                compassrag, read_only_snapshot_contract
            )
        return

    rag_result = compassrag.rag_qa(queries=all_queries, gold_docs=gold_docs, gold_answers=gold_answers)
    if read_only_snapshot_contract is not None:
        assert_read_only_retrieval_snapshot_unchanged(
            compassrag, read_only_snapshot_contract
        )
    if isinstance(rag_result, tuple) and len(rag_result) >= 5:
        query_solutions, all_response_message, all_metadata, overall_retrieval_result, overall_qa_results = rag_result[:5]
        if args.qa_output_path:
            output_dir = os.path.dirname(args.qa_output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(args.qa_output_path, "w", encoding="utf-8") as output_file:
                for idx, query_solution in enumerate(query_solutions):
                    payload = query_solution.to_dict()
                    payload["query_idx"] = idx
                    payload["raw_response"] = all_response_message[idx] if idx < len(all_response_message) else None
                    payload["metadata"] = all_metadata[idx] if idx < len(all_metadata) else None
                    output_file.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
            print(f"Wrote per-query QA results: {args.qa_output_path}", flush=True)
        if overall_retrieval_result is not None:
            print(f"Evaluation results for retrieval: {overall_retrieval_result}", flush=True)
        if overall_qa_results is not None:
            print(f"Evaluation results for QA: {overall_qa_results}", flush=True)

if __name__ == "__main__":
    main()
