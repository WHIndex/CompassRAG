from compassrag.query_plan_bind_probe_join import QueryPlanOntologyBindProbeJoin


def _query_plan():
    return {
        "enabled": True,
        "anchor_entities": ["Hamlet"],
        "query_condition": {
            "answer_type": "location",
            "answer_hints": ["place", "location"],
            "relation_terms": ["born in"],
        },
        "matched_plans": [{
            "id": "author_birth_place",
            "path": [
                {"relation": "written by"},
                {"relation": "born in"},
            ],
        }],
    }


def test_bind_probe_executes_grounded_second_hop_join():
    records = {
        0: {
            "passage": "Hamlet was written by William Shakespeare.",
            "extracted_triples": [["Hamlet", "written by", "William Shakespeare"]],
        },
        1: {
            "passage": "William Shakespeare was born in Stratford-upon-Avon.",
            "extracted_triples": [["William Shakespeare", "was born in", "Stratford-upon-Avon"]],
        },
        2: {
            "passage": "Christopher Marlowe was an English playwright.",
            "extracted_triples": [["Christopher Marlowe", "was", "English playwright"]],
        },
    }

    retriever = QueryPlanOntologyBindProbeJoin(
        anchor_top_k=1,
        max_bindings=2,
        max_probes=2,
        dense_top_k=3,
        max_candidates=2,
        min_binding_score=0.50,
        min_candidate_score=0.60,
        min_relation_overlap=0.20,
    )
    result = retriever.run(
        query="Where was the author of Hamlet born?",
        base_doc_ids=[0, 2],
        query_plan_log=_query_plan(),
        dense_retrieve=lambda _: ([1, 2, 0], [1.0, 0.5, 0.2]),
        get_passage_text=lambda idx: records[idx]["passage"],
        get_openie_record=lambda idx: records[idx],
    )

    assert result.log["binding_count"] == 1
    assert result.log["bindings"][0]["binding_entity"] == "William Shakespeare"
    assert result.log["probes"][0]["second_hop_relation"] == "born in"
    assert [item["passage_idx"] for item in result.added_candidates] == [1]
    context = result.added_candidates[0]["contexts"][0]
    assert context["ontology_bind_probe_join"] is True
    assert context["binding_source_passage_idx"] == 0
    assert context["binding_source_triple"] == ["Hamlet", "written by", "William Shakespeare"]
    assert context["bridge_entity"] == "William Shakespeare"
    assert context["second_hop_openie_hit"] is True
    assert context["second_hop_relation_overlap"] >= 0.20


def test_bind_probe_rejects_unresolved_relation_miss():
    records = {
        0: {
            "passage": "Hamlet was written by William Shakespeare.",
            "extracted_triples": [["Hamlet", "written by", "William Shakespeare"]],
        },
        1: {
            "passage": "William Shakespeare acted in a school play.",
            "extracted_triples": [["William Shakespeare", "acted in", "school play"]],
        },
    }
    retriever = QueryPlanOntologyBindProbeJoin(
        anchor_top_k=1,
        max_bindings=1,
        max_probes=1,
        dense_top_k=1,
        max_candidates=1,
        min_binding_score=0.50,
        min_candidate_score=0.60,
        min_relation_overlap=0.40,
        require_relation_or_answer=True,
    )
    result = retriever.run(
        query="Where was the author of Hamlet born?",
        base_doc_ids=[0],
        query_plan_log=_query_plan(),
        dense_retrieve=lambda _: ([1], [1.0]),
        get_passage_text=lambda idx: records[idx]["passage"],
        get_openie_record=lambda idx: records[idx],
    )

    assert result.added_candidates == []
    assert result.log["rejection_reasons"]["second_hop_relation_answer_miss"] == 1


def test_role_constrained_bind_probe_selects_death_not_birth_date():
    records = {
        0: {
            "passage": "Madame La Presidente was directed by Frank Lloyd.",
            "extracted_triples": [["Madame La Presidente", "directed by", "Frank Lloyd"]],
        },
        1: {
            "passage": "Frank Lloyd was born on 2 February 1886 and died on 10 August 1960.",
            "extracted_triples": [
                ["Frank Lloyd", "was born on", "2 February 1886"],
                ["Frank Lloyd", "died on", "10 August 1960"],
            ],
        },
    }
    plan = {
        "enabled": True,
        "anchor_entities": ["Madame La Presidente"],
        "query_condition": {
            "answer_type": "date",
            "answer_hints": ["date", "year"],


            "relation_terms": ["director"],
        },
        "matched_plans": [{"id": "director", "relation_signatures": ["director"]}],
    }
    retriever = QueryPlanOntologyBindProbeJoin(
        anchor_top_k=1,
        max_bindings=1,
        max_probes=1,
        dense_top_k=1,
        max_candidates=1,
        min_binding_score=0.50,
        min_candidate_score=0.60,
        enable_relation_role_constraints=True,
        require_role_consistent_candidate=True,
        min_terminal_relation_overlap=0.40,
        min_terminal_type_score=0.60,
    )

    result = retriever.run(
        query="What is the date of death of the director of film Madame La Presidente?",
        base_doc_ids=[0],
        query_plan_log=plan,
        dense_retrieve=lambda _: ([1], [1.0]),
        get_passage_text=lambda idx: records[idx]["passage"],
        get_openie_record=lambda idx: records[idx],
    )

    assert result.log["relation_role_resolved"] is True
    assert result.log["terminal_relation_profile"]["key"] == "death_date"
    assert result.log["plan"]["first_hop_relations"] == ["director"]
    assert result.log["probes"][0]["second_hop_relation"] == "died"
    assert [item["passage_idx"] for item in result.added_candidates] == [1]
    context = result.added_candidates[0]["contexts"][0]
    assert context["bind_probe_role_consistent"] is True
    assert context["terminal_triple"] == ["Frank Lloyd", "died on", "10 August 1960"]
    assert context["end_entity"] == "10 August 1960"


def test_role_constrained_bind_probe_rejects_wrong_typed_terminal_fact():
    records = {
        0: {
            "passage": "Lothair II was the son of Ermengarde of Tours.",
            "extracted_triples": [["Lothair II", "mother", "Ermengarde of Tours"]],
        },
        1: {
            "passage": "Ermengarde of Tours was born in Alsace.",
            "extracted_triples": [["Ermengarde of Tours", "was born in", "Alsace"]],
        },
    }
    plan = {
        "enabled": True,
        "anchor_entities": ["Lothair II"],
        "query_condition": {
            "answer_type": "date",
            "answer_hints": ["date", "year"],
            "relation_terms": ["mother", "family"],
        },
        "matched_plans": [],
    }
    retriever = QueryPlanOntologyBindProbeJoin(
        anchor_top_k=1,
        max_bindings=1,
        max_probes=1,
        dense_top_k=1,
        max_candidates=1,
        min_binding_score=0.50,
        min_candidate_score=0.60,
        enable_relation_role_constraints=True,
        require_role_consistent_candidate=True,
    )

    result = retriever.run(
        query="When did Lothair II's mother die?",
        base_doc_ids=[0],
        query_plan_log=plan,
        dense_retrieve=lambda _: ([1], [1.0]),
        get_passage_text=lambda idx: records[idx]["passage"],
        get_openie_record=lambda idx: records[idx],
    )

    assert result.log["terminal_relation_profile"]["key"] == "death_date"
    assert result.log["plan"]["first_hop_relations"][:1] == ["mother"]
    assert result.added_candidates == []
    assert result.log["rejection_reasons"]["terminal_relation_miss"] == 1


def test_role_constrained_bind_probe_respects_bounded_dense_access_path():
    records = {
        0: {
            "passage": "Hamlet was written by William Shakespeare.",
            "extracted_triples": [["Hamlet", "written by", "William Shakespeare"]],
        },
        1: {
            "passage": "William Shakespeare was born in Stratford-upon-Avon.",
            "extracted_triples": [["William Shakespeare", "was born in", "Stratford-upon-Avon"]],
        },
        2: {
            "passage": "An unrelated playwright was born in London.",
            "extracted_triples": [["An unrelated playwright", "was born in", "London"]],
        },
        3: {
            "passage": "Another unrelated playwright was born in Oxford.",
            "extracted_triples": [["Another unrelated playwright", "was born in", "Oxford"]],
        },
    }
    retriever = QueryPlanOntologyBindProbeJoin(
        anchor_top_k=1,
        max_bindings=1,
        max_probes=1,
        dense_top_k=3,
        max_candidates=1,
        min_binding_score=0.50,
        min_candidate_score=0.60,
        enable_relation_role_constraints=True,
        require_role_consistent_candidate=True,
        max_role_candidate_dense_rank=2,
    )

    result = retriever.run(
        query="Where was the author of Hamlet born?",
        base_doc_ids=[0],
        query_plan_log=_query_plan(),
        dense_retrieve=lambda _: ([2, 3, 1], [1.0, 0.9, 0.8]),
        get_passage_text=lambda idx: records[idx]["passage"],
        get_openie_record=lambda idx: records[idx],
    )

    assert result.added_candidates == []
    assert result.log["rejection_reasons"]["outside_role_dense_access_path"] == 1
