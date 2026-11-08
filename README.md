# CompassRAG

CompassRAG constructs executable evidence ontologies from historical queries and uses them to compile and optimize evidence retrieval plans for unseen queries.

## Architecture

The offline pipeline builds a provenance-aware evidence graph, derives positive and hard-negative evidence chains, constructs competency queries, and consolidates reusable evidence patterns into a validated ontology. Historical plan executions provide utility, operator-work cost, and quality-risk labels for grouped out-of-fold training of the physical-plan optimizer.

At query time, CompassRAG extracts a structural signature, matches compatible ontology patterns, compiles a logical evidence plan, enumerates a bounded portfolio of complete physical plans, and selects one under cost and risk constraints. The selected plan produces the ranked evidence passed to the reader.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install local inference dependencies when required:

```bash
pip install -e '.[local]'
```

## Offline Pipeline

The public entry points correspond to the four offline stages:

```bash
python scripts/prepare_historical_workload.py --help
python scripts/build_competency_queries.py --help
python scripts/construct_evidence_ontology.py --help
python scripts/validate_ontology.py --help
python scripts/freeze_ontology.py --help
python scripts/train_plan_optimizer.py --help
```

`MODEL_API_KEY`, `MODEL_BASE_URL`, and `MODEL_NAME` configure a remote chat-completion endpoint. Generated ontologies, execution traces, indexes, and model artifacts are intentionally excluded from version control.

## Online Retrieval and QA

```bash
compassrag \
  --dataset 2wikimultihopqa \
  --query_path data/queries.json \
  --corpus_path data/corpus.json \
  --save_dir outputs/compassrag
```

Use `compassrag --help` for index, ontology, operator, optimizer, and reader options.

## Evaluation

```bash
python scripts/evaluate_logical_plans.py --help
python scripts/evaluate_plan_selection.py --help
pytest
```

Evaluation data and generated result tables are not stored in this repository.
