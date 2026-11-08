from argparse import ArgumentTypeError
from dataclasses import dataclass
from hashlib import md5
from typing import Dict, Any, List, Tuple, Literal, Union, Optional
import numpy as np
import re
import logging

from .typing import Triple
from .llm_utils import filter_invalid_triples

logger = logging.getLogger(__name__)

@dataclass
class NerRawOutput:
    chunk_id: str
    response: str
    unique_entities: List[str]
    metadata: Dict[str, Any]


@dataclass
class TripleRawOutput:
    chunk_id: str
    response: str
    triples: List[List[str]]
    metadata: Dict[str, Any]

@dataclass
class LinkingOutput:
    score: np.ndarray
    type: Literal['node', 'dpr']

@dataclass
class QuerySolution:
    question: str
    docs: List[str]
    doc_scores: np.ndarray = None
    ontology_evidence: Optional[List[str]] = None
    context_docs: Optional[List[str]] = None
    context_doc_scores: np.ndarray = None
    context_builder: Optional[Dict[str, Any]] = None
    answer: str = None
    gold_answers: List[str] = None
    gold_docs: Optional[List[str]] = None


    def to_dict(self):
        context_docs = self.context_docs if self.context_docs is not None else None
        context_scores = self.context_doc_scores if self.context_doc_scores is not None else None
        return {
            "question": self.question,
            "answer": self.answer,
            "gold_answers": self.gold_answers,
            "docs": self.docs[:5],
            "doc_scores": [round(v, 4) for v in self.doc_scores.tolist()[:5]]  if self.doc_scores is not None else None,
            "context_docs": context_docs[:5] if context_docs else None,
            "context_doc_scores": [round(v, 4) for v in context_scores.tolist()[:5]] if context_scores is not None else None,
            "context_builder": self.context_builder,
            "ontology_evidence": self.ontology_evidence[:8] if self.ontology_evidence else None,
            "gold_docs": self.gold_docs,
        }

def text_processing(text):
    if isinstance(text, list):
        return [text_processing(t) for t in text]
    if not isinstance(text, str):
        text = str(text)
    return re.sub('[^A-Za-z0-9 ]', ' ', text.lower()).strip()

def reformat_openie_results(corpus_openie_results) -> (Dict[str, NerRawOutput], Dict[str, TripleRawOutput]):

    ner_output_dict = {
        chunk_item['idx']: NerRawOutput(
            chunk_id=chunk_item['idx'],
            response=None,
            metadata={},
            unique_entities=list(np.unique(chunk_item['extracted_entities']))
        )
        for chunk_item in corpus_openie_results
    }
    triple_output_dict = {
        chunk_item['idx']: TripleRawOutput(
            chunk_id=chunk_item['idx'],
            response=None,
            metadata={},
            triples=filter_invalid_triples(triples=chunk_item['extracted_triples'])
        )
        for chunk_item in corpus_openie_results
    }

    return ner_output_dict, triple_output_dict

def extract_entity_nodes(chunk_triples: List[List[Triple]]) -> (List[str], List[List[str]]):
    chunk_triple_entities = []
    for triples in chunk_triples:
        triple_entities = set()
        for t in triples:
            if len(t) == 3:
                triple_entities.update([t[0], t[2]])
            else:
                logger.warning(f"During graph construction, invalid triple is found: {t}")
        chunk_triple_entities.append(list(triple_entities))
    graph_nodes = list(np.unique([ent for ents in chunk_triple_entities for ent in ents]))
    return graph_nodes, chunk_triple_entities

def flatten_facts(chunk_triples: List[Triple]) -> List[Triple]:
    graph_triples = []
    for triples in chunk_triples:
        graph_triples.extend([tuple(t) for t in triples])
    graph_triples = list(set(graph_triples))
    return graph_triples

def min_max_normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    range_val = max_val - min_val


    if range_val == 0:
        return np.ones_like(x)

    return (x - min_val) / range_val

def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """
    Compute the MD5 hash of the given content string and optionally prepend a prefix.

    Args:
        content (str): The input string to be hashed.
        prefix (str, optional): A string to prepend to the resulting hash. Defaults to an empty string.

    Returns:
        str: A string consisting of the prefix followed by the hexadecimal representation of the MD5 hash.
    """
    return prefix + md5(content.encode()).hexdigest()


def all_values_of_same_length(data: dict) -> bool:
    """
    Return True if all values in 'data' have the same length or data is an empty dict,
    otherwise return False.
    """

    value_iter = iter(data.values())


    try:
        first_length = len(next(value_iter))
    except StopIteration:

        return True


    return all(len(seq) == first_length for seq in value_iter)


def string_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )
