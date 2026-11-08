import json
import difflib
import logging
from pydantic import BaseModel, Field
from copy import deepcopy
from typing import Union, Optional, List, Dict, Any, Tuple, Literal
import re
import ast
from .prompts.filter_default_prompt import best_dspy_prompt

class Fact(BaseModel):
    fact: list[list[str]] = Field(description="A list of facts, each fact is a list of 3 strings: [subject, predicate, object]")


logger = logging.getLogger(__name__)


class DSPyFilter:
    TRIPLE_PATTERN = re.compile(
        r'\[\s*'
        r'(?P<s>"(?:\\.|[^"\\])*")\s*,\s*'
        r'(?P<p>"(?:\\.|[^"\\])*")\s*,\s*'
        r'(?P<o>"(?:\\.|[^"\\])*")\s*'
        r'\]',
        re.DOTALL,
    )

    def __init__(self, compassrag):
        """
        Initializes the object with the necessary configurations and templates for processing input and output messages.

        Parameters:
        compassrag : An object that provides the global configuration and the LLM model required for inference.

        Attributes:
        dspy_file_path : The file path for reranking as specified in the global configuration.
        one_input_template : A string template for formatting the input message with placeholders for specific fields.
        one_output_template : A string template for formatting the output message with specific fields.
        message_template : A template generated using the specified dspy file path.
        llm_infer_fn : A function reference for making inferences using the provided LLM model.
        model_name : The name of the language model as specified in the global configuration.
        default_gen_kwargs : A dictionary for storing the default generation keyword arguments.
        """
        dspy_file_path = compassrag.global_config.rerank_dspy_file_path
        self.one_input_template = """[[ ## question ## ]]\n{question}\n\n[[ ## fact_before_filter ## ]]\n{fact_before_filter}\n\nRespond with the corresponding output fields, starting with the field `[[ ## fact_after_filter ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.\n\nThe value of `fact_after_filter` must be one JSON object only, with this exact shape: {{\"fact\": [[\"subject\", \"predicate\", \"object\"]]}}.\nDo not wrap the JSON in markdown. Do not add explanations. Do not invent facts that are not copied from `fact_before_filter`."""
        self.one_output_template = """[[ ## fact_after_filter ## ]]\n{fact_after_filter}\n\n[[ ## completed ## ]]"""
        self.message_template = self.make_template(dspy_file_path)
        self.llm_infer_fn = compassrag.llm_model.infer
        self.model_name = compassrag.global_config.llm_name
        self.default_gen_kwargs = {}

    def make_template(self, dspy_file_path):
        if dspy_file_path is not None:
            dspy_saved = json.load(open(dspy_file_path, 'r'))
        else:
            dspy_saved = best_dspy_prompt

        system_prompt = dspy_saved['prog']['system']
        message_template = [
            {"role": "system", "content": system_prompt},
        ]
        demos = dspy_saved["prog"]["demos"]
        for demo in demos:
            message_template.append({"role": "user", "content": self.one_input_template.format(question=demo["question"], fact_before_filter=demo["fact_before_filter"])})
            message_template.append({"role": "assistant", "content": self.one_output_template.format(fact_after_filter=demo["fact_after_filter"])})
        return message_template

    @staticmethod
    def _clean_model_value(value: str) -> str:
        cleaned = value.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json|python)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        for marker in ("[[ ## completed ## ]]", "Corrected response:", "\nCorrected response:"):
            marker_idx = cleaned.find(marker)
            if marker_idx != -1:
                cleaned = cleaned[:marker_idx].strip()

        return cleaned.strip()

    @staticmethod
    def _sanitize_facts(facts: Any) -> List[List[str]]:
        sanitized = []
        if not isinstance(facts, list):
            return sanitized

        for fact in facts:
            if not isinstance(fact, (list, tuple)) or len(fact) != 3:
                continue
            subject, predicate, obj = [str(part).strip() for part in fact]
            if subject and predicate and obj:
                sanitized.append([subject, predicate, obj])
        return sanitized

    @classmethod
    def _coerce_fact_payload(cls, parsed_value: Any) -> List[List[str]]:
        if isinstance(parsed_value, Fact):
            return cls._sanitize_facts(parsed_value.fact)
        if isinstance(parsed_value, dict):
            return cls._sanitize_facts(parsed_value.get("fact", []))
        if isinstance(parsed_value, list):
            return cls._sanitize_facts(parsed_value)
        return []

    @classmethod
    def _extract_jsonish_payload(cls, value: str) -> Any:
        cleaned = cls._clean_model_value(value)
        if not cleaned:
            return None

        for candidate in (cleaned, cleaned[cleaned.find("{"):] if "{" in cleaned else cleaned):
            if not candidate:
                continue
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
            try:
                return ast.literal_eval(candidate)
            except (ValueError, SyntaxError):
                pass

            decoder = json.JSONDecoder()
            for start_char in ("{", "["):
                start = candidate.find(start_char)
                if start == -1:
                    continue
                try:
                    parsed, _ = decoder.raw_decode(candidate[start:])
                    return parsed
                except json.JSONDecodeError:
                    continue

        return None

    @classmethod
    def _extract_triples_from_text(cls, value: str) -> List[List[str]]:
        cleaned = cls._clean_model_value(value)
        triples = []
        for match in cls.TRIPLE_PATTERN.finditer(cleaned):
            try:
                triple = json.loads(
                    f'[{match.group("s")}, {match.group("p")}, {match.group("o")}]'
                )
            except json.JSONDecodeError:
                continue
            triples.extend(cls._sanitize_facts([triple]))
        return triples

    @classmethod
    def _parse_fact_after_filter_value(cls, value: str) -> List[List[str]]:
        facts, _ = cls._parse_fact_after_filter_value_with_status(value)
        return facts

    @classmethod
    def _parse_fact_after_filter_value_with_status(cls, value: str) -> Tuple[List[List[str]], bool]:
        parsed_value = cls._extract_jsonish_payload(value)
        if parsed_value is not None:
            return cls._coerce_fact_payload(parsed_value), True

        triples = cls._extract_triples_from_text(value)
        if triples:
            return triples, True

        return [], False

    def parse_filter_with_status(self, response) -> Tuple[List[List[str]], bool]:
        sections = [(None, [])]
        field_header_pattern = re.compile('\\[\\[ ## (\\w+) ## \\]\\]')
        for line in response.splitlines():
            match = field_header_pattern.match(line.strip())
            if match:
                sections.append((match.group(1), []))
            else:
                sections[-1][1].append(line)

        sections = [(k, "\n".join(v).strip()) for k, v in sections]
        parsed = []
        parse_ok = False
        found_fact_section = False
        for k, value in sections:
            if k == "fact_after_filter":
                found_fact_section = True
                parsed, parse_ok = self._parse_fact_after_filter_value_with_status(value)
                if not parse_ok and value.strip():
                    logger.warning("Could not parse fact_after_filter; falling back to empty fact list.")

        if not found_fact_section:
            parsed, parse_ok = self._parse_fact_after_filter_value_with_status(response)

        return parsed, parse_ok

    def parse_filter(self, response):
        parsed, _ = self.parse_filter_with_status(response)
        return parsed

    @staticmethod
    def _normalize_fact(fact: Any) -> Tuple[str, str, str] | None:
        if not isinstance(fact, (list, tuple)) or len(fact) != 3:
            return None
        return tuple(str(part).strip().lower() for part in fact)

    @staticmethod
    def _fact_to_string(fact: Any) -> str:
        normalized = DSPyFilter._normalize_fact(fact)
        if normalized is None:
            return ""
        return " | ".join(normalized)

    def _match_generated_fact_to_candidate(
        self,
        generated_fact: Any,
        candidate_items: List[Tuple],
        candidate_strings: List[str],
        exact_lookup: Dict[Tuple[str, str, str], int],
        min_similarity: float = 0.60,
    ) -> int | None:
        normalized = self._normalize_fact(generated_fact)
        if normalized is None:
            return None
        if normalized in exact_lookup:
            return exact_lookup[normalized]

        generated_string = self._fact_to_string(generated_fact)
        if not generated_string:
            return None

        best_idx = None
        best_score = 0.0
        for idx, candidate_string in enumerate(candidate_strings):
            score = difflib.SequenceMatcher(None, generated_string, candidate_string).ratio()
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_score < min_similarity:
            logger.debug(
                "Discarding generated fact below similarity threshold %.2f: %s",
                best_score,
                generated_fact,
            )
            return None
        return best_idx

    def llm_call(self, question, fact_before_filter):

        messages = deepcopy(self.message_template)
        messages.append({"role": "user", "content": self.one_input_template.format(question=question, fact_before_filter=fact_before_filter)})

        self.default_gen_kwargs['max_completion_tokens'] = 512

        response = self.llm_infer_fn(
            messages=messages,
            model=self.model_name,
            **self.default_gen_kwargs
        )

        if len(response) > 1:
            return response[0]
        return response

    def __call__(self, *args, **kwargs):
        return self.rerank(*args, **kwargs)

    def rerank(self,
               query: str,
               candidate_items: List[Tuple],
               candidate_indices: List[int],
               len_after_rerank: int =None) -> Tuple[List[int], List[Tuple], dict]:
        fact_before_filter = {"fact": [list(candidate_item) for candidate_item in candidate_items]}
        try:

            response = self.llm_call(query, json.dumps(fact_before_filter))
            generated_facts, parse_ok = self.parse_filter_with_status(response)
        except Exception as e:
            logger.warning("Fact rerank LLM call or parsing failed: %s", e)
            generated_facts = []
            parse_ok = False

        candidate_strings = [self._fact_to_string(candidate_item) for candidate_item in candidate_items]
        exact_lookup = {
            self._normalize_fact(candidate_item): idx
            for idx, candidate_item in enumerate(candidate_items)
            if self._normalize_fact(candidate_item) is not None
        }
        result_indices = []
        for generated_fact in generated_facts:
            matched_idx = self._match_generated_fact_to_candidate(
                generated_fact=generated_fact,
                candidate_items=candidate_items,
                candidate_strings=candidate_strings,
                exact_lookup=exact_lookup,
            )
            if matched_idx is not None and matched_idx not in result_indices:
                result_indices.append(matched_idx)

        if not result_indices and candidate_items and not parse_ok:
            logger.warning("Fact rerank parsing failed; keeping top candidate facts as fallback.")
            fallback_len = len_after_rerank or len(candidate_items)
            result_indices = list(range(min(fallback_len, len(candidate_items))))
        elif generated_facts and candidate_items and not result_indices:
            logger.warning("Fact rerank output did not match candidate facts; keeping top candidate facts as fallback.")
            fallback_len = len_after_rerank or len(candidate_items)
            result_indices = list(range(min(fallback_len, len(candidate_items))))

        sorted_candidate_indices = [candidate_indices[i] for i in result_indices]
        sorted_candidate_items = [candidate_items[i] for i in result_indices]
        return sorted_candidate_indices[:len_after_rerank], sorted_candidate_items[:len_after_rerank], {'confidence': None}
