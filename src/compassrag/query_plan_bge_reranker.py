"""Offline/local BGE cross-encoder scoring for evidence candidates.

The scorer deliberately has no knowledge of gold evidence or dataset names.  It
only maps a (query, passage) pair to a semantic relevance score.  Loading is
lazy so existing retrieval runs that do not opt into BGE incur no dependency,
GPU, or memory cost.
"""

from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from typing import List, Sequence, Tuple


class LocalBGEReranker:
    """Score query/passage pairs with a locally cached sequence classifier."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "auto",
        batch_size: int = 16,
        max_length: int = 512,
        cache_size: int = 8192,
    ) -> None:
        self.model_path = str(model_path or "")
        self.device_name = str(device or "auto")
        self.batch_size = max(int(batch_size), 1)
        self.max_length = min(max(int(max_length), 32), 512)
        self.cache_size = max(int(cache_size), 0)
        self.tokenizer = None
        self.model = None
        self.device = None
        self._cache: OrderedDict[str, float] = OrderedDict()

    def _load(self) -> None:
        if self.model is not None:
            return
        if not self.model_path:
            raise ValueError("BGE reranker model_path is empty")
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if self.device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_name = self.device_name
        self.device = torch.device(device_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            local_files_only=True,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
        )
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _key(query: str, passage: str) -> str:
        payload = f"{query}\0{passage}".encode("utf-8", errors="ignore")
        return hashlib.sha1(payload).hexdigest()

    def _remember(self, key: str, value: float) -> None:
        if self.cache_size <= 0:
            return
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def score_pairs(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        """Return sigmoid-normalized relevance scores in input order."""
        if not pairs:
            return []
        outputs: List[float | None] = [None] * len(pairs)
        missing: List[Tuple[int, str, str, str]] = []
        for index, (query, passage) in enumerate(pairs):
            key = self._key(str(query), str(passage))
            cached = self._cache.get(key)
            if cached is None:
                missing.append((index, str(query), str(passage), key))
            else:
                self._cache.move_to_end(key)
                outputs[index] = cached

        if missing:
            self._load()
            import torch

            for offset in range(0, len(missing), self.batch_size):
                batch = missing[offset:offset + self.batch_size]
                queries = [item[1] for item in batch]
                passages = [item[2] for item in batch]
                encoded = self.tokenizer(
                    queries,
                    passages,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                with torch.inference_mode():
                    logits = self.model(**encoded).logits
                if logits.ndim == 2 and logits.shape[1] > 1:
                    values = torch.softmax(logits.float(), dim=-1)[:, -1]
                else:
                    values = torch.sigmoid(logits.float().reshape(-1))
                for item, value in zip(batch, values.detach().cpu().tolist()):
                    score = float(value) if math.isfinite(float(value)) else 0.0
                    outputs[item[0]] = score
                    self._remember(item[3], score)

        return [float(value or 0.0) for value in outputs]

    def score(self, query: str, passages: Sequence[str]) -> List[float]:
        return self.score_pairs([(query, passage) for passage in passages])
