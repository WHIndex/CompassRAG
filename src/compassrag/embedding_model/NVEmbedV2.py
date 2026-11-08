from copy import deepcopy
import os
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel

from ..utils.config_utils import BaseConfig
from ..utils.logging_utils import get_logger
from .base import BaseEmbeddingModel, EmbeddingConfig, make_cache_embed

logger = get_logger(__name__)


class NVEmbedV2EmbeddingModel(BaseEmbeddingModel):

    def __init__(self, global_config: Optional[BaseConfig] = None, embedding_model_name: Optional[str] = None) -> None:
        super().__init__(global_config=global_config)

        if embedding_model_name is not None:
            self.embedding_model_name = embedding_model_name
            logger.debug(f"Overriding {self.__class__.__name__}'s embedding_model_name with: {self.embedding_model_name}")

        self._init_embedding_config()


        logger.debug(f"Initializing {self.__class__.__name__}'s embedding model with params: {self.embedding_config.model_init_params}")

        self.embedding_model = AutoModel.from_pretrained(**self.embedding_config.model_init_params)
        self.embedding_dim = self.embedding_model.config.hidden_size

    def _init_embedding_config(self) -> None:
        """
        Extract embedding model-specific parameters to init the EmbeddingConfig.

        Returns:
            None
        """

        device_map = os.environ.get("COMPASSRAG_EMBEDDING_DEVICE_MAP", "auto").strip() or "auto"
        config_dict = {
            "embedding_model_name": self.embedding_model_name,
            "norm": self.global_config.embedding_return_as_normalized,

            "model_init_params": {

                "pretrained_model_name_or_path": self.embedding_model_name,
                "trust_remote_code": True,
                "device_map": device_map,
                "torch_dtype": self.global_config.embedding_model_dtype,

            },
            "encode_params": {
                "max_length": self.global_config.embedding_max_seq_len,
                "instruction": "",
                "batch_size": self.global_config.embedding_batch_size,
                "num_workers": 32
            },
        }

        self.embedding_config = EmbeddingConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s embedding_config: {self.embedding_config}")





    def batch_encode(self, texts: List[str], **kwargs) -> None:
        if isinstance(texts, str): texts = [texts]

        params = deepcopy(self.embedding_config.encode_params)
        if kwargs: params.update(kwargs)

        if "instruction" in kwargs:
            if kwargs["instruction"] != '':
                params["instruction"] = f"Instruct: {kwargs['instruction']}\nQuery: "


        batch_size = params.pop("batch_size", 16)

        logger.debug(f"Calling {self.__class__.__name__} with:\n{params}")
        if len(texts) <= batch_size:
            params["prompts"] = texts
            results = self.embedding_model.encode(**params)
        else:
            pbar = tqdm(total=len(texts), desc="Batch Encoding")
            results = []
            for i in range(0, len(texts), batch_size):
                params["prompts"] = texts[i:i + batch_size]
                results.append(self.embedding_model.encode(**params))
                pbar.update(batch_size)
            pbar.close()
            results = torch.cat(results, dim=0)

        if isinstance(results, torch.Tensor):
            results = results.cpu()
            results = results.numpy()
        if self.embedding_config.norm:
            results = (results.T / np.linalg.norm(results, axis=1)).T

        return results
