import os

from ..utils.logging_utils import get_logger
from ..utils.config_utils import BaseConfig

from .remote_client import CachedRemoteClient
from .base import BaseLLM
from .bedrock_llm import BedrockLLM
from .transformers_llm import TransformersLLM


logger = get_logger(__name__)


def _get_llm_class(config: BaseConfig):
    if config.llm_base_url is not None and 'localhost' in config.llm_base_url and os.getenv('MODEL_API_KEY') is None:
        os.environ['MODEL_API_KEY'] = 'sk-'

    if config.llm_name.startswith('bedrock'):
        return BedrockLLM(config)

    if config.llm_name.startswith('Transformers/'):
        return TransformersLLM(config)

    return CachedRemoteClient.from_experiment_config(config)
