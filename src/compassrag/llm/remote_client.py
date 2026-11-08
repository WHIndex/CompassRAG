import functools
import hashlib
import json
import os
import sqlite3
from copy import deepcopy
from typing import List, Tuple

import httpx
from filelock import FileLock
from tenacity import retry, stop_after_attempt, wait_fixed

from ..remote import RemoteEndpoint
from ..utils.config_utils import BaseConfig
from ..utils.llm_utils import (
    TextChatMessage
)
from ..utils.logging_utils import get_logger
from .base import BaseLLM, LLMConfig

logger = get_logger(__name__)

def _env_int(name: str, default: int, minimum: int = None) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None or str(raw_value).strip() == "":
        return default
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        logger.warning(f"Ignoring invalid integer env {name}={raw_value!r}; using {default}.")
        return default
    if minimum is not None:
        value = max(value, minimum)
    return value

def _env_float(name: str, default: float, minimum: float = None) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None or str(raw_value).strip() == "":
        return default
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        logger.warning(f"Ignoring invalid float env {name}={raw_value!r}; using {default}.")
        return default
    if minimum is not None:
        value = max(value, minimum)
    return value

def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None or str(raw_value).strip() == "":
        return default
    return str(raw_value).strip().lower() in {"1", "true", "yes", "y", "on"}

def cache_response(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):

        if args:
            messages = args[0]
        else:
            messages = kwargs.get("messages")
        if messages is None:
            raise ValueError("Missing required 'messages' parameter for caching.")


        gen_params = getattr(self, "llm_config", {}).generate_params if hasattr(self, "llm_config") else {}
        model = kwargs.get("model", gen_params.get("model"))
        seed = kwargs.get("seed", gen_params.get("seed"))
        temperature = kwargs.get("temperature", gen_params.get("temperature"))


        key_data = {
            "messages": messages,
            "model": model,
            "seed": seed,
            "temperature": temperature,
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()


        lock_file = self.cache_file_name + ".lock"


        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()

            c.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT,
                    metadata TEXT
                )
            """)
            conn.commit()
            c.execute("SELECT message, metadata FROM cache WHERE key = ?", (key_hash,))
            row = c.fetchone()
            conn.close()
            if row is not None:
                message, metadata_str = row
                metadata = json.loads(metadata_str)

                return message, metadata, True

        if _env_bool("COMPASSRAG_REMOTE_CACHE_ONLY", False):
            raise RuntimeError(
                "Model cache miss while COMPASSRAG_REMOTE_CACHE_ONLY is enabled: "
                f"model={model!r} key={key_hash} cache={self.cache_file_name}"
            )


        result = func(self, *args, **kwargs)
        message, metadata = result


        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()

            c.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT,
                    metadata TEXT
                )
            """)
            metadata_str = json.dumps(metadata)
            c.execute("INSERT OR REPLACE INTO cache (key, message, metadata) VALUES (?, ?, ?)",
                      (key_hash, message, metadata_str))
            conn.commit()
            conn.close()

        return message, metadata, False

    return wrapper

def dynamic_retry_decorator(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        max_retries = _env_int(
            "COMPASSRAG_REMOTE_RETRY_ATTEMPTS",
            getattr(self, "max_retries", 5),
            minimum=1,
        )
        wait_seconds = _env_float("COMPASSRAG_REMOTE_RETRY_WAIT_SECONDS", 1.0, minimum=0.0)
        dynamic_retry = retry(stop=stop_after_attempt(max_retries), wait=wait_fixed(wait_seconds))
        decorated_func = dynamic_retry(func)
        return decorated_func(self, *args, **kwargs)
    return wrapper

class CachedRemoteClient(BaseLLM):
    """Cached client for a chat-completion endpoint."""
    @classmethod
    def from_experiment_config(cls, global_config: BaseConfig) -> "CachedRemoteClient":
        config_dict = global_config.__dict__
        config_dict['max_retries'] = global_config.max_retry_attempts
        cache_dir = os.path.join(global_config.save_dir, "llm_cache")
        return cls(cache_dir=cache_dir, global_config=global_config)

    def __init__(self, cache_dir, global_config, cache_filename: str = None,
                 high_throughput: bool = True,
                 **kwargs) -> None:

        super().__init__()
        self.cache_dir = cache_dir
        self.global_config = global_config

        self.llm_name = global_config.llm_name
        self.llm_base_url = global_config.llm_base_url

        os.makedirs(self.cache_dir, exist_ok=True)
        if cache_filename is None:
            cache_filename = f"{self.llm_name.replace('/', '_')}_cache.sqlite"
        self.cache_file_name = os.path.join(self.cache_dir, cache_filename)

        self._init_llm_config()
        if high_throughput:
            limits = httpx.Limits(max_connections=500, max_keepalive_connections=100)
            trust_env = _env_bool("COMPASSRAG_REMOTE_TRUST_ENV", True)
            default_timeout = 5 * 60
            connect_timeout = _env_float(
                "COMPASSRAG_REMOTE_CONNECT_TIMEOUT_SECONDS",
                _env_float("COMPASSRAG_REMOTE_TIMEOUT_SECONDS", default_timeout, minimum=1.0),
                minimum=1.0,
            )
            read_timeout = _env_float(
                "COMPASSRAG_REMOTE_READ_TIMEOUT_SECONDS",
                _env_float("COMPASSRAG_REMOTE_TIMEOUT_SECONDS", default_timeout, minimum=1.0),
                minimum=1.0,
            )
            write_timeout = _env_float(
                "COMPASSRAG_REMOTE_WRITE_TIMEOUT_SECONDS",
                _env_float("COMPASSRAG_REMOTE_TIMEOUT_SECONDS", default_timeout, minimum=1.0),
                minimum=1.0,
            )
            pool_timeout = _env_float(
                "COMPASSRAG_REMOTE_POOL_TIMEOUT_SECONDS",
                _env_float("COMPASSRAG_REMOTE_TIMEOUT_SECONDS", default_timeout, minimum=1.0),
                minimum=1.0,
            )
            timeout = httpx.Timeout(
                default_timeout,
                connect=connect_timeout,
                read=read_timeout,
                write=write_timeout,
                pool=pool_timeout,
            )
        else:
            limits = None
            timeout = float(_env_float("COMPASSRAG_REMOTE_TIMEOUT_SECONDS", 300.0, minimum=1.0))

        self.max_retries = _env_int(
            "COMPASSRAG_REMOTE_RETRY_ATTEMPTS",
            kwargs.get("max_retries", getattr(global_config, "max_retry_attempts", 5)),
            minimum=1,
        )
        endpoint = self.global_config.azure_endpoint or self.llm_base_url
        self.client = RemoteEndpoint(
            base_url=endpoint,
            api_key=os.environ.get("MODEL_API_KEY", ""),
            timeout=timeout,
            limits=limits,
        )

    def _init_llm_config(self) -> None:
        config_dict = self.global_config.__dict__

        config_dict['llm_name'] = self.global_config.llm_name
        config_dict['llm_base_url'] = self.global_config.llm_base_url
        config_dict['generate_params'] = {
                "model": self.global_config.llm_name,
                "max_tokens": config_dict.get("max_new_tokens", 400),
                "n": config_dict.get("num_gen_choices", 1),
                "seed": config_dict.get("seed", 0),
                "temperature": config_dict.get("temperature", 0.0),
            }

        self.llm_config = LLMConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Init {self.__class__.__name__}'s llm_config: {self.llm_config}")

    @cache_response
    @dynamic_retry_decorator
    def infer(
        self,
        messages: List[TextChatMessage],
        **kwargs
    ) -> Tuple[List[TextChatMessage], dict]:
        params = deepcopy(self.llm_config.generate_params)
        if kwargs:
            params.update(kwargs)
        params["messages"] = messages
        logger.debug("Calling the configured chat-completion endpoint with %s", params)

        messages = params.pop("messages")
        model = params.pop("model")
        response = self.client.chat(model=model, messages=messages, **params)
        choice = response["choices"][0]
        response_message = choice.get("message", {}).get("content", "")
        assert isinstance(response_message, str), "response_message should be a string"
        usage = response.get("usage", {}) or {}
        metadata = {
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "finish_reason": choice.get("finish_reason"),
        }

        return response_message, metadata
