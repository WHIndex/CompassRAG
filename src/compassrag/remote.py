from __future__ import annotations

from typing import Any, Mapping, Sequence

import httpx


class RemoteEndpoint:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float | httpx.Timeout = 300.0,
        limits: httpx.Limits | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("A model endpoint is required")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(
            base_url=base_url.rstrip("/") + "/",
            headers=headers,
            timeout=timeout,
            limits=limits,
        )

    def chat(
        self,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        **parameters: Any,
    ) -> dict[str, Any]:
        payload = {"model": model, "messages": list(messages), **parameters}
        response = self._client.post("chat/completions", json=payload)
        response.raise_for_status()
        value = response.json()
        if not isinstance(value, dict) or not value.get("choices"):
            raise ValueError("The chat endpoint returned no choices")
        return value

    def embeddings(self, model: str, texts: Sequence[str]) -> list[list[float]]:
        response = self._client.post(
            "embeddings",
            json={"model": model, "input": list(texts)},
        )
        response.raise_for_status()
        value = response.json()
        rows = value.get("data", []) if isinstance(value, dict) else []
        return [list(row["embedding"]) for row in rows]

    def close(self) -> None:
        self._client.close()
