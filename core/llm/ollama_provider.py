from __future__ import annotations

import json
from typing import Dict, List, AsyncGenerator, Optional

import aiohttp

from .base_provider import BaseLLMProvider, LLMMessage, LLMResponse


class OllamaProvider(BaseLLMProvider):
    def __init__(self, config: Dict[str, object]):
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:11434").rstrip("/")

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def supports_streaming(self) -> bool:
        return True

    async def complete(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {
                "temperature": self.temperature if temperature is None else temperature,
                "num_predict": self.max_tokens if max_tokens is None else max_tokens,
            },
        }
        try:
            timeout = aiohttp.ClientTimeout(total=60)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 404:
                        raise RuntimeError(f"Ollama model '{self.model}' not found. Run: ollama pull {self.model}")
                    if response.status >= 400:
                        body = await response.text()
                        raise RuntimeError(f"Ollama error {response.status}: {body[:300]}")
                    data = await response.json()
        except aiohttp.ClientConnectorError as exc:
            raise RuntimeError(
                "Ollama not detected at localhost:11434. Install https://ollama.ai, then run: ollama pull llama3.2"
            ) from exc

        content = str(data.get("message", {}).get("content", ""))
        prompt_tokens = int(data.get("prompt_eval_count", 0) or 0)
        output_tokens = int(data.get("eval_count", 0) or 0)
        return LLMResponse(
            content=content,
            model=self.model,
            provider=self.provider_name,
            input_tokens=prompt_tokens,
            output_tokens=output_tokens,
            cost_estimate=0.0,
        )

    async def stream(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
            "options": {
                "temperature": self.temperature if temperature is None else temperature,
                "num_predict": self.max_tokens if max_tokens is None else max_tokens,
            },
        }

        timeout = aiohttp.ClientTimeout(total=120)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as response:
                if response.status == 404:
                    raise RuntimeError(f"Ollama model '{self.model}' not found. Run: ollama pull {self.model}")
                if response.status >= 400:
                    body = await response.text()
                    raise RuntimeError(f"Ollama error {response.status}: {body[:300]}")
                async for raw in response.content:
                    line = raw.decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    chunk = str(data.get("message", {}).get("content", ""))
                    if chunk:
                        yield chunk

    async def health_check(self) -> bool:
        try:
            async with aiohttp.ClientSession() as session:
                timeout = aiohttp.ClientTimeout(total=2)
                async with session.get(f"{self.base_url}/api/tags", timeout=timeout) as response:
                    return response.status == 200
        except Exception:
            return False

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return 0.0

    async def aclose(self) -> None:
        return
