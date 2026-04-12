from __future__ import annotations

import os
from typing import AsyncGenerator, Dict, List, Optional, Tuple

from openai import APIError, AsyncOpenAI, AuthenticationError, InternalServerError, RateLimitError

from .base_provider import BaseLLMProvider, LLMMessage, LLMResponse


class OpenAIProvider(BaseLLMProvider):
    PRICE_PER_MILLION: Dict[str, Tuple[float, float]] = {
        "gpt-4o": (5.0, 15.0),
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4-turbo": (10.0, 30.0),
        "o1-mini": (3.0, 12.0),
        "o1-preview": (15.0, 60.0),
    }

    def __init__(self, config: Dict[str, object]):
        super().__init__(config)
        api_key = (config.get("api_key") or "").strip()
        env_token = (os.getenv("GITHUB_TOKEN") or "").strip()
        default_base = config.get("base_url", "https://api.openai.com/v1")

        self._is_github_models = bool(
            api_key.startswith("github_") or (env_token and api_key == env_token) or "models.inference.ai.azure.com" in default_base
        )
        if self._is_github_models and "base_url" not in config:
            default_base = "https://models.inference.ai.azure.com"

        self.client = AsyncOpenAI(api_key=api_key, base_url=default_base)
        self.base_url = default_base

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def supports_streaming(self) -> bool:
        return True

    async def complete(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        try:
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                temperature=self.temperature if temperature is None else temperature,
                max_tokens=self.max_tokens if max_tokens is None else max_tokens,
            )
        except RateLimitError as exc:
            raise RuntimeError(f"OpenAI/GitHub rate limited: {exc}") from exc
        except AuthenticationError as exc:
            raise RuntimeError("OpenAI/GitHub authentication failed. Check api_key.") from exc
        except (InternalServerError, APIError) as exc:
            # Retry once for transient server errors.
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                temperature=self.temperature if temperature is None else temperature,
                max_tokens=self.max_tokens if max_tokens is None else max_tokens,
            )

        raw = completion.choices[0].message.content
        content = raw if isinstance(raw, str) else ""
        usage = completion.usage
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

        if self._is_github_models:
            cost = 0.0
        else:
            cost = self.estimate_cost(input_tokens=input_tokens, output_tokens=output_tokens)

        return LLMResponse(
            content=content,
            model=self.model,
            provider=self.provider_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_estimate=cost,
        )

    async def stream(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                temperature=self.temperature if temperature is None else temperature,
                max_tokens=self.max_tokens if max_tokens is None else max_tokens,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta
        except RateLimitError as exc:
            raise RuntimeError(f"OpenAI/GitHub rate limited: {exc}") from exc
        except AuthenticationError as exc:
            raise RuntimeError("OpenAI/GitHub authentication failed. Check api_key.") from exc

    async def health_check(self) -> bool:
        try:
            await self.complete(messages=[LLMMessage(role="user", content="Say ok")], max_tokens=4, temperature=0.0)
            return True
        except Exception:
            return False

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        if self._is_github_models:
            return 0.0
        in_price, out_price = self.PRICE_PER_MILLION.get(self.model, (5.0, 15.0))
        return round((input_tokens / 1_000_000.0) * in_price + (output_tokens / 1_000_000.0) * out_price, 6)

    async def aclose(self) -> None:
        try:
            await self.client.close()
        except Exception:
            pass
