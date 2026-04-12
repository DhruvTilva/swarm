from __future__ import annotations

from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

try:
    from anthropic import APIError, AsyncAnthropic, AuthenticationError, RateLimitError
except ImportError:  # pragma: no cover - dependency optional
    APIError = Exception
    AuthenticationError = Exception
    RateLimitError = Exception
    AsyncAnthropic = None

from .base_provider import BaseLLMProvider, LLMMessage, LLMResponse


class AnthropicProvider(BaseLLMProvider):
    PRICE_PER_MILLION: Dict[str, Tuple[float, float]] = {
        "claude-opus-4-5": (15.0, 75.0),
        "claude-sonnet-4-20250514": (3.0, 15.0),
        "claude-haiku-4-5-20251001": (0.25, 1.25),
    }

    def __init__(self, config: Dict[str, object]):
        super().__init__(config)
        if AsyncAnthropic is None:
            raise RuntimeError("anthropic package not installed. Install with: pip install anthropic")
        self.client = AsyncAnthropic(api_key=config.get("api_key", ""))

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def supports_streaming(self) -> bool:
        return True

    async def complete(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        system_message, msg_payload = self._to_anthropic_messages(messages)
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens if max_tokens is None else max_tokens,
                system=system_message,
                messages=msg_payload,
                temperature=self.temperature if temperature is None else temperature,
                stream=False,
            )
        except RateLimitError as exc:
            raise RuntimeError(f"Anthropic rate limited: {exc}") from exc
        except AuthenticationError as exc:
            raise RuntimeError("Anthropic authentication failed. Check api_key.") from exc
        except APIError as exc:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens if max_tokens is None else max_tokens,
                system=system_message,
                messages=msg_payload,
                temperature=self.temperature if temperature is None else temperature,
                stream=False,
            )

        content = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
        input_tokens = int(getattr(response.usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(response.usage, "output_tokens", 0) or 0)
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
        system_message, msg_payload = self._to_anthropic_messages(messages)
        async with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
            system=system_message,
            messages=msg_payload,
            temperature=self.temperature if temperature is None else temperature,
        ) as stream:
            async for text in stream.text_stream:
                if text:
                    yield text

    async def health_check(self) -> bool:
        try:
            await self.complete([LLMMessage(role="user", content="Say ok")], max_tokens=4, temperature=0.0)
            return True
        except Exception:
            return False

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        in_price, out_price = self.PRICE_PER_MILLION.get(self.model, (3.0, 15.0))
        return round((input_tokens / 1_000_000.0) * in_price + (output_tokens / 1_000_000.0) * out_price, 6)

    @staticmethod
    def _to_anthropic_messages(messages: List[LLMMessage]) -> Tuple[str, List[Dict[str, Any]]]:
        system_chunks = [m.content for m in messages if m.role == "system"]
        system_message = "\n\n".join(system_chunks)
        payload: List[Dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                continue
            role = "assistant" if m.role == "assistant" else "user"
            payload.append({"role": role, "content": m.content})
        if not payload:
            payload = [{"role": "user", "content": "Continue."}]
        return system_message, payload

    async def aclose(self) -> None:
        return
