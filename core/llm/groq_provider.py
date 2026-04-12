from __future__ import annotations

from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

try:
    from groq import APIError, AsyncGroq, AuthenticationError, RateLimitError
except ImportError:  # pragma: no cover - dependency optional
    APIError = Exception
    AuthenticationError = Exception
    RateLimitError = Exception
    AsyncGroq = None

from .base_provider import BaseLLMProvider, LLMMessage, LLMResponse


class GroqProvider(BaseLLMProvider):
    PRICE_PER_MILLION: Dict[str, Tuple[float, float]] = {
        "llama-3.3-70b-versatile": (0.59, 0.79),
        "llama-3.1-8b-instant": (0.05, 0.08),
        "mixtral-8x7b-32768": (0.24, 0.24),
        "gemma2-9b-it": (0.20, 0.20),
    }

    def __init__(self, config: Dict[str, object]):
        super().__init__(config)
        if AsyncGroq is None:
            raise RuntimeError("groq package not installed. Install with: pip install groq")
        self.client = AsyncGroq(api_key=config.get("api_key", ""))

    @property
    def provider_name(self) -> str:
        return "groq"

    @property
    def supports_streaming(self) -> bool:
        return True

    async def complete(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        payload = self._to_groq_messages(messages)
        try:
            completion = await self.client.chat.completions.create(
                model=self.model,
                messages=payload,
                temperature=self.temperature if temperature is None else temperature,
                max_tokens=self.max_tokens if max_tokens is None else max_tokens,
            )
        except RateLimitError as exc:
            raise RuntimeError(f"Groq rate limited: {exc}") from exc
        except AuthenticationError as exc:
            raise RuntimeError("Groq authentication failed. Check api_key.") from exc
        except APIError as exc:
            raise RuntimeError(f"Groq API error: {exc}") from exc

        raw = completion.choices[0].message.content
        content = raw if isinstance(raw, str) else ""
        usage = completion.usage
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
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
        payload = self._to_groq_messages(messages)
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=payload,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                yield delta

    async def health_check(self) -> bool:
        try:
            await self.complete([LLMMessage(role="user", content="Say ok")], max_tokens=4, temperature=0.0)
            return True
        except Exception:
            return False

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        in_price, out_price = self.PRICE_PER_MILLION.get(self.model, (0.59, 0.79))
        return round((input_tokens / 1_000_000.0) * in_price + (output_tokens / 1_000_000.0) * out_price, 6)

    @staticmethod
    def _to_groq_messages(messages: List[LLMMessage]) -> List[Dict[str, Any]]:
        system = [m.content for m in messages if m.role == "system"]
        non_system = [m for m in messages if m.role != "system"]
        if not non_system:
            non_system = [LLMMessage(role="user", content="Continue.")]

        payload = [{"role": m.role, "content": m.content} for m in non_system]
        if system:
            first_user_idx = next((idx for idx, item in enumerate(payload) if item["role"] == "user"), None)
            merged = "\n\n".join(system)
            if first_user_idx is None:
                payload.insert(0, {"role": "user", "content": merged})
            else:
                payload[first_user_idx]["content"] = f"{merged}\n\n{payload[first_user_idx]['content']}"
        return payload

    async def aclose(self) -> None:
        return
