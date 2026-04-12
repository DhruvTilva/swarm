from __future__ import annotations

from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

try:
    import google.generativeai as genai
    from google.api_core.exceptions import PermissionDenied, ResourceExhausted
except ImportError:  # pragma: no cover - dependency optional
    genai = None
    PermissionDenied = Exception
    ResourceExhausted = Exception

from .base_provider import BaseLLMProvider, LLMMessage, LLMResponse


class GeminiProvider(BaseLLMProvider):
    PRICE_PER_MILLION: Dict[str, Tuple[float, float]] = {
        "gemini-1.5-pro": (3.50, 10.50),
        "gemini-2.0-flash": (0.075, 0.30),
        "gemini-1.5-flash": (0.075, 0.30),
    }

    def __init__(self, config: Dict[str, object]):
        super().__init__(config)
        if genai is None:
            raise RuntimeError("google-generativeai package not installed. Install with: pip install google-generativeai")
        genai.configure(api_key=config.get("api_key", ""))

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def supports_streaming(self) -> bool:
        return True

    async def complete(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        system_instruction, contents = self._to_gemini(messages)
        model = genai.GenerativeModel(
            self.model,
            system_instruction=system_instruction if system_instruction else None,
            generation_config={
                "temperature": self.temperature if temperature is None else temperature,
                "max_output_tokens": self.max_tokens if max_tokens is None else max_tokens,
            },
        )
        try:
            response = await model.generate_content_async(contents=contents, stream=False)
        except ResourceExhausted as exc:
            raise RuntimeError(f"Gemini resource exhausted: {exc}") from exc
        except PermissionDenied as exc:
            raise RuntimeError("Gemini authentication failed. Check api_key.") from exc

        content = response.text or ""
        usage = getattr(response, "usage_metadata", None)
        input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
        output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
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
        system_instruction, contents = self._to_gemini(messages)
        model = genai.GenerativeModel(
            self.model,
            system_instruction=system_instruction if system_instruction else None,
            generation_config={
                "temperature": self.temperature if temperature is None else temperature,
                "max_output_tokens": self.max_tokens if max_tokens is None else max_tokens,
            },
        )
        response = await model.generate_content_async(contents=contents, stream=True)
        async for chunk in response:
            text = getattr(chunk, "text", None)
            if text:
                yield text

    async def health_check(self) -> bool:
        try:
            await self.complete([LLMMessage(role="user", content="Say ok")], max_tokens=4, temperature=0.0)
            return True
        except Exception:
            return False

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        in_price, out_price = self.PRICE_PER_MILLION.get(self.model, (0.075, 0.30))
        return round((input_tokens / 1_000_000.0) * in_price + (output_tokens / 1_000_000.0) * out_price, 6)

    @staticmethod
    def _to_gemini(messages: List[LLMMessage]) -> Tuple[str, List[Dict[str, Any]]]:
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        contents: List[Dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                continue
            role = "model" if m.role == "assistant" else "user"
            contents.append({"role": role, "parts": [m.content]})
        if not contents:
            contents = [{"role": "user", "parts": ["Continue."]}]
        return system, contents

    async def aclose(self) -> None:
        return
