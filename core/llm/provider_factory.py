from __future__ import annotations

import os
from .base_provider import BaseLLMProvider


def _detect_provider(config: dict) -> str:
    explicit = (config.get("provider") or "").strip().lower()
    if explicit:
        return explicit

    api_key = (config.get("api_key") or "").strip()
    env_token = (os.getenv("GITHUB_TOKEN") or "").strip()

    if api_key.startswith("sk-ant-"):
        return "anthropic"
    if api_key.startswith("AIza"):
        return "gemini"
    if api_key.startswith("gsk_"):
        return "groq"
    if api_key.startswith("github_") or (env_token and api_key == env_token):
        return "openai"
    if api_key.startswith("sk-"):
        return "openai"
    if (config.get("provider") or "").lower() == "ollama":
        return "ollama"
    return "openai"


def create_provider(config: dict) -> BaseLLMProvider:
    provider_name = _detect_provider(config)
    if provider_name == "openai":
        from .openai_provider import OpenAIProvider

        return OpenAIProvider(config)
    if provider_name == "anthropic":
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(config)
    if provider_name == "gemini":
        from .gemini_provider import GeminiProvider

        return GeminiProvider(config)
    if provider_name == "groq":
        from .groq_provider import GroqProvider

        return GroqProvider(config)
    if provider_name == "ollama":
        from .ollama_provider import OllamaProvider

        return OllamaProvider(config)

    supported = ["openai", "anthropic", "gemini", "groq", "ollama"]
    if provider_name not in supported:
        raise ValueError(
            f"Unknown provider: {provider_name}. Supported: {supported}"
        )
    raise ValueError(f"Unknown provider: {provider_name}")
