from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from core.build_engine import BuildEngine, BuildResult, SwarmSettings
from core.llm import create_provider
from core.message_bus import MessageBus
from ui.dashboard import DashboardApp


def load_settings(
    config_path: Path,
    provider_override: str = "",
    model_override: str = "",
    output_override: str = "",
) -> SwarmSettings:
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in configuration file: {config_path}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"Configuration root must be a mapping in: {config_path}")

    config: Dict[str, Any] = raw

    provider = (provider_override or config.get("provider") or "openai").strip().lower()
    model = (model_override or config.get("model") or "gpt-4o").strip()
    base_url = (config.get("base_url") or "https://api.openai.com/v1").strip()

    file_key = (config.get("api_key") or "").strip()
    env_key = ""
    if provider == "openai":
        env_key = (os.getenv("OPENAI_API_KEY") or os.getenv("GITHUB_TOKEN") or "").strip()
    elif provider == "anthropic":
        env_key = (os.getenv("ANTHROPIC_API_KEY") or "").strip()
    elif provider == "gemini":
        env_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    elif provider == "groq":
        env_key = (os.getenv("GROQ_API_KEY") or "").strip()
    elif provider == "ollama":
        env_key = ""

    api_key = file_key or env_key
    output_dir = Path(output_override) if output_override else Path(config.get("output_dir", "output"))

    return SwarmSettings(
        provider=provider,
        api_key=api_key,
        model=model,
        base_url=base_url,
        temperature=float(config.get("temperature", 0.35)),
        max_tokens=int(config.get("max_tokens", 1400)),
        database_path=Path(config.get("database_path", "swarm.db")),
        output_dir=output_dir,
        agent_delay_seconds=float(config.get("agent_delay_seconds", 0.25)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Swarm: multi-agent AI software factory")
    parser.add_argument("task", help="Plain-English software request")
    parser.add_argument("--config", default="swarm.config.yaml", help="Path to swarm config YAML file")
    parser.add_argument("--headless", action="store_true", help="Run without textual dashboard")
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "gemini", "groq", "ollama"],
        help="Override provider from config",
    )
    parser.add_argument("--model", help="Override model from config")
    parser.add_argument("--output", help="Override output directory")
    return parser.parse_args()


async def check_provider_health(settings: SwarmSettings) -> bool:
    try:
        provider = create_provider(
            {
                "provider": settings.provider,
                "api_key": settings.api_key,
                "model": settings.model,
                "base_url": settings.base_url,
                "temperature": settings.temperature,
                "max_tokens": settings.max_tokens,
            }
        )
    except Exception as exc:
        print(f"Provider setup failed: {exc}")
        print("Falling back to deterministic mode.")
        return False

    print(f"Checking {provider.provider_name}...")
    healthy = await provider.health_check()
    if not healthy:
        print(f"{provider.provider_name} unreachable.")
        print("Check your API key and internet connection.")
        print("Falling back to deterministic mode.")
    else:
        print(f"{provider.provider_name} connected. Model: {settings.model}")

    await provider.aclose()
    return healthy


async def run_headless(engine: BuildEngine) -> BuildResult:
    await check_provider_health(engine.settings)
    return await engine.run_headless()


def run_dashboard(task: str, engine: BuildEngine) -> None:
    app = DashboardApp(task=task, engine=engine)
    app.run()


def build_settings_from_args(args: argparse.Namespace) -> Optional[SwarmSettings]:
    try:
        return load_settings(
            Path(args.config),
            provider_override=args.provider or "",
            model_override=args.model or "",
            output_override=args.output or "",
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Configuration error: {exc}")
        return None


def main() -> None:
    args = parse_args()
    settings = build_settings_from_args(args)
    if settings is None:
        return

    bus = MessageBus(settings.database_path)
    engine = BuildEngine(task=args.task, settings=settings, bus=bus)

    if args.headless:
        asyncio.run(run_headless(engine))
        return

    run_dashboard(args.task, engine)


if __name__ == "__main__":
    main()
