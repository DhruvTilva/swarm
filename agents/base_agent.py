from __future__ import annotations

import asyncio
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from core.message_bus import SwarmMessage
from core.llm import LLMMessage, LLMResponse, create_provider


@dataclass
class AgentSettings:
    provider: str
    api_key: str
    model: str
    base_url: str
    temperature: float
    max_tokens: int
    delay_seconds: float


class BaseAgent:
    name: str = "Agent"
    emoji: str = "🤖"
    personality: str = "Balanced and helpful"

    def __init__(self, bus: Any, settings: AgentSettings) -> None:
        self.bus = bus
        self.settings = settings
        self.last_output: str = ""
        self.provider = None
        self.llm_calls: int = 0
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.total_cost: float = 0.0
        self._last_llm_error: str = ""
        try:
            self.provider = create_provider(
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
            self._last_llm_error = f"Provider setup failed: {exc}"

    @property
    def system_prompt(self) -> str:
        return (
            "You are a member of SWARM, a multi-agent software factory. "
            f"Your name is {self.name}. Personality: {self.personality}. "
            "Always provide concrete technical output with sharp tradeoffs. "
            "Write short, high-signal updates optimized for real-time dashboards."
        )

    def _fallback_lines(self, phase: str, task: str, context: Dict[str, Any]) -> List[str]:
        raise NotImplementedError

    async def stream_phase_lines(
        self,
        phase: str,
        task: str,
        context: Dict[str, Any],
        max_lines: int = 5,
    ) -> List[str]:
        if self.provider is None:
            lines = self._fallback_lines(phase, task, context)
            if self._last_llm_error:
                lines.insert(0, self._last_llm_error)
            self.last_output = "\n".join(lines)
            await asyncio.sleep(self.settings.delay_seconds)
            return lines[:max_lines]

        try:
            prompt = self._build_phase_prompt(phase=phase, task=task, context=context)
            response = await self.call_llm_response(
                messages=[
                    LLMMessage(role="system", content=self.system_prompt),
                    LLMMessage(role="user", content=prompt),
                ],
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens,
            )
            text = response.content
            lines = self._split_to_lines(text)
            if not lines:
                lines = self._fallback_lines(phase, task, context)
            self.last_output = "\n".join(lines)
            return lines[:max_lines]
        except Exception as exc:
            fallback = self._fallback_lines(phase, task, context)
            fallback.insert(0, f"LLM unavailable: {exc}")
            self.last_output = "\n".join(fallback)
            return fallback[:max_lines]

    async def publish_lines(self, phase: str, lines: List[str], status: str = "working") -> None:
        for line in lines:
            await self.bus.publish(
                SwarmMessage(
                    source=self.name,
                    target="all",
                    phase=phase,
                    kind="agent_log",
                    status=status,
                    text=line,
                )
            )
            await asyncio.sleep(self.settings.delay_seconds)

    async def react(self, phase: str, incoming: SwarmMessage) -> None:
        reactions = self._reaction_templates(incoming)
        if not reactions:
            return
        line = random.choice(reactions)
        await self.bus.publish(
            SwarmMessage(
                source=self.name,
                target=incoming.source,
                phase=phase,
                kind="reaction",
                status="arguing",
                text=line,
            )
        )

    def _reaction_templates(self, incoming: SwarmMessage) -> List[str]:
        return []

    def _build_phase_prompt(self, phase: str, task: str, context: Dict[str, Any]) -> str:
        requirement_updates = context.get("requirement_updates", [])
        shared_notes = context.get("notes", {})
        return (
            f"Task: {task}\n"
            f"Phase: {phase}\n"
            f"Requirement updates: {requirement_updates}\n"
            f"Shared notes keys: {list(shared_notes.keys())}\n"
            "Respond with 4-6 short bullet lines, each line under 110 chars, each line practical."
        )

    @staticmethod
    def _split_to_lines(text: str) -> List[str]:
        chunks = re.split(r"[\n\r]+|(?<=[.!?])\s+", text)
        clean = [c.strip(" -\t") for c in chunks if c and c.strip()]
        return clean

    async def call_llm_response(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        if self.provider is None:
            raise RuntimeError(self._last_llm_error or "LLM provider unavailable")
        try:
            response = await self.provider.complete(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            self._record_usage(response)
            return response
        except Exception as exc:
            self._last_llm_error = str(exc)
            raise

    async def call_llm(
        self,
        messages: List[LLMMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        response = await self.call_llm_response(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.content

    async def stream_llm(
        self,
        messages: List[LLMMessage],
        on_chunk: Optional[Any] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        full_response = ""
        if self.provider is None:
            raise RuntimeError(self._last_llm_error or "LLM provider unavailable")
        async for chunk in self.provider.stream(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            full_response += chunk
            if on_chunk is not None:
                await on_chunk(chunk)
        return full_response

    def _record_usage(self, response: LLMResponse) -> None:
        self.llm_calls += 1
        self.input_tokens += int(response.input_tokens)
        self.output_tokens += int(response.output_tokens)
        self.total_cost += float(response.cost_estimate)

    async def aclose(self) -> None:
        if self.provider is None:
            return
        try:
            await self.provider.aclose()
        except Exception:
            # Client shutdown failures should not break the overall build flow.
            pass
