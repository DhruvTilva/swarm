from __future__ import annotations

import asyncio
import json
import re
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import aiosqlite

from agents.architect import ArchitectAgent
from agents.backend import BackendAgent
from agents.base_agent import AgentSettings, BaseAgent
from agents.docs import DocsAgent
from agents.frontend import FrontendAgent
from agents.pm import PMAgent
from agents.tester import TesterAgent
from core.file_writer import FileWriter
from core.llm import BaseLLMProvider, LLMResponse, create_provider
from core.message_bus import MessageBus, SwarmMessage


@dataclass
class SwarmSettings:
    provider: str
    api_key: str
    model: str
    base_url: str
    temperature: float
    max_tokens: int
    database_path: Path
    output_dir: Path
    agent_delay_seconds: float


@dataclass
class BuildResult:
    project_path: Path
    message_count: int
    duration_seconds: float
    requirement_updates: List[str]
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BuildState:
    task: str
    project_name: str
    project_slug: str
    started_at: datetime

    provider: Optional[BaseLLMProvider] = None
    provider_name: str = "unknown"

    product_prd: str = ""
    architect_design: str = ""

    agent_outputs: Dict[str, str] = field(default_factory=dict)
    agent_files: Dict[str, Dict[str, str]] = field(default_factory=dict)

    backend_endpoints: List[Dict[str, Any]] = field(default_factory=list)
    backend_schemas: Dict[str, Any] = field(default_factory=dict)
    implementation_summary: str = ""

    ui_summary: str = ""
    ui_complexity: str = "MEDIUM"

    bugs_found: List[Dict[str, Any]] = field(default_factory=list)
    bugs_fixed: List[str] = field(default_factory=list)
    coverage_report: Dict[str, Any] = field(default_factory=dict)
    tester_verdict: str = "PENDING"
    tester_signed_off: bool = False

    pm_scorecard: Dict[str, Any] = field(default_factory=dict)
    pm_ship_decision: str = "PENDING"
    pm_delivery_summary: str = ""
    requirement_change: str = ""

    docs_files: Dict[str, str] = field(default_factory=dict)

    total_cost: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    api_calls_made: int = 0
    loc_generated: int = 0

    current_phase: str = "INIT"
    phase_history: List[Dict[str, Any]] = field(default_factory=list)

    all_generated_files: Dict[str, str] = field(default_factory=dict)


AgentRunner = Callable[[Dict[str, Any]], Awaitable[List[str]]]


class BuildEngine:
    PHASES = [
        "PLANNING",
        "ARCHITECTURE_REVIEW",
        "IMPLEMENTATION",
        "TESTING",
        "DOCUMENTATION",
        "PACKAGING",
        "COMPLETE",
    ]

    def __init__(self, task: str, settings: SwarmSettings, bus: MessageBus) -> None:
        self.task = task.strip()
        self.settings = settings
        self.bus = bus

        slug = self._slugify(self._derive_project_name(self.task))
        self.output_dir = settings.output_dir / slug

        self.provider: Optional[BaseLLMProvider] = None
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
        except Exception:
            self.provider = None

        self.state = BuildState(
            task=self.task,
            project_name=self._derive_project_name(self.task),
            project_slug=slug,
            started_at=datetime.now(timezone.utc),
            provider=self.provider,
            provider_name=settings.provider,
        )

        self.writer = FileWriter(
            output_root=settings.output_dir,
            provider=settings.provider,
            api_key=settings.api_key,
            model=settings.model,
            base_url=settings.base_url,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )

        agent_settings = AgentSettings(
            provider=settings.provider,
            api_key=settings.api_key,
            model=settings.model,
            base_url=settings.base_url,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            delay_seconds=settings.agent_delay_seconds,
        )

        self.architect = ArchitectAgent(bus=bus, settings=agent_settings)
        self.backend = BackendAgent(bus=bus, settings=agent_settings)
        self.frontend = FrontendAgent(bus=bus, settings=agent_settings)
        self.tester = TesterAgent(bus=bus, settings=agent_settings)
        self.docs = DocsAgent(bus=bus, settings=agent_settings)
        self.pm = PMAgent(bus=bus, settings=agent_settings)

        self._agents_by_name: Dict[str, BaseAgent] = {
            "Architect": self.architect,
            "Backend": self.backend,
            "Frontend": self.frontend,
            "Tester": self.tester,
            "Docs": self.docs,
            "PM": self.pm,
        }

        self._usage_seen: Dict[str, Dict[str, float]] = {
            name: {"llm_calls": 0.0, "input_tokens": 0.0, "output_tokens": 0.0, "total_cost": 0.0}
            for name in self._agents_by_name
        }

        self._db: Optional[aiosqlite.Connection] = None
        self._build_id: str = ""
        self._next_phase_index: int = 0
        self._phase_started_at = time.monotonic()

    async def run(self) -> BuildResult:
        started_at = time.monotonic()
        await self.bus.initialize()
        await self.bus.subscribe(self._on_bus_message)

        try:
            await self._init_persistence_tables()
            await self._start_or_resume_build()

            await self._publish("SYSTEM", "BOOT", "system", "working", f"Swarm started for task: {self.state.task}")
            await self._publish(
                "SYSTEM",
                "BOOT",
                "system",
                "working",
                f"Provider: {self.state.provider_name} | Model: {self.settings.model}",
            )

            for index in range(self._next_phase_index, len(self.PHASES)):
                phase = self.PHASES[index]
                self.state.current_phase = phase
                self._phase_started_at = time.monotonic()

                await self.bus.set_phase(phase, index + 1, len(self.PHASES))
                await self._publish("SYSTEM", phase, "phase", "working", f"Phase {index + 1}/7: {phase}")

                await self.run_phase(phase)
                await self.persist_phase(phase)
                await self._update_build_row("RUNNING")

            duration = time.monotonic() - started_at
            summary = self._build_summary(duration)
            await self._update_build_row("COMPLETE", summary)

            return BuildResult(
                project_path=self.output_dir,
                message_count=self.bus.message_count,
                duration_seconds=duration,
                requirement_updates=[self.state.requirement_change] if self.state.requirement_change else [],
                total_input_tokens=self.state.total_input_tokens,
                total_output_tokens=self.state.total_output_tokens,
                total_cost=self.state.total_cost,
                summary=summary,
            )
        except Exception:
            await self._update_build_row("INTERRUPTED")
            raise
        finally:
            await self._shutdown()

    async def run_headless(self) -> BuildResult:
        print("Swarm starting in headless mode")
        print(f"Task: {self.state.task}")
        print(f"Provider: {self.state.provider_name}")
        print()
        result = await self.run()
        self.print_completion_summary()
        return result

    def print_completion_summary(self) -> None:
        s = self.state
        print("\n" + "=" * 50)
        print("SWARM BUILD COMPLETE")
        print("=" * 50)
        print(f"Output: {self.output_dir}")
        print(f"Time: {self.get_elapsed_str()}")
        print(f"Provider: {s.provider_name}")
        print(f"Coverage: {s.coverage_report.get('coverage_percentage', 0)}%")
        print(f"Bugs: {len(s.bugs_found)} found, {len(s.bugs_fixed)} fixed")
        print(f"Cost: ${s.total_cost:.4f}")
        print(f"LOC: {s.loc_generated}")
        print(f"Ship: {s.pm_ship_decision}")
        print("=" * 50)

    async def run_phase(self, phase: str) -> None:
        if phase == "PLANNING":
            await self._run_planning_phase()
            await self.complete_phase(phase, "PRD and architecture design published.")
            return
        if phase == "ARCHITECTURE_REVIEW":
            await self._run_architecture_review_phase()
            await self.complete_phase(phase, "Architecture reviewed and all agents briefed.")
            return
        if phase == "IMPLEMENTATION":
            await self._run_implementation_phase()
            await self.complete_phase(phase, "Backend and frontend completed with merged outputs.")
            return
        if phase == "TESTING":
            await self._run_testing_phase()
            if self.state.tester_signed_off:
                await self.complete_phase(phase, "Tester signed off.")
            else:
                await self.complete_phase(phase, "Max retries reached, continued with warnings.")
            return
        if phase == "DOCUMENTATION":
            await self._run_documentation_phase()
            await self.complete_phase(phase, "Documentation suite generated.")
            return
        if phase == "PACKAGING":
            await self._run_packaging_phase()
            await self.complete_phase(phase, "Project packaged and PM final review completed.")
            return
        if phase == "COMPLETE":
            await self._run_complete_phase()
            await self.complete_phase(phase, "Build summary generated and published.")

    async def _run_planning_phase(self) -> None:
        await self.run_agent_safe(self.pm, "PM", self.get_agent_context("PM"), self._execute_pm_planning)
        if not self.state.product_prd:
            self.state.product_prd = getattr(self.pm, "prd", "") or self.pm.last_output

        await self._publish("SYSTEM", "PLANNING", "summary", "working", "PM PRD published. Architect can start now.")

        await self.run_agent_safe(
            self.architect,
            "Architect",
            self.get_agent_context("Architect"),
            self._execute_architect_planning,
        )

        if not self.state.architect_design:
            self.state.architect_design = self.architect.last_output

        await self._publish("SYSTEM", "PLANNING", "summary", "done", "Architect design published to all agents.")

    async def _run_architecture_review_phase(self) -> None:
        await self.run_agent_safe(
            self.pm,
            "PM",
            self.get_agent_context("PM"),
            self._execute_pm_architecture_review,
        )

        await asyncio.gather(
            *(
                self.run_agent_safe(
                    self._agents_by_name[name],
                    name,
                    self.get_agent_context(name),
                    self._make_stream_runner(name, "ARCHITECTURE"),
                )
                for name in ["Architect", "Backend", "Frontend", "Tester", "Docs"]
            )
        )

        if not self.state.requirement_change:
            update = await self.pm.maybe_inject_requirement_change("ARCHITECTURE", self.state.task)
            if update:
                self.state.requirement_change = update
                await self.backend.react(
                    phase="ARCHITECTURE_REVIEW",
                    incoming=SwarmMessage(
                        source="PM",
                        target="Backend",
                        phase="ARCHITECTURE_REVIEW",
                        kind="requirement_change",
                        status="arguing",
                        text=update,
                    ),
                )
                await self.frontend.react(
                    phase="ARCHITECTURE_REVIEW",
                    incoming=SwarmMessage(
                        source="PM",
                        target="Frontend",
                        phase="ARCHITECTURE_REVIEW",
                        kind="requirement_change",
                        status="arguing",
                        text=update,
                    ),
                )

    async def _run_implementation_phase(self) -> None:
        await self.run_agent_safe(
            self.backend,
            "Backend",
            self.get_agent_context("Backend"),
            self._make_stream_runner("Backend", "IMPLEMENTATION"),
        )

        endpoints, schemas = self._extract_backend_contracts(self.backend.generated_files, self.state.architect_design)
        self.state.backend_endpoints = endpoints
        self.state.backend_schemas = schemas

        await self._publish(
            "Backend",
            "IMPLEMENTATION",
            "summary",
            "working",
            f"Published backend contracts: {len(endpoints)} endpoints, {len(schemas)} schemas.",
        )

        await self.run_agent_safe(
            self.frontend,
            "Frontend",
            self.get_agent_context("Frontend"),
            self._make_stream_runner("Frontend", "IMPLEMENTATION"),
        )

        if not self.state.requirement_change:
            update = await self.pm.maybe_inject_requirement_change("IMPLEMENTATION", self.state.task)
            if update:
                self.state.requirement_change = update
                await self._publish(
                    "SYSTEM",
                    "IMPLEMENTATION",
                    "summary",
                    "arguing",
                    "Requirement update injected. Re-running impacted implementation agents.",
                )
                await self.run_agent_safe(
                    self.backend,
                    "Backend",
                    self.get_agent_context("Backend"),
                    self._make_stream_runner("Backend", "IMPLEMENTATION"),
                )
                await self.run_agent_safe(
                    self.frontend,
                    "Frontend",
                    self.get_agent_context("Frontend"),
                    self._make_stream_runner("Frontend", "IMPLEMENTATION"),
                )

        self.state.all_generated_files.update(self.state.agent_files.get("Backend", {}))
        self.state.all_generated_files.update(self.state.agent_files.get("Frontend", {}))
        self.state.loc_generated = self._count_loc(self.state.all_generated_files)

    async def _run_testing_phase(self) -> None:
        await self._write_output(include_docs=False)

        max_phase_retries = 2
        retries = 0
        while retries <= max_phase_retries:
            await self.run_agent_safe(
                self.tester,
                "Tester",
                self.get_agent_context("Tester"),
                self._execute_tester_quality_gate,
            )

            await self.run_agent_safe(
                self.pm,
                "PM",
                self.get_agent_context("PM"),
                self._execute_pm_triage,
            )

            self.state.tester_signed_off = bool(getattr(self.tester, "sign_off", False))
            if self.state.tester_signed_off:
                return

            blocking = [
                bug for bug in self.state.bugs_found if str(bug.get("severity", "")).lower() in {"critical", "high"}
            ]
            for bug in blocking:
                assigned_to = str(bug.get("assigned_to", "Backend"))
                if assigned_to not in {"Backend", "Frontend"}:
                    assigned_to = "Backend"
                agent = self._agents_by_name[assigned_to]
                bug_id = str(bug.get("id", "BUG"))

                for _ in range(3):
                    await self.run_agent_safe(
                        agent,
                        assigned_to,
                        {**self.get_agent_context(assigned_to), "bug": bug},
                        self._make_stream_runner(assigned_to, "TESTING"),
                    )
                    if bug_id and bug_id not in self.state.bugs_fixed:
                        self.state.bugs_fixed.append(bug_id)

                    await self._write_output(include_docs=False)
                    await self.run_agent_safe(
                        self.tester,
                        "Tester",
                        self.get_agent_context("Tester"),
                        self._execute_tester_quality_gate,
                    )

                    still_blocking = any(
                        str(item.get("id", "")) == bug_id
                        and str(item.get("severity", "")).lower() in {"critical", "high"}
                        for item in self.state.bugs_found
                    )
                    if not still_blocking:
                        break

            self.state.tester_signed_off = bool(getattr(self.tester, "sign_off", False))
            if self.state.tester_signed_off:
                return

            retries += 1
            if retries <= max_phase_retries:
                await self._publish(
                    "SYSTEM",
                    "TESTING",
                    "summary",
                    "arguing",
                    f"Tester did not sign off. Retrying TESTING phase ({retries}/{max_phase_retries}).",
                )

        await self._publish(
            "SYSTEM",
            "TESTING",
            "warning",
            "arguing",
            "Max TESTING retries reached. Forcing continuation with warnings.",
        )

    async def _run_documentation_phase(self) -> None:
        await self.run_agent_safe(
            self.docs,
            "Docs",
            self.get_agent_context("Docs"),
            self._execute_docs_generation,
        )
        self.state.all_generated_files.update(self.state.docs_files)
        self.state.loc_generated = self._count_loc(self.state.all_generated_files)

    async def _run_packaging_phase(self) -> None:
        await self._write_output(include_docs=True)

        await self.run_agent_safe(
            self.pm,
            "PM",
            self.get_agent_context("PM"),
            self._execute_pm_packaging_review,
        )

        startup_ok, error_text = await self._quick_startup_check(self.output_dir)
        if not startup_ok:
            await self._publish(
                "SYSTEM",
                "PACKAGING",
                "warning",
                "arguing",
                f"Startup verification warning: {error_text}",
            )

    async def _run_complete_phase(self) -> None:
        summary = self._build_summary(time.monotonic() - self._phase_started_at)
        await self._publish("SYSTEM", "COMPLETE", "summary", "done", "Build complete.")
        await self._publish("SYSTEM", "COMPLETE", "summary", "done", f"Output: {self.output_dir}")
        await self._publish(
            "SYSTEM",
            "COMPLETE",
            "summary",
            "done",
            f"Time: {self.get_elapsed_str()} | Cost: ${self.state.total_cost:.4f} | Coverage: {summary['test_coverage']}%",
        )

    def get_agent_context(self, agent_name: str) -> Dict[str, Any]:
        base_context: Dict[str, Any] = {
            "task": self.state.task,
            "product_prd": self.state.product_prd,
            "architect_design": self.state.architect_design,
            "requirement_change": self.state.requirement_change,
        }

        extra: Dict[str, Dict[str, Any]] = {
            "Architect": {},
            "Backend": {
                "architect_design": self.state.architect_design,
            },
            "Frontend": {
                "backend_endpoints": self.state.backend_endpoints,
                "backend_schemas": self.state.backend_schemas,
                "implementation_summary": self.state.implementation_summary,
            },
            "Tester": {
                "output_dir": self.output_dir,
                "backend_endpoints": self.state.backend_endpoints,
                "all_generated_files": self.state.all_generated_files,
            },
            "Docs": {
                "output_dir": self.output_dir,
                "all_generated_files": self.state.all_generated_files,
                "coverage_report": self.state.coverage_report,
                "bugs_found": self.state.bugs_found,
                "bugs_fixed": self.state.bugs_fixed,
                "elapsed_time": self.get_elapsed_str(),
                "api_calls_made": self.state.api_calls_made,
                "loc_generated": self.state.loc_generated,
            },
            "PM": {
                "current_phase": self.state.current_phase,
                "agent_outputs": self.state.agent_outputs,
                "bugs_found": self.state.bugs_found,
                "tester_verdict": self.state.tester_verdict,
                "coverage_report": self.state.coverage_report,
            },
        }
        return {**base_context, **extra.get(agent_name, {})}

    async def collect_agent_output(self, agent: BaseAgent, agent_name: str) -> None:
        self.state.agent_outputs[agent_name] = agent.last_output

        if hasattr(agent, "generated_files"):
            files = getattr(agent, "generated_files") or {}
            if isinstance(files, dict):
                self.state.agent_files[agent_name] = files
                self.state.all_generated_files.update(files)

        if agent_name == "Architect":
            self.state.architect_design = agent.last_output

        elif agent_name == "Backend":
            if hasattr(agent, "implementation_summary"):
                self.state.implementation_summary = str(getattr(agent, "implementation_summary", ""))
            extracted_endpoints, extracted_schemas = self._extract_backend_contracts(
                self.state.agent_files.get("Backend", {}), self.state.architect_design
            )
            self.state.backend_endpoints = extracted_endpoints
            self.state.backend_schemas = extracted_schemas

        elif agent_name == "Frontend":
            if hasattr(agent, "ui_summary"):
                self.state.ui_summary = str(getattr(agent, "ui_summary", ""))
            if hasattr(agent, "complexity_level"):
                self.state.ui_complexity = str(getattr(agent, "complexity_level", "MEDIUM"))

        elif agent_name == "Tester":
            self.state.bugs_found = list(getattr(agent, "bugs_found", []))
            self.state.bugs_fixed = list(getattr(agent, "bugs_fixed", []))
            self.state.coverage_report = dict(getattr(agent, "coverage_report", {}))
            self.state.tester_verdict = str(getattr(agent, "verdict", "PENDING"))
            self.state.tester_signed_off = bool(getattr(agent, "sign_off", False))

        elif agent_name == "PM":
            self.state.product_prd = str(getattr(agent, "prd", self.state.product_prd))
            self.state.requirement_change = str(getattr(agent, "requirement_change", self.state.requirement_change))
            self.state.pm_scorecard = dict(getattr(agent, "scorecard", {}))
            self.state.pm_ship_decision = str(getattr(agent, "ship_decision", self.state.pm_ship_decision))
            self.state.pm_delivery_summary = str(
                getattr(agent, "delivery_summary", self.state.pm_delivery_summary)
            )

        elif agent_name == "Docs":
            docs_files = getattr(agent, "generated_files", {})
            if isinstance(docs_files, dict):
                self.state.docs_files = docs_files
                self.state.all_generated_files.update(docs_files)

        self.state.loc_generated = self._count_loc(self.state.all_generated_files)
        await self._sync_agent_usage(agent_name)

    async def track_llm_call(self, response: LLMResponse) -> None:
        self.state.api_calls_made += 1
        self.state.total_input_tokens += int(response.input_tokens)
        self.state.total_output_tokens += int(response.output_tokens)
        self.state.total_cost += float(response.cost_estimate)
        await self._publish_metrics()

    async def run_agent_safe(
        self,
        agent: BaseAgent,
        agent_name: str,
        context: Dict[str, Any],
        runner: AgentRunner,
    ) -> bool:
        try:
            lines = await runner(context)
            if lines:
                await agent.publish_lines(phase=self.state.current_phase, lines=lines, status="working")
            await self.collect_agent_output(agent, agent_name)
            return True
        except Exception as exc:
            await self._publish(
                "SYSTEM",
                self.state.current_phase,
                "error",
                "arguing",
                f"{agent_name} encountered an error: {exc}",
            )
            if hasattr(agent, "get_fallback_output"):
                try:
                    agent.last_output = str(agent.get_fallback_output(context))
                    await self.collect_agent_output(agent, agent_name)
                except Exception:
                    pass
            return False

    async def complete_phase(self, phase: str, summary: str) -> None:
        self.state.phase_history.append(
            {
                "phase": phase,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": self.get_phase_duration(),
                "summary": summary,
            }
        )
        await self._publish_phase(phase, summary)
        await self._publish(
            "SYSTEM",
            phase,
            "phase_complete",
            "done",
            f"Phase {self.get_phase_number(phase)}/7 [{phase}] complete. {summary}",
        )

    async def persist_phase(self, phase: str) -> None:
        if self._db is None:
            return
        await self._db.execute(
            """
            INSERT INTO phases (phase, progress, total, timestamp, state_json, build_id, summary)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                phase,
                self.get_phase_number(phase),
                7,
                time.time(),
                json.dumps(self._serialize_state()),
                self._build_id,
                self.state.phase_history[-1]["summary"] if self.state.phase_history else "",
            ),
        )
        await self._db.commit()

    async def persist_message(self, message: SwarmMessage) -> None:
        if self._db is None:
            return
        await self._db.execute(
            """
            INSERT INTO messages (source, target, phase, kind, status, text, timestamp, build_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message.source,
                message.target,
                self.state.current_phase,
                message.kind,
                message.status,
                message.text,
                message.timestamp,
                self._build_id,
            ),
        )
        await self._db.commit()

    async def replay_history(self, build_id: str) -> List[Dict[str, Any]]:
        if self._db is None:
            return []
        cursor = await self._db.execute(
            "SELECT source, target, phase, kind, status, text, timestamp FROM messages WHERE build_id = ? ORDER BY id ASC",
            (build_id,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "source": row[0],
                "target": row[1],
                "phase": row[2],
                "kind": row[3],
                "status": row[4],
                "text": row[5],
                "timestamp": row[6],
            }
            for row in rows
        ]

    def get_elapsed_str(self) -> str:
        elapsed = datetime.now(timezone.utc) - self.state.started_at
        total_seconds = int(elapsed.total_seconds())
        mins, secs = divmod(total_seconds, 60)
        hours, mins = divmod(mins, 60)
        if hours:
            return f"{hours:02d}:{mins:02d}:{secs:02d}"
        return f"{mins:02d}:{secs:02d}"

    def get_phase_duration(self) -> int:
        return int(time.monotonic() - self._phase_started_at)

    def get_phase_number(self, phase: str) -> int:
        try:
            return self.PHASES.index(phase) + 1
        except ValueError:
            return 0

    async def _execute_pm_planning(self, _context: Dict[str, Any]) -> List[str]:
        return await self.pm.run_planning(
            task=self.state.task,
            project_name=self.state.project_name,
            agent_messages=list(self.state.agent_outputs.values()),
        )

    async def _execute_pm_architecture_review(self, _context: Dict[str, Any]) -> List[str]:
        return await self.pm.review_architecture(self.state.architect_design)

    async def _execute_pm_triage(self, _context: Dict[str, Any]) -> List[str]:
        return self.pm.triage_bugs(self.state.bugs_found)

    async def _execute_pm_packaging_review(self, _context: Dict[str, Any]) -> List[str]:
        score_lines = self.pm.pre_ship_review(
            tester_report=self.state.coverage_report,
            tester_verdict=self.state.tester_verdict,
            backend_generated_files=self.state.agent_files.get("Backend", {}),
            frontend_generated_files=self.state.agent_files.get("Frontend", {}),
            docs_generated_files=self.state.docs_files,
        )
        delivery = self.pm.build_delivery_summary(
            project_name=self.state.project_slug,
            elapsed_text=self.get_elapsed_str(),
            tester_report=self.state.coverage_report,
        )
        return score_lines + [line for line in delivery.splitlines() if line.strip()]

    async def _execute_architect_planning(self, context: Dict[str, Any]) -> List[str]:
        return await self.architect.stream_phase_lines(
            phase="PLANNING",
            task=self.state.task,
            context=context,
            max_lines=60,
        )

    async def _execute_tester_quality_gate(self, _context: Dict[str, Any]) -> List[str]:
        frontend_calls = self.tester.frontend_calls_from_files(self.state.agent_files.get("Frontend", {}))
        endpoint_lines = [f"{i.get('method', 'GET')} {i.get('path', '')}" for i in self.state.backend_endpoints]
        return await self.tester.run_quality_gate(
            task=self.state.task,
            architect_design=self.state.architect_design,
            output_dir=self.output_dir,
            backend_endpoints=endpoint_lines,
            frontend_calls=frontend_calls,
            all_generated_files=self.state.all_generated_files,
        )

    async def _execute_docs_generation(self, _context: Dict[str, Any]) -> List[str]:
        return await self.docs.run_documentation(
            task=self.state.task,
            architect_design=self.state.architect_design,
            output_dir=self.output_dir,
            all_generated_files=self.state.all_generated_files,
            tester_coverage_report=self.state.coverage_report,
            tester_bugs_found=self.state.bugs_found,
            tester_bugs_fixed=self.state.bugs_fixed,
            elapsed_time=self.get_elapsed_str(),
            api_calls_made=self.state.api_calls_made,
        )

    def _make_stream_runner(self, name: str, phase: str) -> AgentRunner:
        agent = self._agents_by_name[name]

        async def runner(context: Dict[str, Any]) -> List[str]:
            return await agent.stream_phase_lines(phase=phase, task=self.state.task, context=context, max_lines=8)

        return runner

    async def _write_output(self, include_docs: bool) -> Path:
        elapsed = int((datetime.now(timezone.utc) - self.state.started_at).total_seconds())
        path = await self.writer.write_project(
            task=self.state.task,
            project_name=self.state.project_slug,
            agent_outputs=self.state.agent_outputs,
            build_messages=[entry["summary"] for entry in self.state.phase_history],
            elapsed_seconds=elapsed,
            api_calls=self.state.api_calls_made,
            frontend_generated_files=self.state.agent_files.get("Frontend", {}),
            docs_generated_files=self.state.docs_files if include_docs else {},
            tester_coverage_report=self.state.coverage_report,
            tester_bugs=self.state.bugs_found,
            tester_verdict=self.state.tester_verdict,
            pm_delivery_summary=self.state.pm_delivery_summary,
            pm_scorecard=self.state.pm_scorecard,
            pm_requirement_change=self.state.requirement_change,
        )
        self.output_dir = path
        return path

    async def _quick_startup_check(self, target_dir: Path) -> Tuple[bool, str]:
        if not (target_dir / "app" / "main.py").exists():
            return False, "app/main.py missing"
        proc = None
        try:
            proc = subprocess.Popen(
                [
                    "python",
                    "-m",
                    "uvicorn",
                    "app.main:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "8779",
                    "--log-level",
                    "error",
                ],
                cwd=target_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            await asyncio.sleep(2)
            if proc.poll() is not None:
                stderr = (proc.stderr.read() if proc.stderr else "")[:500]
                return False, f"startup process exited early: {stderr}"
            return True, "ok"
        except Exception as exc:
            return False, str(exc)
        finally:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()

    async def _sync_agent_usage(self, agent_name: str) -> None:
        agent = self._agents_by_name[agent_name]
        seen = self._usage_seen[agent_name]

        current_calls = float(getattr(agent, "llm_calls", 0))
        current_input = float(getattr(agent, "input_tokens", 0))
        current_output = float(getattr(agent, "output_tokens", 0))
        current_cost = float(getattr(agent, "total_cost", 0.0))

        delta_calls = int(current_calls - seen["llm_calls"])
        delta_input = int(current_input - seen["input_tokens"])
        delta_output = int(current_output - seen["output_tokens"])
        delta_cost = float(current_cost - seen["total_cost"])

        if delta_calls > 0:
            self.state.api_calls_made += delta_calls
            self.state.total_input_tokens += max(delta_input, 0)
            self.state.total_output_tokens += max(delta_output, 0)
            self.state.total_cost += max(delta_cost, 0.0)
            await self._publish_metrics()

        seen["llm_calls"] = current_calls
        seen["input_tokens"] = current_input
        seen["output_tokens"] = current_output
        seen["total_cost"] = current_cost

    async def _publish_metrics(self) -> None:
        await self._publish(
            "SYSTEM",
            self.state.current_phase,
            "metrics",
            "working",
            (
                f"Metrics => api_calls={self.state.api_calls_made} "
                f"tokens={self.state.total_input_tokens + self.state.total_output_tokens} "
                f"cost=${self.state.total_cost:.4f} provider={self.state.provider_name}"
            ),
        )

    async def _publish_phase(self, phase: str, summary: str) -> None:
        await self._publish(
            "SYSTEM",
            phase,
            "phase",
            "done",
            f"Phase update {self.get_phase_number(phase)}/7: {phase} | duration={self.get_phase_duration()}s | {summary}",
        )

    async def _publish(self, source: str, phase: str, kind: str, status: str, text: str) -> None:
        await self.bus.publish(
            SwarmMessage(
                source=source,
                target="all",
                phase=phase,
                kind=kind,
                status=status,
                text=text,
            )
        )

    async def _on_bus_message(self, message: SwarmMessage) -> None:
        await self.persist_message(message)

    async def _init_persistence_tables(self) -> None:
        self._db = await aiosqlite.connect(self.settings.database_path.as_posix())

        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS builds (
                build_id TEXT PRIMARY KEY,
                task TEXT NOT NULL,
                project_slug TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                summary_json TEXT
            )
            """
        )
        await self._add_column_if_missing("messages", "build_id", "TEXT")
        await self._add_column_if_missing("phases", "state_json", "TEXT")
        await self._add_column_if_missing("phases", "build_id", "TEXT")
        await self._add_column_if_missing("phases", "summary", "TEXT")
        await self._db.commit()

    async def _add_column_if_missing(self, table: str, column: str, column_type: str) -> None:
        if self._db is None:
            return
        cursor = await self._db.execute(f"PRAGMA table_info({table})")
        cols = await cursor.fetchall()
        existing = {str(row[1]).lower() for row in cols}
        if column.lower() in existing:
            return
        await self._db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")

    async def _start_or_resume_build(self) -> None:
        if self._db is None:
            self._build_id = str(uuid.uuid4())
            return

        cursor = await self._db.execute(
            """
            SELECT build_id FROM builds
            WHERE task = ? AND project_slug = ? AND status = 'RUNNING'
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (self.state.task, self.state.project_slug),
        )
        row = await cursor.fetchone()

        if row:
            self._build_id = str(row[0])
            await self._try_resume_state()
            await self._publish(
                "SYSTEM",
                "BOOT",
                "system",
                "working",
                f"Resumed interrupted build {self._build_id} from phase index {self._next_phase_index + 1}.",
            )
            return

        self._build_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            """
            INSERT INTO builds (build_id, task, project_slug, status, started_at, updated_at, summary_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (self._build_id, self.state.task, self.state.project_slug, "RUNNING", now, now, ""),
        )
        await self._db.commit()

    async def _try_resume_state(self) -> None:
        if self._db is None or not self._build_id:
            return
        cursor = await self._db.execute(
            """
            SELECT phase, state_json
            FROM phases
            WHERE build_id = ? AND state_json IS NOT NULL
            ORDER BY id DESC
            LIMIT 1
            """,
            (self._build_id,),
        )
        row = await cursor.fetchone()
        if not row:
            self._next_phase_index = 0
            return

        phase = str(row[0])
        state_json = row[1]
        if state_json:
            try:
                self._hydrate_state(json.loads(state_json))
            except Exception:
                pass

        phase_number = self.get_phase_number(phase)
        self._next_phase_index = min(phase_number, len(self.PHASES) - 1)

    async def _update_build_row(self, status: str, summary: Optional[Dict[str, Any]] = None) -> None:
        if self._db is None or not self._build_id:
            return
        await self._db.execute(
            """
            UPDATE builds
            SET status = ?, updated_at = ?, summary_json = ?
            WHERE build_id = ?
            """,
            (status, datetime.now(timezone.utc).isoformat(), json.dumps(summary or {}), self._build_id),
        )
        await self._db.commit()

    def _serialize_state(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        for field_name in BuildState.__dataclass_fields__.keys():
            if field_name == "provider":
                continue
            value = getattr(self.state, field_name)
            if isinstance(value, datetime):
                payload[field_name] = value.isoformat()
            else:
                payload[field_name] = value
        payload["provider_model"] = getattr(self.provider, "model", self.settings.model)
        return payload

    def _hydrate_state(self, payload: Dict[str, Any]) -> None:
        for field_name in BuildState.__dataclass_fields__.keys():
            if field_name == "provider":
                continue
            if field_name not in payload:
                continue
            if field_name == "started_at":
                try:
                    setattr(self.state, field_name, datetime.fromisoformat(str(payload[field_name])))
                except Exception:
                    continue
            else:
                setattr(self.state, field_name, payload[field_name])
        self.state.provider = self.provider

    def _build_summary(self, duration_seconds: float) -> Dict[str, Any]:
        return {
            "project": self.state.project_slug,
            "output_path": str(self.output_dir),
            "provider": self.state.provider_name,
            "model": getattr(self.provider, "model", self.settings.model),
            "elapsed_seconds": int(duration_seconds),
            "api_calls": self.state.api_calls_made,
            "total_tokens": self.state.total_input_tokens + self.state.total_output_tokens,
            "estimated_cost": f"${self.state.total_cost:.4f}",
            "loc_generated": self.state.loc_generated,
            "files_generated": len(self.state.all_generated_files),
            "bugs_found": len(self.state.bugs_found),
            "bugs_fixed": len(self.state.bugs_fixed),
            "test_coverage": self.state.coverage_report.get("coverage_percentage", 0),
            "tester_verdict": self.state.tester_verdict,
            "pm_grade": self.state.pm_scorecard.get("grade", "N/A"),
            "ship_decision": self.state.pm_ship_decision,
        }

    def _extract_backend_contracts(
        self,
        generated_files: Dict[str, str],
        architect_design: str,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        endpoints: List[Dict[str, Any]] = []
        schemas: Dict[str, Any] = {}
        seen = set()

        for content in generated_files.values():
            for method, path in re.findall(
                r"@(?:router|app)\.(get|post|put|patch|delete)\(\s*\"([^\"]+)\"",
                content,
                flags=re.IGNORECASE,
            ):
                key = (method.upper(), path)
                if key in seen:
                    continue
                seen.add(key)
                endpoints.append({"method": method.upper(), "path": path})

            for model in re.findall(r"class\s+([A-Za-z_][A-Za-z0-9_]*)\((?:BaseModel|SQLModel)\)", content):
                schemas.setdefault(model, {"source": "generated"})

        for method, path in re.findall(
            r"\[(GET|POST|PUT|PATCH|DELETE)\]\s+(/[^\s]+)",
            architect_design,
            flags=re.IGNORECASE,
        ):
            key = (method.upper(), path)
            if key not in seen:
                seen.add(key)
                endpoints.append({"method": method.upper(), "path": path})

        return endpoints, schemas

    @staticmethod
    def _count_loc(files: Dict[str, str]) -> int:
        return sum(content.count("\n") + 1 for content in files.values() if isinstance(content, str))

    @staticmethod
    def _derive_project_name(task: str) -> str:
        lowered = task.lower().strip()
        lowered = re.sub(r"\b(build|create|make|generate|develop|implement)\b", " ", lowered)
        lowered = re.sub(r"\b(me|a|an|the|with|for|through|using|via|by|to)\b", " ", lowered)
        cleaned = re.sub(r"[^a-z0-9\s-]", " ", lowered)
        words = [word for word in cleaned.split() if word]
        if not words:
            return "generated-project"
        return "-".join(words[:5])

    @staticmethod
    def _slugify(text: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9\-]+", "-", text.strip().lower()).strip("-")
        return slug or "generated-project"

    async def _shutdown(self) -> None:
        for agent in self._agents_by_name.values():
            await agent.aclose()
        await self.writer.aclose()
        if self.provider is not None:
            await self.provider.aclose()
        if self._db is not None:
            await self._db.close()
        await self.bus.close()
