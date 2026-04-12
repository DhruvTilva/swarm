from __future__ import annotations

import asyncio
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from textual import events
from textual.app import App, ComposeResult
from textual.containers import Grid, Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Footer, ProgressBar, RichLog, Static

from core.build_engine import BuildEngine
from core.message_bus import SwarmMessage


@dataclass(frozen=True)
class AgentTheme:
    name: str
    emoji: str
    color: str
    personality_lines: Tuple[str, ...]


AGENT_THEMES: Dict[str, AgentTheme] = {
    "Architect": AgentTheme(
        name="Architect",
        emoji="🏗️",
        color="#00FFFF",
        personality_lines=(
            "I am redesigning the entire data flow. Again.",
            "The monolith must die. We use microservices.",
            "Backend will not like this but I am right.",
            "Drawing system boundaries. This is the real work.",
        ),
    ),
    "Backend": AgentTheme(
        name="Backend",
        emoji="⚙️",
        color="#00FF41",
        personality_lines=(
            "Just write the code. Stop architecting.",
            "I found 3 things Architect over-engineered. Fixing them.",
            "API is live. Tests passing. You are welcome.",
            "Why is PM changing requirements AGAIN.",
        ),
    ),
    "Frontend": AgentTheme(
        name="Frontend",
        emoji="🎨",
        color="#FF006E",
        personality_lines=(
            "Backend API response is a mess. Abstracting it.",
            "Users will not understand this. Redesigning the flow.",
            "Making it beautiful. Performance comes second. Fight me.",
            "PM just added a feature that breaks everything I built.",
        ),
    ),
    "Tester": AgentTheme(
        name="Tester",
        emoji="🧪",
        color="#FFD60A",
        personality_lines=(
            "Found a bug. Backend will not be happy.",
            "Test coverage is 34%. This is unacceptable.",
            "I do not trust code I did not break first.",
            "Running edge cases. Something will fail. It always does.",
        ),
    ),
    "Docs": AgentTheme(
        name="Docs",
        emoji="📚",
        color="#C77DFF",
        personality_lines=(
            "Documenting what was promised, not what was built.",
            "The README contradicts the actual behavior. Again.",
            "Adding examples because the code is not self-explanatory.",
            "Nobody reads docs until something breaks.",
        ),
    ),
    "PM": AgentTheme(
        name="PM",
        emoji="🌀",
        color="#FF6B00",
        personality_lines=(
            "Momentum is great. Adding one small requirement.",
            "Stakeholders want dark mode. Adding it to scope.",
            "This is fine. Everything is fine. Ship it.",
            "I changed the requirements. The team can handle it.",
        ),
    ),
}


ACTIVE_STATUS = {
    "🧠 Thinking...",
    "⚡ Building...",
    "💬 Arguing...",
    "⏳ Waiting...",
}


class AgentPanel(Vertical):
    status_label = reactive("🧠 Thinking...")
    last_action_at = reactive("--:--:--")
    typing = reactive(False)
    heartbeat_on = reactive(True)
    health_state = reactive("alive")

    def __init__(self, theme: AgentTheme) -> None:
        super().__init__(classes=f"agent-panel agent-{theme.name.lower()}")
        self.theme = theme
        self._header_id = f"header-{theme.name.lower()}"
        self._log_id = f"log-{theme.name.lower()}"
        self._meta_id = f"meta-{theme.name.lower()}"
        self._pulsing = False
        self._pulse_phase = 0
        self._typing_phase = 0
        self._argue_task: Optional[asyncio.Task[Any]] = None

    def compose(self) -> ComposeResult:
        yield Static(id=self._header_id, classes="agent-header")
        yield RichLog(id=self._log_id, markup=True, wrap=True, auto_scroll=True, max_lines=600)
        yield Static(id=self._meta_id, classes="agent-meta")

    def on_mount(self) -> None:
        self._refresh_header()
        self._refresh_meta()
        self.set_interval(0.18, self._tick_animation)

    def watch_status_label(self) -> None:
        self._refresh_header()

    def watch_last_action_at(self) -> None:
        self._refresh_meta()

    def watch_typing(self) -> None:
        self._refresh_meta()

    def watch_heartbeat_on(self) -> None:
        self._refresh_header()

    def watch_health_state(self) -> None:
        self._refresh_header()

    def set_status(self, label: str) -> None:
        previous = self.status_label
        self.status_label = label

        if label in ACTIVE_STATUS:
            self.health_state = "alive"
            self._start_pulse()
        elif label == "✅ Done":
            self.health_state = "done"
            self._stop_pulse()
        elif label.startswith("❌"):
            self.health_state = "error"
            self._stop_pulse()
        else:
            self._stop_pulse()

        if label == "💬 Arguing..." and previous != "💬 Arguing...":
            if self._argue_task is None or self._argue_task.done():
                self._argue_task = asyncio.create_task(self._argue_flash())

    def set_typing(self, active: bool) -> None:
        self.typing = active
        if not active:
            self._typing_phase = 0

    def set_timestamp(self, ts_text: str) -> None:
        self.last_action_at = ts_text

    def set_heartbeat(self, on: bool) -> None:
        self.heartbeat_on = on

    def write_line(self, text: str) -> None:
        safe = text.replace("[", "\\[")
        self.query_one(f"#{self._log_id}", RichLog).write(f"[white]{safe}[/white]")

    def add_system_line(self, text: str) -> None:
        safe = text.replace("[", "\\[")
        self.query_one(f"#{self._log_id}", RichLog).write(f"[{self.theme.color}]{safe}[/{self.theme.color}]")

    def _refresh_header(self) -> None:
        dot = self._heartbeat_dot_markup()
        top = (
            f"[bold {self.theme.color}]┏━ {dot} {self.theme.emoji} {self.theme.name.upper()}"
            f" — {self.status_label} ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold {self.theme.color}]"
        )
        self.query_one(f"#{self._header_id}", Static).update(top)

    def _refresh_meta(self) -> None:
        dots = "[dim]   [/dim]"
        if self.typing:
            active = (self._typing_phase % 3) + 1
            dot_parts: List[str] = []
            for idx in range(1, 4):
                if idx <= active:
                    dot_parts.append(f"[bold {self.theme.color}]●[/bold {self.theme.color}]")
                else:
                    dot_parts.append("[dim]●[/dim]")
            dots = " ".join(dot_parts)
        bottom = (
            f"[bold {self.theme.color}]┗━[/bold {self.theme.color}] "
            f"[dim]Last action: {self.last_action_at}[/dim] {dots}"
        )
        self.query_one(f"#{self._meta_id}", Static).update(bottom)

    def _heartbeat_dot_markup(self) -> str:
        if self.health_state == "done":
            return "[bold #00FF41]●[/bold #00FF41]"
        if self.health_state == "error":
            return "[bold #FF3B30]●[/bold #FF3B30]"
        if self.heartbeat_on:
            return f"[bold {self.theme.color}]●[/bold {self.theme.color}]"
        return "[dim]●[/dim]"

    def _tick_animation(self) -> None:
        if self._pulsing:
            self._pulse_phase = (self._pulse_phase + 1) % 2
            if self._pulse_phase == 0:
                self.remove_class("pulse-b")
                self.add_class("pulse-a")
            else:
                self.remove_class("pulse-a")
                self.add_class("pulse-b")

        if self.typing:
            self._typing_phase = (self._typing_phase + 1) % 3
            self._refresh_meta()

    def _start_pulse(self) -> None:
        if self._pulsing:
            return
        self._pulsing = True
        self.add_class("pulse-a")

    def _stop_pulse(self) -> None:
        self._pulsing = False
        self.remove_class("pulse-a")
        self.remove_class("pulse-b")

    async def _argue_flash(self) -> None:
        for _ in range(3):
            self.add_class("argue-flash")
            await asyncio.sleep(0.07)
            self.remove_class("argue-flash")
            await asyncio.sleep(0.07)


class DashboardApp(App):
    CSS_PATH = Path(__file__).with_name("dashboard.tcss")
    CSS = """
    Screen {
        background: #0d1117;
        color: #f0f6fc;
    }
    """
    BINDINGS = [("q", "quit", "Quit")]

    def __init__(self, task: str, engine: BuildEngine) -> None:
        super().__init__()
        print(f"[swarm] DashboardApp init: task={task}", file=sys.stderr)
        self._task = task
        self._engine = engine
        self._css_exists = Path(self.CSS_PATH).exists()
        self._started_at = time.monotonic()
        self._total_phases = len(engine.PHASES)
        self._phase_index = 0
        self._phase_name = "BOOT"
        self._api_call_count = 0
        self._api_call_keys: Set[Tuple[str, str]] = set()
        self._is_complete = False
        self._provider_name = engine.settings.provider

        self._consumer_task: Optional[asyncio.Task[Any]] = None
        self._engine_task: Optional[asyncio.Task[Any]] = None
        self._feed_task: Optional[asyncio.Task[Any]] = None
        self._agent_tasks: List[asyncio.Task[Any]] = []

        self._feed_queue: asyncio.Queue[str] = asyncio.Queue()
        self._agent_queues: Dict[str, asyncio.Queue[SwarmMessage]] = {
            name: asyncio.Queue() for name in AGENT_THEMES
        }
        self._personality_index: Dict[str, int] = {name: 0 for name in AGENT_THEMES}
        self._agent_status: Dict[str, str] = {name: "🧠 Thinking..." for name in AGENT_THEMES}

        self._heartbeat_color = "#58A6FF"
        self._heartbeat_on = True

        self.agent_panels: Dict[str, AgentPanel] = {
            name: AgentPanel(theme) for name, theme in AGENT_THEMES.items()
        }

    def compose(self) -> ComposeResult:
        try:
            with Vertical(id="main-root"):
                yield Static(id="top-bar")

                with Grid(id="main-grid"):
                    for panel in self.agent_panels.values():
                        yield panel

                with Vertical(id="feed-wrap"):
                    yield Static(
                        "[bold #FFD60A]┏━ INTER-AGENT FEED ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold #FFD60A]",
                        id="feed-title",
                    )
                    yield RichLog(id="feed-log", markup=True, wrap=True, auto_scroll=True, max_lines=180)
                    yield Static(
                        "[bold #FFD60A]┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold #FFD60A]",
                        id="feed-bottom",
                    )

                with Horizontal(id="phase-wrap"):
                    yield Static(id="phase-left")
                    yield ProgressBar(total=self._total_phases, show_eta=False, show_percentage=False, id="phase-progress")
                    yield Static(id="phase-right")

                yield Footer()

            # Scanlines are decorative only and can mask rendering if startup fails.
            yield Static(id="scanlines")
        except Exception as exc:
            self._log_runtime_error("compose", exc)
            yield Static(f"COMPOSE ERROR: {exc}", id="compose-error")
            yield Footer()

    def on_ready(self) -> None:
        self.notify("Swarm dashboard loaded successfully")

    async def on_mount(self) -> None:
        try:
            if not self._css_exists:
                self._log_runtime_error("css", FileNotFoundError(f"Missing CSS: {self.CSS_PATH}"))
                self.notify("dashboard.tcss missing, running with fallback CSS", severity="warning")

            # Render immediately; no blocking boot animation before first paint.
            self._set_phase_glow(self._phase_name)
            self._update_top_bar()
            self._update_bottom_bar()

            self._render_scanlines()

            self.set_interval(1.0, self._heartbeat_tick)
            self.set_interval(0.5, self._sync_heartbeat)
            self.set_interval(1.0, self._tick)

            self._feed_task = asyncio.create_task(self._feed_worker())
            for agent_name in AGENT_THEMES:
                self._agent_tasks.append(asyncio.create_task(self._agent_worker(agent_name)))

            self._consumer_task = asyncio.create_task(self._consume_messages())
            self._engine_task = asyncio.create_task(self._run_engine())
        except Exception as exc:
            self._log_runtime_error("on_mount", exc)
            self._show_fatal_screen(f"Mount error: {exc}")

    async def on_unmount(self) -> None:
        tasks: List[asyncio.Task[Any]] = []
        for task in [self._consumer_task, self._engine_task, self._feed_task, *self._agent_tasks]:
            if task is not None and not task.done():
                task.cancel()
                tasks.append(task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def on_resize(self, _: events.Resize) -> None:
        self._render_scanlines()

    async def _boot_sequence(self) -> None:
        boot_log = self.query_one("#boot-log", RichLog)
        boot_lines = [
            "[bold #FFD60A]🐝 SWARM INITIALIZING...[/bold #FFD60A]",
            "[bold #58A6FF]▓▓▓▓▓▓▓▓▓▓ AGENTS LOADING...[/bold #58A6FF]",
            "[bold #C77DFF]⚡ DEPLOYING ARCHITECT... BACKEND... FRONTEND...[/bold #C77DFF]",
            "[bold #00FF41]🔥 ALL SYSTEMS ONLINE. BEGINNING BUILD.[/bold #00FF41]",
        ]

        for line in boot_lines:
            boot_log.write(line)
            await asyncio.sleep(0.48)

        await asyncio.sleep(0.22)

    async def _run_engine(self) -> None:
        try:
            await self._engine.run()
        except Exception as exc:
            self._log_runtime_error("_run_engine", exc)
            safe_error = str(exc).replace("[", "\\[")
            await self._feed_queue.put(
                f"[bold red]✅ SYSTEM:[/bold red] Build failed: {safe_error}"
            )

    async def _consume_messages(self) -> None:
        try:
            async for message in self._engine.bus.stream():
                self.call_later(self._route_message, message)
        except Exception as exc:
            self._log_runtime_error("_consume_messages", exc)
            self._show_fatal_screen(f"Message stream error: {exc}")

    def _route_message(self, message: SwarmMessage) -> None:
        if message.kind == "phase":
            self._apply_phase_message(message)

        if message.source in self._agent_queues:
            self._agent_queues[message.source].put_nowait(message)

        self._feed_queue.put_nowait(self._format_feed_line(message))

        if message.phase == "COMPLETE" and message.kind == "summary":
            self.call_later(self._mark_complete, message.text)

    def _apply_phase_message(self, message: SwarmMessage) -> None:
        parsed = re.search(r"Phase\s+(\d+)/(\d+):\s+([A-Z_]+)", message.text)
        if parsed:
            self._phase_index = int(parsed.group(1))
            self._total_phases = int(parsed.group(2))
            self._phase_name = parsed.group(3)
            progress = self.query_one("#phase-progress", ProgressBar)
            progress.update(total=self._total_phases, progress=self._phase_index)
            self._set_phase_glow(self._phase_name)
            self._update_bottom_bar()

    async def _agent_worker(self, agent_name: str) -> None:
        queue = self._agent_queues[agent_name]
        panel = self.agent_panels[agent_name]

        while True:
            message = await queue.get()
            status = self._status_for_message(message)
            panel.set_status(status)
            panel.set_typing(True)
            panel.set_timestamp(time.strftime("%H:%M:%S", time.localtime(message.timestamp)))

            self._agent_status[agent_name] = status
            self._track_api_calls(message)

            lines = self._split_stream_lines(message.text)
            for line in lines:
                panel.write_line(line)
                await asyncio.sleep(0.05)

            if "llm unavailable" in message.text.lower():
                personality = self._next_personality_line(agent_name)
                panel.add_system_line(f"↳ {personality}")
                await self._feed_queue.put(
                    self._format_feed_line(
                        SwarmMessage(
                            source=agent_name,
                            target="all",
                            phase=message.phase,
                            kind="fallback",
                            status="waiting",
                            text=personality,
                        )
                    )
                )

            panel.set_typing(False)
            self._update_top_bar()
            self._update_bottom_bar()

    async def _feed_worker(self) -> None:
        log = self.query_one("#feed-log", RichLog)
        while True:
            line = await self._feed_queue.get()
            log.write(f"[dim]▸[/dim] {line}")
            await asyncio.sleep(0.025)
            log.write(line)
            await asyncio.sleep(0.02)

    def _status_for_message(self, message: SwarmMessage) -> str:
        text_low = message.text.lower()
        if "rate limit" in text_low or "too many requests" in text_low:
            return "⏳ Waiting..."
        if "error" in text_low and "llm unavailable" not in text_low:
            return "❌ Error"
        if message.kind in {"reaction", "reaction_prompt"} or message.status == "arguing":
            return "💬 Arguing..."
        if message.phase == "PLANNING" or message.status == "thinking":
            return "🧠 Thinking..."
        if message.phase == "COMPLETE" or message.status in {"done", "complete"}:
            return "✅ Done"
        return "⚡ Building..."

    def _track_api_calls(self, message: SwarmMessage) -> None:
        if message.source not in AGENT_THEMES:
            return
        key = (message.source, message.phase)
        if key not in self._api_call_keys:
            self._api_call_keys.add(key)
            self._api_call_count = len(self._api_call_keys)

    def _next_personality_line(self, agent_name: str) -> str:
        lines = AGENT_THEMES[agent_name].personality_lines
        idx = self._personality_index[agent_name] % len(lines)
        self._personality_index[agent_name] += 1
        return lines[idx]

    def _format_feed_line(self, message: SwarmMessage) -> str:
        ts = time.strftime("%H:%M:%S", time.localtime(message.timestamp))
        src = self._styled_actor(message.source)
        target = self._styled_actor(message.target)
        safe_text = message.text.replace("[", "\\[")
        return f"[dim][{ts}][/dim] {src} → {target}: \"[white]{safe_text}[/white]\""

    def _styled_actor(self, actor: str) -> str:
        theme = AGENT_THEMES.get(actor)
        if theme is None:
            if actor.lower() == "all":
                return "[dim]ALL[/dim]"
            safe_actor = actor.replace("[", "\\[")
            return f"[white]{safe_actor}[/white]"
        return f"[bold {theme.color}]{theme.emoji} {theme.name}[/bold {theme.color}]"

    @staticmethod
    def _split_stream_lines(text: str) -> List[str]:
        chunks = re.split(r"\r?\n+", text)
        clean = [chunk.strip() for chunk in chunks if chunk.strip()]
        return clean or [text]

    def _tick(self) -> None:
        self._update_top_bar()
        self._update_bottom_bar()

    def _heartbeat_tick(self) -> None:
        self._heartbeat_on = not self._heartbeat_on
        for panel in self.agent_panels.values():
            panel.set_heartbeat(self._heartbeat_on)
        self._update_top_bar()

    def _sync_heartbeat(self) -> None:
        leader = self._pick_heartbeat_leader()
        if leader is not None:
            self._heartbeat_color = AGENT_THEMES[leader].color
            if self._heartbeat_on:
                self.add_class("heartbeat-a")
                self.remove_class("heartbeat-b")
            else:
                self.add_class("heartbeat-b")
                self.remove_class("heartbeat-a")
        else:
            self._heartbeat_color = "#58A6FF"
            self.remove_class("heartbeat-a")
            self.remove_class("heartbeat-b")

        self._update_top_bar()

    def _pick_heartbeat_leader(self) -> Optional[str]:
        priority = ["💬 Arguing...", "🧠 Thinking...", "⚡ Building...", "⏳ Waiting..."]
        for state in priority:
            for name in ["Architect", "Backend", "Frontend", "Tester", "Docs", "PM"]:
                if self._agent_status.get(name) == state:
                    return name
        return None

    def _update_top_bar(self) -> None:
        elapsed = int(time.monotonic() - self._started_at)
        api_calls = int(getattr(self._engine, "api_calls", self._api_call_count))
        task = self._task.strip()
        task_short = task if len(task) <= 58 else f"{task[:55]}..."
        safe_task = task_short.replace("[", "\\[")

        glow_color = self._heartbeat_color if self._heartbeat_on else "#8B949E"

        if self._is_complete:
            text = (
                "[bold #FFD60A]🐝 SWARM — BUILD COMPLETE 🎉[/bold #FFD60A]"
                f"   [dim]⏱️ {elapsed}s[/dim]   "
                f"[bold {glow_color}]🔥 API calls: {api_calls}[/bold {glow_color}]"
                f"   [dim]🔌 {self._provider_name}[/dim]"
            )
        else:
            text = (
                "[bold #FFD60A]🐝 SWARM[/bold #FFD60A]"
                f"   [white]{safe_task}[/white]"
                f"   [dim]⏱️ {elapsed}s[/dim]"
                f"   [bold {glow_color}]🔥 API calls: {api_calls}[/bold {glow_color}]"
                f"   [dim]🔌 {self._provider_name}[/dim]"
            )
        self.query_one("#top-bar", Static).update(text)

    def _update_bottom_bar(self) -> None:
        elapsed = max(1, int(time.monotonic() - self._started_at))
        api_calls = int(getattr(self._engine, "api_calls", self._api_call_count))
        phase_text = f"Phase {self._phase_index}/{self._total_phases}: {self._phase_name}"

        percent = int((self._phase_index / max(1, self._total_phases)) * 100)
        eta = "--"
        if self._phase_index > 0 and self._phase_index < self._total_phases:
            estimate = int((elapsed / self._phase_index) * (self._total_phases - self._phase_index))
            eta = f"{estimate}s"
        elif self._phase_index >= self._total_phases:
            eta = "0s"

        total_tokens = int(getattr(self._engine, "total_input_tokens", 0) + getattr(self._engine, "total_output_tokens", 0))
        cost_text = self._format_cost_text()

        self.query_one("#phase-left", Static).update(f"[bold #58A6FF]{phase_text}[/bold #58A6FF]")
        self.query_one("#phase-right", Static).update(
            (
                f"[bold #58A6FF]{percent}%[/bold #58A6FF] "
                f"[dim]ETA {eta} | API Calls: {api_calls} | Tokens: {total_tokens} | "
                f"Est. Cost: {cost_text} | Provider: {self._provider_name}[/dim]"
            )
        )

    def _set_phase_glow(self, phase_name: str) -> None:
        for cls in ["glow-planning", "glow-implementation", "glow-testing", "glow-complete"]:
            self.remove_class(cls)

        normalized = phase_name.upper()
        if normalized in {"PLANNING", "ARCHITECTURE", "DOCUMENTATION"}:
            self.add_class("glow-planning")
        elif normalized in {"IMPLEMENTATION", "PACKAGING"}:
            self.add_class("glow-implementation")
        elif normalized == "TESTING":
            self.add_class("glow-testing")
        elif normalized == "COMPLETE":
            self.add_class("glow-complete")

    def _mark_complete(self, summary_text: str) -> None:
        self._is_complete = True
        self._set_phase_glow("COMPLETE")

        progress = self.query_one("#phase-progress", ProgressBar)
        progress.update(total=self._total_phases, progress=self._total_phases)

        out_match = re.search(r"Output:\s*(.+)$", summary_text)
        output_path = out_match.group(1).strip() if out_match else "output/"
        file_count = self._count_output_files(output_path)
        elapsed = int(time.monotonic() - self._started_at)

        self.query_one("#phase-left", Static).update("[bold #00FF41]BUILD COMPLETE[/bold #00FF41]")
        self.query_one("#phase-right", Static).update(
            "[bold #00FF41]"
            f"files {file_count} | time {elapsed}s | api {int(getattr(self._engine, 'api_calls', self._api_call_count))} | tokens {int(getattr(self._engine, 'total_input_tokens', 0) + getattr(self._engine, 'total_output_tokens', 0))} | cost {self._format_cost_text()}"
            "[/bold #00FF41]"
        )
        safe_output_path = output_path.replace("[", "\\[")
        self._feed_queue.put_nowait(
            "[bold #00FF41]✅ SYSTEM:[/bold #00FF41] "
            f"Your project is ready at {safe_output_path}. Press Q to quit."
        )
        asyncio.create_task(self._cinematic_complete_wave())
        self._update_top_bar()

    async def _cinematic_complete_wave(self) -> None:
        panel_order = ["Architect", "Backend", "Frontend", "Tester", "Docs", "PM"]

        for name in panel_order:
            panel = self.agent_panels[name]
            panel.add_class("complete-wave")
            panel.set_status("✅ Done")
            self._agent_status[name] = "✅ Done"
            await asyncio.sleep(0.09)

        await asyncio.sleep(0.18)

        for panel in self.agent_panels.values():
            panel.add_class("complete-flash")

        await asyncio.sleep(0.22)

        for panel in self.agent_panels.values():
            panel.remove_class("complete-wave")
            panel.remove_class("complete-flash")

    def _count_output_files(self, output_path: str) -> int:
        path = Path(output_path)
        if not path.is_absolute():
            path = Path.cwd() / output_path
        if not path.exists():
            return 0
        return sum(1 for p in path.rglob("*") if p.is_file())

    def _format_cost_text(self) -> str:
        provider = str(getattr(self._engine.settings, "provider", "openai")).lower()
        base_url = str(getattr(self._engine.settings, "base_url", "")).lower()
        if provider in {"ollama", "groq"}:
            return "FREE"
        if provider == "openai" and "models.inference.ai.azure.com" in base_url:
            return "FREE"
        return f"${float(getattr(self._engine, 'total_cost', 0.0)):.4f}"

    def _render_scanlines(self) -> None:
        scan_nodes = self.query("#scanlines")
        if len(scan_nodes) == 0:
            return
        scan = self.query_one("#scanlines", Static)
        width = max(20, self.size.width)
        height = max(10, self.size.height)
        bar = "─" * width
        lines: List[str] = []
        for idx in range(height):
            if idx % 2 == 0:
                lines.append(f"[color(#151b23)]{bar}[/color(#151b23)]")
            else:
                lines.append("")
        scan.update("\n".join(lines))

    def _show_fatal_screen(self, message: str) -> None:
        safe = str(message).replace("[", "\\[")
        try:
            self.query_one("#top-bar", Static).update(f"[bold red]SWARM DASHBOARD ERROR[/bold red] {safe}")
        except Exception:
            pass
        try:
            self.query_one("#feed-log", RichLog).write(f"[bold red]FATAL:[/bold red] {safe}")
        except Exception:
            pass

    @staticmethod
    def _log_runtime_error(where: str, exc: Exception) -> None:
        try:
            log_path = Path(__file__).with_name("dashboard_error.log")
            trace = traceback.format_exc()
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {where}: {exc}\n")
                handle.write(trace)
                handle.write("\n")
        except Exception:
            pass
        print(f"[swarm] dashboard error in {where}: {exc}", file=sys.stderr)
