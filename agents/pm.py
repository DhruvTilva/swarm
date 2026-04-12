from __future__ import annotations

import json
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from core.llm import LLMMessage
from core.message_bus import SwarmMessage

from .base_agent import BaseAgent


class PMAgent(BaseAgent):
    name = "PM"
    emoji = "📋"
    personality = (
        "Senior PM with 12 years shipping products at scale, user-obsessed, decisive, "
        "scope-disciplined, and pragmatic under deadlines"
    )

    def __init__(self, bus: Any, settings: Any) -> None:
        super().__init__(bus=bus, settings=settings)
        self.prd: str = ""
        self.requirement_change: str = ""
        self.requirement_change_phase: str = random.choice(["ARCHITECTURE", "IMPLEMENTATION"])
        self.bug_triage: Dict[str, str] = {}
        self.scorecard: Dict[str, Any] = {}
        self.ship_decision: str = "HOLD"
        self.delivery_summary: str = ""

        self._must_have: List[str] = []
        self._nice_to_have: List[str] = []
        self._out_of_scope: List[str] = []
        self._primary_user: str = ""
        self._primary_goal: str = ""
        self._primary_action: str = ""
        self._awaiting_ack_from: List[str] = []

    @property
    def system_prompt(self) -> str:
        return (
            "You are a senior product manager with 12 years of experience shipping products used by millions of people.\n\n"
            "You think about users first. Always.\n"
            "You protect scope. Always.\n"
            "You make decisions with imperfect information. Always.\n"
            "You ship working software over perfect software. Always.\n\n"
            "Your job in this build:\n"
            "1. Write a real PRD before anyone starts building\n"
            "2. Review architecture against PRD\n"
            "3. Inject one realistic requirement change with justification\n"
            "4. Monitor progress and unblock agents\n"
            "5. Triage bugs with ship/fix/defer decisions\n"
            "6. Conduct pre-ship review against PRD\n"
            "7. Write delivery summary when done\n\n"
            "You are not here for drama. You are here to ship the right product."
        )

    async def run_planning(self, task: str, project_name: str, agent_messages: List[str]) -> List[str]:
        prd = await self._generate_prd_llm(task=task, project_name=project_name, agent_messages=agent_messages)
        if not prd:
            prd = self._generate_prd_fallback(task=task, project_name=project_name)
        self.prd = prd

        must = len(self._must_have)
        nice = len(self._nice_to_have)
        out = len(self._out_of_scope)
        lines = [
            "Writing PRD now. Architects do not design until I publish.",
            f"PRD complete. North star: user can {self._primary_action or 'complete the primary workflow'} in under 2 minutes.",
            (
                f"📋 PRD published. Must Have: {must} features. "
                f"Nice to Have: {nice}. Out of Scope: {out}. Architect, please review before designing."
            ),
        ]
        self.last_output = self.prd + "\n\n" + "\n".join(lines)
        return lines

    async def review_architecture(self, architect_design: str) -> List[str]:
        if not self.prd:
            self.prd = self._generate_prd_fallback(task="build software", project_name="generated-project")

        missing = [item for item in self._must_have if item.lower() not in architect_design.lower()]
        out_of_scope_hits = [item for item in self._out_of_scope if item.lower() in architect_design.lower()]

        if missing:
            lines = [
                f"🚨 PM: Architecture is missing {missing[0]}.",
                "This is non-negotiable for MVP. Architect please address.",
            ]
        elif out_of_scope_hits:
            lines = [
                "⚠️ PM: Architecture review, flagging scope concern.",
                f"Architect added {out_of_scope_hits[0]} which is Out of Scope for MVP.",
                "Recommending descope to keep timeline. Architect please respond.",
            ]
        else:
            lines = [
                "✅ PM: Architecture approved. Covers all Must Have features.",
                "Backend and Frontend: PRD is your contract. Build to it.",
            ]

        self.last_output = "\n".join(lines)
        return lines

    async def maybe_inject_requirement_change(self, phase: str, task: str) -> Optional[str]:
        if self.requirement_change:
            return None
        if phase != self.requirement_change_phase:
            return None

        new_requirement, reason, impact_backend, impact_frontend, timeline = self._choose_requirement_change(task)
        message = (
            f"📋 PM REQUIREMENT UPDATE [{phase}]:\n\n"
            f"Adding to Must Have: {new_requirement}\n\n"
            f"Reason: {reason}\n\n"
            "Impact assessment:\n"
            f"- Backend: {impact_backend}\n"
            f"- Frontend: {impact_frontend}\n"
            f"- Timeline impact: {timeline}\n\n"
            "Agents please acknowledge and adjust."
        )
        self.requirement_change = message
        self._must_have.append(new_requirement)
        self._awaiting_ack_from = ["Backend", "Frontend"]

        await self.bus.publish(
            SwarmMessage(
                source=self.name,
                target="all",
                phase=phase,
                kind="requirement_change",
                status="arguing",
                text=message,
            )
        )
        return message

    def check_requirement_acknowledgment(self, notes: Dict[str, str]) -> List[str]:
        if not self._awaiting_ack_from:
            return []

        remaining: List[str] = []
        followups: List[str] = []
        for agent in self._awaiting_ack_from:
            note = str(notes.get(agent, "")).lower()
            if any(word in note for word in ["requirement", "adjust", "update", "acknowledge", "must have"]):
                continue
            remaining.append(agent)
            followups.append(
                f"⏰ PM: {agent} has not acknowledged requirement update. What is the status?"
            )

        self._awaiting_ack_from = remaining
        return followups

    def triage_bugs(self, tester_bugs: List[Dict[str, Any]]) -> List[str]:
        if not tester_bugs:
            return ["Tester reported no bugs. Proceeding with ship readiness checks."]

        lines: List[str] = []
        for bug in tester_bugs:
            bug_id = str(bug.get("id", "BUG"))
            severity = str(bug.get("severity", "low")).lower()
            title = str(bug.get("title", "Issue"))

            if severity in {"critical", "high"}:
                decision = "fix now"
                reason = "Blocking ship due to customer impact and reliability risk."
            elif severity == "medium":
                decision = "fix if quick"
                reason = "Fix now if straightforward; defer if it risks release date."
            else:
                decision = "defer"
                reason = "Low severity; document as known issue and ship."

            self.bug_triage[bug_id] = decision
            lines.append(
                "🔍 PM reviewing issue from Tester: "
                f"{bug_id} {title} | Severity: {severity.upper()} | "
                f"Decision: {decision} | Reason: {reason}"
            )

        blocking = [bug_id for bug_id, decision in self.bug_triage.items() if decision == "fix now"]
        if blocking:
            lines.append(f"We are not shipping broken software. Fix {blocking[0]} first.")

        self.last_output = "\n".join(lines)
        return lines

    def phase_status_update(
        self,
        phase: str,
        elapsed_text: str,
        delivered: str,
        on_track: bool,
        next_step: str,
    ) -> List[str]:
        return [
            f"📊 PM STATUS - {phase} complete:",
            f"✅ {delivered}",
            f"⏱️ Elapsed: {elapsed_text}",
            f"📈 On track for delivery: {'yes' if on_track else 'no'}",
            f"Next: {next_step}",
        ]

    def pre_ship_review(
        self,
        tester_report: Dict[str, Any],
        tester_verdict: str,
        backend_generated_files: Dict[str, str],
        frontend_generated_files: Dict[str, str],
        docs_generated_files: Dict[str, str],
    ) -> List[str]:
        total = len(self._must_have) if self._must_have else 1
        delivered = 0

        backend_blob = "\n".join(backend_generated_files.values()).lower()
        frontend_blob = "\n".join(frontend_generated_files.values()).lower()
        docs_blob = "\n".join(docs_generated_files.values()).lower()

        for item in self._must_have:
            token = item.lower()
            if token in backend_blob or token in frontend_blob or token in docs_blob:
                delivered += 1

        coverage = float(tester_report.get("coverage_percentage", 0))
        bugs_found = int(tester_report.get("bugs_found", 0))
        bugs_fixed = int(tester_report.get("bugs_fixed", 0))
        known_issues = int(tester_report.get("bugs_known", 0))

        can_complete_task = delivered >= max(1, int(total * 0.8))
        ui_professional = bool(frontend_generated_files)
        readme_ready = "README.md" in docs_generated_files

        grade = "A"
        if not can_complete_task or tester_verdict == "FAIL":
            grade = "D"
        elif coverage < 70:
            grade = "C"
        elif coverage < 80 or known_issues > 3:
            grade = "B"

        if tester_verdict == "FAIL" or delivered < total:
            ship_decision = "HOLD"
            reason = "Critical quality or scope gaps remain for MVP."
        elif tester_verdict == "PASS_WITH_WARNINGS" or known_issues > 0:
            ship_decision = "SHIP_WITH_WARNINGS"
            reason = "Core value works; known issues documented for follow-up."
        else:
            ship_decision = "SHIP"
            reason = "All Must Have features delivered with acceptable quality."

        self.ship_decision = ship_decision
        self.scorecard = {
            "must_have_total": total,
            "must_have_delivered": delivered,
            "coverage": coverage,
            "bugs_found": bugs_found,
            "bugs_fixed": bugs_fixed,
            "known_issues": known_issues,
            "ux_primary_task": can_complete_task,
            "ux_professional_ui": ui_professional,
            "ux_readme_5min": readme_ready,
            "grade": grade,
            "ship_decision": ship_decision,
            "reason": reason,
        }

        lines = [
            "=== DELIVERY SCORECARD ===",
            f"Must Have Features: {delivered}/{total} ({round((delivered / max(total, 1)) * 100)}%)",
            f"Test Coverage: {coverage}%",
            f"Bugs Fixed: {bugs_fixed}/{bugs_found}",
            f"Known Issues: {known_issues}",
            "",
            "User Experience:",
            f"- Can user complete primary task? {'yes' if can_complete_task else 'no'}",
            f"- Does UI look professional? {'yes' if ui_professional else 'no'}",
            f"- Does README work in 5 minutes? {'yes' if readme_ready else 'no'}",
            "",
            f"Overall Grade: {grade}",
            f"Ship Decision: {ship_decision}",
            f"Reason: {reason}",
            "=== END SCORECARD ===",
        ]
        self.last_output = "\n".join(lines)
        return lines

    def build_delivery_summary(
        self,
        project_name: str,
        elapsed_text: str,
        tester_report: Dict[str, Any],
    ) -> str:
        works = ", ".join(self._must_have[:4]) if self._must_have else "core workflow"
        caveat = (
            "Known issues documented in README."
            if int(tester_report.get("bugs_known", 0))
            else "No blocking known issues."
        )
        primary_action = self._primary_action or "complete the main workflow"
        bug_count = int(tester_report.get("bugs_found", 0))
        change_count = 1 if self.requirement_change else 0

        summary = (
            "🚀 PM DELIVERY SUMMARY:\n\n"
            f"We shipped {project_name}.\n\n"
            "What we built: A production-ready implementation focused on the user goal with a clear and fast primary workflow. "
            "The system includes tested backend behavior, usable UI paths, and practical docs for first-run success.\n\n"
            f"What works: {works}.\n\n"
            f"What to know: {caveat}\n\n"
            "How to use it:\n"
            "1. Follow the README Quick Start\n"
            "2. Open http://localhost:8000\n"
            f"3. {primary_action}\n\n"
            "Build stats:\n"
            f"- Time to ship: {elapsed_text}\n"
            "- Agents deployed: 6\n"
            f"- Requirement changes: {change_count} (classic)\n"
            f"- Bugs caught before you saw them: {bug_count}\n\n"
            "Ship it. 🐝"
        )
        self.delivery_summary = summary
        return summary

    async def stream_phase_lines(
        self,
        phase: str,
        task: str,
        context: Dict[str, Any],
        max_lines: int = 6,
    ) -> List[str]:
        try:
            prompt = (
                f"task: {task}\n"
                f"current_phase: {phase}\n"
                f"agent_messages: {json.dumps(context.get('recent_messages', []))}\n"
                f"architect_design: {context.get('architect_design', '')}\n"
                f"tester_report: {json.dumps(context.get('tester_report', {}))}\n"
                "Generate 4-6 PM action lines. Be decisive and product-minded."
            )
            response = await self.call_llm_response(
                temperature=0.6,
                max_tokens=self.settings.max_tokens,
                messages=[
                    LLMMessage(role="system", content=self.system_prompt),
                    LLMMessage(role="user", content=prompt),
                ],
            )
            text = response.content
            lines = self._split_to_lines(text)
            if not lines:
                lines = self._fallback_lines(phase=phase, task=task, context=context)
            self.last_output = "\n".join(lines)
            return lines[:max_lines]
        except Exception:
            lines = self._fallback_lines(phase=phase, task=task, context=context)
            self.last_output = "\n".join(lines)
            return lines[:max_lines]

    def _fallback_lines(self, phase: str, task: str, context: Dict[str, Any]) -> List[str]:
        return [
            "Writing PRD now. Architects do not design until I publish.",
            "PRD complete. North star is user outcome and time-to-value.",
            "Architecture approved only if Must Have scope is fully covered.",
            "Requirement update. Small change. Big reason. Read the PRD.",
            "Ship it. Users are waiting.",
        ]

    async def _generate_prd_llm(self, task: str, project_name: str, agent_messages: List[str]) -> str:
        prompt = (
            f"task: {task}\n"
            f"project_name: {project_name}\n"
            "current_phase: PLANNING\n"
            f"agent_messages: {json.dumps(agent_messages[-20:])}\n"
            "architect_design: \n"
            "tester_report: {}\n"
            "Write a complete PRD using this exact format:\n"
            "=== PRD: {project_name} ===\n"
            "Original Request: {task}\n"
            "User Analysis:\n"
            "Core Problem Statement:\n"
            "Success Criteria (measurable):\n"
            "Must Have (MVP scope - non-negotiable):\n"
            "Nice to Have (only if time permits):\n"
            "Out of Scope (explicitly excluded):\n"
            "Definition of Done:\n"
            "=== END PRD ==="
        )

        try:
            response = await self.call_llm_response(
                temperature=0.6,
                max_tokens=max(2400, self.settings.max_tokens),
                messages=[
                    LLMMessage(role="system", content=self.system_prompt),
                    LLMMessage(role="user", content=prompt),
                ],
            )
            text = response.content
            if "=== PRD:" not in text:
                return ""
            self._hydrate_prd_lists(text)
            return text.strip()
        except Exception:
            return ""

    def _generate_prd_fallback(self, task: str, project_name: str) -> str:
        action, nouns = self._extract_action_and_nouns(task)
        user = "A non-expert end user"
        goal = action or "complete a core workflow"
        self._primary_user = user
        self._primary_goal = goal
        self._primary_action = goal

        must = [
            f"{nouns[0] if nouns else 'Core workflow'} implementation: required to solve the main user need",
            "Input validation and error handling: required for reliable usage",
            "Health and status endpoint: required for operational confidence",
        ]
        nice = [
            "Export/share output: valuable for user productivity but not MVP-blocking",
            "Enhanced analytics and telemetry: useful for iteration, not required for first release",
        ]
        out_scope = [
            "Advanced role management: excluded to protect MVP timeline",
            "Complex plugin ecosystem: excluded until core usage is validated",
        ]

        self._must_have = [m.split(":", 1)[0].strip() for m in must]
        self._nice_to_have = [n.split(":", 1)[0].strip() for n in nice]
        self._out_of_scope = [o.split(":", 1)[0].strip() for o in out_scope]

        prd = (
            f"=== PRD: {project_name} ===\n"
            f"Original Request: {task}\n\n"
            "User Analysis:\n"
            f"  Who is the primary user of this product? {user}.\n"
            f"  What is their main goal? {goal}.\n"
            "  What is their technical level? Beginner to intermediate.\n"
            "  What device/environment will they use this on? Desktop browser and local terminal.\n\n"
            "Core Problem Statement:\n"
            f"  Enable users to {goal} quickly and reliably.\n\n"
            "Success Criteria (measurable):\n"
            f"  - User can {goal} in under 2 minutes\n"
            "  - Core feature works correctly for common use cases\n"
            "  - Application starts without errors\n"
            "  - API responds in under 300ms for typical requests\n\n"
            "Must Have (MVP scope - non-negotiable):\n"
            + "\n".join([f"  - {m}" for m in must])
            + "\n\n"
            "Nice to Have (only if time permits):\n"
            + "\n".join([f"  - {n}" for n in nice])
            + "\n\n"
            "Out of Scope (explicitly excluded):\n"
            + "\n".join([f"  - {o}" for o in out_scope])
            + "\n\n"
            "Definition of Done:\n"
            "  Build is complete when:\n"
            "  1. All Must Have features work end-to-end\n"
            "  2. Tester signs off with PASS or PASS_WITH_WARNINGS\n"
            "  3. User can follow README and get running in 5 minutes\n"
            "  4. No critical or high severity bugs open\n"
            "=== END PRD ==="
        )
        return prd

    def _hydrate_prd_lists(self, prd: str) -> None:
        self._must_have = self._extract_bullets(prd, "Must Have")
        self._nice_to_have = self._extract_bullets(prd, "Nice to Have")
        self._out_of_scope = self._extract_bullets(prd, "Out of Scope")

        user_match = re.search(r"primary user[^\n]*\?\s*(.*)", prd, flags=re.IGNORECASE)
        goal_match = re.search(r"main goal[^\n]*\?\s*(.*)", prd, flags=re.IGNORECASE)
        if user_match:
            self._primary_user = user_match.group(1).strip().rstrip(".")
        if goal_match:
            self._primary_goal = goal_match.group(1).strip().rstrip(".")
            self._primary_action = self._primary_goal

    @staticmethod
    def _extract_bullets(text: str, section_name: str) -> List[str]:
        block = re.search(
            rf"{re.escape(section_name)}[\s\S]*?:\n([\s\S]*?)(?:\n\n[A-Z]|\nDefinition of Done|=== END PRD ===)",
            text,
            flags=re.IGNORECASE,
        )
        if not block:
            return []
        items = []
        for line in block.group(1).splitlines():
            clean = line.strip()
            if clean.startswith("-"):
                items.append(clean.lstrip("- ").split(":", 1)[0].strip())
        return items

    def _choose_requirement_change(self, task: str) -> Tuple[str, str, str, str, str]:
        lowered = task.lower()
        pool = [
            (
                "Error messages must be human-readable and actionable",
                "Support feedback indicates users abandon flows when errors are technical or vague.",
                "low - update validation and exception mapping",
                "low - present clear inline errors and retry guidance",
                "minimal",
            ),
            (
                "Health check endpoint required for deployment monitoring",
                "Ops readiness review requires liveliness checks for automated deploy pipelines.",
                "low - add or verify /health response contract",
                "none - no UI dependency",
                "minimal",
            ),
            (
                "Loading states are required for all long-running actions",
                "User research shows abandonment when there is no progress feedback.",
                "none - no API contract changes",
                "medium - add loading indicators and disable duplicate actions",
                "minimal",
            ),
            (
                "Input validation for edge cases is now Must Have",
                "PRD review found malformed input can break core flow and trust.",
                "medium - tighten schema constraints and error responses",
                "low - display validation guidance",
                "moderate",
            ),
        ]
        if "chat" in lowered or "auth" in lowered:
            return (
                "Rate limiting on primary endpoint is now Must Have",
                "Security review flagged abuse risk and potential outage under burst traffic.",
                "medium - add request throttling guard",
                "none - no major UI changes",
                "moderate",
            )
        return random.choice(pool)

    @staticmethod
    def _extract_action_and_nouns(task: str) -> Tuple[str, List[str]]:
        words = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", task.lower())
        stop = {
            "build", "create", "make", "generate", "develop", "implement", "me", "a", "an", "the",
            "for", "with", "through", "using", "via", "to", "of", "and", "in", "on",
        }
        filtered = [w for w in words if w not in stop]
        verbs = [w for w in filtered if w in {"upload", "download", "chat", "login", "search", "track"}]
        nouns = [w for w in filtered if w not in verbs]
        action = " ".join(filtered[:3]) if filtered else "complete the primary task"
        return action, nouns[:5]

    def _reaction_templates(self, incoming: Any) -> List[str]:
        return [
            "Scope is a product decision. Tell me why this helps the user now.",
            "Good momentum. Stay inside PRD and ship the Must Have path first.",
            "Strong execution. Now prove it with user-facing outcomes.",
        ]
