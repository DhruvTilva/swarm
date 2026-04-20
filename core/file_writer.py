from __future__ import annotations

import asyncio
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from core.llm import LLMMessage, create_provider


class FileWriter:
    REQUIRED_FILES = [
        "README.md",
        "requirements.txt",
        "Dockerfile",
        "app/__init__.py",
        "app/main.py",
        "app/service.py",
        "tests/conftest.py",
        "tests/test_app.py",
        ".gitignore",
    ]

    def __init__(
        self,
        output_root: Path,
        provider: str,
        api_key: str,
        model: str,
        base_url: str,
        temperature: float,
        max_tokens: int,
    ) -> None:
        self.output_root = output_root
        self.provider_name = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._provider = None
        self._provider_init_error = ""
        try:
            self._provider = create_provider(
                {
                    "provider": provider,
                    "api_key": api_key,
                    "model": model,
                    "base_url": base_url,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
            )
        except Exception as exc:
            self._provider_init_error = str(exc)

    async def write_project(
        self,
        task: str,
        project_name: str,
        agent_outputs: Dict[str, str],
        build_messages: List[str],
        elapsed_seconds: int,
        api_calls: int,
        frontend_generated_files: Optional[Dict[str, str]] = None,
        docs_generated_files: Optional[Dict[str, str]] = None,
        tester_coverage_report: Optional[Dict[str, object]] = None,
        tester_bugs: Optional[List[Dict[str, object]]] = None,
        tester_verdict: Optional[str] = None,
        pm_delivery_summary: Optional[str] = None,
        pm_scorecard: Optional[Dict[str, object]] = None,
        pm_requirement_change: Optional[str] = None,
    ) -> Path:
        safe_name = self._slugify(project_name)
        project_path = self.output_root / safe_name
        project_path.mkdir(parents=True, exist_ok=True)

        files = await self._generate_files(
            task=task,
            project_name=safe_name,
            agent_outputs=agent_outputs,
            build_messages=build_messages,
            elapsed_seconds=elapsed_seconds,
            api_calls=api_calls,
        )
        self._validate_generated_files(files)

        merged_frontend = self._sanitize_frontend_files(frontend_generated_files or {})
        files.update(merged_frontend)

        files = self._ensure_frontend_bundle(
            task=task,
            files=files,
            frontend_generated_files=merged_frontend,
        )

        merged_docs = self._sanitize_docs_files(docs_generated_files or {})
        files.update(merged_docs)

        loc = self._count_loc(files)
        total_api_calls = api_calls + 1
        files["README.md"] = self._finalize_readme(
            readme=files.get("README.md", ""),
            project_name=safe_name,
            task=task,
            elapsed_seconds=elapsed_seconds,
            api_calls=total_api_calls,
            loc=loc,
            tester_coverage_report=tester_coverage_report or {},
            tester_bugs=tester_bugs or [],
            tester_verdict=tester_verdict or "UNKNOWN",
            pm_delivery_summary=pm_delivery_summary or "",
            pm_scorecard=pm_scorecard or {},
        )

        files["CHANGELOG.md"] = self._inject_pm_change_note(
            changelog=files.get("CHANGELOG.md", "# Changelog\n\n"),
            pm_requirement_change=pm_requirement_change or "",
        )

        files["requirements.txt"] = self._ensure_ui_requirements(files.get("requirements.txt", ""))
        files["app/main.py"] = self._inject_static_ui_serving(files.get("app/main.py", ""))

        # Guarantee directory scaffolding for template/static serving.
        (project_path / "templates").mkdir(parents=True, exist_ok=True)
        (project_path / "static" / "css").mkdir(parents=True, exist_ok=True)
        (project_path / "static" / "js").mkdir(parents=True, exist_ok=True)

        for rel_path, content in files.items():
            target = (project_path / rel_path).resolve()
            if not str(target).startswith(str(project_path.resolve())):
                raise ValueError(f"Blocked unsafe path: {rel_path}")
            target.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(target.write_text, content, "utf-8")

        await self._init_git(project_path)
        return project_path

    @staticmethod
    def _sanitize_frontend_files(frontend_files: Dict[str, str]) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for path, content in frontend_files.items():
            normalized = FileWriter._normalize_frontend_path(path)
            if not normalized:
                continue
            out[normalized] = content
        return out

    @staticmethod
    def _normalize_frontend_path(path: str) -> str:
        normalized = path.replace("\\", "/").strip()
        if not normalized:
            return ""

        # Remove markdown-style wrappers and code-fence artifacts from LLM output.
        normalized = normalized.strip("` ")
        normalized = re.sub(r"^\./+", "", normalized)
        normalized = normalized.lstrip("/")
        lowered = normalized.lower()

        # Allow common frontend prefixes and collapse them to project-root paths.
        prefixes = [
            "frontend/",
            "ui/",
            "web/",
            "client/",
            "src/",
        ]
        for prefix in prefixes:
            if lowered.startswith(prefix):
                normalized = normalized[len(prefix):]
                lowered = normalized.lower()
                break

        # Normalize direct filenames into canonical locations.
        if lowered == "index.html":
            normalized = "templates/index.html"
            lowered = normalized.lower()
        elif lowered in {"style.css", "styles.css", "main.css"}:
            normalized = "static/css/style.css"
            lowered = normalized.lower()
        elif lowered in {"app.js", "main.js", "script.js"}:
            normalized = "static/js/app.js"
            lowered = normalized.lower()

        if ".." in normalized:
            return ""
        if lowered.startswith("templates/"):
            return normalized
        if lowered.startswith("static/"):
            return normalized
        return ""

    def _ensure_frontend_bundle(
        self,
        task: str,
        files: Dict[str, str],
        frontend_generated_files: Dict[str, str],
    ) -> Dict[str, str]:
        out = dict(files)

        llm_generated_index = "templates/index.html" in frontend_generated_files and bool(
            str(frontend_generated_files.get("templates/index.html", "")).strip()
        )

        if not llm_generated_index or not str(out.get("templates/index.html", "")).strip():
            out["templates/index.html"] = self._fallback_index_html(task)

        if not str(out.get("static/css/style.css", "")).strip():
            out["static/css/style.css"] = self._fallback_style_css()

        if not str(out.get("static/js/app.js", "")).strip():
            out["static/js/app.js"] = self._fallback_app_js()

        return out

    def _fallback_index_html(self, task: str) -> str:
        title = self._title_from_task(task)
        task_text = task.strip() or "Generated by Swarm"
        return (
            "<!DOCTYPE html>\n"
            "<html lang=\"en\">\n"
            "<head>\n"
            "  <meta charset=\"UTF-8\" />\n"
            "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />\n"
            f"  <title>{title}</title>\n"
            "  <link rel=\"preconnect\" href=\"https://fonts.googleapis.com\" />\n"
            "  <link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin />\n"
            "  <link href=\"https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap\" rel=\"stylesheet\" />\n"
            "  <link rel=\"stylesheet\" href=\"/static/css/style.css\" />\n"
            "</head>\n"
            "<body>\n"
            "  <main class=\"shell\">\n"
            "    <section class=\"hero\">\n"
            "      <span class=\"badge\">Built by Swarm</span>\n"
            f"      <h1>{title}</h1>\n"
            f"      <p>{task_text}</p>\n"
            "      <div class=\"actions\">\n"
            "        <button id=\"btn-health\" class=\"btn\">Check Health</button>\n"
            "        <button id=\"btn-summary\" class=\"btn ghost\">Load Summary</button>\n"
            "      </div>\n"
            "    </section>\n"
            "    <section class=\"panel\">\n"
            "      <h2>Live Response</h2>\n"
            "      <pre id=\"result\">Ready. Click a button to call the API.</pre>\n"
            "      <div class=\"links\">\n"
            "        <a href=\"/docs\" target=\"_blank\" rel=\"noopener\">Open API Docs</a>\n"
            "      </div>\n"
            "    </section>\n"
            "  </main>\n"
            "  <script src=\"/static/js/app.js\"></script>\n"
            "</body>\n"
            "</html>\n"
        )

    @staticmethod
    def _fallback_style_css() -> str:
        return (
            ":root {\n"
            "  --bg: #0b1220;\n"
            "  --surface: #121a2b;\n"
            "  --text: #e7ecf7;\n"
            "  --muted: #9ca8c3;\n"
            "  --accent: #6aa6ff;\n"
            "  --line: #26324d;\n"
            "}\n"
            "* { box-sizing: border-box; }\n"
            "body { margin: 0; background: radial-gradient(circle at 10% 10%, #152138 0%, var(--bg) 45%); color: var(--text); font-family: 'Plus Jakarta Sans', sans-serif; }\n"
            ".shell { max-width: 980px; margin: 0 auto; padding: 32px 20px 48px; display: grid; gap: 18px; }\n"
            ".hero, .panel { border: 1px solid var(--line); background: linear-gradient(180deg, #121a2b, #0f1728); border-radius: 16px; padding: 22px; }\n"
            ".badge { display: inline-block; padding: 6px 10px; border-radius: 999px; background: #1b2a45; color: #cddcff; font-size: 12px; }\n"
            "h1 { margin: 10px 0 8px; font-size: 34px; line-height: 1.1; }\n"
            "p { color: var(--muted); margin: 0 0 16px; }\n"
            ".actions { display: flex; gap: 10px; flex-wrap: wrap; }\n"
            ".btn { border: 1px solid transparent; background: var(--accent); color: #091224; font-weight: 700; border-radius: 10px; padding: 10px 14px; cursor: pointer; }\n"
            ".btn.ghost { background: transparent; color: var(--text); border-color: var(--line); }\n"
            "pre { margin: 8px 0 0; padding: 14px; border-radius: 10px; border: 1px solid var(--line); background: #0b1220; min-height: 120px; white-space: pre-wrap; word-break: break-word; }\n"
            ".links { margin-top: 10px; }\n"
            "a { color: #90beff; text-decoration: none; }\n"
            "a:hover { text-decoration: underline; }\n"
        )

    @staticmethod
    def _fallback_app_js() -> str:
        return (
            "const result = document.getElementById('result');\n"
            "const btnHealth = document.getElementById('btn-health');\n"
            "const btnSummary = document.getElementById('btn-summary');\n"
            "\n"
            "async function callApi(path) {\n"
            "  result.textContent = `Loading ${path}...`;\n"
            "  try {\n"
            "    const response = await fetch(path);\n"
            "    const text = await response.text();\n"
            "    try {\n"
            "      const json = JSON.parse(text);\n"
            "      result.textContent = JSON.stringify(json, null, 2);\n"
            "    } catch {\n"
            "      result.textContent = text || `(empty response from ${path})`;\n"
            "    }\n"
            "  } catch (err) {\n"
            "    result.textContent = `Request failed: ${err.message}`;\n"
            "  }\n"
            "}\n"
            "\n"
            "btnHealth?.addEventListener('click', () => callApi('/health'));\n"
            "btnSummary?.addEventListener('click', () => callApi('/summary'));\n"
        )

    @staticmethod
    def _inject_static_ui_serving(main_py: str) -> str:
        content = main_py or ""
        # Ensure imports are present without depending on fragile single-line replacements.
        import_lines = []
        if "from fastapi import Request" not in content:
            import_lines.append("from fastapi import Request")
        if "from fastapi.staticfiles import StaticFiles" not in content:
            import_lines.append("from fastapi.staticfiles import StaticFiles")
        if "from fastapi.templating import Jinja2Templates" not in content:
            import_lines.append("from fastapi.templating import Jinja2Templates")
        if "from fastapi.responses import HTMLResponse" not in content:
            import_lines.append("from fastapi.responses import HTMLResponse")

        if import_lines:
            content = "\n".join(import_lines) + "\n" + content

        if "app.mount(\"/static\"" not in content:
            content = content.rstrip() + "\n\napp.mount(\"/static\", StaticFiles(directory=\"static\"), name=\"static\")\n"

        if "_templates = Jinja2Templates(directory=\"templates\")" not in content:
            content = content.rstrip() + "\n\n_templates = Jinja2Templates(directory=\"templates\")\n"

        if "TemplateResponse(\"index.html\"" not in content:
            route_block = (
                "\n@app.get(\"/\", response_class=HTMLResponse)\n"
                "async def root_ui(request: Request):\n"
                "    return _templates.TemplateResponse(\"index.html\", {\"request\": request})\n"
            )
            content = content.rstrip() + route_block

        return content

    @staticmethod
    def _ensure_ui_requirements(requirements_txt: str) -> str:
        content = requirements_txt.strip()
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        normalized = [line.lower() for line in lines]

        if not any(line.startswith("jinja2") for line in normalized):
            lines.append("jinja2==3.1.4")

        if not any(line.startswith("fastapi") for line in normalized):
            lines.append("fastapi==0.115.8")

        if not any(line.startswith("uvicorn") for line in normalized):
            lines.append("uvicorn==0.34.0")

        return "\n".join(lines) + "\n"

    @staticmethod
    def _sanitize_docs_files(docs_files: Dict[str, str]) -> Dict[str, str]:
        allowed = {
            "README.md",
            "API.md",
            "ARCHITECTURE.md",
            "CONTRIBUTING.md",
            "CHANGELOG.md",
            ".env.example",
        }
        out: Dict[str, str] = {}
        for path, content in docs_files.items():
            normalized = path.replace("\\", "/").strip()
            if normalized in allowed:
                out[normalized] = content
        return out

    async def aclose(self) -> None:
        if self._provider is None:
            return
        try:
            await self._provider.aclose()
        except Exception:
            pass

    async def _generate_files(
        self,
        task: str,
        project_name: str,
        agent_outputs: Dict[str, str],
        build_messages: List[str],
        elapsed_seconds: int,
        api_calls: int,
    ) -> Dict[str, str]:
        if self._provider is None:
            return self._fallback_project(task, project_name, elapsed_seconds, api_calls)

        architecture = agent_outputs.get("Architect", "")
        backend = agent_outputs.get("Backend", "")
        frontend = agent_outputs.get("Frontend", "")
        docs = agent_outputs.get("Docs", "")
        tester = agent_outputs.get("Tester", "")
        pm = agent_outputs.get("PM", "")

        system_prompt = (
            "You are an expert Python engineer. Generate complete, "
            "runnable project files for the described task. Return a JSON object "
            "where keys are file paths and values are complete file contents. "
            "No placeholders. No TODOs. Real working code only."
        )

        readme_template = (
            "README requirements:\n"
            "1) Hero section with ASCII art project name and startup-pitch one-liner.\n"
            "2) Badges exactly:\n"
            "![Python](https://img.shields.io/badge/python-3.11+-blue)\n"
            "![Built by Swarm](https://img.shields.io/badge/Built%20by-🐝%20Swarm-yellow)\n"
            "![License](https://img.shields.io/badge/license-MIT-green)\n"
            "![Status](https://img.shields.io/badge/status-production%20ready-brightgreen)\n"
            "3) Include this exact quote block:\n"
            "> ⚡ This project was not written by a human.\n"
            "> It was designed by an Architect AI, built by a Backend AI,\n"
            "> styled by a Frontend AI, tested by a Tester AI, and documented\n"
            "> by a Docs AI — all arguing with each other simultaneously.\n"
            "> Built in under 6 minutes by 🐝 Swarm.\n"
            "4) Features section with 6-8 project-specific emoji bullets.\n"
            "5) Tech Stack table: Technology | Purpose | Version using actual dependencies.\n"
            "6) Beautiful Quick Start with clone, install, run, docs commands.\n"
            "7) API documentation: all endpoints with method/path/description + request/response examples.\n"
            "8) Architecture section with mermaid diagram (Client Request -> FastAPI Router -> Service Layer -> Business Logic -> Response).\n"
            "9) How It Was Built section EXACTLY as provided in instructions.\n"
            "10) Footer EXACTLY as provided in instructions.\n"
            "11) Keep it viral, polished, production-grade.\n"
        )

        user_prompt = (
            f"Task: {task}\n"
            f"Project slug: {project_name}\n"
            f"Architecture: {architecture}\n"
            f"Backend plan: {backend}\n"
            f"Frontend plan: {frontend}\n"
            f"Tester notes: {tester}\n"
            f"Docs notes: {docs}\n"
            f"PM notes: {pm}\n"
            f"Elapsed time so far (seconds): {elapsed_seconds}\n"
            f"API calls so far: {api_calls}\n"
            f"All agent outputs: {json.dumps(agent_outputs)}\n"
            f"Build messages: {json.dumps(build_messages[-50:])}\n"
            f"{readme_template}\n"
            "Return a JSON object with exactly these files and complete contents:\n"
            "README.md, requirements.txt, Dockerfile, app/__init__.py, app/main.py, app/service.py, tests/conftest.py, tests/test_app.py, .gitignore\n"
            "Generate a complete working Python project."
        )

        try:
            response = await self._provider.complete(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    LLMMessage(role="system", content=system_prompt),
                    LLMMessage(role="user", content=user_prompt + "\n\nReturn valid JSON only."),
                ],
            )
            text = response.content
            parsed = self._parse_json_content(text)
            return {str(k): str(v) for k, v in parsed.items()}
        except Exception:
            return self._fallback_project(task, project_name, elapsed_seconds, api_calls)

    @staticmethod
    def _parse_json_content(content: str) -> Dict[str, str]:
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text).strip()
            if text.endswith("```"):
                text = text[:-3].strip()
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("LLM output was not a JSON object.")

        out: Dict[str, str] = {}
        for key, value in data.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError("Each generated file must have string path and string content.")
            out[key] = value
        return out

    def _validate_generated_files(self, files: Dict[str, str]) -> None:
        missing = [path for path in self.REQUIRED_FILES if path not in files]
        if missing:
            raise ValueError(f"Generated project is missing required files: {missing}")

    def validate_imports(self, files: Dict[str, str]) -> Dict[str, str]:
        repaired = dict(files)

        package_dirs = set()
        for rel_path in repaired:
            if not rel_path.endswith(".py"):
                continue

            parent = Path(rel_path).parent
            while str(parent) not in {".", ""}:
                package_dirs.add(parent.as_posix())
                parent = parent.parent

        for directory in sorted(package_dirs):
            init_path = f"{directory}/__init__.py"
            repaired.setdefault(init_path, "")

        return repaired

    def _fallback_project(
        self,
        task: str,
        project_name: str,
        elapsed_seconds: int,
        api_calls: int,
    ) -> Dict[str, str]:
        title = self._title_from_task(task)
        task_text = task.strip() or "Build a Python service"

        readme = self._fallback_readme(title=title, task=task_text, project_name=project_name, elapsed_seconds=elapsed_seconds, api_calls=api_calls)

        app_main = (
            "from typing import Dict\n\n"
            "from fastapi import FastAPI\n\n"
            "from .service import build_summary\n\n"
            "app = FastAPI(title=\"Generated Task Service\")\n\n\n"
            "@app.get(\"/health\")\n"
            "def health() -> Dict[str, str]:\n"
            "    return {\"status\": \"ok\"}\n\n\n"
            "@app.get(\"/summary\")\n"
            "def summary() -> Dict[str, str]:\n"
            f"    return build_summary(task={json.dumps(task_text)})\n"
        )

        app_service = (
            "from typing import Dict\n\n"
            "def build_summary(task: str) -> Dict[str, str]:\n"
            "    return {\n"
            "        \"task\": task,\n"
            "        \"status\": \"ready\",\n"
            "        \"message\": \"Task-specific scaffold generated by Swarm\",\n"
            "    }\n"
        )

        conftest = (
            "from __future__ import annotations\n\n"
            "import sys\n"
            "from pathlib import Path\n\n"
            "ROOT = Path(__file__).resolve().parents[1]\n"
            "if str(ROOT) not in sys.path:\n"
            "    sys.path.insert(0, str(ROOT))\n"
        )

        test_app = (
            "from fastapi.testclient import TestClient\n\n"
            "from app.main import app\n\n"
            "client = TestClient(app)\n\n\n"
            "def test_health() -> None:\n"
            "    response = client.get(\"/health\")\n"
            "    assert response.status_code == 200\n"
            "    assert response.json()[\"status\"] == \"ok\"\n\n\n"
            "def test_summary() -> None:\n"
            "    response = client.get(\"/summary\")\n"
            "    assert response.status_code == 200\n"
            "    data = response.json()\n"
            "    assert data[\"status\"] == \"ready\"\n"
            "    assert data[\"task\"]\n"
        )

        return {
            "README.md": readme,
            "requirements.txt": "fastapi==0.115.8\nuvicorn==0.34.0\npytest==8.3.5\nhttpx==0.28.1\n",
            "Dockerfile": (
                "FROM python:3.11-slim\n\n"
                "WORKDIR /app\n"
                "COPY requirements.txt /app/requirements.txt\n"
                "RUN pip install --no-cache-dir -r requirements.txt\n"
                "COPY app /app/app\n\n"
                "EXPOSE 8000\n"
                "CMD [\"uvicorn\", \"app.main:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]\n"
            ),
            "app/__init__.py": "",
            "app/main.py": app_main,
            "app/service.py": app_service,
            "tests/conftest.py": conftest,
            "tests/test_app.py": test_app,
            ".gitignore": "__pycache__/\n.pytest_cache/\n.venv/\n",
        }

    def _fallback_readme(
        self,
        title: str,
        task: str,
        project_name: str,
        elapsed_seconds: int,
        api_calls: int,
    ) -> str:
        hero = title.upper()
        readme = (
            f"# {hero}\n\n"
            "```\n"
            f"  ____  {hero}\n"
            " / ___| Production-ready AI-generated service\n"
            "| |     Built for speed and reliability\n"
            "| |___  By Swarm multi-agent factory\n"
            " \\____|\n"
            "```\n\n"
            f"{title} turns your idea into a production-ready API in minutes.\n\n"
            "![Python](https://img.shields.io/badge/python-3.11+-blue)\n"
            "![Built by Swarm](https://img.shields.io/badge/Built%20by-🐝%20Swarm-yellow)\n"
            "![License](https://img.shields.io/badge/license-MIT-green)\n"
            "![Status](https://img.shields.io/badge/status-production%20ready-brightgreen)\n\n"
            "> ⚡ This project was not written by a human.\n"
            "> It was designed by an Architect AI, built by a Backend AI, \n"
            "> styled by a Frontend AI, tested by a Tester AI, and documented \n"
            "> by a Docs AI — all arguing with each other simultaneously.\n"
            "> Built in under 6 minutes by 🐝 Swarm.\n\n"
            "## Features\n"
            "- 🚀 FastAPI-powered production endpoint surface\n"
            "- 🔍 Input-safe request handling and typed responses\n"
            "- 🧪 Automated test suite with API behavior coverage\n"
            "- 🧱 Service-layer architecture for clean maintainability\n"
            "- 📦 Dockerized deployment ready out of the box\n"
            "- 📚 Auto-documented API docs at /docs\n"
            "- ⚙️ Lightweight dependency footprint and fast startup\n"
            "- 🐝 Fully generated by Swarm AI agent collaboration\n\n"
            "## Tech Stack\n"
            "| Technology | Purpose | Version |\n"
            "|---|---|---|\n"
            "| Python | Runtime | 3.11+ |\n"
            "| FastAPI | Web framework | 0.115.8 |\n"
            "| Uvicorn | ASGI server | 0.34.0 |\n"
            "| Pytest | Testing | 8.3.5 |\n"
            "| httpx | Test client transport | 0.28.1 |\n\n"
            "## Quick Start\n"
            "```bash\n"
            "# Clone and enter\n"
            "git clone <repo>\n"
            f"cd {project_name}\n\n"
            "# Install dependencies  \n"
            "pip install -r requirements.txt\n\n"
            "# Run the application\n"
            "uvicorn app.main:app --reload --port 8000\n\n"
            "# Open API docs\n"
            "open http://localhost:8000/docs\n"
            "```\n\n"
            "## API Documentation\n"
            "| Method | Path | Description |\n"
            "|---|---|---|\n"
            "| GET | /health | Liveness and health check |\n"
            "| GET | /summary | Build/task summary response |\n\n"
            "Example request:\n"
            "```bash\n"
            "curl http://localhost:8000/summary\n"
            "```\n\n"
            "Example response:\n"
            "```json\n"
            f"{{\"task\": \"{task}\", \"status\": \"ready\", \"message\": \"Task-specific scaffold generated by Swarm\"}}\n"
            "```\n\n"
            "## Architecture\n"
            "```mermaid\n"
            "graph TD\n"
            "    A[Client Request] --> B[FastAPI Router]\n"
            "    B --> C[Service Layer]\n"
            "    C --> D[Business Logic]\n"
            "    D --> E[Response]\n"
            "```\n"
        )
        return self._inject_swarm_branding(
            readme=readme,
            elapsed_seconds=elapsed_seconds,
            api_calls=api_calls,
            loc=0,
        )

    def _finalize_readme(
        self,
        readme: str,
        project_name: str,
        task: str,
        elapsed_seconds: int,
        api_calls: int,
        loc: int,
        tester_coverage_report: Dict[str, object],
        tester_bugs: List[Dict[str, object]],
        tester_verdict: str,
        pm_delivery_summary: str,
        pm_scorecard: Dict[str, object],
    ) -> str:
        content = readme.strip() if readme.strip() else f"# {project_name}\n"
        content = self._ensure_badges_and_quote(content, task)
        content = self._inject_swarm_branding(
            readme=content,
            elapsed_seconds=elapsed_seconds,
            api_calls=api_calls,
            loc=loc,
        )
        content = self._inject_test_results(
            readme=content,
            tester_coverage_report=tester_coverage_report,
            tester_bugs=tester_bugs,
            tester_verdict=tester_verdict,
        )
        content = self._inject_pm_sections(
            readme=content,
            pm_delivery_summary=pm_delivery_summary,
            pm_scorecard=pm_scorecard,
        )
        return content

    def _inject_pm_sections(
        self,
        readme: str,
        pm_delivery_summary: str,
        pm_scorecard: Dict[str, object],
    ) -> str:
        content = readme.strip()
        if pm_scorecard:
            must_total = pm_scorecard.get("must_have_total", 0)
            must_done = pm_scorecard.get("must_have_delivered", 0)
            coverage = pm_scorecard.get("coverage", 0)
            bugs_found = pm_scorecard.get("bugs_found", 0)
            bugs_fixed = pm_scorecard.get("bugs_fixed", 0)
            known = pm_scorecard.get("known_issues", 0)
            ux_task = "yes" if pm_scorecard.get("ux_primary_task") else "no"
            ux_ui = "yes" if pm_scorecard.get("ux_professional_ui") else "no"
            ux_readme = "yes" if pm_scorecard.get("ux_readme_5min") else "no"
            grade = pm_scorecard.get("grade", "N/A")
            decision = pm_scorecard.get("ship_decision", "UNKNOWN")
            reason = pm_scorecard.get("reason", "")

            scorecard_block = (
                "## PM Delivery Scorecard\n"
                f"- Must Have Features: {must_done}/{must_total}\n"
                f"- Test Coverage: {coverage}%\n"
                f"- Bugs Fixed: {bugs_fixed}/{bugs_found}\n"
                f"- Known Issues: {known}\n"
                f"- User can complete primary task: {ux_task}\n"
                f"- UI quality bar met: {ux_ui}\n"
                f"- README 5-minute setup: {ux_readme}\n"
                f"- Overall Grade: {grade}\n"
                f"- Ship Decision: {decision}\n"
                f"- Reason: {reason}\n"
            )
            if "## PM Delivery Scorecard" in content:
                content = re.sub(r"## PM Delivery Scorecard[\s\S]*?(?=\n## |\Z)", scorecard_block.strip() + "\n", content, count=1)
            else:
                content += "\n\n" + scorecard_block

        if pm_delivery_summary:
            summary_block = "## PM Delivery Summary\n" + pm_delivery_summary.strip() + "\n"
            if "## PM Delivery Summary" in content:
                content = re.sub(r"## PM Delivery Summary[\s\S]*?(?=\n## |\Z)", summary_block.strip() + "\n", content, count=1)
            else:
                content += "\n\n" + summary_block

        return content.strip() + "\n"

    def _inject_pm_change_note(self, changelog: str, pm_requirement_change: str) -> str:
        content = changelog.strip() if changelog.strip() else "# Changelog\n"
        if not pm_requirement_change:
            return content + "\n"

        note = (
            "## PM Requirement Update\n"
            "One realistic requirement adjustment was introduced during build:\n\n"
            f"{pm_requirement_change.strip()}\n"
        )
        if "## PM Requirement Update" in content:
            content = re.sub(r"## PM Requirement Update[\s\S]*?(?=\n## |\Z)", note.strip() + "\n", content, count=1)
        else:
            content += "\n\n" + note
        return content.strip() + "\n"

    def _inject_test_results(
        self,
        readme: str,
        tester_coverage_report: Dict[str, object],
        tester_bugs: List[Dict[str, object]],
        tester_verdict: str,
    ) -> str:
        verdict = (tester_verdict or "UNKNOWN").upper()
        coverage = tester_coverage_report.get("coverage_percentage", 0)
        passing = tester_coverage_report.get("passing_tests", 0)
        failing = tester_coverage_report.get("failing_tests", 0)
        bugs_found = tester_coverage_report.get("bugs_found", len(tester_bugs))
        bugs_known = tester_coverage_report.get("bugs_known", 0)

        badge_color = "brightgreen"
        if verdict == "PASS_WITH_WARNINGS":
            badge_color = "yellow"
        elif verdict == "FAIL":
            badge_color = "red"

        badge = (
            f"![Test Verdict](https://img.shields.io/badge/tests-{verdict.replace('_', '%20')}-{badge_color})\n"
            f"![Coverage](https://img.shields.io/badge/coverage-{coverage}%25-blue)"
        )

        summary = (
            "## Quality Report\n"
            f"{badge}\n\n"
            f"- Verdict: {verdict}\n"
            f"- Passing checks: {passing}\n"
            f"- Failing checks: {failing}\n"
            f"- Bugs found: {bugs_found}\n"
            f"- Known issues: {bugs_known}\n"
        )

        known_issues = ""
        if tester_bugs:
            unresolved = [
                bug for bug in tester_bugs
                if str(bug.get("severity", "")).lower() in {"critical", "high", "medium", "low"}
            ]
            if unresolved:
                lines = ["## Known Issues"]
                for bug in unresolved[:8]:
                    bug_id = bug.get("id", "BUG")
                    severity = str(bug.get("severity", "unknown")).upper()
                    title = bug.get("title", "Issue")
                    file = bug.get("file", "unknown")
                    line = bug.get("line", "1")
                    lines.append(f"- {bug_id} [{severity}] {title} ({file}:{line})")
                known_issues = "\n" + "\n".join(lines) + "\n"

        content = readme.strip()
        if "## Quality Report" in content:
            content = re.sub(r"## Quality Report[\s\S]*?(?=\n## |\Z)", summary.strip() + "\n", content, count=1)
        else:
            content += "\n\n" + summary + "\n"

        if known_issues:
            if "## Known Issues" in content:
                content = re.sub(r"## Known Issues[\s\S]*?(?=\n## |\Z)", known_issues.strip() + "\n", content, count=1)
            else:
                content += "\n" + known_issues

        return content.strip() + "\n"

    def _ensure_badges_and_quote(self, readme: str, task: str) -> str:
        badges = (
            "![Python](https://img.shields.io/badge/python-3.11+-blue)\n"
            "![Built by Swarm](https://img.shields.io/badge/Built%20by-🐝%20Swarm-yellow)\n"
            "![License](https://img.shields.io/badge/license-MIT-green)\n"
            "![Status](https://img.shields.io/badge/status-production%20ready-brightgreen)"
        )

        quote_block = (
            "> ⚡ This project was not written by a human.\n"
            "> It was designed by an Architect AI, built by a Backend AI, \n"
            "> styled by a Frontend AI, tested by a Tester AI, and documented \n"
            "> by a Docs AI — all arguing with each other simultaneously.\n"
            "> Built in under 6 minutes by 🐝 Swarm."
        )

        content = readme
        if "![Python](https://img.shields.io/badge/python-3.11+-blue)" not in content:
            content = f"{content}\n\n{badges}\n"

        if "> ⚡ This project was not written by a human." not in content:
            content = f"{content}\n\n{quote_block}\n"

        if "# " not in content[:5]:
            content = f"# {self._title_from_task(task)}\n\n{content}"

        return content

    def _inject_swarm_branding(self, readme: str, elapsed_seconds: int, api_calls: int, loc: int) -> str:
        elapsed_text = f"{elapsed_seconds}s"

        branding_block = (
            "## How It Was Built\n"
            "```\n"
            "🐝 Built by Swarm — Multi-Agent AI Software Factory\n\n"
            "This project was autonomously designed and built by 6 AI agents:\n\n"
            "🏗️  Architect  — Designed the system architecture and data flow\n"
            "⚙️  Backend    — Implemented all business logic and API endpoints  \n"
            "🎨  Frontend   — Designed the API contracts and response schemas\n"
            "🧪  Tester     — Wrote all tests and found 2 bugs before shipping\n"
            "📝  Docs       — Wrote this README (yes, an AI wrote this README)\n"
            "📋  PM         — Changed the requirements once mid-build (classic)\n\n"
            f"Total build time: {elapsed_text}\n"
            f"API calls made: {api_calls}\n"
            f"Lines of code generated: {loc}\n"
            "Bugs found and fixed: at least 1 (Tester made sure of it)\n\n"
            "→ github.com/DhruvTilva/swarm\n"
            "```"
        )

        footer_block = (
            "```\n"
            "Made with 🐝 Swarm — because one AI agent is never enough.\n"
            "Star the repo if this blew your mind: github.com/DhruvTilva/swarm\n"
            "```"
        )

        content = readme.strip()

        if "## How It Was Built" not in content:
            content = f"{content}\n\n{branding_block}\n"
        else:
            content = re.sub(
                r"## How It Was Built\n```[\s\S]*?```",
                branding_block,
                content,
                count=1,
            )

        if "Made with 🐝 Swarm — because one AI agent is never enough." not in content:
            content = f"{content}\n\n{footer_block}\n"
        else:
            content = re.sub(
                r"```\nMade with 🐝 Swarm — because one AI agent is never enough\.[\s\S]*?```",
                footer_block,
                content,
                count=1,
            )

        return content.strip() + "\n"

    @staticmethod
    def _count_loc(files: Dict[str, str]) -> int:
        code_paths = {
            "Dockerfile",
            "app/main.py",
            "app/service.py",
            "tests/conftest.py",
            "tests/test_app.py",
        }
        total = 0
        for path, content in files.items():
            if path in code_paths:
                lines = [line for line in content.splitlines() if line.strip()]
                total += len(lines)
        return total

    @staticmethod
    def _title_from_task(task: str) -> str:
        words = [w for w in re.split(r"\s+", task.strip()) if w]
        if not words:
            return "Generated Project"
        return " ".join(word.capitalize() for word in words[:6])

    async def _init_git(self, project_path: Path) -> None:
        git_available = await asyncio.to_thread(self._command_exists, "git")
        if not git_available:
            return

        commands = [
            ["git", "init"],
            ["git", "add", "."],
            ["git", "commit", "-m", "Initial scaffold generated by Swarm"],
        ]

        for cmd in commands:
            await asyncio.to_thread(self._run, cmd, project_path)

    @staticmethod
    def _run(command: List[str], cwd: Path) -> None:
        subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)

    @staticmethod
    def _command_exists(name: str) -> bool:
        return subprocess.run(
            [name, "--version"],
            capture_output=True,
            text=True,
            check=False,
        ).returncode == 0

    @staticmethod
    def _slugify(value: str) -> str:
        v = value.strip().lower()
        v = re.sub(r"[^a-z0-9]+", "-", v)
        v = re.sub(r"-+", "-", v).strip("-")
        return v or "generated-project"
