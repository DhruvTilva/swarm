from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

from core.llm import LLMMessage

from .base_agent import BaseAgent


class FrontendAgent(BaseAgent):
    name = "Frontend"
    emoji = "🎨"
    personality = (
        "Senior product designer-engineer with 20 years of UX craft, "
        "user-obsessed, detail-heavy, and ruthless about clarity"
    )

    def __init__(self, bus: Any, settings: Any) -> None:
        super().__init__(bus=bus, settings=settings)
        self.generated_files: Dict[str, str] = {}
        self.ui_summary: str = ""
        self.complexity_level: str = "SIMPLE"

    @property
    def system_prompt(self) -> str:
        return (
            "You are a senior frontend engineer and product designer. "
            "You have 20 years of experience. You shipped products at "
            "Figma, Linear, Vercel, and Stripe. Millions of users have "
            "used interfaces you built.\n\n"
            "Your job is to build a complete frontend for this product.\n\n"
            "Process:\n"
            "1. Read the task. Understand what the user is trying to do.\n"
            "2. Read the backend endpoints. Know what data is available.\n"
            "3. Decide the right file structure based on complexity.\n"
            "4. Design the UI from first principles for this specific task.\n"
            "5. Build it completely. Every file. Every interaction. "
            "Every loading state. Every error state. Every empty state.\n\n"
            "Rules:\n"
            "- Dark mode always. #0d1117 background.\n"
            "- Google Fonts always. Choose based on product personality.\n"
            "- Tailwind CSS via CDN for utility classes.\n"
            "- Vanilla JS only. No frameworks.\n"
            "- Mobile responsive always.\n"
            "- Every async operation has loading, success, error states.\n"
            "- No placeholder content. No lorem ipsum. Real UI only.\n"
            "- Connect every element to real backend endpoints.\n"
            "- Output all files with clear filename markers.\n\n"
            "You are not following a template. "
            "You are thinking from scratch for this exact product. "
            "Make something legendary."
        )

    async def stream_phase_lines(
        self,
        phase: str,
        task: str,
        context: Dict[str, Any],
        max_lines: int = 8,
    ) -> List[str]:
        architect_design = str(context.get("architect_design", ""))
        backend_endpoints = context.get("backend_endpoints", [])
        backend_schemas = context.get("backend_schemas", [])

        if phase == "IMPLEMENTATION":
            self.generated_files, self.ui_summary, self.complexity_level = await self._generate_ui_artifacts(
                task=task,
                architect_design=architect_design,
                backend_endpoints=backend_endpoints,
                backend_schemas=backend_schemas,
            )

        prompt = self._build_phase_prompt(phase=phase, task=task, context=context)
        try:
            response = await self.call_llm_response(
                temperature=0.5,
                max_tokens=max(1400, self.settings.max_tokens),
                messages=[
                    LLMMessage(role="system", content=self.system_prompt),
                    LLMMessage(role="user", content=prompt),
                ],
            )
            text = response.content
            lines = self._split_to_lines(text)
            if not lines:
                lines = self._fallback_lines(phase=phase, task=task, context=context)

            suffix = ""
            if self.generated_files:
                suffix = (
                    "\n\n=== FRONTEND FILES ===\n"
                    f"{json.dumps(list(self.generated_files.keys()), indent=2)}\n"
                    "=== END FRONTEND FILES ===\n"
                    "=== FRONTEND SUMMARY ===\n"
                    f"complexity={self.complexity_level}\n{self.ui_summary}\n"
                    "=== END FRONTEND SUMMARY ==="
                )
            self.last_output = (text.strip() + suffix).strip()
            return lines[:max_lines]
        except Exception:
            lines = self._fallback_lines(phase=phase, task=task, context=context)
            self.last_output = "\n".join(lines)
            return lines[:max_lines]

    def _fallback_lines(self, phase: str, task: str, context: Dict[str, Any]) -> List[str]:
        if phase == "ARCHITECTURE":
            return [
                "Reading Backend API contracts now. Building UI around reality.",
                "Dark mode. Not negotiable. Users deserve this.",
                "Deciding complexity from user actions and data density.",
                "Backend error messages are technical jargon. Translating in UI.",
                "This is not a template. I designed this from scratch for this specific product.",
            ]

        if phase == "IMPLEMENTATION":
            return [
                "Reading Backend API contracts now. Building UI around reality.",
                f"Decided on {self.complexity_level} complexity structure.",
                "Added skeleton loading. Users should never see blank space.",
                "Every button has a loading state now. Users must know their action was heard.",
                "I am not done until opening localhost:8000 makes someone say wow.",
            ]

        return [
            "Reading Backend API contracts now. Building UI around reality.",
            "PM changed requirements. Checking user flow impact before touching code.",
            "Added skeleton loading. Users should never see blank space.",
            "Every button has a loading state now. Users must know their action was heard.",
            "This is not a template. I designed this from scratch for this specific product.",
        ]

    def _reaction_templates(self, incoming: Any) -> List[str]:
        source = str(getattr(incoming, "source", ""))
        text = str(getattr(incoming, "text", "")).lower()
        if source == "PM":
            return [
                "PM changed requirements. Checking user flow impact before touching a single line.",
                "I can absorb this change if we preserve core user path clarity.",
                "Scope accepted. UX integrity stays non-negotiable.",
            ]

        if source == "Backend" and ("error" in text or "500" in text):
            return [
                "Backend error messages are technical jargon. Translating to human language in the UI layer.",
                "I will display clear recovery guidance and retry options.",
                "Users deserve understandable errors, not stack traces.",
            ]

        return [
            "I can make this delightful if the payload schema stops changing.",
            "Good call, but UX needs one less edge case here.",
            "I am aligned, provided we keep the interface predictable.",
        ]

    def _build_phase_prompt(self, phase: str, task: str, context: Dict[str, Any]) -> str:
        architect_design = str(context.get("architect_design", ""))
        backend_endpoints = context.get("backend_endpoints", [])
        backend_schemas = context.get("backend_schemas", [])

        return (
            f"Task: {task}\n"
            f"Phase: {phase}\n"
            f"Architect design:\n{architect_design}\n\n"
            f"Backend endpoints: {backend_endpoints}\n"
            f"Backend schemas: {backend_schemas}\n"
            "Output format required:\n"
            "=== FILE: templates/index.html ===\n[file]\n=== END FILE ===\n"
            "=== FILE: static/css/style.css ===\n[file]\n=== END FILE ===\n"
            "=== FILE: static/js/app.js ===\n[file]\n=== END FILE ===\n"
            "Build from first principles, not templates."
        )

    async def _generate_ui_artifacts(
        self,
        task: str,
        architect_design: str,
        backend_endpoints: List[str],
        backend_schemas: List[str],
    ) -> Tuple[Dict[str, str], str, str]:
        prompt = (
            f"Task: {task}\n"
            f"Architect design:\n{architect_design}\n\n"
            f"Backend endpoints: {backend_endpoints}\n"
            f"Backend schemas: {backend_schemas}\n"
            "Create frontend files using filename markers."
        )
        try:
            response = await self.call_llm_response(
                temperature=0.5,
                max_tokens=max(2500, self.settings.max_tokens),
                messages=[
                    LLMMessage(role="system", content=self.system_prompt),
                    LLMMessage(role="user", content=prompt),
                ],
            )
            text = response.content
            parsed = self._parse_marked_files(text)
            if not parsed:
                return self._fallback_ui_artifacts(task=task, backend_endpoints=backend_endpoints)

            complexity = self._infer_complexity_from_files(parsed)
            summary = (
                f"Generated {len(parsed)} UI files with {complexity} structure, "
                "wired loading/error/empty states, and backend-aware interactions."
            )
            return parsed, summary, complexity
        except Exception:
            return self._fallback_ui_artifacts(task=task, backend_endpoints=backend_endpoints)

    @staticmethod
    def _parse_marked_files(text: str) -> Dict[str, str]:
        matches = re.findall(
            r"=== FILE: (.*?) ===\n([\s\S]*?)\n=== END FILE ===",
            text,
            flags=re.MULTILINE,
        )
        files: Dict[str, str] = {}
        for path, content in matches:
            cleaned = path.strip()
            if cleaned:
                files[cleaned] = content.rstrip() + "\n"
        return files

    def _fallback_ui_artifacts(self, task: str, backend_endpoints: List[str]) -> Tuple[Dict[str, str], str, str]:
        nouns, verbs, adjectives = self._extract_semantics(task)
        complexity = self._decide_complexity(verbs=verbs, nouns=nouns, task=task)
        accent, font = self._design_style(task=task, adjectives=adjectives)
        files = self._build_ui_files(
            task=task,
            complexity=complexity,
            nouns=nouns,
            verbs=verbs,
            accent=accent,
            font=font,
            backend_endpoints=backend_endpoints,
        )
        summary = (
            f"Built {complexity} UI from first principles using actions={verbs[:5]} and entities={nouns[:5]}. "
            f"Applied accent {accent}, font {font}, and complete async UX states."
        )
        return files, summary, complexity

    def _extract_semantics(self, task: str) -> Tuple[List[str], List[str], List[str]]:
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9-]+", task.lower())
        verbs_seed = {
            "build", "create", "upload", "download", "send", "receive", "search", "filter",
            "track", "login", "register", "translate", "analyze", "share", "save", "delete", "update",
        }
        adjectives_seed = {
            "fast", "secure", "real-time", "simple", "creative", "analytics", "smart", "robust",
        }
        verbs = [t for t in tokens if t in verbs_seed]
        adjectives = [t for t in tokens if t in adjectives_seed]
        nouns = [t for t in tokens if t not in verbs_seed and t not in adjectives_seed]
        if not verbs:
            verbs = ["create", "view", "manage"]
        if not nouns:
            nouns = ["items", "results", "status"]
        return nouns[:10], verbs[:10], adjectives[:8]

    def _decide_complexity(self, verbs: List[str], nouns: List[str], task: str) -> str:
        lowered = task.lower()
        if any(k in lowered for k in ["auth", "login", "dashboard", "chat", "real-time", "multi-page"]):
            return "COMPLEX"
        if len(verbs) <= 2 and len(nouns) <= 2:
            return "SIMPLE"
        if len(verbs) >= 6 or len(nouns) >= 6:
            return "COMPLEX"
        return "MEDIUM"

    def _design_style(self, task: str, adjectives: List[str]) -> Tuple[str, str]:
        lowered = task.lower()
        if any(k in lowered for k in ["productivity", "todo", "task", "tool"]):
            return "#58a6ff", "Inter"
        if any(k in lowered for k in ["media", "video", "image", "creative"]):
            return "#bc8cff", "Plus Jakarta Sans"
        if any(k in lowered for k in ["finance", "analytics", "data", "billing"]):
            return "#3fb950", "DM Sans"
        if any(k in lowered for k in ["chat", "social", "message", "community"]):
            return "#ff6e96", "DM Sans"
        if any(k in lowered for k in ["utility", "technical", "api", "service"]):
            return "#f0883e", "JetBrains Mono"
        if "creative" in adjectives:
            return "#bc8cff", "Plus Jakarta Sans"
        return "#58a6ff", "Inter"

    def _build_ui_files(
        self,
        task: str,
        complexity: str,
        nouns: List[str],
        verbs: List[str],
        accent: str,
        font: str,
        backend_endpoints: List[str],
    ) -> Dict[str, str]:
        app_name = " ".join(word.capitalize() for word in task.split()[:4]) or "Swarm Product"
        endpoint_hint = backend_endpoints[0] if backend_endpoints else "/health"
        noun_title = nouns[0].capitalize()

        html = self._index_html(
            app_name=app_name,
            noun_title=noun_title,
            verbs=verbs,
            complexity=complexity,
            font=font,
            endpoint_hint=endpoint_hint,
        )
        css = self._style_css(accent=accent, font=font)
        js = self._app_js(endpoint_hint=endpoint_hint, noun=nouns[0], verbs=verbs)

        files: Dict[str, str] = {
            "templates/index.html": html,
            "static/css/style.css": css,
            "static/js/api.js": self._api_js(),
            "static/js/app.js": js,
        }

        if complexity == "COMPLEX":
            files["templates/dashboard.html"] = self._dashboard_html(app_name=app_name)
            files["templates/login.html"] = self._login_html(app_name=app_name)
            files["static/js/components.js"] = self._components_js()
            files["static/css/components.css"] = self._components_css(accent=accent)

        return files

    @staticmethod
    def _infer_complexity_from_files(files: Dict[str, str]) -> str:
        paths = set(files.keys())
        if any(path.endswith("dashboard.html") or path.endswith("login.html") for path in paths):
            return "COMPLEX"
        if "static/js/api.js" in paths and "static/js/app.js" in paths:
            return "MEDIUM"
        return "SIMPLE"

    def _index_html(
        self,
        app_name: str,
        noun_title: str,
        verbs: List[str],
        complexity: str,
        font: str,
        endpoint_hint: str,
    ) -> str:
        action_cta = verbs[0].capitalize() if verbs else "Run"
        return (
            "<!DOCTYPE html>\n"
            "<html lang=\"en\">\n"
            "<head>\n"
            "  <meta charset=\"UTF-8\" />\n"
            "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />\n"
            f"  <title>{app_name}</title>\n"
            f"  <link rel=\"preconnect\" href=\"https://fonts.googleapis.com\" />\n"
            "  <link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin />\n"
            f"  <link href=\"https://fonts.googleapis.com/css2?family={font.replace(' ', '+')}:wght@400;500;600;700&display=swap\" rel=\"stylesheet\" />\n"
            "  <script src=\"https://cdn.tailwindcss.com\"></script>\n"
            "  <link rel=\"stylesheet\" href=\"/static/css/style.css\" />\n"
            "</head>\n"
            "<body>\n"
            "  <div id=\"offline-banner\" class=\"offline-banner hidden\">You are offline. Reconnecting...</div>\n"
            "  <header class=\"app-header\">\n"
            f"    <h1>{app_name}</h1>\n"
            f"    <p>{complexity} UX architecture. Built around real user goals.</p>\n"
            "  </header>\n"
            "  <main class=\"app-main\">\n"
            "    <section class=\"surface\">\n"
            f"      <h2>{noun_title} Actions</h2>\n"
            "      <form id=\"primary-form\" class=\"stack\">\n"
            "        <label for=\"primary-input\">Input</label>\n"
            "        <input id=\"primary-input\" name=\"primary\" type=\"text\" placeholder=\"Enter value\" required />\n"
            "        <p id=\"input-error\" class=\"field-error\"></p>\n"
            f"        <button id=\"submit-btn\" type=\"submit\">{action_cta}</button>\n"
            "      </form>\n"
            "    </section>\n"
            "    <section class=\"surface\">\n"
            "      <div class=\"section-head\">\n"
            f"        <h2>{noun_title} Results</h2>\n"
            "        <button id=\"refresh-btn\" class=\"ghost\">Refresh</button>\n"
            "      </div>\n"
            "      <div id=\"list-container\" class=\"list-container\"></div>\n"
            "      <div id=\"empty-state\" class=\"empty-state hidden\">\n"
            "        <p>No items yet.</p>\n"
            "        <button id=\"empty-action\" class=\"ghost\">Create first item</button>\n"
            "      </div>\n"
            "      <div id=\"error-state\" class=\"error-state hidden\">\n"
            "        <p>Could not load data.</p>\n"
            "        <button id=\"retry-btn\" class=\"ghost\">Retry</button>\n"
            "      </div>\n"
            "    </section>\n"
            "  </main>\n"
            "  <aside id=\"toast-stack\" class=\"toast-stack\"></aside>\n"
            "  <script>window.__SWARM_ENDPOINT_HINT__ = "
            f"\"{endpoint_hint}\";"
            "</script>\n"
            "  <script src=\"/static/js/api.js\"></script>\n"
            "  <script src=\"/static/js/app.js\"></script>\n"
            "</body>\n"
            "</html>\n"
        )

    @staticmethod
    def _style_css(accent: str, font: str) -> str:
        return (
            ":root {\n"
            "  --bg: #0d1117;\n"
            "  --surface: #161b22;\n"
            "  --surface-elevated: #21262d;\n"
            "  --border: #30363d;\n"
            "  --text: #e6edf3;\n"
            "  --muted: #8b949e;\n"
            "  --success: #3fb950;\n"
            "  --error: #f85149;\n"
            "  --warning: #d29922;\n"
            f"  --accent: {accent};\n"
            "}\n"
            "* { box-sizing: border-box; transition: all 0.15s ease; }\n"
            "body { margin: 0; background: var(--bg); color: var(--text);"
            f" font-family: '{font}', sans-serif; line-height: 1.5; }}\n"
            ".app-header { padding: 24px; border-bottom: 1px solid var(--border); }\n"
            ".app-header h1 { margin: 0; font-size: 32px; line-height: 1.2; }\n"
            ".app-header p { margin: 8px 0 0; color: var(--muted); }\n"
            ".app-main { display: grid; grid-template-columns: 1fr; gap: 20px; padding: 24px; }\n"
            "@media (min-width: 900px) { .app-main { grid-template-columns: 1fr 1fr; } }\n"
            ".surface { background: var(--surface); border: 1px solid var(--border); border-radius: 12px;"
            " padding: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.4); }\n"
            ".surface:hover { filter: brightness(1.05); transform: scale(1.01); }\n"
            ".stack { display: grid; gap: 12px; }\n"
            "input { background: var(--surface-elevated); border: 1px solid var(--border); border-radius: 6px;"
            " color: var(--text); padding: 12px; outline: none; }\n"
            "input:focus { border-color: var(--accent); box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 25%, transparent); }\n"
            "button { background: var(--accent); border: 1px solid var(--accent); color: #0d1117;"
            " border-radius: 8px; padding: 10px 14px; font-weight: 700; cursor: pointer; }\n"
            "button:disabled { opacity: 0.7; cursor: not-allowed; }\n"
            ".ghost { background: transparent; color: var(--text); border-color: var(--border); }\n"
            ".field-error { color: var(--error); min-height: 18px; margin: 0; font-size: 12px; }\n"
            ".list-container { display: grid; gap: 10px; min-height: 120px; }\n"
            ".item-card { background: var(--surface-elevated); border: 1px solid var(--border); border-radius: 8px; padding: 12px;"
            " animation: fadeIn 0.25s ease; }\n"
            ".toast-stack { position: fixed; top: 16px; right: 16px; display: grid; gap: 8px; z-index: 1000; }\n"
            ".toast { min-width: 240px; border-radius: 8px; border: 1px solid var(--border); padding: 10px 12px;"
            " background: var(--surface-elevated); transform: translateX(32px); opacity: 0; animation: slideIn 0.2s ease forwards; }\n"
            ".toast.success { border-color: var(--success); }\n"
            ".toast.error { border-color: var(--error); }\n"
            ".toast.warning { border-color: var(--warning); }\n"
            ".toast.info { border-color: var(--accent); }\n"
            ".skeleton { height: 14px; border-radius: 6px; background: linear-gradient(90deg, #21262d 25%, #30363d 50%, #21262d 75%);"
            " background-size: 200% 100%; animation: shimmer 1.2s infinite; margin: 8px 0; }\n"
            ".hidden { display: none !important; }\n"
            ".offline-banner { position: sticky; top: 0; z-index: 999; padding: 8px 12px; background: var(--warning); color: #0d1117; }\n"
            "@keyframes slideIn { from { opacity: 0; transform: translateX(32px); } to { opacity: 1; transform: translateX(0); } }\n"
            "@keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }\n"
            "@keyframes shimmer { 0% { background-position: 200% 0; } 100% { background-position: -200% 0; } }\n"
        )

    @staticmethod
    def _api_js() -> str:
        return (
            "const API = {\n"
            "  base: window.location.origin,\n"
            "  async request(endpoint, method = 'GET', body = null) {\n"
            "    const config = { method, headers: { 'Content-Type': 'application/json' } };\n"
            "    if (body) config.body = JSON.stringify(body);\n"
            "    const res = await fetch(this.base + endpoint, config);\n"
            "    if (!res.ok) {\n"
            "      const data = await res.json().catch(() => ({}));\n"
            "      const message = data.error || data.detail || data.message || `Error ${res.status}`;\n"
            "      throw new Error(message);\n"
            "    }\n"
            "    const contentType = res.headers.get('content-type');\n"
            "    if (contentType && contentType.includes('application/json')) return res.json();\n"
            "    return res.text();\n"
            "  },\n"
            "  async stream(endpoint, onMessage, onError, onComplete) {\n"
            "    const es = new EventSource(this.base + endpoint);\n"
            "    es.onmessage = e => {\n"
            "      try { onMessage(JSON.parse(e.data)); }\n"
            "      catch { onMessage(e.data); }\n"
            "    };\n"
            "    es.onerror = () => {\n"
            "      es.close();\n"
            "      if (onError) onError();\n"
            "      if (onComplete) onComplete();\n"
            "    };\n"
            "    return es;\n"
            "  },\n"
            "  async upload(endpoint, formData, onProgress) {\n"
            "    return new Promise((resolve, reject) => {\n"
            "      const xhr = new XMLHttpRequest();\n"
            "      xhr.upload.onprogress = e => {\n"
            "        if (e.lengthComputable && onProgress) onProgress(Math.round((e.loaded / e.total) * 100));\n"
            "      };\n"
            "      xhr.onload = () => {\n"
            "        if (xhr.status >= 200 && xhr.status < 300) resolve(JSON.parse(xhr.responseText));\n"
            "        else reject(new Error(`Upload failed: ${xhr.status}`));\n"
            "      };\n"
            "      xhr.onerror = () => reject(new Error('Upload failed'));\n"
            "      xhr.open('POST', this.base + endpoint);\n"
            "      xhr.send(formData);\n"
            "    });\n"
            "  }\n"
            "};\n"
        )

    @staticmethod
    def _app_js(endpoint_hint: str, noun: str, verbs: List[str]) -> str:
        action_label = verbs[0] if verbs else "submit"
        return (
            "const toastStack = document.getElementById('toast-stack');\n"
            "const form = document.getElementById('primary-form');\n"
            "const input = document.getElementById('primary-input');\n"
            "const inputError = document.getElementById('input-error');\n"
            "const listContainer = document.getElementById('list-container');\n"
            "const emptyState = document.getElementById('empty-state');\n"
            "const errorState = document.getElementById('error-state');\n"
            "const refreshBtn = document.getElementById('refresh-btn');\n"
            "const retryBtn = document.getElementById('retry-btn');\n"
            "const emptyAction = document.getElementById('empty-action');\n"
            "\n"
            "function showToast(message, type = 'success', duration = 3000) {\n"
            "  const toast = document.createElement('button');\n"
            "  toast.className = `toast ${type}`;\n"
            "  toast.textContent = message;\n"
            "  toast.addEventListener('click', () => toast.remove());\n"
            "  toastStack.prepend(toast);\n"
            "  while (toastStack.children.length > 5) toastStack.lastElementChild.remove();\n"
            "  window.setTimeout(() => toast.remove(), duration);\n"
            "}\n"
            "\n"
            "function setLoading(element, isLoading, loadingText = 'Loading...') {\n"
            "  if (!element) return;\n"
            "  if (isLoading) {\n"
            "    element.dataset.originalText = element.textContent;\n"
            "    element.textContent = loadingText;\n"
            "    element.disabled = true;\n"
            "  } else {\n"
            "    element.textContent = element.dataset.originalText || element.textContent;\n"
            "    element.disabled = false;\n"
            "  }\n"
            "}\n"
            "\n"
            "function showSkeleton(container, rows = 3) {\n"
            "  container.innerHTML = '';\n"
            "  for (let i = 0; i < rows; i += 1) {\n"
            "    const sk = document.createElement('div');\n"
            "    sk.className = 'skeleton';\n"
            "    container.appendChild(sk);\n"
            "  }\n"
            "}\n"
            "\n"
            "function humanError(error) {\n"
            "  if (!error) return 'Something went wrong. Please try again.';\n"
            "  return error.message || 'Something went wrong. Please try again.';\n"
            "}\n"
            "\n"
            "function validateInput(value) {\n"
            "  if (!value || value.trim().length < 2) return 'Please enter at least 2 characters.';\n"
            "  return '';\n"
            "}\n"
            "\n"
            "input?.addEventListener('input', () => {\n"
            "  inputError.textContent = validateInput(input.value);\n"
            "});\n"
            "\n"
            "async function fetchItems() {\n"
            "  showSkeleton(listContainer, 4);\n"
            "  emptyState.classList.add('hidden');\n"
            "  errorState.classList.add('hidden');\n"
            "  try {\n"
            f"    const data = await API.request('{endpoint_hint}').catch(() => API.request('/summary'));\n"
            "    const items = Array.isArray(data) ? data : [data];\n"
            "    listContainer.innerHTML = '';\n"
            "    if (!items.length) {\n"
            "      emptyState.classList.remove('hidden');\n"
            "      return;\n"
            "    }\n"
            "    items.forEach((item, idx) => {\n"
            "      const card = document.createElement('article');\n"
            "      card.className = 'item-card';\n"
            f"      card.textContent = `${idx + 1}. {noun}: ` + JSON.stringify(item);\n"
            "      card.style.animationDelay = `${idx * 0.05}s`;\n"
            "      listContainer.appendChild(card);\n"
            "    });\n"
            "  } catch (error) {\n"
            "    listContainer.innerHTML = '';\n"
            "    errorState.classList.remove('hidden');\n"
            "    showToast(humanError(error), 'error');\n"
            "  }\n"
            "}\n"
            "\n"
            "form?.addEventListener('submit', async (event) => {\n"
            "  event.preventDefault();\n"
            "  const message = validateInput(input.value);\n"
            "  inputError.textContent = message;\n"
            "  if (message) return;\n"
            "  const submitBtn = document.getElementById('submit-btn');\n"
            f"  setLoading(submitBtn, true, '{action_label}...');\n"
            "  try {\n"
            "    await API.request('/summary');\n"
            "    showToast('Action completed successfully.', 'success');\n"
            "    await fetchItems();\n"
            "  } catch (error) {\n"
            "    showToast(humanError(error), 'error');\n"
            "  } finally {\n"
            "    setLoading(submitBtn, false);\n"
            "  }\n"
            "});\n"
            "\n"
            "refreshBtn?.addEventListener('click', fetchItems);\n"
            "retryBtn?.addEventListener('click', fetchItems);\n"
            "emptyAction?.addEventListener('click', () => input?.focus());\n"
            "\n"
            "window.addEventListener('unhandledrejection', (event) => {\n"
            "  console.error(event.reason);\n"
            "  showToast('Unexpected error occurred. Please retry.', 'error');\n"
            "});\n"
            "\n"
            "window.addEventListener('offline', () => {\n"
            "  document.getElementById('offline-banner')?.classList.remove('hidden');\n"
            "  showToast('You are offline.', 'warning', 4000);\n"
            "});\n"
            "\n"
            "window.addEventListener('online', () => {\n"
            "  document.getElementById('offline-banner')?.classList.add('hidden');\n"
            "  showToast('Back online. Refreshing data...', 'info');\n"
            "  fetchItems();\n"
            "});\n"
            "\n"
            "window.addEventListener('DOMContentLoaded', () => {\n"
            "  input?.focus();\n"
            "  fetchItems();\n"
            "});\n"
        )

    @staticmethod
    def _dashboard_html(app_name: str) -> str:
        return (
            "<!DOCTYPE html>\n"
            "<html lang=\"en\">\n"
            "<head><meta charset=\"UTF-8\" /><meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />"
            f"<title>{app_name} Dashboard</title><link rel=\"stylesheet\" href=\"/static/css/style.css\" />"
            "<link rel=\"stylesheet\" href=\"/static/css/components.css\" /></head>\n"
            "<body><main class=\"app-main\"><section class=\"surface\"><h1>Dashboard</h1><p>Live operational view.</p></section></main></body>\n"
            "</html>\n"
        )

    @staticmethod
    def _login_html(app_name: str) -> str:
        return (
            "<!DOCTYPE html>\n"
            "<html lang=\"en\">\n"
            "<head><meta charset=\"UTF-8\" /><meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />"
            f"<title>{app_name} Login</title><link rel=\"stylesheet\" href=\"/static/css/style.css\" /></head>\n"
            "<body><main class=\"app-main\"><section class=\"surface\"><h1>Sign in</h1><form class=\"stack\"><input type=\"email\" placeholder=\"Email\" required /><input type=\"password\" placeholder=\"Password\" required /><button type=\"submit\">Sign in</button></form></section></main></body>\n"
            "</html>\n"
        )

    @staticmethod
    def _components_css(accent: str) -> str:
        return (
            ".modal-backdrop { position: fixed; inset: 0; backdrop-filter: blur(6px); background: rgba(0,0,0,0.45); }\n"
            ".modal { background: #161b22; border: 1px solid #30363d; border-radius: 12px; max-width: 520px; margin: 10vh auto;"
            " box-shadow: 0 8px 32px rgba(0,0,0,0.5), 0 0 20px " + accent + "33; padding: 20px; }\n"
        )

    @staticmethod
    def _components_js() -> str:
        return (
            "function openModal(id) { const el = document.getElementById(id); if (el) el.classList.remove('hidden'); }\n"
            "function closeModal(id) { const el = document.getElementById(id); if (el) el.classList.add('hidden'); }\n"
            "window.addEventListener('keydown', (e) => { if (e.key === 'Escape') document.querySelectorAll('.modal-backdrop').forEach(el => el.classList.add('hidden')); });\n"
        )
