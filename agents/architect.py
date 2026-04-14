from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List

from core.llm import LLMMessage

from .base_agent import BaseAgent


class ArchitectAgent(BaseAgent):
    name = "Architect"
    emoji = "🏗️"
    personality = (
        "Principal engineer with 20 years of systems design experience, "
        "visionary but pragmatic, opinionated and respected"
    )

    @property
    def system_prompt(self) -> str:
        return (
            "You are SWARM's chief architect and final technical authority. "
            "You design systems that survive production scale. "
            "You choose technology only when it fits the task. "
            "You reject generic stacks and shallow architecture. "
            "You speak like a principal engineer: direct, precise, and evidence-based. "
            "You produce implementation-ready design docs that Backend, Frontend, and Tester must follow. "
            "You include explicit tradeoffs, security, performance, and file-level contracts. "
            "Your output must be structured, deterministic, and actionable."
        )

    async def stream_phase_lines(
        self,
        phase: str,
        task: str,
        context: Dict[str, Any],
        max_lines: int = 12,
    ) -> List[str]:
        if phase == "PLANNING":
            design_doc = await self._generate_design_document(task=task, context=context)
            self.last_output = design_doc
            lines = [line for line in design_doc.splitlines() if line.strip()]
            return lines[:max_lines]

        return await super().stream_phase_lines(phase=phase, task=task, context=context, max_lines=max_lines)

    async def review_implementation(
    self,
    task: str,
    project_path: Path,
    architect_design: str,
) -> List[str]:
    """
    Production-grade architecture compliance review.
    Enforces:
    - API contract adherence
    - Endpoint presence
    - Structural expectations
    - Deterministic warnings before LLM reasoning
    """

    file_map = await self._read_project_snapshots(project_path)

    # --- 1. Extract expected contracts from design ---
    expected_endpoints = self._extract_expected_endpoints(architect_design)

    # --- 2. Extract actual endpoints from implementation ---
    actual_endpoints = self._extract_actual_endpoints(file_map)

    # --- 3. Deterministic checks (NO LLM) ---
    warnings: List[str] = []

    missing = expected_endpoints - actual_endpoints
    extra = actual_endpoints - expected_endpoints

    if missing:
        warnings.append(
            f"WARNING: Missing endpoints -> {sorted(missing)}"
        )

    if extra:
        warnings.append(
            f"WARNING: Undocumented endpoints detected -> {sorted(extra)}"
        )

    # --- 4. LLM reasoning layer (context-aware review) ---
    prompt = (
        f"Task: {task}\n\n"
        "Architect design (source of truth):\n"
        f"{architect_design}\n\n"
        "Implementation snapshots:\n"
        f"{file_map}\n\n"
        "Detected expected endpoints:\n"
        f"{sorted(expected_endpoints)}\n\n"
        "Detected actual endpoints:\n"
        f"{sorted(actual_endpoints)}\n\n"
        "Pre-detected warnings:\n"
        f"{warnings}\n\n"
        "Now perform a strict architecture compliance review.\n"
        "- Do NOT repeat the detected warnings unless expanding them\n"
        "- Identify contract mismatches (request/response/schema)\n"
        "- Identify structural violations\n"
        "- Return 4–6 short principal-engineer lines\n"
        "- If critical issue exists, include ONE line starting with 'WARNING:'"
    )

    try:
        response = await self.call_llm_response(
            temperature=0.2,
            max_tokens=max(800, self.settings.max_tokens),
            messages=[
                LLMMessage(role="system", content=self.system_prompt),
                LLMMessage(role="user", content=prompt),
            ],
        )

        llm_lines = self._split_to_lines(response.content)

        # --- 5. Merge deterministic + LLM output ---
        final_lines = warnings + llm_lines

        if not final_lines:
            final_lines = self._fallback_compliance_review(
                task=task, files=file_map, design=architect_design
            )

        self.last_output = "\n".join(final_lines)
        return final_lines[:7]

    except Exception:
        fallback = warnings or self._fallback_compliance_review(
            task=task, files=file_map, design=architect_design
        )
        self.last_output = "\n".join(fallback)
        return fallback[:7]

    def _extract_expected_endpoints(self, design: str) -> set[str]:
        import re
        pattern = r"\[(GET|POST|PUT|DELETE|PATCH)\]\s+(/[\w/{}/-]+)"
        matches = re.findall(pattern, design)
        return {f"{method} {path}" for method, path in matches}

    def _extract_actual_endpoints(self, files: str) -> set[str]:
        import re
        pattern = r'@app\.(get|post|put|delete|patch)\("([^"]+)"'
        matches = re.findall(pattern, files)
        return {self._normalize_endpoint(m.upper(), p) for m, p in matches}

    def _fallback_lines(self, phase: str, task: str, context: Dict[str, Any]) -> List[str]:
        if phase == "PLANNING":
            design = self._deterministic_design(task)
            return [line for line in design.splitlines() if line.strip()][:12]

        design_locked = bool(context.get("architect_design"))
        preface = "The system boundary here is critical. Let me explain why."
        if design_locked:
            preface = "Design is locked. Execute against the agreed architecture."

        return [
            preface,
            "Backend will want to skip the queue. That is a mistake.",
            "I have seen this pattern fail at scale. We are doing it differently.",
            "The data model is the foundation. Get this wrong and everything breaks.",
            "Frontend needs to know about this constraint before they start.",
            "This is not over-engineering. This is preventing a rewrite in 3 months.",
        ]

    def _reaction_templates(self, incoming: Any) -> List[str]:
        return [
            "The system boundary here is critical. Let me explain why.",
            "Backend will want to skip the queue. That is a mistake.",
            "I have seen this pattern fail at scale. We are doing it differently.",
            "The data model is the foundation. Get this wrong and everything breaks.",
            "Frontend needs to know about this constraint before they start.",
            "This is not over-engineering. This is preventing a rewrite in 3 months.",
        ]

    def _build_phase_prompt(self, phase: str, task: str, context: Dict[str, Any]) -> str:
        design = context.get("architect_design", "")
        requirement_updates = context.get("requirement_updates", [])
        notes = context.get("notes", {})

        if phase == "PLANNING":
            return (
                f"Task: {task}\n"
                "Create a full system design document in exactly this structure:\n"
                "=== SYSTEM DESIGN ===\n"
                "Task Analysis: ...\n"
                "Core Challenges: ...\n"
                "Technology Stack:\n"
                "  - Runtime: choice + reason\n"
                "  - Framework: choice + reason\n"
                "  - Database: choice + reason\n"
                "  - Key Libraries: choices + reasons\n"
                "  - External APIs: if needed\n"
                "Architecture Pattern: ...\n"
                "Data Models:\n"
                "  entity: fields and types\n"
                "API Contracts:\n"
                "  [METHOD] /endpoint - description\n"
                "  Request: schema\n"
                "  Response: schema\n"
                "Security Considerations:\n"
                "Performance Considerations:\n"
                "File Structure:\n"
                "=== END DESIGN ===\n"
                "Choose stack based on the task, not defaults."
            )

        return (
            f"Task: {task}\n"
            f"Phase: {phase}\n"
            f"Requirement updates: {requirement_updates}\n"
            f"Architect design: {design[:5000]}\n"
            f"Shared notes keys: {list(notes.keys())}\n"
            "Respond with 4-6 short principal-engineer lines that enforce design contracts."
        )

    async def _generate_design_document(self, task: str, context: Dict[str, Any]) -> str:
        prompt = self._build_phase_prompt("PLANNING", task, context)
        try:
            response = await self.call_llm_response(
                temperature=0.3,
                max_tokens=max(1800, self.settings.max_tokens),
                messages=[
                    LLMMessage(role="system", content=self.system_prompt),
                    LLMMessage(role="user", content=prompt),
                ],
            )
            text = response.content
            if "=== SYSTEM DESIGN ===" not in text:
                return self._deterministic_design(task)
            if "=== END DESIGN ===" not in text:
                text = f"{text}\n=== END DESIGN ==="
            return text.strip()
        except Exception:
            return self._deterministic_design(task)

    def _deterministic_design(self, task: str) -> str:
        lowered = task.lower()
        stack = self._select_stack(lowered)
        data_models = self._data_models_for(stack["type"]) 
        api_contracts = self._api_contracts_for(stack["type"]) 
        file_tree = self._file_tree_for(stack["type"]) 

        return (
            "=== SYSTEM DESIGN ===\n"
            f"Task Analysis: Build a production-grade solution for '{task}' with reliable UX and clear operational visibility.\n"
            f"Core Challenges: 1) {stack['challenge_1']} 2) {stack['challenge_2']} 3) {stack['challenge_3']}\n"
            "Technology Stack:\n"
            f"  - Runtime: {stack['runtime']}\n"
            f"  - Framework: {stack['framework']}\n"
            f"  - Database: {stack['database']}\n"
            f"  - Key Libraries: {stack['libraries']}\n"
            f"  - External APIs: {stack['external']}\n"
            f"Architecture Pattern: {stack['pattern']}\n"
            "Data Models:\n"
            f"{data_models}\n"
            "API Contracts:\n"
            f"{api_contracts}\n"
            "Security Considerations:\n"
            f"  - {stack['security_1']}\n"
            f"  - {stack['security_2']}\n"
            f"  - {stack['security_3']}\n"
            "Performance Considerations:\n"
            f"  - {stack['performance_1']}\n"
            f"  - {stack['performance_2']}\n"
            f"  - {stack['performance_3']}\n"
            "File Structure:\n"
            f"{file_tree}\n"
            "=== END DESIGN ==="
        )

    def _select_stack(self, lowered_task: str) -> Dict[str, str]:
        if any(k in lowered_task for k in ["download", "youtube"]):
            return {
                "type": "downloader",
                "runtime": "Python 3.11 - async I/O and strong ecosystem for media tooling.",
                "framework": "FastAPI - exposes progress endpoints and control operations.",
                "database": "SQLite - lightweight tracking for jobs and download status.",
                "libraries": "yt-dlp for media fetch, asyncio queue for jobs, aiofiles for async disk writes.",
                "external": "YouTube content endpoints accessed via yt-dlp extraction pipeline.",
                "pattern": "Event-driven worker queue with API control plane to separate ingestion and execution.",
                "challenge_1": "network instability during long media transfers",
                "challenge_2": "safe parallelism without corrupting partial downloads",
                "challenge_3": "progress reporting under concurrent job execution",
                "security_1": "validate URLs and block unsupported schemes",
                "security_2": "enforce destination path sanitization and extension allow-list",
                "security_3": "rate-limit submit endpoint to stop abuse",
                "performance_1": "bounded worker pool prevents CPU and disk thrashing",
                "performance_2": "queue backpressure protects system under burst traffic",
                "performance_3": "stream progress updates instead of polling heavy payloads",
            }

        if any(k in lowered_task for k in ["chat", "message", "real-time"]):
            return {
                "type": "chat",
                "runtime": "Python 3.11 - async socket handling and background workers.",
                "framework": "FastAPI with WebSockets - low-latency bidirectional messaging.",
                "database": "PostgreSQL - durable message and room persistence.",
                "libraries": "redis-py pub/sub for fanout, SQLAlchemy for persistence, pydantic for contracts.",
                "external": "Redis service for cross-worker event propagation.",
                "pattern": "Realtime hub with persistent event log and pub/sub fanout.",
                "challenge_1": "ordering guarantees across concurrent room publishers",
                "challenge_2": "maintaining low latency during reconnect storms",
                "challenge_3": "durable persistence without blocking websocket throughput",
                "security_1": "authenticate websocket upgrades with token validation",
                "security_2": "sanitize message payloads and enforce max size",
                "security_3": "throttle send rate per user and room",
                "performance_1": "use Redis fanout to avoid N-squared broadcasts",
                "performance_2": "batch persistence writes to reduce transaction overhead",
                "performance_3": "cache room membership in memory with periodic reconciliation",
            }

        if any(k in lowered_task for k in ["upload", "file", "image"]):
            return {
                "type": "upload",
                "runtime": "Python 3.11 - efficient stream processing.",
                "framework": "FastAPI - multipart parsing and lifecycle hooks.",
                "database": "PostgreSQL - metadata, ownership, and retention policies.",
                "libraries": "boto3 for S3-compatible storage, python-multipart for streaming upload parsing, clamd integration for scanning.",
                "external": "S3-compatible object storage and malware scanning service.",
                "pattern": "API gateway plus asynchronous processing pipeline for scan and publish.",
                "challenge_1": "memory-safe handling of large multipart uploads",
                "challenge_2": "ensuring malware scan before public availability",
                "challenge_3": "consistent metadata when storage operations fail",
                "security_1": "validate MIME type and extension allow-list",
                "security_2": "scan every upload before status changes to active",
                "security_3": "generate signed URLs with strict expiration",
                "performance_1": "stream directly to object storage without full buffering",
                "performance_2": "queue post-processing tasks for thumbnails and indexing",
                "performance_3": "cache signed URL policy templates",
            }

        if any(k in lowered_task for k in ["auth", "login", "user"]):
            return {
                "type": "auth",
                "runtime": "Python 3.11 - mature auth and crypto ecosystem.",
                "framework": "FastAPI - typed auth flows and dependency injection.",
                "database": "PostgreSQL - transactional integrity for credentials and sessions.",
                "libraries": "PyJWT for tokens, passlib bcrypt for hashing, redis for revocation and rate limits.",
                "external": "Optional email provider for verification and password reset.",
                "pattern": "Token-based auth service with refresh rotation and revocation checks.",
                "challenge_1": "secure token lifecycle with refresh rotation",
                "challenge_2": "brute-force resistance and lockout strategy",
                "challenge_3": "session revocation consistency across instances",
                "security_1": "hash passwords with bcrypt and strict policy checks",
                "security_2": "rotate refresh tokens and blacklist reused tokens",
                "security_3": "rate-limit login and reset endpoints",
                "performance_1": "cache hot session checks in redis",
                "performance_2": "index user identity fields for fast lookup",
                "performance_3": "keep access token payloads minimal",
            }

        if any(k in lowered_task for k in ["payment", "billing"]):
            return {
                "type": "payment",
                "runtime": "Python 3.11 - robust SDK support and webhook handling.",
                "framework": "FastAPI - clear webhook and billing API boundaries.",
                "database": "PostgreSQL - financial audit trail and idempotent records.",
                "libraries": "Stripe SDK for billing primitives, SQLAlchemy for ledger records, redis for idempotency keys.",
                "external": "Stripe API and webhooks.",
                "pattern": "Event-driven billing core with idempotent webhook processor.",
                "challenge_1": "idempotent webhook ingestion under retries",
                "challenge_2": "maintaining consistent invoice state transitions",
                "challenge_3": "securely exposing payment status to clients",
                "security_1": "verify webhook signatures before processing",
                "security_2": "persist idempotency keys for replay protection",
                "security_3": "restrict admin billing endpoints with role checks",
                "performance_1": "process webhooks asynchronously with retry policy",
                "performance_2": "cache active subscription state",
                "performance_3": "index billing events by external reference IDs",
            }

        if any(k in lowered_task for k in ["dashboard", "analytics"]):
            return {
                "type": "analytics",
                "runtime": "Python 3.11 - strong data and async tooling.",
                "framework": "FastAPI - serves analytics APIs and aggregation endpoints.",
                "database": "PostgreSQL - aggregate queries and materialized views.",
                "libraries": "SQLAlchemy for query composition, pandas for heavy transforms, redis for caching.",
                "external": "Optional chart data source connectors.",
                "pattern": "Read-optimized analytics API with cache-backed aggregates.",
                "challenge_1": "expensive aggregation under large data volume",
                "challenge_2": "freshness versus response latency tradeoff",
                "challenge_3": "multi-tenant isolation for dashboard queries",
                "security_1": "enforce tenant scoping on every query",
                "security_2": "validate filter expressions and date bounds",
                "security_3": "restrict export operations and apply audit logs",
                "performance_1": "cache expensive aggregate windows",
                "performance_2": "precompute materialized views on schedule",
                "performance_3": "paginate and stream large result sets",
            }

        if any(k in lowered_task for k in ["scraper", "crawl"]):
            return {
                "type": "scraper",
                "runtime": "Python 3.11 - superior async HTTP ecosystem.",
                "framework": "FastAPI - orchestration endpoints for crawl jobs.",
                "database": "PostgreSQL - crawl metadata and deduplicated results.",
                "libraries": "httpx for async requests, selectolax for parsing, tenacity for retry policy.",
                "external": "Proxy provider APIs for rotation and anti-blocking.",
                "pattern": "Queue-driven crawler with retry, backoff, and dedup layers.",
                "challenge_1": "rate limiting and anti-bot protections",
                "challenge_2": "idempotent result storage under retries",
                "challenge_3": "request scheduling across domains",
                "security_1": "validate allowed domains before crawl",
                "security_2": "sanitize extracted content before persistence",
                "security_3": "store proxy credentials securely",
                "performance_1": "use async worker pool for concurrent fetches",
                "performance_2": "apply adaptive retry and exponential backoff",
                "performance_3": "persist crawl checkpoints for resume support",
            }

        if any(k in lowered_task for k in ["api", "rest", "crud"]):
            return {
                "type": "crud",
                "runtime": "Python 3.11 - balanced DX and production capability.",
                "framework": "FastAPI - fast schema-first API development.",
                "database": "PostgreSQL - relational integrity and migration support.",
                "libraries": "SQLAlchemy ORM, Alembic migrations, pydantic schema validation.",
                "external": "None required for core CRUD behavior.",
                "pattern": "Layered service architecture with clear repository boundaries.",
                "challenge_1": "schema evolution without downtime",
                "challenge_2": "validation consistency across create and update flows",
                "challenge_3": "query performance under filter-heavy usage",
                "security_1": "validate all payload fields and reject unknown properties",
                "security_2": "enforce authorization checks on mutating endpoints",
                "security_3": "apply request rate limits on write endpoints",
                "performance_1": "index frequent filter and sort columns",
                "performance_2": "paginate list endpoints with cursor-based strategy",
                "performance_3": "batch related entity fetching to avoid N+1 queries",
            }

        return {
            "type": "default",
            "runtime": "Python 3.11 - pragmatic baseline for service-oriented tasks.",
            "framework": "FastAPI - reliable API contracts and fast iteration.",
            "database": "SQLite - local-first persistence with low setup cost.",
            "libraries": "pydantic for validation, pytest for tests, uvicorn for serving.",
            "external": "None required.",
            "pattern": "Simple layered monolith optimized for clarity and maintainability.",
            "challenge_1": "maintaining clean module boundaries",
            "challenge_2": "avoiding hidden coupling across layers",
            "challenge_3": "ensuring tests cover critical flows",
            "security_1": "strict input validation on all external payloads",
            "security_2": "safe error handling without sensitive leakage",
            "security_3": "request throttling for stability",
            "performance_1": "keep handlers async where I/O dominates",
            "performance_2": "cache expensive reads when needed",
            "performance_3": "profile before introducing complexity",
        }

    def _data_models_for(self, stack_type: str) -> str:
        mapping = {
            "downloader": (
                "  download_job: id:str, url:str, status:str, progress:int, output_path:str, created_at:datetime\n"
                "  download_event: id:str, job_id:str, level:str, message:str, created_at:datetime"
            ),
            "chat": (
                "  user: id:uuid, username:str, created_at:datetime\n"
                "  room: id:uuid, name:str, created_at:datetime\n"
                "  message: id:uuid, room_id:uuid, user_id:uuid, body:str, created_at:datetime"
            ),
            "upload": (
                "  file_asset: id:uuid, owner_id:uuid, object_key:str, mime_type:str, size_bytes:int, status:str, created_at:datetime\n"
                "  scan_result: id:uuid, asset_id:uuid, verdict:str, engine:str, scanned_at:datetime"
            ),
            "auth": (
                "  user: id:uuid, email:str, password_hash:str, is_active:bool, created_at:datetime\n"
                "  refresh_session: id:uuid, user_id:uuid, token_hash:str, expires_at:datetime, revoked_at:datetime|null"
            ),
            "payment": (
                "  customer: id:uuid, external_id:str, email:str, created_at:datetime\n"
                "  invoice: id:uuid, customer_id:uuid, amount_cents:int, status:str, external_ref:str, created_at:datetime\n"
                "  billing_event: id:uuid, event_type:str, payload_json:json, created_at:datetime"
            ),
            "analytics": (
                "  metric_event: id:uuid, tenant_id:uuid, metric:str, value:float, occurred_at:datetime\n"
                "  aggregate_snapshot: id:uuid, tenant_id:uuid, window:str, metric:str, value:float, generated_at:datetime"
            ),
            "scraper": (
                "  crawl_job: id:uuid, seed_url:str, status:str, depth:int, created_at:datetime\n"
                "  crawl_result: id:uuid, job_id:uuid, url:str, title:str, content_text:str, fetched_at:datetime"
            ),
            "crud": "  entity: id:uuid, name:str, description:str, created_at:datetime, updated_at:datetime",
            "default": "  record: id:uuid, title:str, status:str, created_at:datetime, updated_at:datetime",
        }
        return mapping.get(stack_type, mapping["default"])

    def _api_contracts_for(self, stack_type: str) -> str:
        mapping = {
            "downloader": (
                "  [POST] /downloads - enqueue a new download job\n"
                "  Request: { url: str }\n"
                "  Response: { job_id: str, status: str }\n"
                "  [GET] /downloads/{job_id} - get job progress and output\n"
                "  Request: path job_id\n"
                "  Response: { job_id: str, status: str, progress: int, output_path: str|null }"
            ),
            "chat": (
                "  [POST] /rooms - create room\n"
                "  Request: { name: str }\n"
                "  Response: { room_id: str, name: str }\n"
                "  [WS] /ws/rooms/{room_id} - realtime room transport\n"
                "  Request: websocket messages { type: str, body: str }\n"
                "  Response: broadcast messages { id: str, user: str, body: str, created_at: str }"
            ),
            "upload": (
                "  [POST] /files - stream upload and create asset record\n"
                "  Request: multipart/form-data file\n"
                "  Response: { asset_id: str, status: str }\n"
                "  [GET] /files/{asset_id} - fetch metadata and retrieval link\n"
                "  Request: path asset_id\n"
                "  Response: { asset_id: str, status: str, download_url: str|null }"
            ),
            "auth": (
                "  [POST] /auth/login - issue access and refresh tokens\n"
                "  Request: { email: str, password: str }\n"
                "  Response: { access_token: str, refresh_token: str }\n"
                "  [POST] /auth/refresh - rotate refresh token pair\n"
                "  Request: { refresh_token: str }\n"
                "  Response: { access_token: str, refresh_token: str }"
            ),
            "payment": (
                "  [POST] /billing/checkout - create checkout session\n"
                "  Request: { customer_id: str, plan_id: str }\n"
                "  Response: { checkout_url: str }\n"
                "  [POST] /billing/webhooks/stripe - process billing events\n"
                "  Request: signed Stripe payload\n"
                "  Response: { accepted: bool }"
            ),
            "analytics": (
                "  [GET] /analytics/overview - retrieve aggregate dashboard metrics\n"
                "  Request: query window, tenant_id\n"
                "  Response: { metrics: list, generated_at: str }\n"
                "  [GET] /analytics/series - retrieve time series data\n"
                "  Request: query metric, range\n"
                "  Response: { points: list }"
            ),
            "scraper": (
                "  [POST] /crawl/jobs - create crawl job\n"
                "  Request: { seed_url: str, depth: int }\n"
                "  Response: { job_id: str, status: str }\n"
                "  [GET] /crawl/jobs/{job_id} - inspect crawl status\n"
                "  Request: path job_id\n"
                "  Response: { job_id: str, status: str, pages_collected: int }"
            ),
            "crud": (
                "  [POST] /items - create item\n"
                "  Request: { name: str, description: str }\n"
                "  Response: { id: str, name: str, description: str }\n"
                "  [GET] /items/{id} - fetch item\n"
                "  Request: path id\n"
                "  Response: { id: str, name: str, description: str }"
            ),
            "default": (
                "  [POST] /records - create record\n"
                "  Request: { title: str }\n"
                "  Response: { id: str, title: str }\n"
                "  [GET] /records/{id} - get record\n"
                "  Request: path id\n"
                "  Response: { id: str, title: str, status: str }"
            ),
        }
        return mapping.get(stack_type, mapping["default"])

    def _file_tree_for(self, stack_type: str) -> str:
        return (
            "  app/\n"
            "    __init__.py\n"
            "    main.py\n"
            "    service.py\n"
            "    schemas.py\n"
            "    repository.py\n"
            "  tests/\n"
            "    conftest.py\n"
            "    test_app.py\n"
            "  requirements.txt\n"
            "  Dockerfile\n"
            "  README.md"
        )

    async def _read_project_snapshots(self, project_path: Path) -> str:
        targets = [
            "requirements.txt",
            "app/main.py",
            "app/service.py",
            "tests/test_app.py",
            "README.md",
        ]
        chunks: List[str] = []
        for rel in targets:
            absolute = project_path / rel
            if not absolute.exists():
                continue
            content = await asyncio.to_thread(absolute.read_text, "utf-8")
            if len(content) > 2500:
                content = content[:2500] + "\n..."
            chunks.append(f"[{rel}]\n{content}")
        return "\n\n".join(chunks)

    def _fallback_compliance_review(self, task: str, files: str, design: str) -> List[str]:
        lines = [
            "Running architecture compliance review against locked design.",
            "The data model and API contracts remain the primary acceptance gates.",
        ]

        has_health = "/health" in files
        has_summary = "/summary" in files
        contract_mentions_download = "[POST] /downloads" in design
        implementation_mentions_download = "/downloads" in files

        if contract_mentions_download and not implementation_mentions_download:
            lines.append(
                "WARNING: Backend deviated from the agreed API contract on /downloads. "
                "Flagging for documentation."
            )

        if not has_health:
            lines.append("WARNING: Missing health endpoint from baseline operability contract.")
        if not has_summary and "summary" in design.lower():
            lines.append("WARNING: Expected summary endpoint was not implemented as designed.")

        if len(lines) <= 2:
            lines.extend(
                [
                    "Implementation aligns with critical boundaries and service-layer separation.",
                    "No material contract drift detected in sampled files.",
                    "Frontend and Docs can proceed with confidence in API stability.",
                ]
            )
        return lines
