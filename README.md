<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%2B-3B82F6?style=flat-square" alt="Python 3.11+" />
  <img src="https://img.shields.io/badge/License-MIT-22C55E?style=flat-square" alt="MIT License" />
  <img src="https://img.shields.io/github/stars/DhruvTilva/swarm?style=flat-square&color=FFD700" alt="GitHub stars" />
  <img src="https://img.shields.io/badge/Mode-Multi--Agent-F97316?style=flat-square" alt="Multi-Agent" />
</p>

<h1 align="center">🐝 SWARM</h1>

<p align="center">
  <img src="https://readme-typing-svg.herokuapp.com?size=25&center=true&vCenter=true&width=600&lines=Builds+itself.+Adapts+instantly.;That’s+AI-native." />
</p>


<div align="center">
<pre>
███████╗██╗    ██╗ █████╗ ██████╗ ███╗   ███╗
██╔════╝██║    ██║██╔══██╗██╔══██╗████╗ ████║
███████╗██║ █╗ ██║███████║██████╔╝██╔████╔██║
╚════██║██║███╗██║██╔══██║██╔══██╗██║╚██╔╝██║
███████║╚███╔███╔╝██║  ██║██║  ██║██║ ╚═╝ ██║
╚══════╝ ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝
</pre>
</div>


<p align="center"><strong>Six AI agents. One command. Real working software.</strong></p>
<p align="center"><strong>Not a prototype. Not a demo. Production code ships in minutes.</strong></p>

## ⚡ See It In Action

![Swarm Terminal Dashboard](./doc/demo.gif)

> 🐝 Six agents argue, build, and ship together.

## 🤖 Meet The Team

| Agent | Role | Superpower |
|-------|------|----------|
| 🏗️ Architect | System designer | Analyzes task deeply, chooses right stack, produces design all agents follow |
| ⚙️ Backend | API engineer | Reads Architect's design, generates production-grade task-specific code |
| 🎨 Frontend | UI engineer | Thinks from first principles, generates stunning dark-mode UI for any task |
| 🧪 Tester | QA enforcer | Actually RUNS the code, finds bugs, assigns fixes, blocks ship if broken |
| 📝 Docs | Technical writer | Reads actual code, documents reality not promises, Stripe-level quality |
| 📋 PM | Product manager | Writes real PRD, protects scope, triages bugs, makes ship decisions |

## 🔥 Why Swarm Hits Different

| Feature | ChatGPT | GitHub Copilot | Cursor | 🐝 Swarm |
|---------|---------|----------------|--------|----------|
| Multi-agent collaboration | ❌ | ❌ | ❌ | ✅ |
| Agents argue & debate decisions | ❌ | ❌ | ❌ | ✅ |
| Real-time cinematic terminal UI | ❌ | ❌ | ❌ | ✅ |
| Generates complete project structure | ❌ | ⚠️ | ⚠️ | ✅ |
| PM agent breaks requirements mid-build | ❌ | ❌ | ❌ | ✅ 😈 |
| Runs on GitHub Models API (free) | ❌ | ✅ | ❌ | ✅ |
| Open source & self-hostable | ❌ | ❌ | ❌ | ✅ |
| Builds Dockerfile + tests + docs | ❌ | ⚠️ | ⚠️ | ✅ |
| Tracks real cost per build | ❌ | ❌ | ❌ | ✅ |
| Works with 6 different LLMs | ❌ | ❌ | ❌ | ✅ |
| Tester actually runs the code | ❌ | ❌ | ❌ | ✅ |
| PM writes real PRD first | ❌ | ❌ | ❌ | ✅ |

> No tool matches this PM chaos. 😈
> It changes requirements mid-build to mirror real teams.

## ⚡ Build Phases

PLANNING → ARCHITECTURE REVIEW → IMPLEMENTATION → TESTING → DOCUMENTATION → PACKAGING → COMPLETE

| Phase | What Happens |
|-------|-------------|
| Planning | PM writes PRD. Architect designs system. |
| Architecture Review | PM approves design. All agents briefed. |
| Implementation | Backend builds API. Frontend builds UI. |
| Testing | Tester RUNS code. Finds bugs. Assigns fixes. Signs off. |
| Documentation | Docs reads actual code. Writes accurate docs. |
| Packaging | All files written. Git initialized. PM final review. |
| Complete | Working product delivered. |

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/DhruvTilva/swarm

cd swarm

# 2. Install
pip install -r requirements.txt

# 3. Build anything
python main.py "build me a YouTube video downloader with format selector"
```

# 🔥 Swarm handles anything. Try these:
```bash
python main.py "build me a real-time crypto price tracker with alerts"
python main.py "build me an AI-powered resume analyzer with scoring"
python main.py "build me a GitHub repository analytics dashboard"
python main.py "build me a personal finance dashboard with charts"
python main.py "build me an API rate limiter with Redis and analytics"
```
# 🧠 Complex? No problem.
```bash
python main.py "build me a multi-tenant SaaS boilerplate with auth, billing, and role-based access"
```
# 🎯 One command. Six agents. Real software.
# No matter what you describe — Swarm figures it out.

## 🛠️ Installation & Configuration

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

swarm.config.yaml:

```yaml
# GitHub Models (FREE — moderate quality)
provider: openai
api_key: "your_github_token"
model: gpt-4o
base_url: "https://models.inference.ai.azure.com"

# Groq (FREE tier — fastest inference)
# provider: groq
# api_key: "gsk_..."
# model: llama-3.3-70b-versatile

# Ollama (FREE forever — fully local and private)
# provider: ollama
# model: llama3.2
# base_url: "http://localhost:11434"

# Anthropic Claude (paid — highest quality)
# provider: anthropic
# api_key: "sk-ant-..."
# model: claude-sonnet-4-20250514

# Google Gemini (free tier available)
# provider: gemini
# api_key: "AIza..."
# model: gemini-2.0-flash

# OpenAI (paid - highest quality)
# provider: openai
# api_key: "sk-..."
# model: gpt-4o

temperature: 0.7
max_tokens: 2000
database_path: "swarm.db"
output_dir: "output"
```

Provider override from CLI:

```bash
python main.py "build me a todo app" --provider groq
python main.py "build me a todo app" --provider ollama
python main.py "build me a todo app" --model gpt-4o
python main.py "build me a todo app" --output ./my-output
```

## 🔌 Supported LLM Providers

| Provider | Cost | Speed | Quality | Get Started |
|----------|------|-------|---------|-------------|
| GitHub Models | ✅ FREE | Fast | Moderate | github.com/settings/tokens |
| Groq | ✅ FREE tier | ⚡ Fastest | Good | console.groq.com |
| Ollama | ✅ FREE forever | Medium | Good | ollama.ai |
| Google Gemini | ✅ Free tier | Fast | Good | aistudio.google.com |
| OpenAI | 💳 Paid | Fast | Excellent | platform.openai.com |
| Anthropic Claude | 💳 Paid | Fast | Excellent | console.anthropic.com |

## 💰 Cost Transparency

Swarm tracks every API call and shows estimated cost in the dashboard in real time.

Typical build costs:

| Provider | Avg Cost Per Build |
|----------|--------------------|
| GitHub Models | FREE |
| Groq free tier | FREE |
| Ollama | FREE |
| Gemini free tier | FREE |
| GPT-4o | ~$0.05 - $0.15 |
| Claude Sonnet | ~$0.03 - $0.10 |

Swarm is designed to be cost-efficient.
Most users pay nothing.

## 📦 Output Structure

```text
output/your-project/
├── README.md          ← viral-worthy, AI-written
├── requirements.txt   ← actual dependencies
├── Dockerfile         ← production ready
├── app/
│   ├── main.py        ← working FastAPI app
│   └── service.py     ← business logic
├── tests/
│   └── test_app.py    ← real tests
└── .gitignore
```

## 🧠 Built Different

Most AI coding tools feel like autocomplete on steroids.
Swarm moves differently.

You do not prompt one assistant.
You deploy a team.

The Architect draws system boundaries first.
The Backend challenges over-engineered design decisions.
The Frontend protects clean API contracts.
The Tester catches bugs before production sees them.
The Docs agent records what ships.
The PM changes requirements mid-build.

They communicate.
They argue.
They resolve.
They ship.
You watch.

## 🗺️ Roadmap

- [x] 6-agent multi-agent build system
- [x] Real-time cinematic terminal dashboard
- [x] GitHub Models API (free with Copilot Pro)
- [x] SQLite build memory and persistence
- [x] Cinematic boot sequence and animations
- [x] Multi-LLM support (Claude, Gemini, Groq, Ollama, OpenAI)
- [x] Intelligent Architect with real system design authority
- [x] Production-grade Backend with task-specific code generation
- [x] God-tier Frontend with first-principles UI generation
- [x] World-class Tester that actually runs and verifies code
- [x] Stripe-level Docs that reads actual code
- [x] PM with real PRD and ship decision authority
- [x] Cost tracking per LLM call
- [x] Provider health check on startup
- [ ] Web UI dashboard (browser-based)
- [ ] Custom agent personalities via config

## 🤝 Contributing

Swarm is early.
Join us and build real developer superpowers.

- Build magical AI collaboration workflows.
- Craft sharper terminal UI moments.
- Ship open-source tools that work on day one.

How to contribute:

1. Fork the repo.
2. Create your feature branch.
3. Build something legendary.
4. Open a PR with clear intent.

We review PRs within 48 hours.

## 📢 Share

If Swarm blew your mind:
→ ⭐ Star it
→ 🐦 Tweet it: "Watched 6 AI agents argue and ship software."
→  Share it on LinkedIn, Reddit r/programming, HN

## Footer

<p align="center">🐝 Swarm — because one AI is never enough</p>
<p align="center">If this blew your mind, star the repo. It takes 1 second.</p>
