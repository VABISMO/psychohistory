<div align="center">

# 🔮 Psychohistory Harness

**A hybrid mathematical–LLM engine for grounded socio-political forecasting.**

Inspired by Asimov's *Foundation*, it projects branching futures from any scenario —
every number grounded in the live web, every probability computed by a transparent model.

[![Backend](https://img.shields.io/badge/backend-FastAPI-009688?logo=fastapi&logoColor=white)](#-architecture)
[![Frontend](https://img.shields.io/badge/frontend-Next.js%2016-000000?logo=next.js&logoColor=white)](#-architecture)
[![Inference](https://img.shields.io/badge/inference-Ollama%20(local)-FF6F00)](https://ollama.com)
[![Tests](https://img.shields.io/badge/tests-24-brightgreen)](#-testing)
[![License](https://img.shields.io/badge/license-MIT-blue)](#-license)

</div>

---

## ✨ Features

- **Branching forecast trees** — depth `D` produces `3^d` nodes per level (`3 → 12 → 39 → 120`), each spawn split into *continuation / optimistic / pessimistic*.
- **Real web grounding** — live Google News + Wikipedia (ES + EN) retrieval, so events name **real actors**.
- **No impossible numbers** — institutional maxima (e.g. a chamber's **350 seats**) are fetched online and injected as hard caps. *Nothing is hardcoded.*
- **Transparent mathematics** — probabilities from a normal‑CDF model with trend, radiant and depth‑decay modulation.
- **Resilient by design** — a brace‑aware JSON *salvage* parser recovers complete events even from truncated LLM output; missing slots are back‑filled mathematically so a forecast is **always** returned.
- **Global summary modal** — one paragraph for the single most plausible trajectory.
- **Local & private** — inference runs on your machine via Ollama; MongoDB falls back to in‑memory `mongomock`.
- **Hardened backend** — bottleneck serialisation, sliding‑window rate limits, exponential‑backoff retries.
- **Dynamic model selector** — the UI lists the models actually installed in your Ollama registry.
- **Production-ready frontend** — optimized static build, all pages prerendered, no dev tooling.

---

## 🏗 Architecture

```
┌──────────────────────────┐        ┌───────────────────────────────────────────┐
│        Frontend          │  HTTP  │                 Backend (FastAPI)          │
│  Next.js + React Flow    │ ─────► │  /predict   /models   /predictions         │
│  prompt · depth · model  │        │                                            │
│  interactive graph+modal │        │   retrieve_context ─┐  live web (News+Wiki) │
└──────────────────────────┘        │   retrieve_constraints┘  ES + EN           │
                                    │            │                               │
                                    │            ▼                               │
                                    │   single constrained LLM call (Ollama)     │
                                    │            │                               │
                                    │   salvage-parse → assemble tree → math     │
                                    │            │                               │
                                    │            ▼                               │
                                    │   MongoDB  (or in-memory mongomock)        │
                                    └───────────────────────────────────────────┘
```

**Request lifecycle:** retrieve real context → discover real‑world bounds → one constrained LLM call → resilient parse → assemble tree → attach mathematical indices → persist.

---

## 🧮 The Mathematics

The core is fully deterministic and documented in the in‑app whitepaper (`/paper`).

**Normal CDF operator**

```
Φ(x) = ½ · (1 + erf(x / √2))
```

**Base branch probabilities** — with magnitude `m ∈ [0,1]`, threshold `C_c = 0.5`, dispersion `σ_C = 0.15`:

```
C_t    = 0.5 + 0.3 · m
b_same = Φ( (C_t − C_c) / σ_C )
b_max  = b_same · ( 0.7 + 0.2 · (1 − 0.15 · depth) )
b_min  = max( 0.05 , b_same · ( 0.15 + 0.01 · depth ) )
```

**Trend / radiant modulation** — shifts for `rising`, `declining`, `volatile`; a *radiant* event adds `+0.25 / +0.15 / −0.10`.

**Depth decay** — distant futures are systematically less certain:

```
δ(depth) = exp( −0.3 · depth )      b_* ← clamp01( b_* · δ )
```

**Global indices**

```
λ(node) = b_same(magnitude, trend, radiant, depth = level − 1)   # stability
κ       = (1/N) · Σ λ_i                                          # coherence ∈ [0,1]
```

**Tree size**

```
N(D) = Σ_{d=1}^{D} 3^d = (3^(D+1) − 3) / 2
```

> Full derivations, parameter tables and references live at **`/paper`** in the running app.

---

## 📦 Dependencies

**Backend** — Python **3.11+** (tested on 3.13). Installed from `backend/requirements.txt`:

| Package        | Role                                  |
| -------------- | ------------------------------------- |
| `fastapi`      | HTTP API                              |
| `uvicorn`      | ASGI server                           |
| `httpx`        | async web retrieval (News/Wikipedia)  |
| `pymongo`      | MongoDB driver                        |
| `mongomock`    | in‑memory Mongo fallback              |
| `pydantic`     | data models / validation              |
| `python-dotenv`| `.env` loading                        |
| `pytest`       | test runner                           |

**Frontend** — Node.js **20+** (tested on 26). Key packages from `frontend/package.json`:

| Package                    | Role                          |
| -------------------------- | ----------------------------- |
| `next` (16) / `react` (19) | framework                     |
| `@xyflow/react`            | interactive prediction graph  |
| `tailwindcss` (v4)         | styling                       |
| `lucide-react`             | icons                         |
| `@radix-ui/*`              | accessible primitives         |

**Runtime services**

| Service  | Required | Notes                                              |
| -------- | -------- | -------------------------------------------------- |
| Ollama   | **yes**  | local LLM inference; pull at least one model       |
| MongoDB  | no       | optional; falls back to in‑memory `mongomock`      |

---

## 🚀 Quickstart

### 0. Clone

```bash
git clone https://github.com/VABISMO/psychohistory.git
cd psychohistory
```

### 1. Install Ollama and a model

Install [Ollama](https://ollama.com), make sure the daemon is running, then pull a model:

```bash
ollama pull llama3
```

### 2. Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # adjust if needed
uvicorn psychohistory.main:app --host 0.0.0.0 --port 8000
```

### 3. Frontend

```bash
cd frontend
npm install
npm run dev -- --port 3000
```

### 4. One‑shot launchers

```bash
./start.sh          # development (hot reload)
./start-prod.sh     # production (optimized build, no dev tooling)
```

Both start backend + frontend in the background and log to `.logs/`.

| Service    | URL                          |
| ---------- | ---------------------------- |
| Frontend   | http://localhost:3000        |
| Whitepaper | http://localhost:3000/paper  |
| Backend    | http://localhost:8000        |
| API docs   | http://localhost:8000/docs   |

---

## 🖥️ Usage

1. Open **http://localhost:3000**.
2. **Write a scenario** in the prompt bar — or click one of the example chips
   (US midterms, Russia/Ukraine, China 21st Party Congress, Iran, Israel, Europe, Spain 29‑Nov‑2026).
3. Choose the **Depth** (1–5) — how many generations of future events to project.
4. Pick a **model** from the dropdown (populated from your local Ollama registry).
5. Press **Generate prediction** (or `Cmd/Ctrl + Enter`). CPU inference takes a few minutes.
6. **Explore the graph** — pan/zoom; click any node for its details: probability,
   magnitude, trend, estimated **date** and **duration**.
7. Click **Summary** to open the modal with the single most plausible future.
8. The whitepaper (**`/paper`**) is linked discreetly in the header ("whitepaper").

---

## 🏭 Production

Build once, serve optimized:

```bash
./start-prod.sh
```

Or manually:

```bash
# frontend
cd frontend
npm run build
npm start -- --port 3000      # or: ./node_modules/.bin/next start --port 3000

# backend
cd backend
venv/bin/uvicorn psychohistory.main:app --host 0.0.0.0 --port 8000
```

`npm run build` prerenders both routes as static (`○ /`, `○ /paper`) and runs
TypeScript checks. In production there is **no developer overlay or dev‑tools
indicator** at all.

For a hardened backend deployment, put the FastAPI app behind a process manager
and a reverse proxy, and set `MONGO_URI` to a real MongoDB instance.

---

## 🔌 API

| Method | Endpoint       | Description                                            |
| ------ | -------------- | ------------------------------------------------------ |
| `POST` | `/predict`     | Generate a forecast tree. Body: `prompt`, `depth`, `model?` |
| `GET`  | `/models`      | List installed Ollama models, best‑first + `default`   |
| `GET`  | `/predictions` | Recent persisted predictions                           |
| `GET`  | `/`            | Health check                                           |

**Example**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Spain snap general election 29 November 2026", "depth": 2}'
```

---

## ⚙️ Configuration

Configuration is read from `backend/.env`:

| Variable      | Default                     | Description                                   |
| ------------- | --------------------------- | --------------------------------------------- |
| `MONGO_URI`   | `mongodb://mongo:27017/`    | Mongo connection; falls back to `mongomock`   |
| `OLLAMA_HOST` | `http://localhost:11434`    | Ollama endpoint                               |

Tuning knobs (in `psychohistory/services/predictor.py`): `PREFERRED_MODELS`, `SEMANTIC_MAX_DEPTH`, sampling options and token caps.

---

## 🧪 Testing

Backend (24 tests — probability math, normalisation, JSON salvage/grouping, tree assembly):

```bash
cd backend
venv/bin/pytest tests/ -v
```

Frontend type‑check and lint:

```bash
cd frontend
npx tsc --noEmit
npm run lint
```

---

## 🛠 Troubleshooting

| Symptom | Fix |
| ------- | --- |
| Model dropdown says **"no local models"** | Start Ollama (`ollama serve`) and `ollama pull llama3`. |
| **Predictions are very slow** | CPU inference (~6–7 tok/s). Use a smaller model and lower depth. |
| MongoDB not installed | Nothing to do — the backend auto‑falls back to in‑memory `mongomock`. |
| `Address already in use` | Another process holds port 3000/8000; kill it or change the port. |
| Stale/odd frontend behaviour in dev | Delete `frontend/.next` and restart. |
| Dev overlay / issue badge appearing | It is Next's error indicator — it only shows on a real error. Production builds never show it. |

---

## 📁 Project Structure

```
psychohistory/
├── backend/
│   ├── psychohistory/
│   │   ├── main.py              # FastAPI app, routes, rate-limit middleware
│   │   ├── models.py            # Pydantic models (Event, Node, Result)
│   │   ├── config_db.py         # Mongo with mongomock fallback
│   │   └── services/
│   │       ├── predictor.py     # engine: prompt, salvage parser, math fallback
│   │       ├── search.py        # News + Wikipedia, constraint discovery
│   │       └── limiter.py       # Bottleneck, RateLimiter, retries
│   ├── tests/test_math.py       # 24 tests
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   └── src/
│       ├── app/page.tsx             # main UI
│       ├── app/paper/page.tsx       # whitepaper
│       ├── components/prediction-graph.tsx
│       └── lib/api.ts
├── start.sh                     # development launcher
├── start-prod.sh                # production launcher (build + serve)
├── ROADMAP.md                   # future development checklist
└── README.md
```

---

## 🧭 Limitations & Ethics

Psychohistory is fiction; this is an **exploratory reasoning aid, not an oracle**. Probabilities are heuristic, reflect modelling choices, and are not measured frequencies. LLM narratives can inherit bias from training data and retrieved sources. Grounding reduces — but cannot eliminate — hallucination. Use outputs to inform human judgement, never to replace it, and never to defame real people. See [`ROADMAP.md`](ROADMAP.md) for planned improvements.

---

## 📜 License

Released under the **MIT License**. See `LICENSE`.

<div align="center"><sub>Generated locally · deterministically documented.</sub></div>
