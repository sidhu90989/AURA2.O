# AURA 2.0

> Intelligent multi-modal assistant (NLP, Vision, Voice, System Monitoring, Web Control, Maps, Memory) — initial repository setup.

## 🚀 Features (Planned / Present Code Modules)
- `nlp_processor.py` – Natural language understanding / intent processing.
- `vision_processor.py` – Face recognition & visual analysis (uses `vision_db/face_vectors.db`).
- `voice_interface.py` – Speech input/output handling.
- `memory_manager.py` – Long/short term memory utilities.
- `map_service.py` – Mapping & geospatial utilities.
- `neuro_web_controller.py` – Web automation / control.
- `system_monitor.py` – Host system metrics & health.
- `ethical.py` / `security.py` / `access.py` – Guardrails, permissions & policies.

## ⚠️ Security & Privacy
Biometric data (faces) and encryption keys SHOULD NOT be committed.
This repo's `.gitignore` excludes:
- `nlp-env/` (virtual environment)
- `known_faces/` and encrypted face store `known_faces_.enc/`
- `*.key` encryption keys
- `*.log` runtime logs

If you need to provide sample data, create sanitized assets under `samples/` and whitelist them.

## 🔐 Secrets & Keys
Store runtime secrets in environment variables or a `.env` file that remains untracked.
Example:
```
AURA_OPENAI_API_KEY=sk-...
AURA_ENV=dev
```
Load with `python-dotenv` or your own config loader.

## 🛠️ Quick Start
Minimal core (lightweight orchestrator only):
```bash
python -m venv aura-min
source aura-min/bin/activate  # Windows: .\\aura-min\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt   # minimal requirements (requests + dotenv, optional cryptography)
python main.py
```

Full feature install (adds NLP models, vision, speech, web automation, graph DB):
```bash
python -m venv aura-full
source aura-full/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-extras.txt
python main.py
```

## 📦 Dependencies
Slim workspace strategy:
- `requirements.txt`: minimal runtime for orchestrator scaffold.
- `requirements-extras.txt`: heavy / optional components (ML, vision, speech, selenium, graph, etc.).

Add or upgrade a package (full mode):
```bash
pip install somepackage==1.2.3
pip freeze | grep somepackage
```
Then append to `requirements-extras.txt` (or `requirements.txt` only if truly minimal core).

Rebuild minimal environment:
```bash
python -m venv aura-min
source aura-min/bin/activate
pip install -r requirements.txt
```

## 🆕 Backend (FastAPI) Architecture (WIP)
New backend scaffold under `backend/app` provides modular services:
- `core/` – config + database session
- `models/` – SQLAlchemy models (`User`, `Memory`)
- `services/vectorstore.py` – FAISS (or fallback) embedding store
- `services/memory_service.py` – memory persistence + similarity search
- `services/nlp_service.py` – wrapper to existing `EnhancedNLPEngine` (placeholder)
- `routers/` – `memory`, `nlp`, `voice`, `system` endpoints

Run (after installing `backend/requirements.txt`):
```bash
uvicorn backend.app.main:app --reload
```

Environment variables (example):
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=...
DATABASE_URL=postgresql+psycopg2://postgres:postgres@localhost:5432/aura
VECTOR_STORE_DIR=./vector_store
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### Data Flow (Chat)
1. Client sends messages -> `/nlp/chat`
2. Service retrieves semantic neighbors from vector store
3. Context packaged for provider (future real GPT/Claude call)
4. Response returned + optionally stored as memory

### Memory Strategy
- Short-term: last N exchanges (in-process or Redis planned)
- Long-term: `memories` table + FAISS vectors + (future) knowledge graph (Neo4j from `memory_manager.py`)
- Retrieval: hybrid lexical + embedding (current: embedding only placeholder)

## 🌐 Frontend (Next.js + Tailwind) Dashboard
Implemented futuristic HUD dashboard located in `frontend/`.

Structure (current):
```
frontend/
  app/
    layout.tsx          # Root layout shell + dynamic status bar
    page.tsx            # Main dashboard grid
  components/
    ChatPanel.tsx       # Chat with memory context
    VoicePanel.tsx      # Placeholder voice control (STT/TTS roadmap)
    SystemPanel.tsx     # Live system metrics via /system/snapshot
    MemoryPanel.tsx     # Memory search UI
    GraphPanel.tsx      # Knowledge graph store/query
    StatusBar.tsx       # Polls backend health + system readiness
  lib/
    api.ts              # Fetch/post helpers
  styles/
    globals.css         # Tailwind + custom glass / neon styles
  next-env.d.ts
  tsconfig.json
```

Install & Run:
```bash
cd frontend
npm install
npm run dev
```
Environment variable:
```
NEXT_PUBLIC_AURA_API_BASE=http://localhost:8000
```
Adjust the value if the backend runs elsewhere (container, remote host, etc.).

Status Bar Logic:
- API: /health (online/error)
- MEM: /system/snapshot success => ready
- NLP: placeholder (will update once streaming/queue added)

Planned Enhancements:
- WebSocket streaming for token-by-token responses
- Real-time transcription waveform
- Memory timeline with consolidation markers
- Graph visualization (force layout) replacing raw JSON list

## 🧠 Human-Like Cognition Layer (Design Roadmap)
| Capability | Mechanism | Status |
|------------|----------|--------|
| Emotional state tracking | Sentiment + affect model mapping to session state | Planned |
| Context graph | Neo4j `KnowledgeGraph` + entity linking | Existing module (integration pending) |
| Memory consolidation | Nightly job merging semantically similar memories | Planned |
| Reflection loop | Periodic summarizer stores distilled insights | Planned |
| Explainability | Chain-of-thought metadata (internal) + user-facing rationale | Planned |
| Adaptive persona | Personality vector adjusted by feedback | Planned |

Pipeline concept:
```
Input -> NLP Analysis -> Emotion/Intent -> Memory Retrieval -> Reasoning Core -> Response Draft -> Safety/Ethics Filter -> Output (Text/Voice)
```

## 🔐 Security Evolution
Reuse `security.py` for:
- Session encryption tokens
- Audit logging of sensitive operations
- Future: JWT / OAuth2 layer for external clients

## ✅ Upcoming Tasks
- Migrations (Alembic) + indexing strategy
- Real provider integration (OpenAI/Anthropic) in `nlp_service`
- Whisper local/remote STT
- ElevenLabs / Coqui TTS streaming
- Hybrid retrieval (BM25 + dense)
- Frontend Next.js scaffold + WebSocket streaming
- Auth & RBAC for multi-user usage

## 🧪 Testing (Planned Additions)
`backend/app/tests/` will include:
- Memory add + search
- Health + system snapshot

## 🧪 Quick Smoke Test (API)
```bash
uvicorn backend.app.main:app --reload
curl http://localhost:8000/health
curl -X POST http://localhost:8000/memory/ -H 'Content-Type: application/json' -d '{"text":"Hello memory"}'
```

## 🔌 API Endpoint Summary (Updated)
| Category | Endpoint | Method | Description |
|----------|----------|--------|-------------|
| Core | `/health` | GET | Health check |
| Memory | `/memory/` | POST | Store memory (text) |
| Memory | `/memory/search` | GET | Similarity search (q,k) |
| NLP | `/nlp/chat` | POST | Chat with contextual memory retrieval |
| Voice | `/voice/stt` | POST | Speech-to-text (placeholder) |
| Voice | `/voice/tts` | POST | Text-to-speech (placeholder) |
| System | `/system/snapshot` | GET | Basic system metrics |
| Security | `/security/encrypt` | POST | Encrypt arbitrary string |
| Security | `/security/decrypt` | POST | Decrypt token |
| Graph | `/graph/store` | POST | Store interaction in knowledge graph |
| Graph | `/graph/related` | GET | Query related nodes by text fragment |
| Graph | `/graph/emotion` | GET | Query interactions by emotion |
| Vision | `/vision/identify` | POST | Upload image (placeholder processing) |

### Example: Chat with Memory
```bash
curl -X POST http://localhost:8000/nlp/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages":["Hello AURA, summarize yourself."], "store": true, "include_memories": true}'
```

### Example: Encrypt / Decrypt
```bash
TOKEN=$(curl -s -X POST http://localhost:8000/security/encrypt -H 'Content-Type: application/json' -d '{"data":"classified"}' | jq -r .token)
curl -X POST http://localhost:8000/security/decrypt -H 'Content-Type: application/json' -d '{"token":"'$TOKEN'"}'
```

If modules (security, graph, vision) are not fully configured, endpoints return fallback statuses rather than failing.

## 🗺️ Suggested Next Steps
- Add a LICENSE (e.g., MIT) if you plan to make this public/open-source.
- Add structured logging configuration (rotate, JSON logs).
- Implement unit tests for safety-critical modules (`security.py`, `ethical.py`).
- Provide an architecture diagram.
- Add type hints & mypy config.

## 🤝 Contributing
1. Fork
2. Create feature branch
3. Commit with conventional style (`feat:`, `fix:`)
4. Open PR

## 📄 License
Not yet selected. Add a `LICENSE` file (MIT/Apache-2.0/BSD-3-Clause) before broad sharing.

---
Initial scaffold generated. Update this README as functionality evolves.
