# Video AI PoC

**English** ｜ [中文](README.zh-CN.md)

A video-understanding proof of concept: **cheap CV carries the volume, tracking reuse handles repetition, a vector memory remembers subjects, and the LLM is only called when it matters** — accurate recognition at low cost.

> 🔗 Repo: https://github.com/Zhijing-W/video-ai-poc

---

## Runtime logic flow (currently implemented)

![Runtime logic tree](video-understanding-poc/docs/phase3-logic-flow.png)

**Feature-toggle decision tree (multi-branch + path merging):**

![Decision tree](video-understanding-poc/docs/phase3-decision-tree.png)

## Phase status

| Phase | Scope | Status |
|---|---|---|
| **Phase 1** | LLM-first MVP: video → ffmpeg frames → gpt-4o → structured JSON | ✅ Done |
| **Phase 2** | Cost-controlled hybrid: YOLO detection + event gating + smart frame sampling; LLM called per-event only | ✅ Done |
| **Phase 3** | Track-and-identify + subject memory: ByteTrack, ReID FAISS gallery, multi-frame fusion, live identity integration, eval script | ✅ Done |
| **Phase 4** | Customer-aligned identity-aware multi-frame event understanding (face + body + gait → identity → event understanding) | 📝 Design |
| **Phase 5** | End-to-end on Azure (ingest → AML inference → playback) | 📝 Design |

## Key capabilities delivered in Phase 3

| Capability | Module |
|---|---|
| Multi-object tracking (ByteTrack, stable track_id, per-session isolation) | `app/tracker.py`, `/track` |
| Three-clock decoupling / track gating (reuse conclusion when tracks unchanged; call LLM only for new subjects) | `app/services/track_gate.py` |
| Fine-grained perception v1 (YOLO-Pose torso color sampling, fixes color misjudgement) | `app/pose.py` |
| Subject-memory ReID vector store (FAISS cosine + open-set enrollment + quality gate + negative cache) | `app/gallery.py`, `app/reid.py`, `/identify` |
| Multi-cue fusion + best-frame voting (temporal / ReID / color / motion; face slot reserved) | `app/track_fusion.py`, `/fusion` |
| Live identity integration (recognize returning subjects, cross-track reuse, frontend display) | `app/services/identity_integration.py` |
| Evaluation harness (recognition precision/recall + LLM calls per video; proves savings without losing accuracy) | `scripts/eval_phase3.py` |

## Quick start

```powershell
cd video-understanding-poc

# 1) Create a virtual env and install deps
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2) Configure Azure OpenAI
copy .env.example .env
#   Edit .env: AZURE_OPENAI_ENDPOINT / API_KEY / DEPLOYMENT (a vision model)

# 3) Run
.\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
```

Open `http://127.0.0.1:8000/` (live monitor) — Swagger at `/docs`, health at `/health`.

## Branches

| Branch | Purpose |
|---|---|
| `main` | Integration trunk |
| `snapshot/baseline-phase1-3` | 🧊 Frozen snapshot — restore point before the task pivot |
| `feature/event-understanding` | 🚧 Ongoing work — the new event-understanding direction |

## Docs

- `docs/phase/` — Phase 1–5 design documents
- `assets/` — per-phase architecture diagrams

---

> 🧭 Motto: cheap models carry the volume, tracking reuse handles repetition, the vector store remembers, the LLM only adjudicates the grey zone.
