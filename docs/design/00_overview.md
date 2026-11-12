# MushRoom Academy — Design Overview

## Vision

Transform the mushroom safety project from a static Gradio demo into an interactive **training simulation** where users guide a cartoon cat character ("Professor Whiskers") from mushroom novice to expert mycologist. The journey mirrors the active learning pipeline: the cat learns from each mushroom the user presents, makes predictions (sometimes wrong), and gradually becomes more accurate — visualizing the real AL algorithm underneath.

---

## Why Rebuild the Stack

| Current (Gradio) | Limitation |
|---|---|
| Single-page, server-rendered | No persistent state, no animations, no character |
| No session/user state | Can't track a training journey across interactions |
| Limited layout control | Can't do sprite animations, progress maps, custom UI |
| No separation of concerns | Frontend and backend tightly coupled in one Python file |

**New stack: React (frontend) + FastAPI (backend)** — gives us full control over animations, state, routing, and deploys cleanly on AWS.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     React Frontend                       │
│  ┌───────────┐  ┌──────────┐  ┌───────────────────────┐ │
│  │ Cat Agent  │  │ Training │  │ Mushroom Detection    │ │
│  │ Animations │  │ Journey  │  │ Playground            │ │
│  └───────────┘  └──────────┘  └───────────────────────┘ │
│         ↕              ↕               ↕                 │
│              Zustand State Manager                       │
│                        ↕                                 │
│                   REST / WebSocket                       │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────┼────────────────────────────────┐
│                   FastAPI Backend                         │
│  ┌─────────────┐  ┌──────────┐  ┌────────────────────┐  │
│  │ Session Mgr  │  │ AL Engine│  │ Gemini Bridge      │  │
│  │ (per-user    │  │ (pluggable│  │ (feature extract   │  │
│  │  cat state)  │  │  models)  │  │  + direct judge)   │  │
│  └─────────────┘  └──────────┘  └────────────────────┘  │
│         ↕              ↕               ↕                 │
│              ML Package Layer (versioned)                 │
│  ┌─────────────┐  ┌──────────┐  ┌────────────────────┐  │
│  │ Preprocessor │  │ Learners │  │ Query Strategies   │  │
│  └─────────────┘  └──────────┘  └────────────────────┘  │
└──────────────────────────────────────────────────────────┘
                         │
┌────────────────────────┼────────────────────────────────┐
│                      AWS                                 │
│   S3 (models/assets) ─ ECS Fargate ─ CloudFront (CDN)   │
└──────────────────────────────────────────────────────────┘
```

---

## Document Index

| Document | Contents |
|---|---|
| [01_app_architecture.md](01_app_architecture.md) | Frontend/backend structure, tech stack, API routes, state management |
| [02_cat_agent.md](02_cat_agent.md) | Cat character design, personality, animation states, all art assets with AI generation prompts |
| [03_training_journey.md](03_training_journey.md) | Full gameplay loop — how a user trains the cat from novice to expert, level system, active learning integration |
| [04_ml_packages.md](04_ml_packages.md) | ML module interfaces, packaging strategy, versioning, how to swap models independently |
| [05_aws_deployment.md](05_aws_deployment.md) | Docker setup, AWS services, CI/CD, environment configuration |

---

## Tech Stack Summary

| Layer | Technology | Why |
|---|---|---|
| Frontend | React 18 + TypeScript | Component model, animation ecosystem, huge community |
| Styling | Tailwind CSS | Rapid UI, responsive, utility-first |
| Animations | Framer Motion + Lottie | Sprite transitions, cat reactions, smooth state changes |
| State | Zustand | Lightweight, no boilerplate, perfect for game-like state |
| Backend | FastAPI (Python) | Async, fast, native Pydantic validation, keeps ML in Python |
| ML Runtime | scikit-learn, XGBoost (packaged) | Existing models, pluggable interface |
| Vision AI | Google Gemini API | Already integrated, feature extraction + direct judge |
| Database | SQLite (dev) / PostgreSQL (prod) | Session persistence for training journeys |
| Deployment | Docker + AWS ECS Fargate | Serverless containers, auto-scaling, no server management |
| CDN | AWS CloudFront + S3 | Static assets (cat sprites, sounds) served fast globally |

---

## Core User Flows

### Flow 1: Training Journey (Main Experience)
```
Start New Journey → Meet the Cat (Level 0: Clueless)
    → User uploads mushroom photo
    → Cat examines it (animation)
    → Cat makes prediction (often wrong at first)
    → User reveals ground truth (oracle label)
    → Cat reacts (happy if right, sick if false negative on poison)
    → Model retrains with new data point
    → Repeat until cat reaches expert level
    → See learning curves, compare strategies
```

### Flow 2: Expert Mode (Original Demo, Enhanced)
```
Upload photo → Both judges analyze → Cat presents verdict → Final consensus
```

### Flow 3: Strategy Comparison (Research Visualization)
```
Pick 2+ query strategies → Watch cats train in parallel → Compare learning curves
```
