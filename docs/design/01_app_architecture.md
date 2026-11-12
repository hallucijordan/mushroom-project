# App Architecture

## Directory Structure

```
mushroom-project/
├── frontend/                    # React app
│   ├── public/
│   │   └── assets/
│   │       ├── cat/             # All cat sprite sheets & Lottie JSON
│   │       ├── mushrooms/       # Mushroom icons, backgrounds
│   │       ├── ui/              # Buttons, frames, badges
│   │       └── sounds/          # SFX (meow, munch, sick, levelup)
│   ├── src/
│   │   ├── components/
│   │   │   ├── cat/
│   │   │   │   ├── CatAvatar.tsx        # Main cat renderer (picks sprite by state)
│   │   │   │   ├── CatDialogue.tsx      # Speech bubble with typewriter effect
│   │   │   │   ├── CatReaction.tsx      # Reaction animations (happy, sick, thinking)
│   │   │   │   └── CatProgressBadge.tsx # Level badge overlay
│   │   │   ├── training/
│   │   │   │   ├── TrainingArena.tsx    # Main training view (cat + mushroom + controls)
│   │   │   │   ├── MushroomUploader.tsx # Image upload with preview
│   │   │   │   ├── PredictionCard.tsx   # Cat's prediction display
│   │   │   │   ├── OracleReveal.tsx     # User reveals ground truth
│   │   │   │   ├── LearningCurve.tsx    # Real-time accuracy chart
│   │   │   │   └── StrategyPicker.tsx   # Choose query strategy
│   │   │   ├── expert/
│   │   │   │   ├── ExpertMode.tsx       # Original dual-judge enhanced view
│   │   │   │   ├── JudgePanel.tsx       # Single judge's verdict display
│   │   │   │   └── ConsensusBar.tsx     # Final weighted verdict
│   │   │   ├── journey/
│   │   │   │   ├── JourneyMap.tsx       # Visual progress through levels
│   │   │   │   ├── LevelCard.tsx        # Individual level details
│   │   │   │   └── Achievements.tsx     # Unlocked milestones
│   │   │   └── shared/
│   │   │       ├── Header.tsx
│   │   │       ├── Footer.tsx
│   │   │       ├── LoadingSpinner.tsx
│   │   │       └── MushroomCard.tsx     # Reusable mushroom info display
│   │   ├── pages/
│   │   │   ├── HomePage.tsx             # Landing — meet the cat, choose mode
│   │   │   ├── TrainingPage.tsx         # Active learning training session
│   │   │   ├── ExpertPage.tsx           # Expert detection mode
│   │   │   ├── JourneyPage.tsx          # View progress & stats
│   │   │   └── ComparisonPage.tsx       # Strategy comparison dashboard
│   │   ├── stores/
│   │   │   ├── catStore.ts              # Cat state (mood, level, XP)
│   │   │   ├── trainingStore.ts         # AL session state (pool, labeled, model)
│   │   │   └── sessionStore.ts          # User session, preferences
│   │   ├── hooks/
│   │   │   ├── useCatAnimation.ts       # Animation state machine
│   │   │   ├── useTrainingSession.ts    # Training loop lifecycle
│   │   │   └── useWebSocket.ts          # Real-time updates from backend
│   │   ├── services/
│   │   │   └── api.ts                   # Axios/fetch wrapper for backend
│   │   ├── types/
│   │   │   ├── cat.ts                   # CatState, CatMood, CatLevel types
│   │   │   ├── training.ts              # TrainingSession, QueryStrategy types
│   │   │   └── mushroom.ts              # Mushroom, Prediction, Verdict types
│   │   ├── utils/
│   │   │   ├── animations.ts            # Animation timing constants
│   │   │   └── formatters.ts            # Display helpers
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   └── tailwind.config.js
│
├── backend/                     # FastAPI app
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI app factory, CORS, lifespan
│   │   ├── config.py            # Settings via pydantic-settings (env vars)
│   │   ├── routers/
│   │   │   ├── training.py      # /api/training/* — AL session endpoints
│   │   │   ├── detection.py     # /api/detect/*   — single-image detection
│   │   │   ├── session.py       # /api/session/*  — user session CRUD
│   │   │   └── models.py        # /api/models/*   — model info, swap, retrain
│   │   ├── services/
│   │   │   ├── al_engine.py     # Active learning orchestrator
│   │   │   ├── gemini_bridge.py # Wraps Gemini calls (extract + direct judge)
│   │   │   ├── cat_brain.py     # Cat personality + dialogue generation
│   │   │   └── session_mgr.py   # Session persistence (SQLite/Postgres)
│   │   ├── schemas/
│   │   │   ├── training.py      # Pydantic models for training endpoints
│   │   │   ├── detection.py     # Pydantic models for detection endpoints
│   │   │   ├── cat.py           # Cat state schemas
│   │   │   └── session.py       # Session schemas
│   │   ├── db/
│   │   │   ├── database.py      # SQLAlchemy async engine setup
│   │   │   ├── models.py        # ORM models (Session, TrainingRun, etc.)
│   │   │   └── migrations/      # Alembic migrations
│   │   └── ws/
│   │       └── training_ws.py   # WebSocket for live training updates
│   ├── requirements.txt
│   └── Dockerfile
│
├── ml/                          # ML package (independently versioned)
│   ├── mushroom_ml/
│   │   ├── __init__.py
│   │   ├── preprocessor.py      # Migrated from src/data/preprocessor.py
│   │   ├── learners/
│   │   │   ├── __init__.py
│   │   │   ├── base.py          # Abstract learner interface
│   │   │   ├── random_forest.py # RF implementation
│   │   │   ├── logistic.py      # Bagged LR implementation
│   │   │   └── mlp.py           # Neural network implementation
│   │   ├── strategies/
│   │   │   ├── __init__.py
│   │   │   ├── base.py          # Abstract strategy interface
│   │   │   ├── random.py        # Random baseline
│   │   │   ├── uncertainty.py   # Uncertainty sampling
│   │   │   ├── committee.py     # Query-by-Committee
│   │   │   └── bald.py          # BALD (Bayesian AL by Disagreement)
│   │   ├── active_learning.py   # AL loop orchestrator
│   │   └── evaluation.py        # Metrics, learning curve data
│   ├── setup.py                 # pip install -e ml/
│   ├── pyproject.toml
│   └── tests/
│
├── data/                        # Unchanged — CSV data files
├── models/                      # Serialized model artifacts (.joblib)
├── config/                      # API keys (gitignored)
├── docs/design/                 # These design docs
├── docker-compose.yml           # Local dev: frontend + backend + db
├── Dockerfile                   # Production multi-stage build
└── deploy/                      # AWS deployment configs
    ├── ecs-task-definition.json
    ├── cloudformation.yml
    └── .env.example
```

---

## API Endpoints

### Training Session

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/training/start` | Start a new training session — initializes cat at Level 0, seeds initial labeled pool |
| `GET` | `/api/training/{session_id}/state` | Get current session state (cat level, accuracy, pool size, labeled count) |
| `POST` | `/api/training/{session_id}/predict` | Upload a mushroom image — cat makes a prediction using current model |
| `POST` | `/api/training/{session_id}/reveal` | User reveals ground truth — model retrains, cat reacts |
| `POST` | `/api/training/{session_id}/query` | Cat requests next batch from pool using selected strategy |
| `GET` | `/api/training/{session_id}/curve` | Get learning curve data (accuracy vs. labels) |
| `POST` | `/api/training/{session_id}/strategy` | Change query strategy mid-session |
| `DELETE` | `/api/training/{session_id}` | Abandon session |

### Detection (Expert Mode)

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/detect/analyze` | Upload image — run both judges, return verdicts + consensus |
| `POST` | `/api/detect/extract-features` | Extract 20 features only (no classification) |

### Session & User

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/session/create` | Create anonymous session (no auth needed — course project) |
| `GET` | `/api/session/{id}` | Get session with cat state, history |
| `GET` | `/api/session/{id}/achievements` | Get unlocked achievements |

### Model Management

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/models/available` | List available learner types and strategies |
| `GET` | `/api/models/current` | Get current model info (type, accuracy, training size) |

---

## State Management (Frontend — Zustand)

### `catStore`
```typescript
interface CatStore {
  level: number;           // 0-5
  xp: number;              // Points toward next level
  mood: CatMood;           // 'idle' | 'thinking' | 'happy' | 'sick' | 'proud' | 'curious' | 'sleeping'
  name: string;            // "Professor Whiskers" (customizable)
  accuracy: number;        // Current model accuracy (0-1)
  totalPredictions: number;
  correctPredictions: number;
  currentDialogue: string; // What the cat is saying
  // Actions
  setMood: (mood: CatMood) => void;
  addXP: (amount: number) => void;
  levelUp: () => void;
  setDialogue: (text: string) => void;
}
```

### `trainingStore`
```typescript
interface TrainingStore {
  sessionId: string | null;
  phase: 'idle' | 'predicting' | 'waiting_reveal' | 'retraining' | 'querying';
  strategy: QueryStrategy;
  labeledCount: number;
  poolRemaining: number;
  learningCurve: { labels: number; accuracy: number }[];
  currentMushroom: MushroomImage | null;
  currentPrediction: Prediction | null;
  // Actions
  startSession: (strategy: QueryStrategy, learner: LearnerType) => Promise<void>;
  predict: (image: File) => Promise<Prediction>;
  reveal: (groundTruth: 'edible' | 'poisonous') => Promise<CatReaction>;
  queryNext: () => Promise<void>;
}
```

---

## WebSocket Protocol

Used for live training updates (retraining can take a few seconds).

```
Client → Server:  { type: "start_retrain", session_id: "..." }
Server → Client:  { type: "retrain_progress", progress: 0.5 }
Server → Client:  { type: "retrain_complete", new_accuracy: 0.82, cat_reaction: "happy" }
Server → Client:  { type: "level_up", new_level: 3, dialogue: "I can smell the spores now!" }
```

---

## Key Design Decisions

1. **No authentication** — Course project; sessions are anonymous, identified by UUID. No passwords, no OAuth. Session ID stored in localStorage.

2. **SQLite for dev, PostgreSQL for prod** — Single-file DB locally, real DB on AWS. SQLAlchemy abstracts the difference.

3. **Vite + React** over Next.js — No SSR needed. Pure SPA is simpler and the cat animations need full client-side control.

4. **Zustand over Redux** — Less boilerplate, simpler mental model for game-like state (mood, XP, level).

5. **FastAPI over Flask/Django** — Async support for Gemini API calls, native Pydantic for request/response validation, WebSocket support built-in.

6. **WebSocket for retraining** — Model retraining takes 2-10 seconds depending on data size. WebSocket lets us show the cat "thinking" animation in real-time without polling.
