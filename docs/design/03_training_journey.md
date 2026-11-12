# Training Journey Design — From Clueless Kitten to Fungal Master

## Core Concept

The user doesn't just "use" a classifier — they **raise** one. The cat starts knowing nothing and the user guides it through an active learning loop, choosing which mushrooms to study, revealing labels, and watching the cat grow. This makes the AL research tangible: every XP point is a real labeled sample, every level-up is a real accuracy milestone.

---

## Level System

| Level | Title | Accuracy Threshold | Labeled Samples (approx.) | Unlock |
|---|---|---|---|---|
| 0 | Clueless Kitten | 0% – 59% | 0-200 | Training begins |
| 1 | Curious Nibbler | 60% – 74% | ~200-400 | Strategy picker (can switch from Random) |
| 2 | Spore Spotter | 75% – 84% | ~400-800 | Learning curve chart visible |
| 3 | Gill Inspector | 85% – 89% | ~800-1500 | Expert mode unlocked (dual-judge) |
| 4 | Myco Apprentice | 90% – 94% | ~1500-3000 | Strategy comparison view |
| 5 | Fungal Master | 95%+ | ~3000+ | Full dashboard, certificate |

**XP System:**
- +10 XP per correctly labeled sample
- +5 XP per incorrectly labeled sample (still learning)
- +25 XP bonus for a streak of 5 correct
- -0 XP (never punish — the cat always grows, just slower)

Level thresholds are based on **model cross-validation accuracy** — the real AL metric. XP is a softer parallel that keeps the user engaged even when accuracy plateaus.

---

## Session Lifecycle

### Phase 1: Onboarding

```
┌─────────────────────────────────────────────┐
│              MEET YOUR CAT                   │
│                                              │
│     [Cat idle animation - Level 0]           │
│                                              │
│  "Mew! I'm Professor Whiskers.              │
│   I don't know ANY mushrooms yet.            │
│   Can you teach me which ones are safe?"     │
│                                              │
│  Name your cat: [Professor Whiskers___]      │
│                                              │
│  Choose starting strategy:                   │
│  ○ Random (simple — I'll pick randomly)      │
│  ○ D-Optimal (smart — maximize coverage)     │
│                                              │
│  Choose my brain:                            │
│  ○ Random Forest (reliable, fast)            │
│  ○ Logistic Regression (simple, calibrated)  │
│  ○ Neural Network (powerful, slower)         │
│                                              │
│          [ Start Training! ]                 │
└─────────────────────────────────────────────┘
```

**What happens on "Start Training":**
1. Backend creates a session in the DB
2. Initial 200 samples selected from pool (Random or D-Optimal)
3. Model trains on initial set
4. Cat enters `curious` state
5. Frontend navigates to Training Arena

### Phase 2: Training Loop (Automatic — User Watches)

> **Design rationale:** Pool samples come from `secondary_data.csv`, which already has true labels.
> Asking the user to manually reveal labels would be absurd — users visit this app precisely
> *because* they can't tell edible from poisonous. Training is therefore **fully automatic**.
> The user interaction (upload your own photo, get a verdict) belongs in **Expert Mode** only.

The cat trains itself. The user controls speed and strategy, and watches the loop:

```
┌────────────────────────────────────────────────────────────────┐
│  TRAINING ARENA                                    Level: ★☆☆  │
│                                                                │
│  ┌──────────────────┐    ┌─────────────────┐  ┌────────────┐  │
│  │  Specimen #4217  │    │   Cat Avatar    │  │  Progress  │  │
│  │  cap: convex     │    │   (thinking)    │  │  Accuracy  │  │
│  │  color: brown    │    │                 │  │  XP bar    │  │
│  │  habitat: woods  │    │  💬 "Definitely │  │  Curve     │  │
│  │                  │    │  edible — the   │  │  (Lv 2+)   │  │
│  │  Last result:    │    │  convex cap     │  └────────────┘  │
│  │  ✅ Correct!     │    │  hints at it."  │                  │
│  └──────────────────┘    │                 │                  │
│                          │  [⏸ Pause]      │                  │
│  Strategy picker         │  🐢 Med ⚡ Fast  │                  │
│  (disabled while         └─────────────────┘                  │
│   running)                                                     │
│  ┌────────────────────────────────────────────────────────────┐│
│  │ Labeled: 245 │ Pool: 60,755 │ ✅ 18/20 recent             ││
│  └────────────────────────────────────────────────────────────┘│
└────────────────────────────────────────────────────────────────┘
```

#### Automatic Round Flow (`/api/training/{id}/step`):

**Step 1 — Query** (`curious` animation)
- Backend selects next sample from pool using the chosen strategy
- Frontend shows mushroom specimen card

**Step 2 — Predict** (`thinking` animation)
- Backend: `model.predict_proba(X)` on selected sample
- Cat speaks its prediction + confidence dialogue

**Step 3 — Auto-Reveal** (`happy` / `sick` / `confused` animation)
- True label is already known (from dataset) — no user input needed
- Backend compares prediction to true label, returns `wasCorrect`
- Correct → `happy` + +10 XP
- Wrong (predicted edible, was poisonous) → `sick` + +5 XP + dramatic dialogue
- Wrong (predicted poisonous, was edible) → `confused` + +5 XP

**Step 4 — Retrain** (brief `thinking`)
- Sample added to labeled set
- Every 50 samples: model fully retrains; accuracy updated via WebSocket
- Otherwise: accumulate for next batch

**Step 5 — Level Check**
- If accuracy crosses threshold → `proud` animation → level-up notification
- New features unlock

**Step 6 — Loop**
- Immediately repeat from Step 1

User controls:
- **Start / Pause** button
- **Speed** selector: Slow / Medium / Fast (controls animation duration per step)
- **Strategy** picker (disabled while running; change takes effect on next start)

### Phase 3: Batch Query Mode

For efficiency, the user can also process batches:

```
┌────────────────────────────────────────────────────┐
│  BATCH QUERY                                        │
│                                                     │
│  Strategy: [Uncertainty Sampling ▼]                 │
│  Batch size: [50]                                   │
│                                                     │
│  Cat says: "I've picked 50 mushrooms I'm most      │
│  uncertain about. Let me study them all at once!"   │
│                                                     │
│  [ Query & Auto-Label Batch ]                       │
│                                                     │
│  Progress: ████████████░░░░░░░░  24/50 labeled      │
│  Current accuracy: 78.3% → 79.1% (↑0.8%)           │
└────────────────────────────────────────────────────┘
```

In batch mode, the pool labels are auto-revealed (simulating oracle) and the model retrains on the full batch. This is closer to real AL experiments and lets users rapidly progress.

### Phase 4: Expert Mode (Unlocked at Level 3)

The original dual-judge system, enhanced with the cat character:

```
┌────────────────────────────────────────────────────────────────┐
│  EXPERT MODE — Professor Whiskers (Level 4: Myco Apprentice)   │
│                                                                │
│  [Upload Real Mushroom Photo 📷]                               │
│                                                                │
│  ┌─────────────────────┐  ┌─────────────────────┐             │
│  │ Judge A (ML Model)  │  │ Judge B (Gemini)     │             │
│  │                     │  │                      │             │
│  │ Features extracted: │  │ Visual assessment:   │             │
│  │ • cap: convex       │  │ "Brown convex cap    │             │
│  │ • gills: close      │  │ with close gills..." │             │
│  │ • stem: 8cm         │  │                      │             │
│  │                     │  │ POISONOUS 92%        │             │
│  │ POISONOUS 88%       │  │                      │             │
│  └─────────────────────┘  └─────────────────────┘             │
│                                                                │
│  ┌────────────── Cat's Take ──────────────────────┐           │
│  │ 🐱 "Both my analytical and visual sides agree:  │           │
│  │  this is POISONOUS. Don't eat it! The close     │           │
│  │  gill spacing and brown convex cap are classic   │           │
│  │  warning signs I've learned."                    │           │
│  │                                                  │           │
│  │  ┌──────────────────────────────────────────┐   │           │
│  │  │ ☠️ FINAL VERDICT: POISONOUS (90% conf)   │   │           │
│  │  └──────────────────────────────────────────┘   │           │
│  └──────────────────────────────────────────────────┘          │
└────────────────────────────────────────────────────────────────┘
```

### Phase 5: Strategy Comparison (Unlocked at Level 4)

Side-by-side comparison of AL strategies — the research visualization.

```
┌──────────────────────────────────────────────────────────────┐
│  STRATEGY SHOWDOWN                                            │
│                                                               │
│  Compare how different strategies teach the cat:              │
│                                                               │
│  ☑ Random    ☑ Uncertainty    ☑ QBC    ☐ BALD                │
│  Classifier: [Random Forest ▼]    Budget: [2000 labels]      │
│                                                               │
│  [ Run Comparison ]                                           │
│                                                               │
│  ┌──────────────── Learning Curves ─────────────────────┐    │
│  │  Acc                                                  │    │
│  │  95%|                          ──── Uncertainty       │    │
│  │  90%|                    ──────────── QBC             │    │
│  │  85%|              ─────────────── Random             │    │
│  │  80%|         ────                                    │    │
│  │  75%|    ────                                         │    │
│  │     └─────────────────────────────────────────        │    │
│  │      200   500   800  1000  1500  2000  labels        │    │
│  └───────────────────────────────────────────────────────┘    │
│                                                               │
│  Summary: Uncertainty sampling reached 95% with 1,247 labels  │
│  vs. Random needing 2,891 — a 57% reduction in labeling!     │
└──────────────────────────────────────────────────────────────┘
```

This runs the full AL simulation on the backend (not interactive — just computation) and returns the curves.

---

## Data Flow: Pool Samples vs. Photo Uploads

### Pool Samples (tabular data from secondary_data.csv)

```
Pool (61k unlabeled) ──strategy query──► Selected Sample
                                              │
                                         ┌────▼────┐
                                         │ Backend  │ ← label is known but hidden
                                         │ predicts │
                                         └────┬────┘
                                              │
                                    Cat shows prediction
                                              │
                                    User clicks "reveal"
                                              │
                                    Backend reveals true label
                                              │
                                    Sample moves to labeled set
                                              │
                                    Model retrains (if batch full)
```

### Photo Uploads (real images via Gemini)

```
User uploads photo ──► Gemini extracts 20 features ──► Feature vector
                                                            │
                                                       ┌────▼────┐
                                                       │ Backend  │
                                                       │ predicts │
                                                       └────┬────┘
                                                            │
                                                  Cat shows prediction
                                                            │
                                                  User reports ground truth
                                                  (or Gemini acts as oracle)
                                                            │
                                                  Feature vector + label
                                                  added to labeled set
                                                            │
                                                  Model retrains
```

---

## Achievements System

Unlockable milestones displayed on the Journey page.

| Achievement | Condition | Icon |
|---|---|---|
| First Steps | Label first sample | 🐾 |
| Getting Warmer | Reach 60% accuracy | 🌡️ |
| Spore Sense | Reach 75% accuracy | 🔬 |
| Lab Certified | Reach 85% accuracy | 🥼 |
| Sharp Eyes | 10 correct in a row | 👁️ |
| Poison Survivor | Survive 5 false negatives | 💚 |
| Speed Learner | Reach Level 3 under 500 labels | ⚡ |
| Strategist | Try all 4 query strategies | 🧠 |
| Photo Detective | Upload 10 real mushroom photos | 📸 |
| Fungal PhD | Reach Level 5 | 🎓 |
| The Full Course | Complete strategy comparison | 📊 |

---

## Session Persistence (Database Schema)

```sql
-- Core session
CREATE TABLE training_session (
    id              TEXT PRIMARY KEY,  -- UUID
    cat_name        TEXT DEFAULT 'Professor Whiskers',
    cat_level       INTEGER DEFAULT 0,
    cat_xp          INTEGER DEFAULT 0,
    learner_type    TEXT NOT NULL,     -- 'random_forest' | 'logistic' | 'mlp'
    strategy        TEXT NOT NULL,     -- 'random' | 'uncertainty' | 'committee' | 'bald'
    init_strategy   TEXT NOT NULL,     -- 'random' | 'd_optimal'
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tracks each labeled sample in the session
CREATE TABLE labeled_sample (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT REFERENCES training_session(id),
    sample_index    INTEGER,           -- index into the pool dataframe
    true_label      INTEGER,           -- 0=edible, 1=poisonous
    predicted_label INTEGER,
    predicted_conf  REAL,
    was_correct     BOOLEAN,
    labeled_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Snapshots of model performance at each retrain
CREATE TABLE training_snapshot (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT REFERENCES training_session(id),
    labeled_count   INTEGER,
    cv_accuracy     REAL,
    cv_std          REAL,
    f1_score        REAL,
    roc_auc         REAL,
    snapshot_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Unlocked achievements
CREATE TABLE achievement (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT REFERENCES training_session(id),
    achievement_key TEXT NOT NULL,
    unlocked_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## Mushroom Display for Pool Samples

Since pool samples are tabular (not images), we display them as stylized "specimen cards":

```
┌─────────────────────────────────┐
│  🍄 Specimen #4,217             │
│  ┌────────────────────────────┐ │
│  │ [Generated mushroom icon   │ │
│  │  based on cap-shape &      │ │
│  │  cap-color features]       │ │
│  └────────────────────────────┘ │
│                                 │
│  Cap: convex, brown, scaly      │
│  Gills: close, buff             │
│  Stem: 12cm × 15mm, bulbous    │
│  Habitat: woods                 │
│  Season: autumn                 │
│                                 │
│  [What does the cat think? →]   │
└─────────────────────────────────┘
```

The mushroom icon is a simple SVG composed from cap-shape + cap-color, giving visual variety without needing 61k images.
