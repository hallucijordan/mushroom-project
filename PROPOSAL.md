# Mushroom Edibility Classifier with Active Learning & Vision Agent

## Overview

This project builds an end-to-end system to classify whether a mushroom is **edible or poisonous**, combining:
- A machine learning classifier trained via **active learning**
- A **Gemini Vision agent** that bridges the gap between real-world mushroom photos and structured model input

The core insight: instead of requiring users to manually fill in 20 botanical features, a vision agent extracts those features automatically from a photo — making the classifier practically usable in the real world.

---

## Problem Statement

Existing mushroom classifiers rely on structured feature inputs (cap shape, gill color, habitat, etc.) that require expert botanical knowledge to fill in. This creates a usage barrier for non-experts. Meanwhile, labeled mushroom data from real-world images is scarce and expensive to obtain.

**This project addresses both problems:**
1. Use active learning to maximize classifier performance with minimal labeled data
2. Use a vision LLM to auto-extract structured features from photos, removing the input barrier

---

## Dataset

| Dataset | Rows | Description |
|---|---|---|
| `primary_data.csv` | 173 | Real species from *Mushrooms & Toadstools* (Hardin, 1999). Feature ranges per species. Used as initial labeled set. |
| `secondary_data.csv` | 61,069 | Simulated individual mushrooms sampled from primary ranges. Labels hidden to serve as the unlabeled pool. |

**Target variable:** `class` — `e` (edible) or `p` (poisonous)

**20 input features:** cap-diameter, cap-shape, cap-surface, cap-color, does-bruise-or-bleed, gill-attachment, gill-spacing, gill-color, stem-height, stem-width, stem-root, stem-surface, stem-color, veil-type, veil-color, has-ring, ring-type, spore-print-color, habitat, season

---

## System Architecture

```
[User Photo]
     │
     ▼
┌─────────────────────────────┐
│     Gemini Vision Agent     │  ← Prompt-engineered to output
│  (feature extraction layer) │    the 20 structured features
└─────────────────────────────┘
     │
     │  structured feature vector
     ▼
┌─────────────────────────────┐
│     ML Classifier           │  ← Trained via active learning
│  (edible / poisonous)       │    on primary + queried secondary
└─────────────────────────────┘
     │
     ▼
[Prediction + Confidence Score]
     │
     ├── High confidence → Return result to user
     │
     └── Low confidence → Flag for human review
                              └── Add to active learning pool
```

---

## Component 1: Active Learning Pipeline

### Setup
- **Initial labeled set:** Primary data (173 species), preprocessed from range values to representative single values
- **Unlabeled pool:** Secondary data (61,069 rows) with labels hidden
- **Oracle:** Ground-truth labels revealed on query (simulating lab verification)

### Query Strategy
Start with **uncertainty sampling** (least confidence), then evaluate:
- Query-by-committee (ensemble disagreement)
- BALD (Bayesian Active Learning by Disagreement)

### Evaluation
Plot **learning curves**: model accuracy vs. number of labeled samples queried — comparing active learning against random sampling baseline.

### Classifier Candidates
- Random Forest (interpretable baseline)
- Gradient Boosting (XGBoost / LightGBM)
- Logistic Regression (uncertainty calibration)

---

## Component 2: Gemini Vision Agent

### Role
Given a mushroom photo (and optionally context like location, season), the agent outputs a structured JSON of the 20 features required by the classifier.

### Example Interaction
**Input:** Photo of a mushroom + "Found in woods, autumn"

**Agent Output:**
```json
{
  "cap-diameter": 12.5,
  "cap-shape": "x",
  "cap-surface": "g",
  "cap-color": "n",
  "does-bruise-or-bleed": "f",
  "gill-attachment": "e",
  "gill-spacing": "c",
  "gill-color": "w",
  "stem-height": 9.0,
  "stem-width": 14.0,
  "stem-root": "s",
  "stem-surface": "y",
  "stem-color": "w",
  "veil-type": "u",
  "veil-color": "w",
  "has-ring": "t",
  "ring-type": "g",
  "spore-print-color": "w",
  "habitat": "d",
  "season": "a"
}
```

### Prompt Design
The agent will be given:
- Full feature codebooks (all valid values and their meanings)
- Instructions to estimate metrical values (cm/mm) from visual cues
- Instructions to mark uncertain features explicitly (enabling downstream uncertainty handling)

---

## Component 3: Application

### Interface (TBD — CLI or Web)
1. User uploads a mushroom photo (+ optional: location, season)
2. Gemini extracts features → displayed to user for review/correction
3. Classifier runs on features → returns `edible` / `poisonous` + confidence
4. Low-confidence results flagged; user can confirm label to grow the training set

### Active Feedback Loop
User-confirmed labels from the app feed back into the active learning pool, enabling the classifier to improve over time with real-world data.

---

## Technical Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Environment | Conda (`mushroom-project`) |
| ML | scikit-learn, XGBoost, modAL or custom AL loop |
| Vision Agent | Google Gemini API (gemini-2.0-flash) |
| Data | pandas, numpy |
| Visualization | matplotlib, seaborn |
| App (TBD) | Streamlit or FastAPI + React |
| Version Control | Git + GitHub |

---

## Milestones

- [ ] **M1 — Data Preprocessing:** Parse primary ranges → single values; encode categoricals; align schemas between primary and secondary
- [ ] **M2 — Baseline Classifier:** Train supervised model on full secondary data; establish accuracy ceiling
- [ ] **M3 — Active Learning Loop:** Implement AL pipeline; plot learning curves vs. random sampling
- [ ] **M4 — Gemini Agent:** Prompt engineering; feature extraction from photos; JSON output validation
- [ ] **M5 — Integration:** Connect vision agent output → classifier pipeline
- [ ] **M6 — Application:** Build user-facing interface; implement feedback loop

---

## Key Research Questions

> 1. How many human-verified labels are needed (via active learning) to match the performance of a fully supervised model trained on all 61,069 simulated samples?
> 2. Does the full pipeline (Gemini → features → ML classifier) outperform Gemini making a direct edibility judgment from the photo alone?
> 3. Can a vision agent replace manual feature input entirely, without loss of classification accuracy?

---

## Evaluation Experiments

### Experiment 1: Active Learning Efficiency
Compare learning curves of:
- Active learning (uncertainty sampling) vs. random sampling baseline
- Metric: accuracy vs. number of labeled samples queried

### Experiment 2: System Comparison (Core Contribution)

Three systems evaluated on the same set of mushroom images with known ground-truth labels:

| System | Input | Pipeline |
|---|---|---|
| **A — Gemini Direct** | Photo | Gemini → edible/poisonous |
| **B — Pure ML** | Manually filled 20 features | ML Classifier → edible/poisonous |
| **C — Full System** | Photo | Gemini → 20 features → ML Classifier → edible/poisonous |

**Questions this answers:**
- **C vs. A:** Does the structured ML classifier add value on top of LLM direct judgment?
- **C vs. B:** How much quality is lost when features come from vision extraction vs. human annotation?
- **A vs. B:** LLM visual understanding vs. traditional ML — who has the higher ceiling?

**Why this matters:** Many assume large vision LLMs can replace specialized pipelines. This experiment quantifies whether domain-trained classifiers with structured feature extraction provide measurable gains over end-to-end LLM judgment — a finding applicable to medical imaging, plant disease detection, and other expert classification domains.

### Experiment 3: Feature Extraction Quality
Compare features extracted by Gemini vs. ground-truth features from secondary data:
- Per-feature accuracy (which features does Gemini get right/wrong?)
- Impact of extraction errors on downstream classification

---

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Gemini misidentifies visual features | Allow user to review/edit extracted features before classification |
| Domain gap between primary and secondary data | Normalize features; test AL initialized from secondary-only split as ablation |
| Model overconfidence on simulated data | Calibrate probabilities; test on held-out primary species |
| Gemini API cost | Cache results; batch requests; use gemini-flash for lower cost |
