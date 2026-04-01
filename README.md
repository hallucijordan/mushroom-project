# 🍄 Mushroom Safety Judge

A mushroom edibility classifier combining **active learning research** with a
**Gemini Vision demo app**.  Two independent AI judges examine a mushroom photo
and deliver verdicts through a conversation-style interface.

> ⚠️ **Research demo only — never eat wild mushrooms based solely on AI advice.**

---

## Overview

| Component | What it does |
|---|---|
| **Base Learner** | Random Forest trained on 61,069 simulated mushroom instances |
| **Judge A — Model-backed** | Gemini extracts 20 structured features → ML classifier decides |
| **Judge B — Direct Vision** | Gemini looks at the photo and judges directly (no ML model) |
| **Gradio UI** | Upload a photo, get both verdicts side-by-side with a final consensus |

---

## Quickstart

### 1. Create conda environment

```bash
conda env create -f environment.yml
conda activate mushroom-project
```

### 2. Add your Gemini API key

```bash
cp config/api_keys.example.py config/api_keys.py
# Edit config/api_keys.py and set GEMINI_API_KEY = "AIza..."
```

`config/api_keys.py` is gitignored — your key will never be committed.

### 3. Download the dataset

```bash
python download_data.py
```

Downloads `primary_data.csv` (173 real species) and `secondary_data.csv`
(61,069 simulated instances) into `dt/`.

### 4. Train the base learner

```bash
python train.py
```

Fits the preprocessor and Random Forest on `secondary_data.csv`, evaluates on
a held-out test split, and writes artefacts to `models/`.

### 5. Launch the app

```bash
python app.py
```

Opens the Gradio UI at `http://localhost:7860`.

---

## Project structure

```
mushroom-project/
├── config/
│   ├── api_keys.py          # gitignored — your real key goes here
│   └── api_keys.example.py  # template
├── src/
│   ├── data/
│   │   └── preprocessor.py      # load CSVs, parse numeric ranges, OrdinalEncoder
│   ├── models/
│   │   └── base_learner.py      # RandomForest + isotonic calibration, save/load
│   ├── agents/
│   │   └── gemini_extractor.py  # Gemini Vision → 20-feature JSON
│   └── judges/
│       ├── model_judge.py       # features → classifier → narrative (Judge A)
│       └── direct_judge.py      # direct Gemini vision verdict (Judge B)
├── train.py                 # one-shot training script
├── app.py                   # Gradio UI
├── download_data.py         # Kaggle dataset downloader
├── environment.yml          # conda environment spec
└── requirements.txt         # pip dependencies
```

---

## Dataset

| File | Rows | Role |
|---|---|---|
| `primary_data.csv` | 173 | Real species — initial labeled set |
| `secondary_data.csv` | 61,069 | Simulated instances — training pool |

**Target:** `class` — `e` (edible) or `p` (poisonous)

**20 features:** cap-diameter, cap-shape, cap-surface, cap-color,
does-bruise-or-bleed, gill-attachment, gill-spacing, gill-color,
stem-height, stem-width, stem-root, stem-surface, stem-color,
veil-type, veil-color, has-ring, ring-type, spore-print-color,
habitat, season

---

## How the judges work

```
          ┌──────────────────────────────────────┐
          │           User uploads photo          │
          └────────────────┬─────────────────────┘
                           │
          ┌────────────────┴─────────────────────┐
          │                                       │
   ┌──────▼──────┐                       ┌────────▼──────┐
   │   Judge A   │                       │   Judge B     │
   │ (Model-     │                       │ (Direct       │
   │  backed)    │                       │  Vision)      │
   └──────┬──────┘                       └────────┬──────┘
          │                                       │
   Gemini extracts                        Gemini assesses
   20 features as JSON                    photo directly
          │                                       │
   ML classifier                         Verdict +
   → edible/poisonous                    confidence
   + confidence                                   │
          │                                       │
          └────────────────┬─────────────────────┘
                           │
                  Weighted consensus
                  (60% Judge A, 40% Judge B)
                           │
                    Final Verdict
```

---

## Tuning knobs

Each module exposes its key parameters as class-level constants:

| Location | Parameter | Default |
|---|---|---|
| [src/models/base_learner.py](src/models/base_learner.py) | `N_ESTIMATORS` | 300 |
| [src/models/base_learner.py](src/models/base_learner.py) | `MAX_DEPTH` | None |
| [src/agents/gemini_extractor.py](src/agents/gemini_extractor.py) | `DEFAULT_MODEL` | `gemini-2.0-flash` |
| [src/judges/direct_judge.py](src/judges/direct_judge.py) | `DEFAULT_MODEL` | `gemini-2.0-flash` |
| [app.py](app.py) | consensus weights | 0.6 / 0.4 |

---

## Roadmap (from proposal)

- [x] M1 — Data preprocessing
- [x] M2 — Base classifier
- [x] M4 — Gemini Vision demo (Judge A + B)
- [x] M5 — Application UI
- [ ] M3 — Active learning loop (query strategies + learning curves)
