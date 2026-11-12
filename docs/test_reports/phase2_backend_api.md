# Phase 2 Backend API — Test Report

**Date:** 2026-04-10
**Environment:** local (macOS), `uvicorn backend.app.main:app --reload --port 8000`
**Branch:** v2-restructure

---

## Setup

```bash
# .env at project root
GEMINI_API_KEY=<redacted>
DATA_DIR=./dt/MushroomDataset
MODEL_DIR=./models
```

Dependencies installed in `mushroom-project` conda env:
```bash
pip install -r backend/requirements.txt
pip install -e ml/
```

---

## Test 1 — Health Check

```bash
curl http://localhost:8000/api/health
```

**Response:**
```json
{"status": "ok"}
```

**Result: PASS** — Server starts and responds.

---

## Test 2 — `POST /api/training/start`

```bash
curl -s -X POST http://localhost:8000/api/training/start \
  -H "Content-Type: application/json" \
  -d '{"learner_type": "random_forest", "strategy": "uncertainty", "init_strategy": "random"}'
```

**Response:**
```json
{
  "session_id": "6ca80e58-8782-4c68-8093-2c287ba39002",
  "labeled_count": 200,
  "pool_remaining": 60869,
  "cv_accuracy": 0.7,
  "cv_std": 0.06892024376045112
}
```

**Result: PASS**
- Session created with UUID
- Initial labeled set: 200 samples (random init)
- Pool: 60,869 remaining (full secondary dataset minus 200)
- Initial 5-fold CV accuracy: 70% — expected for a small initial labeled set

---

## Test 3 — `POST /api/training/{session_id}/predict`

```bash
curl -s -X POST http://localhost:8000/api/training/6ca80e58-8782-4c68-8093-2c287ba39002/predict
```

**Response:**
```json
{
  "pool_index": 31279,
  "label": "poisonous",
  "confidence": 0.5000098680294217,
  "p_edible": 0.4999901319705783,
  "p_poisonous": 0.5000098680294217
}
```

**Result: PASS**
- Uncertainty sampling correctly selected a sample near the decision boundary (confidence ≈ 0.50)
- Returns valid pool index, label, and probability breakdown

---

## Test 4 — `POST /api/training/{session_id}/reveal`

```bash
curl -s -X POST http://localhost:8000/api/training/6ca80e58-8782-4c68-8093-2c287ba39002/reveal \
  -H "Content-Type: application/json" \
  -d '{"pool_index": 31279}'
```

**Response:**
```json
{
  "true_label": 0,
  "predicted_label": 1,
  "predicted_conf": 0.5000098680294217,
  "was_correct": false,
  "retrained": false,
  "metrics": null
}
```

**Result: PASS**
- True label revealed (0 = edible), prediction was wrong (poisonous) — expected at low confidence
- `retrained: false` — correct, retraining triggers every 50 labels, only 1 revealed here
- `metrics: null` when no retrain occurred — correct

---

## Issues Encountered & Fixes

| Issue | Root Cause | Fix |
|---|---|---|
| `No module named 'greenlet'` | Missing implicit SQLAlchemy async dependency | `pip install greenlet`, added to `requirements.txt` |
| `No API key was provided` | `.env` file contained bare value without `KEY=VALUE` format | Fixed `.env` to `GEMINI_API_KEY=<value>` |
| `FileNotFoundError: ./data/secondary_data.csv` | `config.py` default `data_dir` didn't match actual data location | Added `DATA_DIR=./dt/MushroomDataset` to `.env` |
| `ModuleNotFoundError: No module named 'src'` | Old `models/preprocessor.joblib` was pickled against the old `src.data.preprocessor` class | Deleted stale `.joblib` files; new preprocessor fit and saved automatically on first run |

---

## Milestone Status

- [x] `uvicorn` starts without errors
- [x] `GET /api/health` returns 200
- [x] `POST /api/training/start` creates session and returns `session_id`
- [x] `POST /api/training/{id}/predict` returns prediction with pool index
- [x] `POST /api/training/{id}/reveal` records label and handles retrain trigger correctly
