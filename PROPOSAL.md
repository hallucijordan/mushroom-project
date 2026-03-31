# Mushroom Edibility Classification via Active Learning

## Overview

This project investigates **active learning** as a strategy for training a mushroom edibility classifier (edible vs. poisonous) with minimal labeled data. The core research question is how much annotation effort active learning can save compared to random labeling — evaluated rigorously across multiple query strategies and classifiers.

As a lightweight application demo, a **Gemini Vision agent** is also integrated to extract structured features from mushroom photos, showing how the trained classifier can be used in practice.

---

## Problem Statement

Obtaining labeled biological data is expensive — in the real world, verifying whether a mushroom species is poisonous requires expert mycologists or lab testing. Active learning addresses this by intelligently selecting which samples to label, maximizing model performance per annotation.

**Core question:** Given a large pool of unlabeled mushrooms and a small initial labeled set, how many labels does active learning need to reach the same accuracy as a fully supervised model trained on all data?

---

## Dataset

| Dataset | Rows | Role |
|---|---|---|
| `primary_data.csv` | 173 | Real species data → **initial labeled set** |
| `secondary_data.csv` | 61,069 | Simulated instances → **unlabeled pool** (labels hidden) |

**Target variable:** `class` — `e` (edible) or `p` (poisonous)

**20 input features:** cap-diameter, cap-shape, cap-surface, cap-color, does-bruise-or-bleed, gill-attachment, gill-spacing, gill-color, stem-height, stem-width, stem-root, stem-surface, stem-color, veil-type, veil-color, has-ring, ring-type, spore-print-color, habitat, season

**Active learning simulation setup:**
- Labels in secondary data are hidden; revealed only when queried (simulating oracle annotation)
- Primary data serves as the cold-start labeled set (173 real species examples)

---

## Active Learning Pipeline (Core)

### Batch Configuration
- **Batch size:** 50 instances per query round
- **Initial labeled set:** 173 samples (primary_data.csv)

### Stopping Criteria
The Active Learning loop terminates when **any** of the following conditions is met:

| Criterion | Condition | Rationale |
|-----------|-----------|-----------|
| **Target accuracy reached** | Validation accuracy ≥ 95% | Goal achieved, no need for more labels |
| **Pool exhausted** | Unlabeled pool is empty | No more samples to query |


### Initial Training Set Selection

We compare two initialization strategies:

| Strategy | Method | Deterministic? |
|----------|--------|----------------|
| **Random** | Uniform random sampling | No (seed-controlled) |
| **D-Optimal** | Maximize information matrix determinant | Yes |

Both strategies select **173 samples** as the initial labeled set.


### Query Strategies (compared)

| Strategy | Category | Selection Criterion |
|----------|----------|---------------------|
| Random | Baseline | Uniform random selection |
| Uncertainty sampling | Uncertainty | Lowest prediction confidence |
| Query-by-Committee | Ensemble | Maximum disagreement among committee models |
| BALD | Bayesian | Maximum mutual information |

All strategies select **50 instances per round**.


### Classifiers
- **Random Forest** — ensemble of decision trees; uncertainty via tree disagreement
- **Bagged Logistic Regression** — well-calibrated probabilities; uncertainty via bootstrap ensemble
- **Neural Network (MLP)** — flexible learner; uncertainty via MC Dropout or Deep Ensemble



### Evaluation

#### Primary Metric
- **Label efficiency:** Number of labels required to reach 95% validation accuracy
- **Learning curves:** accuracy vs. number of labeled samples queried
- **AUC-ROC** and **F1** at fixed label budgets (100, 500, 1000, 5000 samples)

---

## Gemini Integration (Demo / Appetizer)

A lightweight demo showing the classifier is usable in practice:

1. User submits a mushroom photo + optional context (habitat, season)
2. Gemini Vision extracts the 20 structured features as JSON
3. The AL-trained classifier predicts edible / poisonous + confidence score

### Bonus Comparison
A quick experiment comparing:

| System | Pipeline |
|---|---|
| **A — Gemini Direct** | Photo → Gemini → edible/poisonous |
| **B — Full System** | Photo → Gemini features → ML Classifier → edible/poisonous |

This answers: does the domain-trained classifier add value over LLM direct judgment? Kept as a secondary finding, not the main contribution.

---

## Technical Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Environment | Conda (`mushroom-project`) |
| ML / AL | scikit-learn, XGBoost, modAL or custom AL loop |
| Vision Agent | Google Gemini API (gemini-2.0-flash) |
| Data | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Version Control | Git + GitHub |

---

## Milestones

- [ ] **M1 — Data Preprocessing:** Parse primary ranges → single values; encode categoricals; align schemas
- [ ] **M2 — Baseline Classifier:** Train on full secondary data; establish accuracy ceiling
- [ ] **M3 — Active Learning Loop:** Implement AL pipeline; compare query strategies; plot learning curves
- [ ] **M4 — Gemini Demo:** Feature extraction from photos; A vs. B comparison (lightweight)
- [ ] **M5 — Application:** Simple interface connecting photo → Gemini → classifier

---

## Key Research Question

> Using active learning, how many labeled samples are required to match the performance of a fully supervised model trained on all 61,069 simulated instances — and which query strategy achieves this most efficiently?

---

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Domain gap between primary and secondary data | Normalize features; ablate AL initialized from secondary-only split |
| Model overconfidence on simulated data | Calibrate probabilities; validate on held-out primary species |
| Gemini API cost | Cache results; use gemini-flash; limit to small photo sample set |
