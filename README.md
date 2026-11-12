# Mushroom Project: Label-Efficient Fungi Classification with Active Learning

Course project for 02-750 Automation of Scientific Research (Carnegie Mellon University, Spring 2026),
by Luci Lu and Ziyan Zhang. The final report is [`main.pdf`](main.pdf).

The main question: **how many labeled images does a mushroom edibility classifier need, and can active
learning reduce that number compared with random labeling?** We test 6 query strategies on 36,361 fungus
images (179 species) with 5 random seeds each.

Two smaller parts: a Gemini Vision demo app that judges a mushroom photo two ways
([Part 2](#part-2-gemini-demo-app-secondary)), and an active-learning web app that is still in progress
([Part 3](#part-3-active-learning-web-app-in-progress)).

> **Research demo only. Never eat a wild mushroom based on AI advice.**

## Main result

| | Labels | Bucket accuracy (4 classes) |
|---|---:|---:|
| Majority-class baseline | 0 | 0.570 |
| Full supervision (whole training pool) | 32,724 | 0.824 |
| Margin sampling: labels to reach 90% of full supervision | **1,340** (4% of the pool) | 0.741 |
| Random sampling: labels to reach 90% of full supervision | 1,400 | 0.741 |
| Margin sampling: labels to reach 95% of full supervision | **3,700** (all 5 seeds) | 0.783 |
| Random sampling: labels to reach 95% of full supervision | 4,550 (only 2 of 5 seeds) | 0.783 |

- **Margin sampling is the most label-efficient strategy** at the 90% and 95% targets.
- **No strategy is best on every metric.** At 5,000 labels, bucket accuracy differs by only 1.4 points, but
  recall on the UNCERTAIN bucket differs by 4.3 points. BALD has the best bucket accuracy, and class-balanced
  entropy has the best UNCERTAIN recall. Margin is close to the best on both.
- **The frozen backbone limits the result more than the strategy does.** Full supervision is still 17 points
  below a perfect classifier.

## Repository layout

| Path | Part | What it is |
|---|---|---|
| `main.pdf`, `main.tex` | 1 | Final report |
| `00_data_analysis.ipynb` | 1 | Edibility label analysis: species to 4 buckets, per-genus and per-image distributions |
| `01b_embedding_picekvit.ipynb` | 1 | Frozen ViT-B/16 embeddings (768-dim) for all 36,361 images, logistic-regression sanity check |
| `02b_active_learning_picekvit.ipynb` | 1 | Full-supervision ceiling, 6 strategies x 5 seeds, learning curves, label efficiency, per-genus analysis |
| `results/` | 1 | Result tables (CSV) and the figures used in the report |
| `main.ipynb` | 1 | Early active-learning experiments on the tabular UCI mushroom data |
| `app.py`, `train.py`, `src/` | 2 | Gemini demo app |
| `ml/` | 3 | `mushroom_ml` package: learners, query strategies, active-learning session, with unit tests |
| `backend/` | 3 | FastAPI service around `mushroom_ml` |
| `frontend/` | 3 | React UI prototype (simulated data) |
| `PROPOSAL.md`, `milestone_1.md` | | Project proposal and milestone report |

## Part 1: Active learning on DF20-Mini images

### Setup

- **Data:** DF20-Mini (Picek et al., 2022), official split: 32,724 training images (the unlabeled pool) and
  3,637 test images, 179 species after removing rows without a species.
- **Labels:** each species is mapped to one of 4 user-facing buckets from an edibility table generated with Claude
  (see [Limitations](#limitations)):

  | Bucket | Meaning | Species | Test images |
  |---|---|---:|---:|
  | SAFE | Edible raw and cooked | 21 | 546 |
  | COOK_FIRST | Edible only when cooked | 37 | 764 |
  | AVOID | Poisonous, deadly or inedible | 96 | 2,073 |
  | UNCERTAIN | Edibility not well documented | 25 | 254 |

- **Features:** frozen ViT-B/16-224 checkpoint from Picek et al., fine-tuned on DF20-Mini. No backbone training
  on our side. A logistic regression on these features reaches 0.626 test species accuracy.
- **Classifier:** MLP (768-512-256-179) with MC dropout (p = 0.4, 20 forward passes), retrained from scratch
  every round and trained on species (179 classes). The species prediction is turned into a bucket in two ways:
  the top species decides the bucket ("hard"), or species probabilities are summed per bucket ("soft").
- **Active learning loop:** 400 initial labels (about 2 per species, stratified), then batches of 100 up to
  5,000 labels (15% of the pool). Strategies: random, entropy, margin, coreset, class-balanced entropy, BALD.
  5 seeds each.

### Results

Full supervision (all 32,724 labels): species top-1 accuracy 0.607, bucket accuracy 0.824 (hard) and 0.831
(soft). Recall is 0.895 for AVOID but only 0.535 for UNCERTAIN.

Labels needed to reach a share of the full-supervision bucket accuracy (mean over the seeds that reached the
target; the number of seeds is shown when not all 5 reached it):

| Target | Random | Entropy | **Margin** | Coreset | CB entropy | BALD |
|---|---:|---:|---:|---:|---:|---:|
| 80% (0.659) | 520 | 620 | **520** | 660 | 660 | 560 |
| 90% (0.741) | 1,400 | 1,940 | **1,340** | 2,500 | 1,980 | 1,660 |
| 95% (0.783) | 4,550 (2/5) | 4,500 | **3,700** | not reached | 4,800 (4/5) | 3,920 |

At 5,000 labels (hard decoding, mean over 5 seeds):

| Strategy | Bucket accuracy | AVOID recall | UNCERTAIN recall |
|---|---:|---:|---:|
| BALD | **0.789** | **0.896** | 0.450 |
| Margin | 0.785 | 0.877 | 0.487 |
| Entropy | 0.783 | 0.875 | 0.477 |
| CB entropy | 0.782 | 0.875 | **0.489** |
| Random | 0.780 | 0.876 | 0.461 |
| Coreset | 0.775 | 0.866 | 0.446 |

All strategies catch about 9 in 10 AVOID images. For the deadliest genus, *Amanita*, AVOID recall is 0.90
with soft decoding, although species accuracy on *Amanita* is only about 0.65. The full tables are in
`results/` and in the report.

### How to reproduce

Data and model files are not in the repository (too large):

| File | Source |
|---|---|
| `data/DF20M/` (images), `data/DF20M-metadata/` | DF20-Mini from the [Danish Fungi dataset](https://github.com/BohemianVRA/DanishFungiDataset) |
| `artifacts/DF20M-ViT_base_patch16_224_best_accuracy.pth` | Downloaded automatically by notebook 01b |
| `artifacts/species_edibility_map.csv` | Edibility table for the 179 species, generated with Claude (Anthropic) (not included) |
| `artifacts/df20m_metadata.csv` | Cleaned metadata from an earlier CLIP-based pipeline (not included). Notebook 01b rebuilds the same table from the DF20-Mini metadata as `df20m_metadata_picekvit.csv`; notebook 00 reads it under the old name. |

Run order: `01b_embedding_picekvit` (embeddings, about 20 min on Apple Silicon), then `00_data_analysis`
(writes the species-to-bucket table), then `02b_active_learning_picekvit` (30 runs; writes `results/`).

The notebooks need `torch`, `torchvision` and `timm` in addition to `requirements.txt`.

## Part 2: Gemini demo app (secondary)

A Gradio app that judges an uploaded mushroom photo two ways:

- **Judge A (model-backed):** Gemini extracts 20 structured features (cap shape, gill color, ...) as JSON, and a
  Random Forest with isotonic calibration, trained on the UCI Secondary Mushroom dataset (61,069 simulated
  mushrooms), predicts edible or poisonous.
- **Judge B (direct vision):** Gemini looks at the photo and decides directly.
- The final verdict weights Judge A at 0.6 and Judge B at 0.4.

```bash
conda env create -f environment.yml
conda activate mushroom-project
cp config/api_keys.example.py config/api_keys.py   # then set GEMINI_API_KEY (file is gitignored)
python train.py                                     # trains the classifier, writes models/
python app.py                                       # opens http://localhost:7860
```

The UCI data is already in `data/MushroomDataset/` (`download_data.py` fetches it again from Kaggle).
`src/models/base_learner.py` was rewritten in November 2026 because the first version was never committed.

## Part 3: Active-learning web app (in progress)

An app for running active learning interactively on the UCI tabular data, in three layers:

- **`ml/` (`mushroom_ml`):** learners (Random Forest with calibration, bagged logistic regression, MLP), query
  strategies (uncertainty, BALD, query by committee, random) and an `ALSession` that runs the loop. No web
  code. 56 unit tests (`pytest ml/tests`).
- **`backend/`:** FastAPI service. It starts training sessions, runs one query-label-retrain step per call,
  returns learning curves, and runs the two-judge photo check (needs `GEMINI_API_KEY`; without a key that
  endpoint returns 503).
- **`frontend/`:** React UI. **This is a prototype: it is not fully connected to the backend yet and shows
  simulated data.** The gaps are listed in [`frontend/README.md`](frontend/README.md).

```bash
pip install -e "ml/[dev]" -r backend/requirements.txt   # Python 3.11+
pytest ml/tests
uvicorn backend.app.main:app --port 8000              # API docs at http://localhost:8000/docs
```

## Limitations

- **The edibility labels come from an LLM (Claude), not from experts.** UNCERTAIN mostly means the LLM had
  low confidence (24 of 25 species). Some labels may be wrong. AVOID mixes 2 deadly species with many
  species that are only poisonous or inedible.
- **The backbone was fine-tuned on the same DF20-Mini training split**, so the embeddings are an upper bound,
  not a true cold-start setting. A frozen backbone also leaves less room for active learning to help. Training
  the backbone inside the loop was not tested.
- One dataset (Danish fungi), one classifier, 5 seeds. The differences in bucket accuracy between strategies
  are small.
- **Demo app:** the Random Forest reaches about 0.9999 test accuracy on a random split of the simulated data.
  This number is not meaningful: the simulated rows come from only 173 species, so test rows are very
  similar to training rows. The classifier is also overconfident (often 100%) when Gemini returns only a few
  features.

## References

- Picek, L. et al. *Danish Fungi 2020: Not Just Another Image Recognition Dataset.* WACV 2022.
- Wagner, D., Heider, D. and Hattab, G. *Mushroom data creation, curation, and simulation to support
  classification tasks.* Scientific Reports 11 (2021).
