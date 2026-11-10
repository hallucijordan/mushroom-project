# Mushroom Project: Label-Efficient Fungi Classification with Active Learning

Course project for 02-750 Automation of Scientific Research (Carnegie Mellon University, Spring 2026),
by Luci Lu and Ziyan Zhang.

The main question: **how many labeled images does a mushroom edibility classifier need, and can active
learning reduce that number compared with random labeling?** We test 6 query strategies on 36,361 fungus
images (179 species) with 5 random seeds each.

A smaller second part is a Gemini Vision demo app that judges a mushroom photo two ways (see
[Part 2](#part-2-gemini-demo-app-secondary)).

> **Research demo only. Never eat a wild mushroom based on AI advice.**

## Main result

| | Labels | Bucket accuracy (4 classes) |
|---|---:|---:|
| Majority-class baseline | 0 | 0.570 |
| Full supervision (whole training pool) | 32,724 | 0.817 |
| Margin sampling: labels to reach 90% of full supervision | **1,300** (4% of the pool) | 0.735 |
| Random sampling: labels to reach 90% of full supervision | 1,380 | 0.735 |

Margin sampling reached the 90% target slightly earlier than random sampling. **Most other uncertainty-based
strategies needed more labels than random** (table below). With a strong pretrained backbone, active learning
gave only a small advantage in this setting.

## Repository layout

| Path | Part | What it is |
|---|---|---|
| `00_data_analysis.ipynb` | 1 | Edibility label analysis: species to 4 buckets, per-genus and per-image distributions |
| `01b_embedding_picekvit.ipynb` | 1 | Frozen ViT-B/16 embeddings (768-dim) for all 36,361 images |
| `02b_active_learning_picekvit.ipynb` | 1 | Full-supervision ceiling, 6 strategies x 5 seeds, learning curves, label efficiency |
| `main.ipynb` | 1 | Active-learning experiments on the tabular UCI mushroom data |
| `app.py`, `train.py`, `src/` | 2 | Gemini demo app |
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
  on our side.
- **Classifier:** MLP (512-256) with MC dropout (p = 0.2), trained on species (179 classes). Species
  probabilities are summed into buckets ("soft") or the top species decides the bucket ("hard").
- **Active learning loop:** 400 initial labels (about 2 per species, stratified), then batches of 100 up to
  3,000 labels (9.2% of the pool). Strategies: random, entropy, margin, coreset, class-balanced entropy, BALD.
  5 seeds each.

### Results

Full supervision (all 32,724 labels): species top-1 accuracy 0.591, bucket accuracy 0.817 (hard) and 0.821
(soft).

Labels needed to reach a share of the full-supervision bucket accuracy (mean over 5 seeds; all seeds reached
each target):

| Target | Random | Entropy | **Margin** | Coreset | CB entropy | BALD |
|---|---:|---:|---:|---:|---:|---:|
| 80% (0.653) | 480 | 540 | **480** | 600 | 560 | 500 |
| 90% (0.735) | 1,380 | 1,660 | **1,300** | 2,260 | 1,900 | 1,480 |

Bucket accuracy at 3,000 labels (hard, mean over 5 seeds): BALD 0.769, margin 0.768, entropy 0.763,
random 0.758, CB entropy 0.753, coreset 0.753. The spread is about 1.6 points.

### How to reproduce

Data and model files are not in the repository (too large):

| File | Source |
|---|---|
| `data/DF20M/` (images), `data/DF20M-metadata/` | DF20-Mini from the [Danish Fungi dataset](https://github.com/BohemianVRA/DanishFungiDataset) |
| `artifacts/DF20M-ViT_base_patch16_224_best_accuracy.pth` | Downloaded automatically by notebook 01b |
| `artifacts/species_edibility_map.csv` | Edibility table for the 179 species, generated with Claude (Anthropic) (not included) |
| `artifacts/df20m_metadata.csv` | Cleaned metadata from an earlier CLIP-based pipeline (not included). Notebook 01b rebuilds the same table from the DF20-Mini metadata as `df20m_metadata_picekvit.csv`; notebook 00 reads it under the old name. |

Run order: `01b_embedding_picekvit` (embeddings, 20-40 min on Apple M4), then `00_data_analysis` (writes the
species-to-bucket table), then `02b_active_learning_picekvit` (30 runs, about 20 s each).

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

## Limitations

- **The edibility labels come from an LLM (Claude), not from experts.** UNCERTAIN mostly means the LLM had
  low confidence (24 of 25 species). Some labels may be wrong.
- **The backbone was fine-tuned on the same DF20-Mini training split**, so the embeddings are an upper bound,
  not a true transfer-learning setting. A strong backbone also leaves less room for active learning to help.
- One dataset, one classifier, frozen features, 5 seeds. The differences between strategies are small.
- **Demo app:** the Random Forest reaches about 0.9999 test accuracy on a random split of the simulated data.
  This number is not meaningful: the simulated rows come from only 173 species, so test rows are very
  similar to training rows. The classifier is also overconfident (often 100%) when Gemini returns only a few
  features.

## References

- Picek, L. et al. *Danish Fungi 2020: Not Just Another Image Recognition Dataset.* WACV 2022.
- Wagner, D., Heider, D. and Hattab, G. *Mushroom data creation, curation, and simulation to support
  classification tasks.* Scientific Reports 11 (2021).
