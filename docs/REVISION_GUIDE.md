# Revision Guide for `main.tex` (Results & Discussion)

Audience: a future agent (or human collaborator) editing the Results and
Discussion sections of `main.tex`. Read this before changing numbers,
swapping figures, or restructuring the narrative.

---

## 1. File map — where every number came from

All numbers in the Results section are pulled from the three notebooks and
the CSVs they save into `results/` and `results/figures/`. **If you change
a number in `main.tex`, change it from a regenerated CSV — do not edit
prose in isolation.**

| Section | Number | Source |
|---|---|---|
| 5.1 species/image bucket counts | 21/37/96/25 species, 5444/7639/20725/2553 images | `00_data_analysis.ipynb` cells 4 & 9 |
| 5.1 per-genus composition | Mycena uniform AVOID, etc. | `00_data_analysis.ipynb` cell 6 |
| 5.1 AVOID internal split | 52 poisonous / 42 inedible / 2 deadly | `00_data_analysis.ipynb` cell 14 |
| 5.1 LLM confidence | 96% UNCERTAIN low, 81% SAFE high | `00_data_analysis.ipynb` cell 18 |
| 5.2 LR sanity check | train 0.981, test 0.626 | `01b_embedding_picekvit.ipynb` cell 16 |
| 5.2 / 5.3 full-sup ceiling | species 0.607/0.484, bucket 0.824/0.744 hard, 0.831/0.750 soft | `02b_active_learning_picekvit.ipynb` cell 8 |
| 5.4 final-round table | per-strategy means | `results/picekvit_final_summary.csv` |
| 5.5 label efficiency | budgets to reach 80/90/95% | `results/picekvit_label_efficiency.csv` |
| 5.6 Pareto / safety | bucket_acc vs UNC recall | `results/uncertain_summary.csv` and `results/full_summary.csv` |
| 5.7 per-genus species accuracy | 6 strategies × 6 genera | `results/picekvit_per_genus.csv` |
| 5.7 Amanita AVOID recall 0.904 | per-genus per-bucket recall | `results/genus_bucket_summary.csv` |

If a CSV is missing or stale, re-run the notebook that produces it
(notebooks are numbered in dependency order — 00 → 01b → 02b).

---

## 2. Figure inventory — what is referenced and where it lives

The Results section currently embeds these eight figures. Paths are
relative to the repository root, which is the `\graphicspath` pdfLaTeX
will see when compiling `main.tex` from the project root.

| LaTeX label | File path | Source notebook |
|---|---|---|
| `fig:bucket-pie` | `results/figures/species_vs_image_bucket_pie.png` | `00` cell 10 |
| `fig:genus-bucket` | `results/figures/species_per_genus_bucket.png` | `00` cell 7 |
| `fig:confidence-bucket` | `results/figures/confidence_per_bucket.png` | `00` cell 18 |
| `fig:learning-curves-hard` | `results/picekvit_learning_curves_hard.png` | `02b` cell 20 |
| `fig:learning-curves-soft` | `results/picekvit_learning_curves_soft.png` | `02b` cell 21 |
| `fig:bucket-recall` | `results/picekvit_per_bucket_recall.png` | `02b` cell 23 |
| `fig:label-eff` | `results/fig_label_efficiency.png` | `02b` cell 28 |
| `fig:pareto` | `results/fig8_uncertain_pareto.png` | `02b` cell 32 |
| `fig:genus-heatmap` | `results/fig11_per_genus_heatmap.png` | `02b` cell 32 |

Available but currently unused (consider for an appendix or to swap in if
a referenced figure is missing):

- `results/dataset_species_gallery.png` — 179-species mosaic, useful for a
  dataset-overview teaser figure if the page budget allows.
- `results/figures/avoid_internal_structure.png` — bar chart of the AVOID
  sub-classes; supports Section 5.1 if the reviewer asks for AVOID detail.
- `results/figures/long_tail_per_bucket.png` — per-bucket species
  long-tail. Good for a methods appendix.
- `results/fig15_safety_buffer.png`, `results/fig9_avoid_recall_curve.png`,
  `results/fig6_uncertain_recall_curve.png` — alternative safety
  visualisations.
- `results/fig5_stage_rank_heatmap.png` — strategy ranking by AL stage
  (early/mid/late); supports the "BALD wins late" sub-claim in the
  Discussion if you want to surface it.

Do **not** reference figures from `results/figures/` and `results/`
interchangeably — they are different directories. Use the table above.

---

## 3. Known inconsistencies to watch out for

1. **Bucket distribution numbers in Background vs. Results.** The
   Background (Section 4.1) currently says "roughly 12% SAFE, 20%
   COOK\_FIRST, 55% AVOID, 13% UNCERTAIN" *at the image level*, but those
   numbers are actually the **species-level** percentages. The image-level
   numbers (which we report in Results) are 15 / 21 / 57 / 7. Either
   correct the Background or rephrase Results to match — but pick one
   convention before submission.

2. **Budget mismatch in Notebook docstrings vs. actual run.** The header
   of `02b_active_learning_picekvit.ipynb` (markdown cell 0) says
   "AL budget: 3000 labels (9.2%)", but the actual config in cell 2 sets
   `MAX_LABELS = 5000` and the numbers in `main.tex` are based on the
   $5{,}000$-label run. Update the notebook header, not the LaTeX.

3. **Picek's reported species accuracy.** The Background quotes Picek et
   al. as "$0.72$ top-1, $0.74$ macro F1" for ViT-B/16-224. The actual
   paper reports $0.7203$ Recall\@1 and $0.7395$ macro F1 for that
   backbone. Numbers are correct; just verify the citation if asked.

4. **Confidence breakdown missing "medium" for some buckets.** In Section
   5.1, the prose says "$81\%$ of SAFE species came back as \emph{high}".
   The remaining $19\%$ are \emph{medium}, not \emph{low} (no SAFE species
   was rated low). This is consistent in the prose but will look wrong if
   someone tries to add up to 100% from low+high alone.

---

## 4. How to safely change a number

1. Re-run the relevant notebook end-to-end (or just the affected cell if
   you know the dependencies).
2. Confirm the CSV(s) listed in §1 have actually been overwritten
   (check `ls -la results/*.csv`).
3. Re-grep `main.tex` for the old number to make sure every occurrence
   is updated. Numbers are repeated across Results, Discussion,
   Significance, and the Abstract-style opening of Section 5; updating
   only one location is the most common failure mode here.
4. If a strategy ranking flips (e.g. margin overtakes BALD on bucket
   accuracy), update the **prose claims** in Section 5.4, the **Discussion
   §6.3 narrative about "Why BALD wins"**, and the **recommendation in
   §6.3 (margin as default)** — they are all interlocking.

---

## 5. How to add a new figure

1. Save the figure into `results/` (or `results/figures/` for the
   data-analysis figures from notebook `00`).
2. Pick a *unique* LaTeX label of the form `fig:<short-name>`. The current
   labels are listed in §2 — do not collide.
3. Use `\includegraphics[width=...\linewidth]{results/<file>.png}`. Stick
   to fractions of `\linewidth` so the figure scales with the column.
   Default width: `0.85` for portrait-oriented figures, `0.95` for
   landscape ones.
4. Reference the figure in prose with `Figure~\ref{fig:<short-name>}` (the
   non-breaking tilde matters).
5. Add a row to §2 of this guide.

---

## 6. How to add a new AL strategy or a new metric

If a future revision adds a seventh strategy or a new metric column:

- The strategy must be added to the `STRATEGIES` dict in
  `02b_active_learning_picekvit.ipynb` cell 17 and given a stable name.
  All downstream CSV columns and figure legends are keyed off that name.
- After re-running cell 18 (the main loop) and cell 25 (the summary
  table), regenerate `results/picekvit_final_summary.csv`,
  `results/picekvit_label_efficiency.csv`, and the figures listed in §2.
- In `main.tex`, add a new row to **Table 4** (`tab:final-summary`),
  **Table 5** (`tab:label-eff`), and **Table 6** (`tab:genus`). Each
  table also needs the bold-best updated.
- Decide whether the new strategy belongs on the Pareto plot
  (`fig:pareto`); if so, regenerate `results/fig8_uncertain_pareto.png`.

For a new metric (e.g. AVOID precision instead of recall):

- Add it to the `evaluate(...)` function in `02b` cell 11.
- Add a column to `results/picekvit_final_summary.csv`.
- Add it to Table 4 if it is a headline number, or to a new sub-table if
  it is a diagnostic.

---

## 7. Section-by-section editing notes

**Section 5.1 (Edibility hierarchy)** — Mostly safe to edit prose. Do not
change the bucket counts without re-running notebook `00`. The
`fig:confidence-bucket` figure is the load-bearing evidence that
UNCERTAIN is a real epistemic category; if it is removed, the "research
signal" claim in §5.1 and §6 must also be softened.

**Section 5.2 (Sanity check + ceiling)** — The full-sup ceiling
(Table 1) is the comparison baseline used throughout the rest of the
Results. If it changes, every "X% of full-sup" budget in Table 5 must be
recomputed by re-running `02b` cell 27.

**Section 5.3 (Learning curves)** — The two figures
(`fig:learning-curves-hard`, `fig:learning-curves-soft`) are saved by
`02b` cells 20 and 21. The shaded bands are $\pm 1$ std across five seeds;
if the seed count changes, update the caption text.

**Section 5.4 (Final summary)** — Bolded entries in Table 4 are the per-
column maximum. There is currently a tie at species accuracy ($0.520$,
margin and BALD) that is bolded twice, which is intentional.

**Section 5.5 (Label efficiency)** — The "N/A" entry for core-set at the
$95\%$ hard target is real (no seed reached the threshold). Do not
silently fill it in; flag it explicitly.

**Section 5.6 (Per-bucket recall + Pareto)** — The Pareto frame is the
single most important conceptual claim of the paper; both the figure and
the prose claim "no single strategy dominates" must travel together.

**Section 5.7 (Per-genus)** — Numbers come from
`results/picekvit_per_genus.csv` (species accuracy per genus per strategy)
and `results/genus_bucket_summary.csv` (per-bucket recall per genus,
strategy-averaged). The $0.904$ Amanita AVOID-recall headline is the
strategy-averaged soft-decoding number from `genus_bucket_summary.csv`.

**Section 6 (Discussion)** — Each subsection is built on a specific
quantitative claim from §5. If you change the claim in §5, change the
matching paragraph in §6:

| §6 subsection | Depends on §5 claim |
|---|---|
| 6.1 Soft decoding | Table 1 hard-vs-soft delta + §5.7 Amanita number |
| 6.2 No best strategy | Table 4 spread + Pareto figure |
| 6.3 Why BALD wins / loses | Table 4 BALD row + UNCERTAIN recall column |
| 6.4 Headroom is in the backbone | LR test acc 0.626 vs full-sup 0.607 |
| 6.5 Implications for deployment | AVOID recall ~0.90 + Amanita 0.904 |
| 6.6 Limitations | Honest caveats — usually safe to edit |

---

## 8. Compilation notes

- The document uses Times New Roman (`mathptmx`), $1$-inch margins,
  double spacing, and `\usepackage{booktabs}` for tables — all already in
  the preamble. Do not switch table styles.
- Equations 1 and 2 (`\eqref{eq:hard}` and `\eqref{eq:soft}`) are
  referenced from both Background, Results §5.7, and Discussion §6.1.
  Renumbering would break three call sites.
- The bibliography is hand-curated in `\section*{References}`. If you
  add a citation, update the numbered list at the bottom and the in-text
  bracket reference.

---

## 9. Quick sanity checklist before submitting any edit

- [ ] Did you re-run the source notebook for any number you changed?
- [ ] Did you grep for the old number in `main.tex` to catch duplicates?
- [ ] Do the figure paths in `\includegraphics` actually exist?
- [ ] Does the bolded "best" entry in each table still match the column max?
- [ ] If a strategy-ranking claim moved, did you update both Results and
  Discussion?
- [ ] Did `pdflatex main.tex` complete without missing-file or
  undefined-reference warnings?
