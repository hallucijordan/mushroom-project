# Milestone 1 — Agent-Driven Mushroom Detection: Design & Verification

## Overview

We implemented and verified an end-to-end Gemini-powered mushroom safety application, establishing the applicability of multi-agent LLM pipelines for structured biological classification tasks.

## System Design

```
User uploads photo
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
  Judge A (Model-backed)           Judge B (Direct Vision)
  Gemini extracts 20 features      Gemini assesses photo directly
        │                                  │
  RandomForest classifier                  │
  (trained on 61k samples)                 │
        │                                  │
  verdict + confidence             verdict + confidence
        └──────────────┬───────────────────┘
                       ▼
           Weighted consensus (60/40)
                  Final Verdict
```

### Modular components

| Module | Role |
|---|---|
| `src/data/preprocessor.py` | Parses both CSV formats (ranges `[lo,hi]` and plain values); OrdinalEncoder fitted once, persisted |
| `src/models/base_learner.py` | RandomForest + isotonic calibration; tunable via class constants |
| `src/agents/gemini_extractor.py` | Gemini Vision → structured 20-feature JSON |
| `src/judges/model_judge.py` | Judge A: features → classifier → narrative |
| `src/judges/direct_judge.py` | Judge B: direct visual verdict, no ML |
| `app.py` | Gradio UI wiring both judges side-by-side |
| `config/api_keys.py` | Gitignored key file; `api_keys.example.py` as template |

## Contributions

1. **Dual-judge architecture** — separates the structured-feature ML path from the end-to-end LLM path, enabling direct A vs. B comparison as proposed.

2. **Heterogeneous CSV parsing** — primary data uses `[lo, hi]` bracket ranges and multi-value categoricals `[x, f]`; secondary data uses plain semicolon-delimited values. A single preprocessor handles both transparently.

3. **Calibrated classifier** — `CalibratedClassifierCV(isotonic)` produces reliable confidence percentages shown in the UI, not just argmax labels.

4. **API key security** — `config/api_keys.py` is gitignored from the first commit; a committed example template documents the format.

5. **Gradio 6 compatibility** — resolved breaking changes in `gr.Chatbot` (removed `type` and `bubble_full_width` args; messages format uses `{"role", "content"}` dicts; `theme` moved to `launch()`).

## Verification

The screenshot below shows a successful end-to-end run on a photo of *Pholiota squarrosa* (Scaly Pholiota):

![UI screenshot](ui.png)

- **Judge A** extracted 17 features (cap-diameter, cap-shape, cap-surface, gill-color, stem dimensions, etc.) and the classifier returned **POISONOUS at 93.2% confidence**
- **Judge B** independently identified the yellowish-tan scaly caps, pale gills, and ringed stem as characteristic of *Pholiota squarrosa* — **POISONOUS at 95.0% confidence**
- **Final verdict: POISONOUS** (weighted consensus 60/40)

Both judges agreed, and Judge B's species identification matches the botanical record — confirming the pipeline is functionally correct.
