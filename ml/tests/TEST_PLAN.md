# ML Package Test Plan

Run all tests from the `ml/` directory:
```bash
cd ml
pip install -e ".[dev]"
pytest tests/ -v
```

---

## test_learners.py — Learner Implementations

Tests run against all three learners (`random_forest`, `logistic`, `mlp`) via `@pytest.mark.parametrize`.

| # | Test | What it checks |
|---|------|----------------|
| L1 | `test_fit_returns_self` | `fit()` returns the learner instance (fluent API) |
| L2 | `test_predict_shape_and_values` | `predict()` returns shape `(n,)` with values in `{0, 1}` |
| L3 | `test_predict_proba_shape` | `predict_proba()` returns shape `(n, 2)` with rows summing to 1.0 |
| L4 | `test_get_uncertainty_shape_and_nonneg` | `get_uncertainty()` returns shape `(n,)` with non-negative values |
| L5 | `test_predict_single_output_format` | `predict_single()` returns dict with correct keys and valid probability values |
| L6 | `test_save_and_load` | Saved and reloaded learner produces identical predictions |
| L7 | `test_get_learner_unknown_key` | `get_learner()` raises `ValueError` for unknown key |
| L8 | `test_learner_registry_keys` | Registry contains exactly `{random_forest, logistic, mlp}` |

---

## test_strategies.py — Query Strategies

Tests run against all four strategies (`random`, `uncertainty`, `committee`, `bald`) via `@pytest.mark.parametrize`.

| # | Test | What it checks |
|---|------|----------------|
| S1 | `test_query_returns_correct_count` | `query()` returns exactly `n_samples` indices |
| S2 | `test_query_indices_in_range` | All returned indices are valid (within pool bounds) |
| S3 | `test_query_no_duplicates` | No duplicate indices in a single query |
| S4 | `test_query_returns_ndarray` | Return type is `np.ndarray` |
| S5 | `test_get_strategy_unknown_key` | `get_strategy()` raises `ValueError` for unknown key |
| S6 | `test_strategy_registry_keys` | Registry contains exactly `{random, uncertainty, committee, bald}` |

---

## test_preprocessor.py — MushroomPreprocessor

| # | Test | What it checks |
|---|------|----------------|
| P1 | `test_fit_transform_output_shape` | Output shape is `(n_samples, n_features)` |
| P2 | `test_fit_transform_no_nans` | No NaN values in output after fit_transform |
| P3 | `test_transform_matches_fit_transform` | `transform()` on same data gives same result as `fit_transform()` |
| P4 | `test_transform_before_fit_raises` | `transform()` before `fit_transform()` raises `RuntimeError` |
| P5 | `test_nan_numeric_imputed` | NaN numeric values are imputed with median |
| P6 | `test_bracket_range_parsed` | `"[2, 4]"` format parsed to midpoint `3.0` |
| P7 | `test_feature_dict_to_array` | `feature_dict_to_array()` returns shape `(1, n_features)` |
| P8 | `test_save_and_load` | Saved and reloaded preprocessor produces identical output |

---

## test_al_session.py — ALSession

| # | Test | What it checks |
|---|------|----------------|
| A1 | `test_initial_labeled_count` | Labeled set size matches initial data size |
| A2 | `test_initial_pool_size` | Pool size is total minus initial |
| A3 | `test_history_recorded_after_init` | History has one entry with `cv_accuracy` after construction |
| A4 | `test_query_batch_returns_valid_indices` | Batch query returns valid pool indices |
| A5 | `test_label_sample_moves_from_pool` | Pool shrinks and labeled set grows by 1 after `label_sample()` |
| A6 | `test_label_sample_result_format` | `label_sample()` returns dict with correct keys and types |
| A7 | `test_retrain_appends_history` | `retrain()` appends a new snapshot to history |
| A8 | `test_predict_sample_output_format` | `predict_sample()` returns valid prediction dict |
| A9 | `test_add_external_sample_grows_labeled` | External sample is added to labeled set |
| A10 | `test_get_learning_curve_structure` | Learning curve is a list of dicts with expected keys |
