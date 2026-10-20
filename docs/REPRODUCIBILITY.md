# Reproduction scope

## Manuscript-facing reporting

The six notebooks in `notebooks/` and the stages exposed by `scripts/reproduce.py` operate on fixed scored records, protocol locks, manifests and compact result files. They reproduce the benchmark audit/count checks, classification and MPRA metrics, validation-only recipe choices, the 14 headline statistical contrasts and the manuscript tables/figures without retraining models.

Generated outputs are written outside the bundled reference inputs, primarily under `results/reproduced/` and `.runtime/`.

## Data reconstruction

Git contains the manifests, hashes, fold/split assignments, saved predictions and derived records needed for manuscript auditing, but it does not contain every raw DNA sequence or large model artifact. Source acquisition is documented in `data/README.md`.

A fresh sequence-level rebuild therefore requires separately obtaining the source data and running the preparation route described in `REPRODUCTION_GUIDE.md`. Exact legacy grouped-fold recreation uses the recorded preparation environment rather than assuming that later scikit-learn versions generate identical shuffled folds.

## Checkpoint inference and retraining

`src/frozen/` preserves the final training/evaluation implementations used for the principal fitted models. Fitted checkpoints and pretrained weights are intentionally not kept in Git history. Fresh checkpoint inference or full retraining requires the separately authenticated artifacts, a compatible GPU environment and the explicit optional scripts described in the reproduction guide.

These GPU routes are not required to regenerate the manuscript tables and figures from the released predictions.

## Validation record

`docs/VALIDATION.json` summarizes the release-preparation checks. The six manuscript-facing notebooks were executed; all 14 headline statistical contrasts reproduced with zero recorded discrepancy; checkpoint inference checks covered 60 classification fold sets and 60 activity evaluation sets with no new optimizer steps; and raw legacy/activity/external reconstruction checks passed.

A fresh rerun of the complete training campaign was not performed during repository preparation. The validation therefore should not be interpreted as adding retraining, model-selection or pretraining uncertainty to the reported inferential intervals.

## Statistical and evaluation scope

The released analyses preserve the manuscript's stated limits. Inference is conditional on fixed fitted models and the evaluated cohorts. Previously examined internal/external cohorts are not described as prospectively untouched confirmatory tests. Single-seed and ensemble results remain distinct, and positive external rank correlation is not treated as evidence of calibrated absolute reporter activity.
