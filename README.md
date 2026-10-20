# Enhancer benchmark audit

Reproducibility materials for **Benchmark audit and model evaluation for enhancer prediction: data overlap, training adequacy and assay transfer**.

This repository contains the manuscript-facing code, fixed manifests, predictions, statistics and figure/table generation used for the final study.

## Quick start

```bash
conda env create -f environment.yml
conda activate enhancer-audit

python -B -X utf8 scripts/reproduce.py audit
python -B -X utf8 scripts/reproduce.py classification
python -B -X utf8 scripts/reproduce.py activity
python -B -X utf8 scripts/reproduce.py external
python -B -X utf8 scripts/reproduce.py statistics --full-statistics
python -B -X utf8 scripts/reproduce.py figures-and-tables
```

These commands reproduce the manuscript-facing analyses from the saved, locked predictions and reference results. They do not retrain models.

## Manuscript notebooks

| Notebook | Main purpose |
|---|---|
| `notebooks/01_benchmark_audit.ipynb` | Benchmark overlap and fixed-group checks |
| `notebooks/02_classification.ipynb` | Enhancer classification evaluation |
| `notebooks/03_activity_models.ipynb` | MPRA activity-model evaluation and training-duration control |
| `notebooks/04_external_transfer.ipynb` | DHS-natural external transfer |
| `notebooks/05_statistics.ipynb` | Bootstrap, permutation and Holm-adjusted headline contrasts |
| `notebooks/06_figures_and_tables.ipynb` | Main/supplementary figures and tables |

## Repository structure

```text
configs/       fixed protocols, seeds, recipes and model/input locks
data/          acquisition notes, hashes, duplicate/fold manifests and MPRA splits
docs/          provenance, validation and reproduction guidance
figures/       manuscript figure exports
notebooks/     six manuscript-facing notebooks
results/       compact predictions, reference inputs, tables and fit summaries
scripts/       reproduction, reconstruction and checkpoint/training wrappers
src/           reporting/statistical code and frozen implementation sources
tests/         numerical and provenance checks
```

## Validation

The release-preparation validation record is in [`docs/VALIDATION.json`](docs/VALIDATION.json). It documents execution of the six manuscript notebooks, reproduction of the 14 headline statistical contrasts, checkpoint-based classification/activity evaluation checks and source-data reconstruction checks.

The validation record also states the limits of release preparation. In particular, it does not claim a fresh rerun of the complete training campaign.

## Data and larger model artifacts

Large raw/source artifacts and fitted checkpoints are not stored in Git history. The repository keeps the compact manifests, hashes, splits, predictions and derived results needed to audit the study. Source acquisition details are in [`data/README.md`](data/README.md), with third-party terms in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

The manuscript reproducibility release is archived on Zenodo:

- **Version v1.0.0 DOI:** https://doi.org/10.5281/zenodo.23503625
- **Concept DOI (all versions):** https://doi.org/10.5281/zenodo.23503624

For the manuscript, cite the version-specific DOI above so the archived software snapshot is fixed.

## Reproduction details

For raw-data reconstruction, checkpoint inference or explicit retraining, see [`docs/REPRODUCTION_GUIDE.md`](docs/REPRODUCTION_GUIDE.md). The reporting environment is defined by `environment.yml` / `requirements.txt`; legacy preparation requirements are separated where exact historical package behavior matters.

## Interpretation limits

The repository preserves the manuscript's stated evaluation limits: previously examined cohorts are not described as prospectively untouched confirmatory tests, bootstrap intervals condition on fixed fitted models/evaluation cohorts, ensemble and single-seed results are kept distinct and positive external rank correlation is not treated as calibrated absolute activity prediction.

## Citation and rights

Citation metadata are in [`CITATION.cff`](CITATION.cff). The root [`LICENSE`](LICENSE) applies to original study code. Third-party software, pretrained models and datasets retain their own terms; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

**Repository:** https://github.com/AryanAditta/enhancer-benchmark-audit
**Zenodo:** https://doi.org/10.5281/zenodo.23503625
