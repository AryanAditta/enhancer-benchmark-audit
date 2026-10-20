# Reproducing the study

Run commands from the repository root. The fastest supported route reproduces the manuscript analyses from the fixed predictions and reference records; the raw-data and GPU routes are optional deeper checks.

## 1. Manuscript reproduction

Create the reporting environment and run the public stages:

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

The same reporting route is available through the six notebooks in `notebooks/`. `scripts/verify_release.py` checks the compact release inputs and frozen implementation hashes before deeper reproduction work.

## 2. Optional raw-data reconstruction

Obtain the source data listed in `data/README.md`, authenticate them against the supplied manifests/hashes and use the dedicated preparation environment:

```bash
python -B -X utf8 -m pip install -r requirements-preparation.txt
python -B -X utf8 scripts/rebuild_datasets.py --artifact-root data/artifacts --output results/reproduced/raw_preparation
```

The final legacy audit recorded scikit-learn 1.6.1 for exact shuffled grouped-fold recreation. Later versions can change fold assignments, so the frozen manifests remain authoritative for the manuscript.

## 3. Optional checkpoint inference

Fresh model scoring requires the separately obtained fitted checkpoints/pretrained weights and a compatible NVIDIA CUDA environment. After authenticating the artifacts against the configuration manifests:

```bash
python -B -X utf8 scripts/prepare_training_workspace.py --source-root data/artifacts
python -B -X utf8 scripts/reproduce_from_checkpoints.py --workspace data/training_workspace --artifact-root data/artifacts --execute
```

This route is optional because the manuscript-facing reporting package already contains the saved scored predictions.

## 4. Optional full retraining

Full retraining is explicit opt-in and computationally expensive:

```bash
python -B -X utf8 scripts/prepare_training_workspace.py --source-root data/artifacts --output data/full_fit_workspace
python -B -X utf8 scripts/retrain.py --task classification --workspace data/full_fit_workspace --execute
python -B -X utf8 scripts/retrain.py --task activity --workspace data/full_fit_workspace --execute
```

The study uses fixed seeds, folds and finite candidate grids. Fresh GPU runs may differ numerically across hardware/software. No fresh rerun of the entire training campaign was performed during public-release preparation.

## 5. Interpretation limits

Reproduction does not convert previously examined cohorts into prospective confirmatory tests. The reported bootstrap/permutation analyses condition on the fixed fitted models and evaluation cohorts and do not include full retraining, model-selection or pretraining uncertainty.
