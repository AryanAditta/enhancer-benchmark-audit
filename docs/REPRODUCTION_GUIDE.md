# Reproducing the study

Run commands from the repository root. Keep the bundled reference results read-only. Generated files belong in the ignored `.runtime/`, `data/artifacts/`, `data/training_workspace/` and `results/reproduced/` directories. No command below changes the study recipes or selects settings from test results.

## 1. Reporting and manuscript analyses (CPU)

Create a repository-local environment with Python 3.12.13:

```powershell
conda create --prefix .runtime/reporting-env python=3.12.13 pip -y
conda activate ./.runtime/reporting-env
python -B -X utf8 -m pip install -r requirements-reporting-lock-win-py312.txt
python -B -X utf8 scripts/verify_release.py
python -B -X utf8 scripts/run_notebooks.py
```

The complete reporting dependency lock was tested on Windows/Python 3.12 in a newly populated, isolated package directory with host site-packages excluded from the Jupyter kernels. Other platforms may use `requirements.txt`; that is a portability route rather than a claim of a tested fresh operating-system installation.

The six current notebooks reproduce the sequence/hash audit, validation-only recipe selection, metrics, 14 headline statistical comparisons, tables and figures. Notebook 05 defaults to the complete 2,000-bootstrap/9,999-permutation calculations. The executed copies are saved separately. Historical notebooks are never executed by this command. For individual commands, see the repository README.

This route uses saved scored predictions. The following routes repeat earlier steps rather than claiming that six reporting notebooks alone retrain the whole project.

## 2. Obtain authenticated inputs and fitted models

`configs/zenodo_artifacts.json` describes the prepared, versioned artifact set. Its record ID and DOI intentionally remain null until actual Zenodo publication. The software cannot download an unpublished record. Once the deposit exists, use its real numeric record ID:

```powershell
python -B -X utf8 scripts/download_artifacts.py --record-id ACTUAL_RECORD_ID --components inputs
python -B -X utf8 scripts/download_artifacts.py --record-id ACTUAL_RECORD_ID --components all --execute
```

The first command only prints a plan. `--components inputs --execute` obtains inputs without fitted models; `activity` or `classification` obtains the corresponding fitted models plus inputs. The downloader authenticates each container and each extracted member against SHA-256 and refuses conflicting existing files. Full acquisition needs space for approximately 44 GB of downloads plus extracted files; it does not delete previous work to make room. Container/member paths and byte sizes are public even before deposit publication.

Maintainers can test acquisition without a network using `--local-upload-plan PATH_TO_PRIVATE_PLAN`; absolute PC paths and access tokens are deliberately excluded from GitHub.

Source attribution and upstream terms remain in [data/README.md](../data/README.md) and [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md). Do not infer a new data/weight license from the root original-code rights notice.

## 3. Rebuild datasets from raw sources

Use a separate environment because the original audit recorded **scikit-learn 1.6.1**. Version 1.8.0 changes shuffled `StratifiedGroupKFold` assignments. The source audit's 1.6.1 setting reproduces every accepted five- and ten-fold assignment exactly. Existing frozen manifests remain authoritative and unchanged.

```powershell
conda create --prefix .runtime/preparation-env python=3.12.13 pip -y
conda activate ./.runtime/preparation-env
python -B -X utf8 -m pip install -r requirements-preparation.txt
python -B -X utf8 scripts/rebuild_datasets.py --artifact-root data/artifacts --output results/reproduced/raw_preparation
```

A working `g++` must already be on PATH for the original exhaustive LCS implementation. The adapter authenticates source notebooks and inputs, executes their original preparation cells, redirects paths and uses the study's preserved read-only TwoBit reader. It rebuilds:

- the eight legacy FASTA sources, exact/reverse-complement overlap, all 5,338,278 Stage 1 sequence pairs, LCS candidate groups, independent gapped alignment and the accepted candidate manifests;
- 76,833 ENCODE activity rows with the original chromosome assignments;
- 29,330 frozen external rows before the declared primary cohort filtering (8,399 per cell).

Numerical values and row order are compared with the frozen originals. CSV byte hashes can differ through platform line endings and float serialization; the validation records report content equality and byte identity separately. Newly rebuilt CSVs do not silently replace the byte-authenticated training inputs. No private Word documents from the legacy source bundle are required or redistributed.

## 4. Fresh inference from fitted checkpoints (GPU)

Use the pinned CUDA training environment. A local prefix named `enhancer` satisfies the frozen implementation's original environment-name check:

```powershell
conda env create --prefix .runtime/enhancer -f configs/training_environment.yml
conda activate ./.runtime/enhancer
python -B -X utf8 scripts/prepare_training_workspace.py --source-root data/artifacts
python -B -X utf8 scripts/reproduce_from_checkpoints.py --workspace data/training_workspace --artifact-root data/artifacts --execute
```

Requires a compatible NVIDIA CUDA GPU; original runs used a 16 GB RTX 5060 Ti. This authenticates all inputs, implementation copies, checkpoints and evaluation IDs before scoring. Old prediction files are read only after fresh inference, for comparison. The release check reproduced all **60 classification fold sets and 60 activity evaluation sets**, with maximum absolute prediction difference **0.0** in the original GPU software/hardware environment. This is not an independently installed GPU-environment test. Other hardware can differ numerically; explicit tolerances and reports are retained.

## 5. Full fitting (GPU, substantial runtime)

Prepare a **new** workspace if the inference workspace already exists. Original training is explicit opt-in:

```powershell
python -B -X utf8 scripts/prepare_training_workspace.py --source-root data/artifacts --output data/full_fit_workspace
python -B -X utf8 scripts/retrain.py --task classification --workspace data/full_fit_workspace --execute
python -B -X utf8 scripts/retrain.py --task activity --workspace data/full_fit_workspace --execute
```

Classification runs the fixed 60 two-epoch jobs for two encoders, two stages, three seeds and five folds. Activity runs 54 fits for three models, two cells, three declared learning rates and three seeds, then authenticates completion **before** validation-only selection and test/external inference. Use `--training-only` to stop activity after authenticated training. Same-workspace continuation reuses authenticated completed jobs and the frozen implementation's resume policy; it never chooses recipes from test outcomes.

The public launcher fixes a missing `finish_training()` orchestration call without modifying frozen training code. Its tests verify ordering and failure handling. A separate integration check executed real setup, reuse/authentication of all 114 existing jobs and both completion routines while optimizer steps were prohibited. This is a completed-job reuse test, **not** a new full fitting experiment. No full 114-fit retraining or independent installation of the CUDA environment was performed during release preparation. The original study's finite probes and serialized-resume checks remain preserved.

## 6. Scope and historical baselines

The two transformer classification models and three prospective activity models have frozen fitting code and fitted-checkpoint manifests. Seven earlier single-run classification baselines contribute archived predictions; their checkpoint completeness and exact fitting provenance differ. See [BASELINE_PROVENANCE.md](BASELINE_PROVENANCE.md). Do not interpret reporting reproduction as fresh fitting of all seven historical baselines.

The 67 archived original notebooks and 83 supporting code versions preserve superseded branches, failures, machine paths and output text. [PHASE_INDEX.md](PHASE_INDEX.md) distinguishes their historical status. They are valuable research records, not 67 promises of portable one-click execution. `scripts/restore_historical_sources.py` reconstructs every inventoried original path into a new empty destination after checking hashes.

Previously examined cohorts, conditional intervals, three seeds, unequal compute and limited candidate grids remain limitations. Reproduction does not turn these analyses into a fresh confirmatory study.
