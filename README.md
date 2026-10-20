# Enhancer benchmark audit

Reproducibility materials for **Benchmark audit and model evaluation for enhancer prediction: data overlap, training adequacy and assay transfer**.

This repository supports the manuscript-facing benchmark audit, grouped enhancer classification, quantitative lentiMPRA activity modelling, training-duration controls, statistical synthesis and external assay-transfer evaluation.

## Start here

If you want to reproduce the **current manuscript results**, use the six manuscript-facing notebooks or the command-line wrappers below. You do **not** need the historical project archive for the main paper.

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

The reporting route uses saved scored predictions and locked reference results. It does **not** retrain models. For dataset reconstruction, checkpoint inference and explicit full training, follow [docs/REPRODUCTION_GUIDE.md](docs/REPRODUCTION_GUIDE.md).

## What this repository reproduces

| Notebook | Manuscript coverage |
|---|---|
| `01_benchmark_audit.ipynb` | Exact/RC overlap audit, fixed group isolation, classification part of Tables 2–3 and Figure 4 |
| `02_classification.ipynb` | Three-seed and archived single-run classification; Table 4 and Figures 5–6 |
| `03_activity_models.ipynb` | All 54 activity fits, validation-only recipe selection, internal metrics and own-prefix duration control; Table 5 and Figure 7 |
| `04_external_transfer.ipynb` | DHS-natural transfer; Table 7 and Figure 8 |
| `05_statistics.ipynb` | Fourteen headline contrasts, conditional bootstrap/permutation tests and Holm adjustment; Table 6 and Supplementary Table S18 |
| `06_figures_and_tables.ipynb` | Current main figures, seven main tables, 19 supplementary tables and supplementary figures |

Table 1 and Supplementary Table S19 are built from preserved, cited literature records. Table 2 combines classification and activity populations.

## Repository structure

```text
configs/       fixed recipes, seeds, grids, selection rules and input/model locks
data/          source URLs, checksums, opaque identity/fold manifests and acquisition notes
docs/          provenance, validation, reproduction scope and release documentation
figures/       current main and supplementary figure exports
notebooks/     six manuscript-facing notebooks plus a clearly separated historical archive
results/       derived manifests, compact prediction files, tables and fit logs
scripts/       reporting commands and explicit optional reconstruction/training wrappers
src/           reporting/statistical code plus frozen implementation sources
tests/         numerical and provenance checks
```

## Validation status

Release-preparation checks are recorded in [docs/VALIDATION.json](docs/VALIDATION.json). The current release-preparation state includes:

- execution of all six manuscript-facing notebooks;
- reproduction of all 14 headline statistical contrasts from fixed fitted predictions;
- fresh checkpoint inference for 60 classification fold sets and 60 activity evaluation sets;
- reconstruction checks for the legacy benchmark, ENCODE activity data and frozen external cohort;
- authenticated reuse of the 60 classification and 54 activity completed jobs without new optimizer steps.

The validation record also states what was **not** repeated. In particular, no fresh full 114-fit retraining campaign and no independent reinstall/test of the CUDA training environment were performed during release preparation.

## Data and model artifacts

Original legacy DNA strings and large fitted model weights are not stored in Git history. Public GitHub content retains hashes, folds, manifests, derived results and compact prediction records. Source acquisition and third-party terms are documented in [data/README.md](data/README.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

A separate Zenodo artifact deposit is planned for larger authenticated inputs/checkpoints. Until that record is actually published, no Zenodo DOI or record ID should be inferred from placeholders or planning files. See [docs/ZENODO_RELEASE.md](docs/ZENODO_RELEASE.md).

## Statistical interpretation

The repository preserves the manuscript's stated limits:

- previously examined cohorts are not treated as prospectively untouched confirmatory tests;
- reported bootstrap intervals condition on fixed fitted models and evaluation cohorts;
- three seeds, finite recipe grids and unequal compute do not establish convergence;
- ensemble results are reported separately from single-seed results;
- positive external rank correlation does not imply calibrated absolute activity prediction or native enhancer function.

## Reproduction routes beyond saved-result reporting

The repository distinguishes four levels of reproducibility:

1. **CPU reporting:** regenerate manuscript metrics, statistics, tables and figures from saved predictions.
2. **Raw-data reconstruction:** rebuild legacy, ENCODE activity and external cohorts from authenticated source artifacts.
3. **Fresh checkpoint inference:** rescore authenticated fitted checkpoints in the pinned GPU environment.
4. **Full fitting:** explicitly retrain the current transformer/activity studies using the frozen training sources and declared recipes.

See [docs/REPRODUCTION_GUIDE.md](docs/REPRODUCTION_GUIDE.md) for tested commands and scope notes.

## Historical research archive

The repository also preserves the original notebook/source history under `notebooks/original_phases/` and archived supporting sources under `src/archive/`. This material documents how the project evolved, including superseded analyses, recovery versions and failed branches.

**These historical files are not the recommended reproduction interface for the manuscript.** The six notebooks listed above and the locked current results are authoritative for the submitted study. [docs/PHASE_INDEX.md](docs/PHASE_INDEX.md) distinguishes historical findings from current manuscript evidence and records known corrections such as the Phase 38 collapse-count erratum.

The archive preserves 62 main notebooks plus five distinct saved/recovery versions and 83 distinct supporting code files. [docs/ORIGINAL_SOURCE_INVENTORY.json](docs/ORIGINAL_SOURCE_INVENTORY.json) and [docs/HISTORICAL_ARCHIVE_VALIDATION.json](docs/HISTORICAL_ARCHIVE_VALIDATION.json) record the preserved source inventory and validation state.

## Citation and rights

Citation metadata are in [CITATION.cff](CITATION.cff). The repository is public for reproducibility, but the root [LICENSE](LICENSE) currently keeps original study code **all rights reserved** until the authors select a reuse license. Third-party components and datasets retain their own terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `third_party/licenses/`.

A versioned GitHub release and Zenodo DOI should be cited once the manuscript release is frozen and the archive is published.

## Release preparation

Before creating `v1.0.0`, the maintainers should:

1. review the final manuscript-facing files and validation report;
2. confirm the intended license/rights statement for original code;
3. verify the Zenodo artifact plan and metadata;
4. create the versioned GitHub release;
5. publish the corresponding Zenodo record;
6. replace manuscript DOI placeholders with the actual archived DOI.
