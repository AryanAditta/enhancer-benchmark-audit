# Enhancer benchmark audit

Code and compact results supporting **Benchmark audit and model evaluation for enhancer prediction: data overlap, training adequacy and assay transfer**.

The study audits the supplied iEnhancer-2L train/test overlap, evaluates grouped Stage 1/Stage 2 classification, compares DNABERT-2, MPRA-LegNet and MPRAnn on measured lentiMPRA activity, tests DNABERT-2 training duration against its own two-epoch prefix, and evaluates DHS-natural assay transfer.

## Setup

```bash
conda env create -f environment.yml
conda activate enhancer-audit
```

Tested reporting versions are pinned in environment.yml and requirements.txt. The original GPU study environment is documented separately in configs/training_environment.yml; it requires the appropriate CUDA PyTorch build and authenticated model/data artifacts.

## Reproduce saved-result analyses

```bash
python -B -X utf8 scripts/reproduce.py audit
python -B -X utf8 scripts/reproduce.py classification
python -B -X utf8 scripts/reproduce.py activity
python -B -X utf8 scripts/reproduce.py external
python -B -X utf8 scripts/reproduce.py statistics --full-statistics
python -B -X utf8 scripts/reproduce.py figures-and-tables
```

Alternatively, run notebooks/01_benchmark_audit.ipynb through 06_figures_and_tables.ipynb. The full statistical command repeats the declared Monte Carlo calculations and can take substantially longer than checking point estimates. Generated output stays in results/reproduced; the supplied reference files are read-only inputs.

## Structure

```text
configs/       fixed recipes, seeds, grids, selection and input/model locks
data/          source URLs, checksums, opaque identity/fold manifests and acquisition notes
src/           reporting functions, statistics, exact frozen training sources and drawing code
notebooks/     six ordered public reproduction notebooks
scripts/       commands for reporting and explicit, authenticated optional retraining
results/       derived manifests, small prediction files, all main/supplementary tables and fit logs
figures/       current main and supplementary figures with captions
docs/          reproduction limits, provenance and public-copy transformation ledger
tests/         numerical and provenance checks
```

## Notebook map

| Notebook | Coverage |
|---|---|
| 01_benchmark_audit | Exact/RC hash overlap, fixed group isolation; classification part of Tables 2-3 and Figure 4 |
| 02_classification | Three-seed and archived single-run classification; Table 4 and Figures 5-6 |
| 03_activity_models | All 54 fit records, validation-only recipes, internal metrics and own-prefix duration; Table 5 and Figure 7 |
| 04_external_transfer | DHS-natural transfer; Table 7 and Figure 8 |
| 05_statistics | Fourteen headline contrasts, conditional bootstrap/permutation tests and Holm adjustment; Table 6 and S18 |
| 06_figures_and_tables | Reproduce current main figure drawings and assemble all seven main/19 supplementary tables and supplementary images |

Table 1 and S19 are built from preserved, cited literature records. Table 2 combines both classification and activity populations.

## Data and models

Read data/README.md and THIRD_PARTY_NOTICES.md. Original legacy DNA and large model weights are not uploaded. The public copy retains hashes/folds and derived results. DNA-valued resampling groups are replaced by opaque labels in identical sorted order. Full GPU retraining requires separate authenticated artifacts; see docs/REPRODUCIBILITY.md and the opt-in preparation/training scripts. No new training is performed by the reporting notebooks.

## Interpretation

Previously examined cohorts are not fresh confirmatory tests. Intervals condition on fitted models; equal candidate counts do not imply equal compute or exhaustive tuning. Retain undefined metrics, all fit records, epoch/grid-boundary flags and the two descriptive near-constant fits. Positive external ranking does not establish calibrated absolute activity or native enhancer function.

## License, citation and archive

Original code currently retains all rights reserved until the authors select a reuse license. Upstream MIT/Apache notices remain in third_party/licenses; data retain their source terms. See LICENSE and CITATION.cff. A Zenodo DOI will be added after the repository is connected and a fixed archival release is created. This initial GitHub upload does not invent a DOI or finalize manuscript authorship.

## Complete scored-prediction records

The public copy includes all 72 MPRA validation-prediction sets, all 60 selected/control seed evaluation sets and the 20 ensemble sets. Notebook 03 independently recomputes the validation correlations, seed metrics and ensemble means/SDs without loading model checkpoints. Classification includes all 12 pooled single-seed sets as well as the 18 ensemble/archived-model rows.

The seven historical classification baselines remain archived single-run predictions; unavailable original fitting code/checkpoints are not recreated. Main figure numeric plotting records are checked independently of font-dependent raster pixels. Canonical figure exports remain unchanged.
