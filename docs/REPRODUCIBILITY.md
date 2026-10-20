# Reproduction scope

## Original source history

[PHASE_INDEX.md](PHASE_INDEX.md) maps all four work stages, all 62 main original notebooks and five additional saved/recovery versions to the current six reproduction notebooks. `ORIGINAL_SOURCE_INVENTORY.json` accounts for all 145 notebook source paths and 129 supporting-code source paths. Byte-identical backups share an archive destination; every differing version is retained, together with its source path, SHA-256 and byte size. Original notebook code, explanatory text, metadata and saved outputs are unchanged.

This archive includes superseded statistical/reporting snapshots, historical test evaluations, early short-duration recipes, the separate 9C locus branch and failed/interrupted attempts. Their historical output/status text is preserved and must be interpreted using the index and [Phase 38 reporting erratum](history/ERRATUM_collapse_count.md). Original scripts retain their machine-specific paths and dependencies. Archive validation checks file identity, notebook schemas and Python syntax; it does not assert that every historical notebook executes in this public reporting environment or validates every historical result anew.

`scripts/restore_historical_sources.py` recreates every inventoried path, including duplicate backup paths, into a new empty directory outside the repository. It checks hashes before and after copying and executes no historical source. Full historical execution additionally needs the original authenticated input/model artifacts and original path/environment assumptions. The six current entry notebooks below remain the tested saved-result reproduction workflow.

## CPU reporting

The six ordered notebooks operate on bundled scored records, frozen settings and compact results. They reproduce benchmark hash/count checks, classification and activity metrics, validation-only recipe choice, the headline statistical family, and publication assets without training. Outputs go to results/reproduced and .runtime, which are ignored by Git.

Public benchmark manifests omit raw DNA. Hash/GC fields are derived from the authenticated original sequence strings. Audit/count and fixed-group isolation checks are rerunnable from those fields. A fresh all-pairs sequence-similarity search requires obtaining the original DNA. Source snapshots show the original sequence-based implementation; do not describe a hash audit as a fresh exhaustive LCS rescan.

DNA-valued MPRA group IDs are replaced by ordinal opaque labels preserving membership and sorted order, so the exact resampling assignment is retained. See docs/public_copy_transformations.json. Source hashes and public-copy hashes are different where this documented transformation occurs.

The primary statistical reproduction uses the preserved 2,000 bootstrap and 9,999 permutation draws for each of 14 headline comparisons. The own-prefix secondary sensitivity has its separate 1,000-draw protocol. Conditional intervals exclude retraining and recipe-search uncertainty; three seeds, dependent folds, previous cohort examination and limited candidate grids remain disclosed. No outcome selects new training or alters a grid.

## Full training and checkpoint inference

Byte-identical frozen training, architecture and evaluation sources are included under src/frozen. The public source copy does not bundle pretrained/fitted weights, raw datasets, optimizer states or every historical source archive. Full training therefore requires the exact separately obtained artifacts, authenticated by the supplied manifests. It is not claimed to have been rerun during repository preparation.

The original implementation expects a study-shaped working directory and an environment named enhancer. prepare_training_workspace.py constructs a separate directory with those paths from supplied authentic artifacts. The frozen source is never patched in place. Classification uses 60 fixed two-epoch fits; MPRA uses all 54 fixed candidate/seed fits and preserves DNABERT-2 own two-epoch prefix checkpoints. Fresh GPU runs can vary across hardware/software; the fixed-prediction reporting route targets the recorded results.

## Figure reproduction

The current manuscript has eight main groups as 14 separate figure objects and eight supplementary figure images. Current figure exports and captions are in figures/. The drawing sources for the current main figures are retained and invoked through a portable adapter. Original historical plotting sources are retained separately for provenance. Supplied supplementary image extraction is an exact-copy operation, not a claim that a new model or new data were generated.

## Attribution and publication

README and LICENSE distinguish public code availability from an author-selected permissive license. CITATION.cff uses collective repository contributors rather than asserting a final manuscript author list. No Zenodo record, DOI, release tag or acceptance status is claimed. Enable the repository in Zenodo before creating the archival GitHub release.

## Complete scored-prediction records

The public copy includes all 72 MPRA validation-prediction sets, all 60 selected/control seed evaluation sets and the 20 ensemble sets. Notebook 03 independently recomputes the validation correlations, seed metrics and ensemble means/SDs without loading model checkpoints. Classification includes all 12 pooled single-seed sets as well as the 18 ensemble/archived-model rows.

The seven historical classification baselines remain archived single-run predictions; unavailable original fitting code/checkpoints are not recreated. Main figure numeric plotting records are checked independently of font-dependent raster pixels. Canonical figure exports remain unchanged.
