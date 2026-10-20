# Source data and acquisition

The reporting route works from the bundled compact predictions, derived manifests and reference tables. No raw legacy DNA strings or model weights are stored in Git; they are separately inventoried for the prepared artifact deposit.

ENCODE source files:
- https://www.encodeproject.org/files/ENCFF774DYO/ (HepG2; ENCSR022GQD)
- https://www.encodeproject.org/files/ENCFF631LHA/ (K562; ENCSR382BVV)

DHS-natural source: Gosai et al. (2024), DOI https://doi.org/10.1038/s41586-024-08070-z, Supplementary Table 12. Use the measured activity columns and the frozen inclusion/split records; the retained cohort has 8,399 elements per cell.

Legacy iEnhancer benchmark files: obtain the exact source version and compare sequence hashes with data/manifests/legacy_hashes.csv before a sequence-level audit. The original source filenames and identities are in the derived manifests. A live download location is not a guarantee of byte identity.

configs/activity_input_manifest.csv records exact derived training-input filenames, hashes and sizes; configs/pretrained_file_manifest.csv records pinned model files. Some derived historical inputs/checkpoints require a separate artifact deposit. No DOI or public artifact URL is invented here. The original local study contains these files; the versioned Zenodo data/checkpoint upload plan is now prepared locally, pending actual deposit publication. See [docs/ZENODO_RELEASE.md](../docs/ZENODO_RELEASE.md).

scripts/prepare_training_workspace.py copies an authenticated, separately obtained study-input tree into a new data/training_workspace directory. It never edits the supplied input tree. scripts/retrain.py is an explicit opt-in wrapper around byte-identical frozen training source; it refuses to start if required inputs are missing or mismatched. See docs/REPRODUCIBILITY.md.

The preparation artifact list also includes the exact legacy FASTA-only repack, original exhaustive-audit sources, hg38 reference and publisher external-source workbook. [REPRODUCTION_GUIDE.md](../docs/REPRODUCTION_GUIDE.md) describes content-verified reconstruction. The original final legacy audit used scikit-learn 1.6.1; install requirements-preparation.txt in its separate environment to reproduce its final folds.
