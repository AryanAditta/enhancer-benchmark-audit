# Source data and acquisition

The reporting route works from the bundled compact predictions, derived manifests and reference tables. No raw legacy DNA strings or model weights are distributed.

ENCODE source files:
- https://www.encodeproject.org/files/ENCFF774DYO/ (HepG2; ENCSR022GQD)
- https://www.encodeproject.org/files/ENCFF631LHA/ (K562; ENCSR382BVV)

DHS-natural source: Gosai et al. (2024), DOI https://doi.org/10.1038/s41586-024-08070-z, Supplementary Table 12. Use the measured activity columns and the frozen inclusion/split records; the retained cohort has 8,399 elements per cell.

Legacy iEnhancer benchmark files: obtain the exact source version and compare sequence hashes with data/manifests/legacy_hashes.csv before a sequence-level audit. The original source filenames and identities are in the derived manifests. A live download location is not a guarantee of byte identity.

configs/activity_input_manifest.csv records exact derived training-input filenames, hashes and sizes; configs/pretrained_file_manifest.csv records pinned model files. Some derived historical inputs/checkpoints require a separate artifact deposit. No DOI or public artifact URL is invented here. The original local study contains these files; a versioned Zenodo data/checkpoint archive is a separate follow-up.

scripts/prepare_training_workspace.py copies an authenticated, separately obtained study-input tree into a new data/training_workspace directory. It never edits the supplied input tree. scripts/retrain.py is an explicit opt-in wrapper around byte-identical frozen training source; it refuses to start if required inputs are missing or mismatched. See docs/REPRODUCIBILITY.md.
