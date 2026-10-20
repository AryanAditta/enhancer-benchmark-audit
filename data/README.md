# Source data and acquisition

This repository tracks the compact hashes, manifests, split/group assignments, predictions and derived records needed to reproduce and audit the manuscript. It does not place all raw DNA sequences, pretrained weights or large fitted checkpoints in Git history.

## ENCODE lentiMPRA

- HepG2: https://www.encodeproject.org/files/ENCFF774DYO/ (experiment `ENCSR022GQD`)
- K562: https://www.encodeproject.org/files/ENCFF631LHA/ (experiment `ENCSR382BVV`)

These measurements are described by Agarwal et al. (2025), DOI `10.1038/s41586-024-08430-9`.

## DHS-natural external cohort

The external DHS-natural elements come from Gosai et al. (2024), DOI `10.1038/s41586-024-08070-z`, Supplementary Table 12. The manuscript's fixed filtering yields 8,399 evaluation elements per cell type.

## Legacy enhancer benchmark

Obtain the legacy iEnhancer benchmark source files from the cited benchmark sources and authenticate the sequence content against `data/manifests/legacy_hashes.csv` before a fresh sequence-level audit. A live download location is not, by itself, evidence of byte identity with the supplied copies audited in the manuscript.

## Reconstruction and larger artifacts

`configs/activity_input_manifest.csv`, the model/pretrained manifests and the run locks record the exact study inputs/configuration expected by the optional reconstruction, checkpoint-inference and retraining scripts. See `docs/REPRODUCTION_GUIDE.md` for those deeper routes.

Third-party data/model terms remain with their original providers; see `THIRD_PARTY_NOTICES.md`. A stable Zenodo manuscript release can archive the public repository and larger authenticated artifacts separately where redistribution is permitted.
