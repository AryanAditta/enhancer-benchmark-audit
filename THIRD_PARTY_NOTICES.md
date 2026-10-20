# Third-party notices

The root `LICENSE` applies only to original study code in this repository. Third-party software, pretrained models and datasets retain their own upstream licenses and terms.

## Model and software sources

- **LegNet**: `autosome-ru/human_legnet`, commit `a20dbc7d0b08daae3af1aebcfb8a71d9f198ab5d` (MIT upstream). The published architecture was reused for the MPRA comparison without architectural changes.
- **MPRAnn / sequence CNN models**: `visze/sequence_cnn_models`, commit `7638fce137db0445123efe8d7e2c35e248fafc5f` (MIT upstream). The Keras `standard()` model was ported to PyTorch for this study.
- **DNABERT-2**: `MAGICS-LAB/DNABERT-2-117M`, revision `ec1f874253852eb3907081f57294991b4280ceb6`. Upstream model/code terms apply.
- **DNABERT**: `zhihan1996/DNA_bert_6`, revision `c56e67ea5827e0ddc67ef059addcf71569b1216e`. Upstream model/code terms apply.

Pinned model-file information used by the study is recorded in the configuration manifests under `configs/`.

## Data sources

- **Legacy enhancer benchmark**: derived from the iEnhancer benchmark literature. Raw DNA strings are not redistributed in Git. Public records include hashes, overlap/group information, fold assignments and saved predictions needed to audit the reported analyses.
- **ENCODE lentiMPRA**: Agarwal et al., *Nature* (2025), DOI `10.1038/s41586-024-08430-9`. HepG2: `ENCFF774DYO` / `ENCSR022GQD`; K562: `ENCFF631LHA` / `ENCSR382BVV`.
- **DHS-natural external cohort**: Gosai et al., *Nature* (2024), DOI `10.1038/s41586-024-08070-z`, Supplementary Table 12.

Source locations and acquisition notes are in `data/README.md`. No blanket relicensing of third-party data, model weights or software is implied by this repository. Larger authenticated artifacts may be archived separately with the manuscript release where redistribution is permitted.
