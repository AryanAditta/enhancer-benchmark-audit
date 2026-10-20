# Third-party sources and rights

Original study code is supplied under the root LICENSE notice; a permissive license remains an author decision. A collective contributor citation is used until the manuscript author list and software citation metadata are finalized.

## Model code

- MPRA-LegNet: autosome-ru/human_legnet, commit a20dbc7d0b08daae3af1aebcfb8a71d9f198ab5d. MIT; preserve third_party/licenses/LegNet_MIT.txt. The copied architecture is in src/frozen/architectures.py and regression_architectures.py.
- MPRAnn: visze/sequence_cnn_models, commit 7638fce137db0445123efe8d7e2c35e248fafc5f, standard() architecture, ported from Keras to PyTorch. MIT; preserve third_party/licenses/MPRAnn_MIT.txt. The port is not a claim of bit-identical training.
- DNABERT-2: MAGICS-LAB/DNABERT-2-117M, revision ec1f874253852eb3907081f57294991b4280ceb6. The upstream Apache 2.0 text is retained in third_party/licenses/DNABERT2_APACHE_2.txt. Obtain pretrained files separately; no pretrained weights are included.
- DNABERT: zhihan1996/DNA_bert_6, revision c56e67ea5827e0ddc67ef059addcf71569b1216e. Obtain model files and comply with their upstream terms; no pretrained weights are included.

## Data

- iEnhancer-2L legacy benchmark: this public copy omits the original DNA strings. It provides derived hashes, fixed folds, labels, predictions and audit counts. Obtain the original files from their source under the applicable terms and authenticate against the manifests before a sequence-level rerun.
- ENCODE lentiMPRA: cite Agarwal et al., Nature 639, 411-420 (2025), DOI 10.1038/s41586-024-08430-9; HepG2 ENCFF774DYO / ENCSR022GQD and K562 ENCFF631LHA / ENCSR382BVV. ENCODE policy: https://www.encodeproject.org/about/data-use-policy/ and https://www.encodeproject.org/help/rest-api/.
- DHS-natural external cohort: Gosai et al., Machine-guided design of cell-type-targeting cis-regulatory elements, Nature 634, 1211-1220 (2024), DOI 10.1038/s41586-024-08070-z, Supplementary Table 12. Retain attribution to the source publication and its applicable license. This copy contains derived scores/observations and opaque group labels, not raw sequence strings.

DNA-valued resampling group IDs are replaced by ordinal opaque labels in the same lexicographic order. Group membership, sorted group order, scores, observations and random-draw assignment are preserved. The transformation ledger documents this separately from source hashes.

Python libraries and fonts are dependencies installed separately under their own terms. They are not relicensed or bundled as runtime binaries. The manuscript draft, author emails, reviewer correspondence, private study notes, model caches, optimizer states and full project archives are not uploaded.
