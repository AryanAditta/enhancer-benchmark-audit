# Supporting source history

This archive contains 83 distinct code files representing all 129 inventoried project-source copies from the original Phases 12–49 and subsequent presentation workspace. Source files are byte-identical; identical copies share a destination. The original source paths and SHA-256 values are recorded in [ORIGINAL_SOURCE_INVENTORY.json](../../docs/ORIGINAL_SOURCE_INVENTORY.json).

`phases12_49_and_presentation/analysis_tools/` retains the main analysis, training, verification and packaging helpers. `phases12_49_and_presentation/presentation/` contains the later figure, table, caption and manuscript formatting scripts. `phases12_49_and_presentation/runtime_validation/` preserves the optimizer/AMP/RNG resumption check. `phases12_49_and_presentation/snapshots/` preserves other and differing historical source versions, using short hash-prefixed filenames to avoid collisions and Windows path-length problems.

Files retain their original imports, machine paths, external inputs and output expectations. No historical code was executed for this upload. Use `scripts/restore_historical_sources.py` to restore the inventoried source layout into a new empty directory; provide the original authenticated artifacts separately before any historical execution. Main frozen training sources and current reporting modules also remain in their existing repository locations.

Upstream LegNet/MPRAnn source copies retain their original licenses and the notices in `THIRD_PARTY_NOTICES.md`. Installed Python libraries, model-library caches and model weights are external dependencies, rather than additional authored project source.
