# Classification model provenance

All nine reported model rows retain their scored records. The primary three-seed transformer fits are a consistent implementation; older single-run baselines remain separately identified. The archive preserves source availability without claiming that a template or matching configuration proves the exact historical executed fit.

| Model | Public record and fitting provenance | Fitted-weight / inference status |
|---|---|---|
| Fine-tuned DNABERT-2 | Frozen Phase 31 code, protocol and all 30 job records; three seeds over two stages/five folds | 30 checkpoint hashes in the artifact plan; fresh inference checked for every fold |
| Fine-tuned DNABERT | Same consistent Phase 31 implementation and all 30 jobs | 30 checkpoint hashes; fresh inference checked for every fold |
| GC logistic regression | Earlier single-run baseline prediction records and preserved original source history | No separately authenticated fitted baseline object in this release; no fresh fit/inference claimed |
| 3-mer logistic regression | Earlier single-run baseline prediction records and preserved original source history | Same limitation |
| Spaced 6-mer/4-position logistic regression | Earlier single-run baseline prediction records and preserved original source history | Same limitation |
| Small CNN | Earlier single-run baseline prediction records and preserved original source history | Same limitation |
| Frozen DNABERT | Earlier frozen-embedding single-run scores; original notebook history retained | No complete authenticated historical fitted head/embedding acquisition route verified in this release |
| Frozen DNABERT-2 | Earlier frozen-embedding single-run scores; original notebook history retained | Same limitation |
| Frozen GENA-LM | Preserved Phase 2 correction contains actual StandardScaler/logistic-regression fitting on original/RC embeddings | Code is available; a complete authenticated original embedding/backbone/head bundle is not established here; no fresh fitting/inference claimed |

The complete source archive is indexed in [PHASE_INDEX.md](PHASE_INDEX.md). In particular, it would be incorrect to say that all historical baseline fitting code is unavailable: the GENA-LM correction is present. It would also be incorrect to promise exact reconstruction of every original baseline fit from saved scores alone.

Reported single-seed means, seed SDs, fold variability and ensemble comparisons remain distinct. Historical predictions are not added to the primary transformer seed pool. This release adds no outcome-driven baseline refitting or new comparison.
