# Classification model provenance

All nine reported classification rows retain the scored records used in the manuscript. The repeated transformer fits have complete locked job/checkpoint provenance; several older single-run baselines are retained as archived prediction records and should not be described as freshly refitted by this repository.

| Model | Public fitting/evaluation provenance |
|---|---|
| Fine-tuned DNABERT-2 | 30 fixed fits: 2 stages × 5 folds × 3 seeds, with checkpoint hashes and fold-level scored records |
| Fine-tuned DNABERT | 30 fixed fits: 2 stages × 5 folds × 3 seeds, with checkpoint hashes and fold-level scored records |
| GC/CpG logistic regression | Single-run saved prediction/evaluation records; no new refit claimed |
| 3-mer logistic regression | Single-run saved prediction/evaluation records; no new refit claimed |
| Spaced 6-mer/4-position logistic regression | Single-run saved prediction/evaluation records; no new refit claimed |
| Small CNN | Single-run saved prediction/evaluation records; no new refit claimed |
| Frozen DNABERT | Saved frozen-embedding prediction/evaluation records; historical fitted-artifact completeness differs from the repeated transformer runs |
| Frozen DNABERT-2 | Saved frozen-embedding prediction/evaluation records; historical fitted-artifact completeness differs from the repeated transformer runs |
| Frozen GENA-LM | Saved prediction/evaluation records with preserved fitting implementation; no claim of a complete authenticated historical backbone/head bundle in Git |

The manuscript therefore keeps repeated-seed transformer results, ensemble results and older single-run baselines explicitly distinct. This release does not add outcome-driven baseline refitting or change any reported model ranking.
